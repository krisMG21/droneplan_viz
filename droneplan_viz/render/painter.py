"""
Painter: orquestador del render. API pública del paquete.

Dos funciones públicas:

- render_snapshot(surface, snapshot, *, theme=None):
    Dibuja un WorldSnapshot estático sobre la Surface. Es la "foto
    completa" del mundo en un instante.

- render_frame(surface, snap_a, snap_b, progress, *, theme=None):
    Dibuja un frame interpolado entre dos snapshots consecutivos.
    progress=0.0 equivale a render_snapshot(snap_a); progress=1.0
    a render_snapshot(snap_b); valores intermedios producen un frame
    "a medio camino" según el tipo de transición clasificado por
    classify_transition().

Orden de dibujo (z-order de menor a mayor):

    1. Fondo (background).
    2. Aristas del grafo de costes (líneas entre Locations).
    3. Locations (círculos grandes).
    4. Transporters y personas (entidades estáticas asociadas a una loc).
    5. Paquetes libres (AtLocation) y paquetes dentro de transporters.
    6. Drones y sus brazos.
    7. Paquetes sostenidos por brazos (encima del drone).
    8. Overlays de estado (X de ERROR ya viene aplicada por draw_drone).

Cada capa se dibuja completa antes de pasar a la siguiente, así nunca
queda un drone "debajo" de una Location. Los paquetes en brazo se pintan
después de los drones para que queden visualmente sobre el brazo.

Filosofía:

- Funciones puras: reciben todos los datos por parámetro, no mantienen
  estado, no cachean nada. Si el render se llama 60 veces por segundo,
  cada llamada es independiente.
- compute_layout() se invoca dentro de cada render_*. No optimizamos
  con cache porque para los tamaños de escenarios docentes el coste
  es despreciable. Si en algún caso se midiera overhead real, se podría
  exponer un compute_layout pre-calculado como argumento opcional.
- El theme por defecto es Theme.default() si el caller no pasa uno.
- Cero pygame.event, cero pygame.display: este módulo NO crea ventanas,
  NO consume eventos. Solo dibuja sobre la Surface recibida.

Sobre interpolación (render_frame):

- progress saneado: <= 0 dibuja snap_a, >= 1 dibuja snap_b. Para evitar
  ambigüedad en los extremos, usamos las cotas estrictas internamente.
- classify_transition(snap_a, snap_b) decide qué interpolar:
    * Static / Failure → dibuja snap_b sin animación (es el "destino").
    * DroneMove → interpola la posición del drone (y del transporter
      si es MoveWithTransporter, y los paquetes dentro de él).
    * PackageMove → interpola la posición del paquete entre los dos
      "places". El drone queda en su loc, en estado INTERACTING.
- Se aplica easing cúbico (ease_in_out_cubic) a la posición interpolada
  para que la animación arranque y termine suavemente. La paleta y
  el resto del world se renderiza sin animación (no interpolamos colores,
  no fundimos).

Sobre el world base que se usa de fondo en render_frame:

- Para DroneMove/PackageMove: dibujamos las Locations, transporters,
  personas y paquetes USANDO snap_a.world. Razón: el snap_b ya tiene
  los efectos aplicados (drone en destino, paquete en su nuevo sitio);
  si dibujáramos snap_b.world y luego sobreescribiéramos la posición
  del drone interpolada, todo lo demás (paquetes que ya cambiaron de
  manos, personas que recibieron, etc.) aparecería como si la acción
  ya hubiera terminado. Mostrar snap_a.world tiene la ventaja de que
  la animación es "estado de origen + entidad que se mueve", lo cual
  es perceptualmente correcto.
- Excepción: el drone que se mueve, su transp (en MoveWithTransporter),
  y/o el paquete (en PackageMove) NO se dibujan en su posición de
  snap_a; se dibujan en la posición interpolada.
"""
from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

import pygame

from droneplan_viz.domain import (
    AtLocation,
    HeldByArm,
    InTransporter,
    World,
)
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.history import WorldSnapshot
from droneplan_viz.render.geometry import (
    Point,
    ViewBox,
    ease_in_out_cubic,
    lateral_tilt_deg,
    lerp_point,
)
from droneplan_viz.render.interpolation import (
    TransitionDroneMove,
    TransitionFailure,
    TransitionPackageMove,
    TransitionStatic,
    classify_transition,
)
from droneplan_viz.render.layout import (
    WorldLayout,
    _CARRIER_MAX,
    _box_capacity,
    _box_depth_key,
    _carrier_depth_key,
    _colocation_groups,
    _packages_at_loc,
    compute_layout,
)
from droneplan_viz.render.sprites import (
    draw_background,
    draw_drone,
    draw_drone_arms,
    draw_edge,
    draw_location,
    draw_package,
    draw_person,
    draw_transporter,
    position_in_transporter,
    position_of_arm,
    set_labels_visible,
)
from droneplan_viz.render.theme import Theme

if TYPE_CHECKING:
    from droneplan_viz.render.sprite_manager import SpriteManager


def _delivered_package_ids(packages, persons) -> "set[str]":
    """Ids de paquetes ya ENTREGADOS (consumidos por una persona).

    Tras un Deliver, el paquete queda AtLocation(persona.position) y su
    contenido pasa a `has_received` de la persona. Visualmente la entrega la
    representa el CAMBIO DE SPRITE de la persona (variante con caja), así que
    el paquete NO debe seguir pintándose como caja en el suelo (si no,
    aparecería duplicado: outline en el suelo + caja en el sprite de la
    persona). Aquí detectamos esos paquetes para omitirlos.

    Detección por loc con MULTIPLICIDAD: por cada location, las personas allí
    aportan un multiset de contenidos recibidos; un paquete AtLocation casa
    (y se marca entregado) si su contenido tiene presupuesto en ese multiset,
    consumiéndolo. Así, si por algún escenario hubiese una caja del mismo
    contenido ESPERANDO en la misma loc que otra ya entregada, solo se
    suprime la entregada. El orden es estable (alfabético, vía
    _packages_at_loc) para que el reparto del presupuesto sea determinista.
    """
    if not persons:
        return set()
    received_by_loc: dict[str, dict[str, int]] = {}
    for person in persons.values():
        if not person.has_received:
            continue
        budget = received_by_loc.setdefault(person.position, {})
        for content in person.has_received:
            budget[content.id] = budget.get(content.id, 0) + 1
    if not received_by_loc:
        return set()
    delivered: set[str] = set()
    for loc_id, ids_here in _packages_at_loc(packages).items():
        budget = dict(received_by_loc.get(loc_id, {}))
        if not budget:
            continue
        for package_id in ids_here:
            cid = packages[package_id].contains.id
            if budget.get(cid, 0) > 0:
                budget[cid] -= 1
                delivered.add(package_id)
    return delivered


def _draw_resting_packages(
    surface,
    packages,
    layout,
    theme: Theme,
    sprite_manager,
    *,
    persons=None,
    exclude=(),
):
    """Dibuja las cajas LIBRES (AtLocation) apiladas en sus locations.

    Punto ÚNICO de pintado de la pila de cajas en reposo: todos los caminos
    del painter (estático y animados) pasan por aquí para garantizar el mismo
    criterio. Por cada location:

      - El índice de slot de cada caja es su posición alfabética entre las
        AtLocation de esa loc (idéntico a `WorldLayout.package_position`, que
        usa el mismo `_packages_at_loc`). El índice se calcula sobre la lista
        COMPLETA —incluidas las excluidas— para no descuadrar los slots.
      - Solo se dibujan las `_box_capacity()` primeras (capas visibles); las
        que rebasan comparten la posición de la caja central de la última capa
        y NO se cargan/dibujan hasta que afloran. Antes esto solo lo hacía el
        camino estático: los animados pintaban TODAS (sombras apiladas en el
        slot central) y sin ordenar.
      - Se pinta atrás→adelante por `_box_depth_key` (capas enteras), no en el
        orden de inserción del dict, para que la oclusión sea correcta.

    `persons`: si se pasan, las cajas ya ENTREGADAS (consumidas por una
        persona co-localizada) NO se dibujan en el suelo — la entrega la
        representa el sprite de la persona (evita la caja duplicada). El
        índice de slot sigue calculándose sobre la lista completa.
    `exclude`: ids que no se dibujan aquí (p.ej. la caja en movimiento/
        coreografía, que el painter dibuja aparte en su posición interpolada).
    """
    skip = set(exclude)
    if persons:
        skip |= _delivered_package_ids(packages, persons)
    cap = _box_capacity()
    for loc_id, ids_here in _packages_at_loc(packages).items():
        if loc_id not in layout.location_positions:
            continue
        visible = [
            i for i in range(len(ids_here))
            if i < cap and ids_here[i] not in skip
        ]
        for i in sorted(visible, key=_box_depth_key):  # atrás → adelante
            package_id = ids_here[i]
            pkg = packages[package_id]
            pos = layout.package_position(package_id, theme)
            draw_package(
                surface, pos, pkg.contains.id, theme,
                shadow=True, sprite_manager=sprite_manager,
            )


def _draw_persons(surface, persons, layout, theme: Theme, sprite_manager):
    """Dibuja las personas (sprite asignado + anillo de needs).

    Punto ÚNICO de pintado de personas (como _draw_resting_packages): los
    cuatro caminos del painter pasan por aquí. Reparte los sets de sprite de
    persona detectados en assets (``personN.png``) entre las personas del
    problema, round-robin por id ORDENADO (asignación estable; si hay más
    personas que sets, se repiten). El índice de reparto es la posición en el
    orden global, así una persona conserva su sprite aunque otras no se
    pinten. Cada persona usa la variante "_box" en cuanto ``has_received`` no
    está vacío (ya recibió su entrega: espera → entrega cumplida).
    """
    bases = sprite_manager.person_sprite_bases() if sprite_manager is not None else ()
    order = sorted(persons)
    for idx, person_id in enumerate(order):
        person = persons[person_id]
        if person.position not in layout.location_positions:
            continue
        base = bases[idx % len(bases)] if bases else None
        draw_person(
            surface,
            layout.person_position(person_id, theme),
            person_id,
            theme=theme,
            sprite_manager=sprite_manager,
            sprite_base=base,
            has_box=bool(person.has_received),
        )


# ---------------------------------------------------------------------------
# Cámara: zoom + pan  (zoom UNIFORME, ver memoria_sesion_d.md §16)
# ---------------------------------------------------------------------------
#
# Decisión (encargo de E2, sustituye a la decisión previa "zoom solo afecta
# al espaciado"): el zoom es UNIFORME — escala TODO junto, como un visor
# estándar (Maps, IDE). Sprites, espaciado del grafo, offsets (hover, anillo,
# suelo), texto, aristas y overlays crecen/encogen con el zoom. El pan sigue
# siendo un desplazamiento en píxeles de la subsurface destino, aplicado
# DESPUÉS del zoom.
#
# Implementación (Estrategia B — post-proceso de la imagen): renderizamos la
# escena a zoom=1 sobre una surface intermedia y luego escalamos ESA imagen
# por `zoom` con pygame.transform.scale (nearest-neighbor, pixel art) y la
# blitteamos desplazada por `pan`. Razón frente a "escalar el Theme":
#   - Uniformidad garantizada y total (también texto, aristas, overlays de
#     estado, coreografía y tilt del dron) sin enumerar campos del Theme.
#   - Evita el doble uso de SpriteAnchors (espacio nativo del compositor del
#     dron Y tamaño de dibujo del carrier en suelo): escalar el Theme
#     rompería la composición de capas del dron.
#   - Un solo punto de escalado nearest-neighbor; YAGNI, sin tocar las
#     funciones draw_* ni el layout.
#
# Modelo matemático (centro de zoom = centro de la surface; pan tras el zoom):
# un punto de la escena en posición p (a zoom=1) aparece en pantalla en
#
#     screen = c + (p - c) * zoom + pan      con c = (W/2, H/2)
#
# Este es EXACTAMENTE el mismo mapeo de posición que la cámara anterior, así
# que locate_entity (que devuelve la posición a zoom=1) y la fórmula de
# focus_on_entity de E2 NO cambian; solo cambia que ahora los TAMAÑOS también
# escalan por zoom.
#
# Invariante crítico: zoom=1.0, pan=(0,0) produce un render PÍXEL-IDÉNTICO al
# render sin estos kwargs. Lo garantiza el fast-path (_camera_active==False),
# que renderiza directamente sobre la surface sin pasar por el escalado.

#: Cota inferior del zoom, AHORA en unidades de escala ABSOLUTA del sprite
#: respecto a su resolución nativa (1.0 = píxel nativo 1:1), no relativa a la
#: vista de ajuste. A 0.3 los sprites quedan pequeños pero legibles; cubre la
#: vista completa de las escenas más densas (cuyo ajuste cae en ~1/SS_MAX ≈
#: 0.33). La UI (app_state) define su rejilla dentro de [ZOOM_MIN, ZOOM_MAX].
ZOOM_MIN: float = 0.3
#: Cota superior del zoom (escala absoluta). A 2.5 los sprites se ven a 2.5× su
#: tamaño nativo (ampliados para inspección de cerca). Es una primitiva GENERAL;
#: la rejilla interactiva de la app es un subconjunto y nunca lo rebasa. Es
#: INDEPENDIENTE de _SUPERSAMPLE_MAX (que acota la resolución del buffer, no la
#: escala de pantalla): por encima de la resolución nativa de la fuente se
#: amplía (sprites mayores, sin más detalle).
ZOOM_MAX: float = 2.5

#: Umbral de zoom por debajo del cual se ocultan las etiquetas de nombre
#: de las entidades. A zoom < 1 el texto se rasteriza a zoom=1 y lo reduce
#: el escalado nearest-neighbor, quedando ilegible; ocultarlo es preferible
#: a mostrarlo distorsionado. A zoom >= 1 el texto se amplia y se lee bien.
_LABEL_MIN_ZOOM: float = 1.0


def _clamp_zoom(zoom: float) -> float:
    """Satura el zoom al rango [ZOOM_MIN, ZOOM_MAX] sin lanzar excepción.

    Defensivo: una UI con slider conectado podría pasar valores fuera de
    rango; en vez de romper, saturamos silenciosamente. Aplicado UNA vez
    en el punto de entrada de render_snapshot/render_frame.
    """
    return max(ZOOM_MIN, min(ZOOM_MAX, zoom))


def _camera_active(zoom: float, pan: tuple[int, int]) -> bool:
    """True si la cámara altera el render (hay que pasar por el escalado).

    Cuando es False (zoom==1.0 y pan==(0,0)) el render es directo sobre la
    surface destino, garantizando el invariante píxel-idéntico.
    """
    return zoom != 1.0 or tuple(pan) != (0, 0)


#: Superficie destino reutilizable para el escalado de escena. transform.scale
#: acepta un tercer argumento DestSurface: reutilizarlo evita asignar una
#: surface del tamaño del viewport en CADA frame (la vista por defecto tiene
#: zoom fijo, así que el tamaño no cambia y se reutiliza siempre). Se reasigna
#: solo cuando cambia (scaled_w, scaled_h), p. ej. al hacer zoom o redimensionar.
_SCALED_DEST: "pygame.Surface | None" = None


def _blit_scaled_scene(
    target: pygame.Surface,
    scene: pygame.Surface,
    zoom: float,
    pan: tuple[int, int],
    theme: Theme,
) -> None:
    """Compone sobre `target` la `scene` (renderizada a zoom=1) aplicando
    zoom UNIFORME nearest-neighbor alrededor del centro de `target` y `pan`.

    Rellena `target` con el fondo del Theme antes del blit para cubrir el
    área no tapada cuando zoom<1 (la escena escalada es menor que el target)
    o cuando el pan la desplaza. El escalado usa pygame.transform.scale
    (nearest-neighbor): preserva el pixel art (a zoom no entero hay "wobble"
    de bordes, asumido por diseño).
    """
    global _SCALED_DEST
    sw, sh = target.get_size()
    cx = sw / 2.0
    cy = sh / 2.0
    scaled_w = max(1, round(sw * zoom))
    scaled_h = max(1, round(sh * zoom))
    if _SCALED_DEST is None or _SCALED_DEST.get_size() != (scaled_w, scaled_h):
        _SCALED_DEST = pygame.Surface((scaled_w, scaled_h))
    scaled = pygame.transform.scale(scene, (scaled_w, scaled_h), _SCALED_DEST)
    # Esquina superior-izquierda donde blittear la imagen escalada para que
    # el punto (cx,cy) de la escena caiga en (cx+pan_x, cy+pan_y) del target:
    #   dest = c*(1-zoom) + pan
    dest_x = round(cx * (1.0 - zoom) + pan[0])
    dest_y = round(cy * (1.0 - zoom) + pan[1])
    target.fill(theme.background)
    target.blit(scaled, (dest_x, dest_y))


# ---------------------------------------------------------------------------
# Viewbox de escena: reserva holgura para las decoraciones de las entidades
# ---------------------------------------------------------------------------
#
# Las locations las coloca compute_layout usando SOLO sus centros, pero las
# entidades dibujan decoraciones que SOBRESALEN del centro de su location:
#   - ARRIBA: el dron flota sobre su pad (offset de suelo + hover) y su lienzo
#     sobresale media altura; el anillo de co-localización lo sube algo más.
#   - ABAJO: el sprite isométrico de la location baja media altura desde su
#     anclaje de suelo, y debajo va su etiqueta.
# Si compute_layout recibe el viewbox simétrico de ViewBox.fit_into, las
# locations extremas (arriba/abajo) quedan tan cerca del borde que esas
# decoraciones se RECORTAN en la vista por defecto (medido: el dron de la
# location superior se sale ~31px por arriba). Con la cámara (Estrategia B) el
# recorte se hornea en la surface intermedia a zoom=1 y el pan ya no lo
# recupera. La raíz es de layout, no de cámara.
#
# Solución: _scene_viewbox reserva esa holgura (asimétrica en vertical;
# horizontal apenas la necesita) ANTES de pasar el viewbox a compute_layout,
# de modo que a zoom=1 nada se recorta → la intermedia queda limpia → el zoom
# y el pan funcionan sobre la escena completa. Se usa de forma IDÉNTICA en
# render_snapshot/render_frame y en locate_entity, así que el contrato de
# locate_entity y el focus de E2 siguen coherentes y el invariante
# "defaults == sin cámara" se mantiene.


def _scene_viewbox(surface_size: tuple[int, int], theme: Theme) -> ViewBox:
    """ViewBox para compute_layout con holgura para las decoraciones.

    Reserva, además del padding del theme, la extensión vertical que las
    entidades dibujan por encima (dron flotante) y por debajo (sprite de
    location + etiqueta) del centro de su location, para que la vista por
    defecto (zoom=1, pan=0) no recorte nada en los bordes superior/inferior.
    """
    w, h = surface_size
    pad = theme.padding
    anchors = theme.sprite_anchors
    # Holgura SUPERIOR: hasta el borde de arriba del lienzo del dron flotante.
    top_dec = (
        abs(theme.location_ground_offset_y)
        + abs(theme.drone_hover_y_offset)
        + anchors.canvas_size[1] // 2
        + abs(theme.drone_colocation_y_lift)
    )
    # Holgura INFERIOR: media altura del sprite de location en pantalla + la
    # etiqueta debajo.
    location_screen_h = theme.location_sprite_native_size[1] * theme.location_sprite_scale
    bottom_dec = location_screen_h // 2 + theme.font_size_label + 8

    top = pad + top_dec
    bottom_margin = pad + bottom_dec
    width = float(w - 2 * pad)
    height = float(h - top - bottom_margin)
    # Surface demasiado pequeña para la holgura: cae al viewbox simétrico
    # (mejor recortar un poco que producir un viewbox degenerado).
    if width <= 0.0 or height <= 0.0:
        return ViewBox.fit_into(surface_size, padding=pad)
    return ViewBox(left=float(pad), top=float(top), width=width, height=height)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def _render_snapshot_plain(
    surface: pygame.Surface,
    snapshot: WorldSnapshot,
    theme: Theme,
    sprite_manager: "SpriteManager | None",
) -> None:
    """Render estático SIN cámara (a zoom=1, pan=(0,0)) sobre `surface`.

    Es el render canónico: la cámara (cuando está activa) lo invoca sobre
    una surface intermedia y luego escala el resultado.
    """
    viewbox = _scene_viewbox(surface.get_size(), theme)
    layout = compute_layout(
        snapshot.world, viewbox, spread_factor=theme.location_spread_factor
    )
    _draw_full_scene(surface, snapshot, layout, theme, sprite_manager=sprite_manager)


def render_snapshot(
    surface: pygame.Surface,
    snapshot: WorldSnapshot,
    *,
    theme: Theme | None = None,
    sprite_manager: "SpriteManager | None" = None,
    zoom: float = 1.0,
    pan: tuple[int, int] = (0, 0),
) -> None:
    """Dibuja el WorldSnapshot sobre la Surface completa.

    Args:
        surface: Surface destino. Su tamaño determina el viewbox.
        snapshot: snapshot a dibujar.
        theme: Theme a aplicar. Si None, Theme.default().
        sprite_manager: si se provee, las entidades que tengan sprite
            disponible se dibujan con él; el resto cae a primitivas. Si
            None, todo se dibuja con primitivas (comportamiento clásico).
        zoom: factor de zoom UNIFORME. 1.0 = render canónico. El zoom escala
            TODO (sprites, espaciado del grafo, offsets, texto, aristas)
            alrededor del centro de la Surface. Se satura a
            [ZOOM_MIN, ZOOM_MAX]. zoom=1.0 (default) → render sin cambios.
        pan: desplazamiento en píxeles (dx, dy) de la imagen renderizada,
            aplicado DESPUÉS del zoom. pan=(0, 0) (default) → sin
            desplazamiento.

    Returns:
        None. La Surface se modifica in-place.

    Invariante: render_snapshot(s, snap) y
    render_snapshot(s, snap, zoom=1.0, pan=(0, 0)) producen surfaces
    píxel-idénticas (fast-path sin escalado).
    """
    if theme is None:
        theme = Theme.default()

    zoom = _clamp_zoom(zoom)
    set_labels_visible(zoom >= _LABEL_MIN_ZOOM)

    if not _camera_active(zoom, pan):
        _render_snapshot_plain(surface, snapshot, theme, sprite_manager)
        return

    # Cámara activa: render a zoom=1 sobre una intermedia y escalado uniforme.
    scene = pygame.Surface(surface.get_size())
    _render_snapshot_plain(scene, snapshot, theme, sprite_manager)
    _blit_scaled_scene(surface, scene, zoom, pan, theme)


def _is_static_only_entering_interacting(
    snap_a: WorldSnapshot, snap_b: WorldSnapshot
) -> bool:
    """¿La Static representa SOLO la entrada de un drone a INTERACTING?

    El runtime emite snapshots intermedios "start" que marcan el inicio
    de una acción no-Move poniendo drone.state := INTERACTING, sin
    ningún otro cambio mecánico (la caja sigue donde estaba, las
    personas sin cambio, etc.). El siguiente snapshot completará la
    acción con la coreografía real. Detectar este caso nos permite
    seguir mostrando el estado pre-acción del drone hasta que la
    coreografía tome el control.

    Heurística estrecha: TRUE solo si:
      - existe al menos un drone que en snap_a NO está INTERACTING y
        en snap_b SÍ lo está,
      - y todo el resto de la mecánica visual (posiciones de drones,
        contenido de cajas, posiciones de transporters, contenido de
        personas) coincide entre ambos.

    En cualquier otro Static (p.ej. ERROR→IDLE de recuperación tras
    fallo, o IDLE→IDLE sin cambios), devolvemos FALSE y el painter
    sigue usando snap_b como destino (comportamiento previo).
    """
    wa, wb = snap_a.world, snap_b.world
    # Tiene que existir al menos un drone que entra a INTERACTING.
    entering_interacting = False
    for did, da in wa.drones.items():
        db = wb.drones.get(did)
        if db is None:
            return False  # cambia el conjunto de drones; no es nuestro caso
        if da.position != db.position:
            return False  # cambia posición de drone → no es nuestro caso
        if da.state != db.state:
            # Único cambio admisible: NO-INTERACTING → INTERACTING.
            if da.state != DroneState.INTERACTING and db.state == DroneState.INTERACTING:
                entering_interacting = True
            else:
                return False
    if not entering_interacting:
        return False
    # Los demás campos del world no deben cambiar (en una Static esto
    # ya está garantizado, pero verificamos package.at y persona.needs
    # por defensa).
    if set(wa.packages.keys()) != set(wb.packages.keys()):
        return False
    for pid, pa in wa.packages.items():
        pb = wb.packages[pid]
        if pa.at != pb.at:
            return False
    return True


def render_frame(
    surface: pygame.Surface,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    progress: float,
    *,
    theme: Theme | None = None,
    sprite_manager: "SpriteManager | None" = None,
    zoom: float = 1.0,
    pan: tuple[int, int] = (0, 0),
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> None:
    """Dibuja un frame interpolado entre snap_a y snap_b.

    En los extremos (progress <= 0, progress >= 1) equivale a
    render_snapshot(snap_a) y render_snapshot(snap_b) respectivamente.
    Para valores intermedios, se aplica easing y se interpola la
    posición del drone o paquete que cambia, según la clasificación
    de la transición.

    Args:
        surface: Surface destino.
        snap_a: snapshot inicial del tramo (típicamente snap_start del
            Command durativo o snap_end del Command anterior).
        snap_b: snapshot final del tramo.
        progress: fracción de avance en [0, 1]. Valores fuera de rango
            se saturan a los extremos (defensa en profundidad; el
            caller debería pasar un valor ya saneado
            o por Timeline).
        theme: Theme a aplicar. Si None, Theme.default().
        sprite_manager: igual que en render_snapshot; se propaga a todos
            los caminos de dibujo (estático e interpolado).
        zoom: factor de zoom UNIFORME (ver render_snapshot). Se satura a
            [ZOOM_MIN, ZOOM_MAX]. El frame entero se renderiza a zoom=1 y
            luego se escala uniformemente, así que la cámara es coherente en
            todas las rutas (extremos, Static/Failure e interpoladas) sin
            tratamiento especial. zoom=1.0 (default) → render sin cambios.
        pan: desplazamiento en píxeles (dx, dy), aplicado tras el zoom
            (ver render_snapshot). pan=(0, 0) (default) → sin desplazamiento.

    Invariante: zoom=1.0, pan=(0, 0) reproduce el render sin cámara
    píxel-idéntico en cualquier progress (fast-path sin escalado).
    """
    if theme is None:
        theme = Theme.default()

    zoom = _clamp_zoom(zoom)
    set_labels_visible(zoom >= _LABEL_MIN_ZOOM)

    if not _camera_active(zoom, pan):
        _render_frame_plain(
            surface, snap_a, snap_b, progress, theme, sprite_manager,
            prev_action=prev_action, next_action=next_action,
        )
        return

    # Cámara activa: render del frame a zoom=1 sobre una intermedia y escalado
    # uniforme. El escalado de la imagen compuesta hace que TODO crezca junto.
    scene = pygame.Surface(surface.get_size())
    _render_frame_plain(
        scene, snap_a, snap_b, progress, theme, sprite_manager,
        prev_action=prev_action, next_action=next_action,
    )
    _blit_scaled_scene(surface, scene, zoom, pan, theme)


def _render_frame_plain(
    surface: pygame.Surface,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    progress: float,
    theme: Theme,
    sprite_manager: "SpriteManager | None",
    *,
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> None:
    """Render interpolado SIN cámara (a zoom=1). La cámara (cuando está
    activa) invoca esto sobre una surface intermedia y escala el resultado.
    """
    transition = classify_transition(snap_a, snap_b)

    # Static "enter-interacting" = la ESPERA (dwell) justo antes de la
    # coreografía de la acción que arranca (el runtime emite un snap_start en
    # el que el dron entra a INTERACTING sin moverse; con delta de tiempo 0,
    # el Timeline le da fallback_step_duration de duración virtual). En vez de
    # dibujar el snapshot estático (dron en el CENTRO de la loc, lo que
    # provocaba el salto durante la pausa), congelamos el FRAME INICIAL
    # (progress 0) de la acción que empieza: ahí el dron está exactamente
    # donde lo dejó la coreografía anterior (su aproximación, vía encadenado)
    # o en el centro si llega de un Move. Cubre TODO el rango [0,1] del
    # intervalo (es una pausa: el dron está quieto en esa posición).
    if isinstance(transition, TransitionStatic) and _is_static_only_entering_interacting(
        snap_a, snap_b
    ):
        if next_action is not None:
            na_a, na_b = next_action
            try:
                nt = classify_transition(na_a, na_b)
            except Exception:
                nt = None
            if isinstance(nt, TransitionPackageMove):
                # La acción anterior de ESTA pausa es también la anterior de
                # la acción que arranca (ambas saltan el mismo Static), así
                # que reusamos prev_action. El next_action de la coreografía
                # no influye en progress 0 (solo afecta al step 3) → None.
                _render_frame_package_move(
                    surface, na_a, na_b, nt, 0.0, theme,
                    raw_progress=0.0, sprite_manager=sprite_manager,
                    prev_action=prev_action, next_action=None,
                )
                return
        # Sin acción siguiente reconocible: render estático de snap_a.
        _render_snapshot_plain(surface, snap_a, theme, sprite_manager)
        return

    # Saneo de extremos: dibujamos directamente el snapshot correspondiente.
    if progress <= 0.0:
        _render_snapshot_plain(surface, snap_a, theme, sprite_manager)
        return
    if progress >= 1.0:
        _render_snapshot_plain(surface, snap_b, theme, sprite_manager)
        return

    # Para transiciones sin animación, dibujamos el destino directamente.
    # Razón: Static = "no hay cambio visual"; Failure = "el drone ya está
    # en ERROR en snap_b, no tiene sentido interpolar un estado intermedio".
    if isinstance(transition, TransitionStatic):
        _render_snapshot_plain(surface, snap_b, theme, sprite_manager)
        return
    if isinstance(transition, TransitionFailure):
        _render_snapshot_plain(surface, snap_b, theme, sprite_manager)
        return

    # Easing aplicado UNA vez aquí; las funciones de dibujo reciben el
    # progress ya curvado. Esto es coherente con la decisión 2.4 de la
    # propuesta: el easing es responsabilidad de painter, no de las
    # primitivas.
    eased = ease_in_out_cubic(progress)

    if isinstance(transition, TransitionDroneMove):
        _render_frame_drone_move(
            surface, snap_a, snap_b, transition, eased, theme,
            raw_progress=progress,
            sprite_manager=sprite_manager,
        )
        return

    if isinstance(transition, TransitionPackageMove):
        _render_frame_package_move(
            surface, snap_a, snap_b, transition, eased, theme,
            raw_progress=progress,
            sprite_manager=sprite_manager,
            prev_action=prev_action, next_action=next_action,
        )
        return

    # Defensa en profundidad: una variante de Transition nueva que no
    # cubrimos aquí. classify_transition ya lanza TypeError ante Commands
    # desconocidos; esta rama es un assert defensivo por si el conjunto
    # de Transition crece.
    raise TypeError(
        f"render_frame no sabe interpolar {type(transition).__name__}"
    )


def locate_entity(
    world: World,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
) -> tuple[int, int]:
    """Devuelve (px, py) donde se dibujaría la entidad con zoom=1, pan=(0,0).

    Wrapper consciente del supersampling: cuando hay tantas locations que el
    render dibuja en un buffer ampliado (viewport × SS), localiza la entidad en
    ese buffer y mapea de vuelta al viewport (÷ SS), de modo que el resultado
    coincide con dónde aparece a zoom=1 y el foco por click cae exacto. Sin
    supersampling delega directamente en el cuerpo.
    """
    if theme is None:
        theme = Theme.default()
    ss = _supersample_factor(world, surface_size, theme)
    if ss > 1.0:
        bw = max(1, round(surface_size[0] * ss))
        bh = max(1, round(surface_size[1] * ss))
        bx, by = _locate_entity_core(world, entity_id, (bw, bh), theme=theme)
        return (round(bx / ss), round(by / ss))
    return _locate_entity_core(world, entity_id, surface_size, theme=theme)


def _locate_entity_core(
    world: World,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
) -> tuple[int, int]:
    """Posición de la entidad a zoom=1 en `surface_size` (sin supersampling).

    Pensada para que la UI (E2) implemente click-to-focus: dado el id de
    una entidad (p.ej. un item del inventario), calcula la posición lógica
    donde se dibujaría SIN cámara; desde ahí E2 computa el `pan` necesario
    para centrarla (típicamente pan = centro_surface - locate_entity(...)
    ajustado por el zoom deseado). Por eso NO se aplica cámara aquí: el
    contrato es "como si zoom=1, pan=(0, 0)".

    El píxel devuelto es el que usaría el render para anclar la entidad
    (las primitivas/sprites redondean el Point a int al dibujar), de modo
    que coincide con dónde aparece realmente en pantalla sin cámara.

    Args:
        world: estado del mundo del que resolver la entidad.
        entity_id: id de la entidad. Puede ser drone, location, transporter,
            person o package (este último SOLO si está AtLocation).
        surface_size: (ancho, alto) de la subsurface destino en píxeles.
        theme: Theme a aplicar. Si None, Theme.default(). Debe ser el MISMO
            que use el render para que la posición coincida.

    Returns:
        (px, py) en píxeles de la subsurface, con zoom=1 y pan=(0, 0).

    Raises:
        KeyError: si entity_id no existe en ninguno de los tipos del world.
        ValueError: si entity_id es un package que NO está AtLocation
            (HeldByArm o InTransporter): su posición depende del drone o
            transportador que lo sostiene; E2 debe resolver primero ese
            portador y enfocar sobre él.
    """
    if theme is None:
        theme = Theme.default()
    # MISMA holgura de decoraciones que el render (ver _scene_viewbox), para
    # que la posición devuelta coincida con dónde se dibuja realmente.
    viewbox = _scene_viewbox(surface_size, theme)
    # SIN cámara: el contrato es "como si zoom=1, pan=0". E2 calcula el pan
    # desde este resultado.
    layout = compute_layout(
        world, viewbox, spread_factor=theme.location_spread_factor
    )

    # Orden de resolución por tipo (drones, locations, transporters,
    # persons, packages). Un id solo puede ser de un tipo en escenarios
    # bien formados; el orden fija el desempate en el caso degenerado.
    if entity_id in world.drones:
        p = layout.drone_position(entity_id, theme)
    elif entity_id in world.locations:
        p = layout.location_position(entity_id)
    elif entity_id in world.transporters:
        p = layout.transporter_position(entity_id, theme)
    elif entity_id in world.persons:
        p = layout.person_position(entity_id, theme)
    elif entity_id in world.packages:
        pkg = world.packages[entity_id]
        if not isinstance(pkg.at, AtLocation):
            raise ValueError(
                f"package {entity_id!r} no está AtLocation (está en "
                f"{type(pkg.at).__name__}); su posición depende del "
                f"drone/transportador que lo sostiene. Resuélvelo "
                f"enfocando primero a ese portador."
            )
        p = layout.package_position(entity_id, theme)
    else:
        raise KeyError(
            f"entity_id {entity_id!r} no encontrado en world "
            f"(ni drone, ni location, ni transporter, ni person, ni package)."
        )
    return p.as_int_tuple()


def locate_entity_interpolated(
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    progress: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> tuple[int, int]:
    """Posición (px, py) EXACTA donde el render dibuja `entity_id` este frame.

    Wrapper consciente del supersampling: si el render usa un buffer ampliado
    (viewport × SS), localiza en ese buffer y mapea de vuelta al viewport
    (÷ SS) para que el seguimiento de cámara caiga donde se dibuja. Sin
    supersampling delega directamente en el cuerpo.

    Raises:
        KeyError: si entity_id no existe en snap_a.world.
    """
    if theme is None:
        theme = Theme.default()
    ss = _supersample_factor(snap_a.world, surface_size, theme)
    if ss > 1.0:
        bw = max(1, round(surface_size[0] * ss))
        bh = max(1, round(surface_size[1] * ss))
        bx, by = _locate_entity_interpolated_core(
            snap_a, snap_b, progress, entity_id, (bw, bh), theme=theme,
            prev_action=prev_action, next_action=next_action,
        )
        return (round(bx / ss), round(by / ss))
    return _locate_entity_interpolated_core(
        snap_a, snap_b, progress, entity_id, surface_size, theme=theme,
        prev_action=prev_action, next_action=next_action,
    )


def _locate_entity_interpolated_core(
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    progress: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> tuple[int, int]:
    """Cuerpo de locate_entity_interpolated SIN supersampling.

    A diferencia de `locate_entity` (estático, solo snap_a y solo paquetes
    AtLocation), esta función reproduce TODA la geometría de interpolación
    del painter para el frame (snap_a→snap_b, progress), de modo que la
    cámara puede SEGUIR a cualquier objeto móvil a lo largo del replay:

      - Drone en DroneMove: posición interpolada origen→destino (eased).
      - Drone en coreografía intra-loc (PickUp/Deliver): su posición de
        approach/retreat; en Load/Unload está quieto junto al transporter.
      - Transporter arrastrado (MoveWithTransporter): sigue al drone.
      - Paquete sostenido por un drone: en el brazo del drone (sigue su
        posición interpolada o coreográfica).
      - Paquete dentro de un transporter: en su slot (sigue al transporter,
        arrastrado o no).
      - Paquete en tránsito (su propio PickUp/Deliver/Load/Unload): la caja
        interpolada entre origen y destino.
      - Entidades quietas u otras no implicadas: su posición estática.

    Espeja a las funciones _render_frame_* (mismo easing y misma fórmula de
    steps de la coreografía); cualquier cambio en aquellas debe replicarse
    en los helpers _package_move_* y _entity_pos_in_transition. Pensada para
    el seguimiento de cámara por frame.

    Args:
        snap_a, snap_b: snapshots origen y destino del frame visible.
        progress: avance en [0, 1] (raw; el easing se aplica internamente).
        entity_id: id de drone, location, transporter, person o package.
            Los paquetes se localizan en CUALQUIER placement (AtLocation,
            HeldByArm, InTransporter): la posición se deriva del portador.
        surface_size: (ancho, alto) de la subsurface del world en píxeles.
        theme: Theme con el que se renderiza (debe ser el mismo).

    Returns:
        (px, py) en píxeles de la subsurface, con zoom=1 y pan=(0, 0).

    Raises:
        KeyError: si entity_id no existe en snap_a.world.
    """
    if theme is None:
        theme = Theme.default()
    raw = max(0.0, min(1.0, progress))
    eased = ease_in_out_cubic(raw)
    world_a = snap_a.world
    viewbox = _scene_viewbox(surface_size, theme)
    layout_a = compute_layout(
        world_a, viewbox, spread_factor=theme.location_spread_factor
    )
    transition = classify_transition(snap_a, snap_b)
    pos = _entity_pos_in_transition(
        transition, snap_a, snap_b, eased, raw, entity_id,
        layout_a, theme, surface_size,
        prev_action=prev_action, next_action=next_action,
    )
    return pos.as_int_tuple()


def _static_package_pos(
    world, layout: WorldLayout, package_id: str, theme: Theme
) -> Point:
    """Posición de un paquete en CUALQUIER placement, derivada del portador.

    - AtLocation: su slot apilado en el suelo de la loc.
    - HeldByArm: el ancla del brazo del drone que lo sostiene.
    - InTransporter: su slot dentro del transporter.
    """
    at = world.packages[package_id].at
    if isinstance(at, AtLocation):
        return layout.package_position(package_id, theme)
    if isinstance(at, HeldByArm):
        dpos = layout.drone_position(at.drone_id, theme)
        arm_ids = [a.id for a in world.drones[at.drone_id].arms]
        idx = arm_ids.index(at.arm_id) if at.arm_id in arm_ids else 0
        return position_of_arm(dpos, idx, max(1, len(arm_ids)), theme)
    if isinstance(at, InTransporter):
        tpos = layout.transporter_position(at.transporter_id, theme)
        in_t = sorted(
            world.packages_in_transporter(at.transporter_id), key=lambda x: x.id
        )
        ids = [x.id for x in in_t]
        idx = ids.index(package_id) if package_id in ids else 0
        return position_in_transporter(tpos, idx, max(1, len(in_t)), theme)
    raise KeyError(f"placement desconocido para package {package_id!r}")


def _static_entity_pos(
    world, layout: WorldLayout, entity_id: str, theme: Theme
) -> Point:
    """Posición estática de cualquier entidad (mismas convenciones que
    locate_entity, pero resolviendo paquetes held/in-transporter)."""
    if entity_id in world.drones:
        return layout.drone_position(entity_id, theme)
    if entity_id in world.locations:
        return layout.location_position(entity_id)
    if entity_id in world.transporters:
        return layout.transporter_position(entity_id, theme)
    if entity_id in world.persons:
        return layout.person_position(entity_id, theme)
    if entity_id in world.packages:
        return _static_package_pos(world, layout, entity_id, theme)
    raise KeyError(
        f"entity_id {entity_id!r} no encontrado en world."
    )


def _entity_pos_in_transition(
    transition,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    eased: float,
    raw: float,
    entity_id: str,
    layout_a: WorldLayout,
    theme: Theme,
    surface_size: tuple[int, int],
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> Point:
    """Posición de `entity_id` según el tipo de transición. Espeja el draw."""
    world_a = snap_a.world

    # --- DroneMove (Move / MoveWithTransporter) ---
    if isinstance(transition, TransitionDroneMove):
        moving = transition.drone_id
        dragged_t = transition.transporter_id

        if dragged_t is not None:
            # MoveWithTransporter: trayectoria por fases (aproximar/transportar/
            # soltar). El dron y el carrier siguen esas posiciones; el carrier
            # arrastrado y los paquetes dentro acompañan al carrier.
            drone_now, carrier_now = _carrier_drag_positions(
                layout_a, world_a, transition, raw, theme
            )
            if entity_id == moving:
                return drone_now
            if entity_id == dragged_t:
                return carrier_now
            if entity_id in world_a.packages:
                at = world_a.packages[entity_id].at
                if isinstance(at, HeldByArm) and at.drone_id == moving:
                    arm_ids = [a.id for a in world_a.drones[moving].arms]
                    idx = arm_ids.index(at.arm_id) if at.arm_id in arm_ids else 0
                    return position_of_arm(drone_now, idx, max(1, len(arm_ids)), theme)
                if isinstance(at, InTransporter) and at.transporter_id == dragged_t:
                    return carrier_now
            return _static_entity_pos(world_a, layout_a, entity_id, theme)

        # Move simple (sin transporter): interpolación directa.
        d_from = _drone_position_in_loc(
            layout_a, world_a, transition.from_loc_id, moving, theme
        )
        d_to = _drone_position_in_loc(
            layout_a, world_a, transition.to_loc_id, moving, theme
        )
        drone_now = lerp_point(d_from, d_to, eased)
        if entity_id == moving:
            return drone_now
        if entity_id in world_a.packages:
            at = world_a.packages[entity_id].at
            # Paquete sostenido por el drone que vuela → va en su brazo.
            if isinstance(at, HeldByArm) and at.drone_id == moving:
                arm_ids = [a.id for a in world_a.drones[moving].arms]
                idx = arm_ids.index(at.arm_id) if at.arm_id in arm_ids else 0
                return position_of_arm(drone_now, idx, max(1, len(arm_ids)), theme)
        return _static_entity_pos(world_a, layout_a, entity_id, theme)

    # --- PackageMove (PickUp / Deliver / Load / Unload) ---
    if isinstance(transition, TransitionPackageMove):
        world_b = snap_b.world
        viewbox = _scene_viewbox(surface_size, theme)
        layout_b = compute_layout(
            world_b, viewbox, spread_factor=theme.location_spread_factor
        )
        protagonist = transition.drone_id
        # Anclas neighbor-aware (mismas que el render): encadenado de acciones
        # consecutivas en la misma loc sin volver al centro.
        center_pos = layout_a.drone_position(protagonist, theme)
        approach_pos = _box_approach_position(
            layout_a, world_a, transition, theme, layout_b=layout_b
        )
        entry_exit = _choreo_entry_exit(
            layout_a, world_a, transition, center_pos, approach_pos,
            prev_action, next_action, viewbox, theme,
        )
        if entity_id == protagonist:
            return _package_move_drone_pos(
                layout_a, layout_b, world_a, transition, raw, theme, entry_exit
            )
        if entity_id == transition.package_id:
            return _package_move_box_pos(
                layout_a, layout_b, world_a, world_b, transition, raw, eased, theme,
                entry_exit,
            )
        # Otro paquete sostenido por el protagonista (p.ej. en el otro brazo):
        # acompaña la posición coreográfica del drone.
        if entity_id in world_a.packages:
            at = world_a.packages[entity_id].at
            if isinstance(at, HeldByArm) and at.drone_id == protagonist:
                dpos = _package_move_drone_pos(
                    layout_a, layout_b, world_a, transition, raw, theme, entry_exit
                )
                arm_ids = [a.id for a in world_a.drones[protagonist].arms]
                idx = arm_ids.index(at.arm_id) if at.arm_id in arm_ids else 0
                return position_of_arm(dpos, idx, max(1, len(arm_ids)), theme)
        return _static_entity_pos(world_a, layout_a, entity_id, theme)

    # --- Static / Failure ---
    # Si es la ESPERA (enter-interacting) antes de una acción, el seguimiento
    # debe ESPEJAR el render: el dron (y lo que sostiene) está congelado en el
    # frame inicial (progress 0) de la acción que arranca, no en el centro.
    if (
        isinstance(transition, TransitionStatic)
        and next_action is not None
        and _is_static_only_entering_interacting(snap_a, snap_b)
    ):
        na_a, na_b = next_action
        try:
            nt = classify_transition(na_a, na_b)
        except Exception:
            nt = None
        if isinstance(nt, TransitionPackageMove):
            layout_na = compute_layout(
                na_a.world, _scene_viewbox(surface_size, theme),
                spread_factor=theme.location_spread_factor,
            )
            return _entity_pos_in_transition(
                nt, na_a, na_b, 0.0, 0.0, entity_id,
                layout_na, theme, surface_size,
                prev_action=prev_action, next_action=None,
            )
    return _static_entity_pos(world_a, layout_a, entity_id, theme)


# ---------------------------------------------------------------------------
# Dibujo de un snapshot completo (orden de capas)
# ---------------------------------------------------------------------------


def _draw_full_scene(
    surface: pygame.Surface,
    snapshot: WorldSnapshot,
    layout: WorldLayout,
    theme: Theme,
    *,
    sprite_manager: "SpriteManager | None" = None,
) -> None:
    """Dibuja el WorldSnapshot completo respetando el z-order.

    Esta función es el "render base". Se llama desde render_snapshot
    directamente y desde las funciones de interpolación con el world
    de snap_a como base + ajustes específicos para la entidad animada.

    sprite_manager se propaga a las entidades que ya soportan sprites
    (drone en el paso 1 del Frente 2; el resto se irá añadiendo). Las
    que aún no lo consumen lo ignoran y dibujan primitiva.
    """
    world = snapshot.world

    # 1. Fondo
    draw_background(surface, theme)

    # 2. Aristas del grafo: línea entre cada par (o, d) con coste definido.
    # Para evitar dibujar dos veces la misma arista (en grafos no dirigidos
    # ambos sentidos están en costs), llevamos un set de pares ordenados.
    _draw_edges(surface, layout, world, theme)

    # 3. Locations
    for loc_id, position in layout.location_positions.items():
        draw_location(surface, position, loc_id, theme, sprite_manager=sprite_manager)

    # 4. Transporters. El nivel de llenado lo comunica el sprite carrier_N;
    #    los paquetes que están DENTRO no se dibujan (quedan ocultos en el
    #    carrier).
    # Apilado tipo lista: por loc, como mucho _CARRIER_MAX visibles (2 capas),
    # dibujados de atrás (capa de arriba, más alejado) hacia delante. Los que
    # rebasan el límite no se dibujan (comparten posición con el último).
    for loc_id, tids in _colocation_groups(world.transporters).items():
        if loc_id not in layout.location_positions:
            continue  # Defensiva: transp con position fuera del world
        visible = [i for i in range(len(tids)) if i < _CARRIER_MAX]
        for i in sorted(visible, key=_carrier_depth_key):  # atrás → adelante
            transporter_id = tids[i]
            transporter = world.transporters[transporter_id]
            tpos = layout.transporter_position(transporter_id, theme)
            in_t = world.packages_in_transporter(transporter_id)
            draw_transporter(
                surface,
                tpos,
                transporter_id,
                contents_count=len(in_t),
                capacity=transporter.capacity,
                theme=theme,
                shadow=True,  # transporter apoyado en el suelo
                sprite_manager=sprite_manager,
            )

    # 5. Personas (sprite asignado + anillo de needs).
    _draw_persons(surface, world.persons, layout, theme, sprite_manager)

    # 6. Paquetes libres (AtLocation): apilados junto a su loc en pirámide
    #    truncada compacta (ver _draw_resting_packages: capacidad + orden).
    _draw_resting_packages(surface, world.packages, layout, theme, sprite_manager,
                           persons=world.persons)

    # 7. Drones (con brazos y paquetes sostenidos)
    for drone_id, drone in world.drones.items():
        if drone.position not in layout.location_positions:
            continue
        _draw_drone_with_arms_and_packages(
            surface, layout, world, drone_id, theme,
            position_override=None,
            sprite_manager=sprite_manager,
        )


def _draw_edges(
    surface: pygame.Surface,
    layout: WorldLayout,
    world,
    theme: Theme,
) -> None:
    """Dibuja una arista por cada par (o, d) con coste definido, sin duplicar.

    Si el grafo es no dirigido y world.costs tiene ambas direcciones
    (caso típico tras el preprocesamiento del facade), dibujamos cada
    arista UNA sola vez ordenando el par lexicográficamente.
    """
    drawn: set[tuple[str, str]] = set()
    for (o, d) in world.costs.keys():
        if o == d:
            continue
        if o not in layout.location_positions or d not in layout.location_positions:
            continue
        key = tuple(sorted((o, d)))
        if key in drawn:
            continue
        drawn.add(key)
        draw_edge(
            surface,
            layout.location_position(o),
            layout.location_position(d),
            theme,
        )


def _drone_object_layer(
    world,
    drone_id: str,
    theme: Theme,
    *,
    dragged_transporter_id: str | None = None,
) -> tuple[str | None, int, int]:
    """Decide la capa de objeto del dron: (held_object, carrier_level, box_count).

    La fuente de verdad para "el dron arrastra el transporter" NO es la
    co-localización (ambigua: los planes reales muestran drones casi
    siempre co-localizados con el transporter SIN arrastrarlo, p.ej.
    durante la fase de carga coger-caja/poner-caja). Es la clasificación
    de transición: solo durante un MoveWithTransporter el dron lo arrastra,
    y eso lo expresa TransitionDroneMove.transporter_id. El caller pasa ese
    id en dragged_transporter_id cuando corresponde; en cualquier otro
    contexto (render estático, Move simple, package move) es None y el
    transporter es entidad del suelo.

    Reglas:
      - dragged_transporter_id no None → "carrier", nivel por llenado.
      - si no, dron sostiene >=1 paquete en brazos → "box", box_count=nº.
      - si no → (None, 0, 0).

    Returns:
        (held_object, carrier_level, box_count):
            held_object: "carrier" | "box" | None.
            carrier_level: 0..5 (solo significativo si "carrier").
            box_count: nº de cajas sueltas (solo significativo si "box").
    """
    if dragged_transporter_id is not None:
        transp = world.transporters.get(dragged_transporter_id)
        inside = len(world.packages_in_transporter(dragged_transporter_id))
        capacity = transp.capacity if transp is not None else 0
        if capacity > 0:
            # Cuantizar a 6 escalones 0..5 (0/20/40/60/80/100%).
            level = round(5 * inside / capacity)
            level = max(0, min(5, level))
        else:
            level = 0
        return ("carrier", level, 0)

    # Cajas sueltas en brazos.
    box_count = 0
    drone = world.drones[drone_id]
    for arm in drone.arms:
        if _held_by(world, drone_id, arm.id) is not None:
            box_count += 1
    if box_count > 0:
        return ("box", 0, box_count)

    return (None, 0, 0)


def _draw_drone_with_arms_and_packages(
    surface: pygame.Surface,
    layout: WorldLayout,
    world,
    drone_id: str,
    theme: Theme,
    *,
    position_override: Point | None = None,
    held_overrides: dict[str, Point] | None = None,
    sprite_manager: "SpriteManager | None" = None,
    direction_angle: float | None = None,
    finished: bool = False,
    dragged_transporter_id: str | None = None,
    force_box_for_package: str | None = None,
    tilt_angle: float = 0.0,
    error_phase: int = 0,
) -> None:
    """Dibuja un drone con su capa de objeto y los paquetes sostenidos.

    Args:
        position_override: si se provee, el drone se dibuja AQUÍ en
            lugar de en su posición lógica. Útil para interpolar Move.
        held_overrides: dict arm_id → Point. Para cada brazo en este
            dict, el paquete sostenido por ese brazo se dibuja en el
            Point indicado en vez de en la posición del brazo. Útil
            para interpolar Pickup/Deliver (paquete a medio camino).
        sprite_manager: si hay sprite compuesto disponible, el dron se
            dibuja como composición de capas (cuerpo + cara + objeto) y
            los marcadores de brazo / paquetes por-brazo se RETIRAN (el
            cuerpo ya muestra los brazos; la caja con su número da el
            recuento). En camino primitiva se conservan.
        direction_angle: ángulo del vector de movimiento (durante Move),
            None si parado. Determina la cara direccional.
        finished: True si el dron terminó su ejecución → cara idle.
        dragged_transporter_id: id del transporter que el dron arrastra
            (solo durante MoveWithTransporter), o None. Fuente: la
            clasificación de transición, no la co-localización.
        force_box_for_package: id de un paquete que, durante la
            coreografía de PickUp/Deliver, ya pertenece visualmente al
            dron aunque el world de snap_a no lo refleje aún. Fuerza la
            capa box y lo suma al recuento.
    """
    drone = world.drones[drone_id]
    if position_override is not None:
        drone_pos = position_override
    else:
        drone_pos = layout.drone_position(drone_id, theme)

    # Decidir la capa de objeto (caja con número, o carrier con nivel).
    held_object, carrier_level, box_count = _drone_object_layer(
        world, drone_id, theme,
        dragged_transporter_id=dragged_transporter_id,
    )
    # Coreografía: si una caja protagonista ya es visualmente del dron
    # (tras el snap) pero el world aún no lo dice, forzamos la capa box.
    if force_box_for_package is not None and held_object != "carrier":
        if held_object != "box":
            held_object, box_count = "box", 1
        elif world.packages[force_box_for_package].at.__class__.__name__ != "HeldByArm":
            box_count += 1

    # ¿Vamos por camino sprite? Lo sabemos preguntando al manager por el
    # compuesto base; si devuelve algo, draw_drone usará sprite y retiramos
    # los adornos por-brazo.
    using_sprite = False
    if sprite_manager is not None:
        interacting = hasattr(drone.state, "name") and drone.state.name == "INTERACTING"
        body_key = "drone_interacting" if interacting else "drone"
        r = theme.drone_radius
        probe = sprite_manager.get_drone_composite(
            body=body_key, face=None, obj=None, dest_size=(2 * r, 2 * r),
        )
        using_sprite = probe is not None

    # Dibujar el drone (compuesto si hay sprite; primitiva si no).
    draw_drone(
        surface, drone_pos, drone_id, drone.state, theme,
        sprite_manager=sprite_manager,
        direction_angle=direction_angle,
        finished=finished,
        held_object=held_object,
        carrier_level=carrier_level,
        box_count=box_count,
        tilt_angle=tilt_angle,
        error_phase=error_phase,
    )

    if using_sprite:
        # Camino sprite: el objeto ya es capa del compuesto.
        #   - box: se pintan hasta 2 cajas (obj "box2"); el número solo
        #     aparece desde 3 (caso raro: dron con >=3 brazos ocupados).
        #   - carrier: número con la cuenta EXACTA de cajas dentro (el
        #     transportador puede llevar muchas; el nivel 0..5 es aproximado).
        if held_object == "box" and box_count >= 3:
            _draw_box_count_overlay(surface, drone_pos, box_count, theme)
        elif held_object == "carrier" and dragged_transporter_id is not None:
            inside = len(world.packages_in_transporter(dragged_transporter_id))
            if inside > 0:
                _draw_box_count_overlay(
                    surface, drone_pos, inside, theme, carrier=True
                )
        # NOTA: durante PackageMove, el paquete protagonista interpolado
        # lo dibuja _render_frame_package_move por separado; aquí no se
        # toca.
        return

    # --- Camino primitiva: comportamiento clásico ---
    arm_ids = [a.id for a in drone.arms]
    holding_ids: list[str | None] = []
    for arm in drone.arms:
        pkg = _held_by(world, drone_id, arm.id)
        holding_ids.append(pkg.contains.id if pkg is not None else None)

    draw_drone_arms(surface, drone_pos, arm_ids, holding_ids, theme)

    n_arms = len(arm_ids)
    for i, arm in enumerate(drone.arms):
        pkg = _held_by(world, drone_id, arm.id)
        if pkg is None:
            continue
        if held_overrides is not None and arm.id in held_overrides:
            pkg_pos = held_overrides[arm.id]
        else:
            pkg_pos = position_of_arm(drone_pos, i, n_arms, theme)
        draw_package(surface, pkg_pos, pkg.contains.id, theme, sprite_manager=sprite_manager)


def _draw_box_count_overlay(
    surface: pygame.Surface,
    drone_pos: Point,
    count: int,
    theme: Theme,
    *,
    carrier: bool = False,
) -> None:
    """Superpone el número de cajas sobre la caja (o el carrier) del dron.

    Overlay dinámico (texto sobre un disco oscuro para que se lea sobre
    cualquier color): no es parte del sprite estático, lo pinta el painter.
    Se coloca sobre la zona del objeto agarrado (más abajo si es el
    carrier, que cuelga más que una caja).
    """
    from droneplan_viz.render.sprites import _render_text, _blit_centered
    r = theme.drone_radius
    y_frac = 0.95 if carrier else 0.55
    overlay_pos = Point(drone_pos.x, drone_pos.y + r * y_frac)
    text_surf = _render_text(
        str(count),
        theme.text,
        theme.font_size_id,
        theme.font_name,
        theme.font_path,
    )
    # Disco oscuro de fondo, para legibilidad sobre caja/carrier de
    # cualquier color.
    radius = max(text_surf.get_width(), text_surf.get_height()) // 2 + 4
    cx, cy = int(overlay_pos.x), int(overlay_pos.y)
    bg = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(bg, (30, 32, 38, 220), (radius, radius), radius)
    surface.blit(bg, (cx - radius, cy - radius))
    _blit_centered(surface, text_surf, overlay_pos)


# ---------------------------------------------------------------------------
# Interpolación: DroneMove
# ---------------------------------------------------------------------------


def _carrier_drag_positions(
    layout: WorldLayout, world, transition, raw_progress: float, theme: Theme
) -> tuple[Point, Point]:
    """Posiciones (drone_pos, carrier_pos) durante un MoveWithTransporter.

    Tres fases sobre el tramo, para que el dron AGARRE el carrier antes de
    llevárselo (en vez de que el carrier se teletransporte a sus brazos):

      - Aproximación [0, 0.25]: el dron baja del centro de la loc hasta
        situarse sobre el carrier (en su posición de reposo). El carrier
        está quieto en el suelo.
      - Transporte [0.25, 0.75]: dron y carrier viajan juntos al destino;
        el carrier cuelga del dron (a `lift` px por debajo de su centro,
        igual que lo colocaría el compuesto).
      - Soltar [0.75, 1]: el dron deja el carrier en su reposo del destino
        y vuelve al centro de la loc.

    En los extremos coincide con los snapshots: en p=0 el dron está en el
    centro de from y el carrier en reposo de from; en p=1, análogo en to.
    """
    a = theme.sprite_anchors
    lift = theme.drone_radius - a.carrier_overlap + a.carrier_size[1] / 2.0
    tid = transition.transporter_id
    center_from = _drone_position_in_loc(
        layout, world, transition.from_loc_id, transition.drone_id, theme
    )
    center_to = _drone_position_in_loc(
        layout, world, transition.to_loc_id, transition.drone_id, theme
    )
    rest_from = _transporter_position_in_loc(layout, transition.from_loc_id, tid, theme)
    rest_to = _transporter_position_in_loc(layout, transition.to_loc_id, tid, theme)
    grab_from = Point(rest_from.x, rest_from.y - lift)
    grab_to = Point(rest_to.x, rest_to.y - lift)
    ga, gb = 0.25, 0.75
    p = max(0.0, min(1.0, raw_progress))
    if p <= ga:
        f = ease_in_out_cubic(p / ga) if ga > 0.0 else 1.0
        return lerp_point(center_from, grab_from, f), rest_from
    if p <= gb:
        f = ease_in_out_cubic((p - ga) / (gb - ga))
        carrier = lerp_point(rest_from, rest_to, f)
        return Point(carrier.x, carrier.y - lift), carrier
    f = ease_in_out_cubic((p - gb) / (1.0 - gb)) if gb < 1.0 else 1.0
    return lerp_point(grab_to, center_to, f), rest_to


def _render_frame_drone_move(
    surface: pygame.Surface,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    transition: TransitionDroneMove,
    eased_progress: float,
    theme: Theme,
    *,
    raw_progress: float = 0.0,
    sprite_manager: "SpriteManager | None" = None,
) -> None:
    """Frame intermedio para Move y MoveWithTransporter (sin cámara).

    Procedimiento:
    1. Layout sobre snap_a (las Locations no cambian; usar snap_a o
       snap_b sería equivalente en este aspecto).
    2. Dibujar fondo, aristas, locations.
    3. Para transporters/personas/paquetes libres: dibujar usando snap_a
       como base. EXCEPCIÓN: el transporter que viaja con el drone (si
       transition.transporter_id is not None) se dibuja en posición
       interpolada, y los paquetes que están dentro de él también.
    4. Para drones: dibujar usando snap_a como base EXCEPTO el drone
       que se mueve, que se dibuja en posición interpolada.

    El zoom/pan NO se tratan aquí: render_frame renderiza este frame a
    zoom=1 y escala la imagen resultante (zoom uniforme). Así la coreografía
    y la geometría se calculan siempre en coordenadas canónicas.
    """
    world_a = snap_a.world
    layout = compute_layout(
        world_a,
        _scene_viewbox(surface.get_size(), theme),
        spread_factor=theme.location_spread_factor,
    )

    # 1-2. Fondo, aristas, locations.
    draw_background(surface, theme)
    _draw_edges(surface, layout, world_a, theme)
    for loc_id, position in layout.location_positions.items():
        draw_location(surface, position, loc_id, theme, sprite_manager=sprite_manager)

    # 3. Transporters. El que viaja con el drone va con posición interpolada.
    moving_transp_id = transition.transporter_id
    # ¿Camino sprite para el dron que arrastra? Si el dron usa sprite
    # compuesto, el transporter arrastrado se dibuja como CAPA del dron
    # (carrier colgando), así que NO lo dibujamos también como entidad
    # separada (evita duplicarlo). En camino primitiva sí se dibuja aparte.
    # El carrier arrastrado se dibuja SIEMPRE standalone (no como capa
    # colgante del compuesto), con una trayectoria de 3 fases
    # (aproximar→transportar→soltar) que evita que se teletransporte a los
    # brazos del dron. Calculamos sus posiciones una vez.
    drag_drone_pos: Point | None = None
    drag_carrier_pos: Point | None = None
    if moving_transp_id is not None:
        drag_drone_pos, drag_carrier_pos = _carrier_drag_positions(
            layout, world_a, transition, raw_progress, theme
        )
    # El carrier "viaja por el aire" (colgando del dron) en la fase de
    # transporte; en aproximación/soltar está apoyado en el suelo.
    carried_phase = 0.25 < max(0.0, min(1.0, raw_progress)) <= 0.75

    for transporter_id, transporter in world_a.transporters.items():
        if transporter.position not in layout.location_positions:
            continue
        # Calcular posición a dibujar:
        if transporter_id == moving_transp_id:
            tpos = drag_carrier_pos
            is_grounded_transp = not carried_phase  # apoyado salvo en transporte
        else:
            tpos = layout.transporter_position(transporter_id, theme)
            is_grounded_transp = True

        in_t = world_a.packages_in_transporter(transporter_id)
        draw_transporter(
            surface,
            tpos,
            transporter_id,
            contents_count=len(in_t),
            capacity=transporter.capacity,
            theme=theme,
            shadow=is_grounded_transp,
            sprite_manager=sprite_manager,
        )
        # Los paquetes dentro NO se dibujan: el sprite carrier_N ya indica
        # el llenado y viajan ocultos dentro del transporter.

    # 4. Personas (no se mueven en este dominio).
    _draw_persons(surface, world_a.persons, layout, theme, sprite_manager)

    # 5. Paquetes libres (estáticos): pila con capacidad + orden de capas.
    _draw_resting_packages(surface, world_a.packages, layout, theme, sprite_manager,
                           persons=world_a.persons)

    # 6. Drones. El que se mueve, en posición interpolada.
    moving_drone_id = transition.drone_id
    for drone_id, drone in world_a.drones.items():
        if drone.position not in layout.location_positions:
            continue
        if drone_id == moving_drone_id:
            if moving_transp_id is not None and drag_drone_pos is not None:
                # MoveWithTransporter: trayectoria de 3 fases (el carrier ya
                # se dibuja standalone arriba). Sin capa de carrier en el
                # compuesto (dragged_transporter_id=None). Dirección/tilt por
                # diferencias finitas de la propia trayectoria por fases.
                override = drag_drone_pos
                eps = 1e-3
                p = max(0.0, min(1.0, raw_progress))
                ahead, _ = _carrier_drag_positions(
                    layout, world_a, transition, min(1.0, p + eps), theme
                )
                behind, _ = _carrier_drag_positions(
                    layout, world_a, transition, max(0.0, p - eps), theme
                )
                dx = ahead.x - behind.x
                dy = ahead.y - behind.y
                direction_angle = (
                    None if (dx == 0 and dy == 0)
                    else math.degrees(math.atan2(-dy, dx)) % 360.0
                )
                tilt = lateral_tilt_deg(
                    dx, dy, p,
                    theme.drone_move_max_tilt_deg,
                    theme.drone_tilt_envelope_steepness,
                )
                _draw_drone_with_arms_and_packages(
                    surface, layout, world_a, drone_id, theme,
                    position_override=override,
                    sprite_manager=sprite_manager,
                    direction_angle=direction_angle,
                    dragged_transporter_id=None,
                    tilt_angle=tilt,
                )
                continue
            # Move simple: interpolar drone_position from_loc → to_loc.
            d_from = _drone_position_in_loc(
                layout, world_a, transition.from_loc_id, drone_id, theme
            )
            d_to = _drone_position_in_loc(
                layout, world_a, transition.to_loc_id, drone_id, theme
            )
            override = lerp_point(d_from, d_to, eased_progress)
            # Dirección: ángulo del vector de movimiento en grados
            # matemáticos (0=E, 90=N). La y de pantalla crece hacia abajo,
            # así que invertimos dy para que "arriba" sea Norte.
            dx = d_to.x - d_from.x
            dy = d_to.y - d_from.y
            if dx == 0 and dy == 0:
                direction_angle = None
            else:
                direction_angle = math.degrees(math.atan2(-dy, dx)) % 360.0
            # Inclinación dinámica: convención drone/avión (positivo =
            # horario yendo a la derecha). PROPORCIONAL a lo horizontal que
            # sea el movimiento (lateral_tilt_deg escala por |dx|/dist): un
            # movimiento casi vertical apenas se inclina; uno horizontal,
            # del todo. Forma temporal: envoltura tanh con pendiente ~0 en
            # los extremos (entra/sale sin tirón). El factor 0.5 mantiene el
            # pico efectivo en 6° (con max_tilt=12°) para no distorsionar la
            # silueta pixel-art; la coreografía intra-loc usa el mismo factor.
            tilt = lateral_tilt_deg(
                dx, dy,
                max(0.0, min(1.0, raw_progress)),
                theme.drone_move_max_tilt_deg,
                theme.drone_tilt_envelope_steepness,
            )
            _draw_drone_with_arms_and_packages(
                surface, layout, world_a, drone_id, theme,
                position_override=override,
                sprite_manager=sprite_manager,
                direction_angle=direction_angle,
                dragged_transporter_id=None,
                tilt_angle=tilt,
            )
        else:
            _draw_drone_with_arms_and_packages(
                surface, layout, world_a, drone_id, theme,
                sprite_manager=sprite_manager,
            )


def _drone_position_in_loc(
    layout: WorldLayout,
    world,
    loc_id: str,
    drone_id: str,
    theme: Theme,
) -> Point:
    """Calcula dónde estaría el drone si estuviera en loc_id, respetando co-loc.

    No muta nada: solo aplica la fórmula de co-localización con los
    drones que estarían co-localizados en loc_id si el drone_id se
    moviera allí. Para Move/MoveWithTransporter, los demás drones que
    ya estén en loc_id forman el anillo de co-localización.

    Esto es ligeramente más fino que llamar a layout.drone_position
    (que usa la posición ACTUAL del drone): durante la interpolación,
    el drone conceptualmente "ya está saliendo" de from_loc y "ya está
    llegando" a to_loc, así que en los extremos del tramo los anillos
    de co-localización son los de from_loc / to_loc respectivamente.

    Para simplicidad y coherencia con drone_position, ordenamos los
    co-localizados alfabéticamente. También aplicamos el lift vertical
    del Theme cuando hay 2+ drones, idéntico al de
    WorldLayout.drone_position: la animación llega justo a la posición
    final correcta sin "saltar" en el último frame.
    """
    from droneplan_viz.render.layout import _ring_offset

    # Centro LÓGICO (rombo del sprite isométrico), CONSISTENTE con
    # WorldLayout.drone_position. Aplica hover (siempre) + lift de
    # co-localización (cuando hay anillo). Si la lógica difiere de la
    # del layout, el dron salta al terminar la transición.
    ground = layout._location_ground_center(loc_id, theme)
    hover_center = Point(ground.x, ground.y + theme.drone_hover_y_offset)
    # Drones que están en loc_id en el world dado, AÑADIENDO el drone_id
    # si no estaba allí (para el extremo de destino).
    base = {d.id for d in world.drones.values() if d.position == loc_id}
    base.add(drone_id)
    colocated = sorted(base)
    if len(colocated) >= 2:
        ring_center = Point(
            hover_center.x,
            hover_center.y + theme.drone_colocation_y_lift,
        )
    else:
        ring_center = hover_center
    return _ring_offset(
        center=ring_center,
        entity_id=drone_id,
        colocated_ids=colocated,
        radius=theme.drone_colocation_offset,
    )


def _transporter_position_in_loc(
    layout: WorldLayout,
    loc_id: str,
    transporter_id: str,
    theme: Theme,
) -> Point:
    """Calcula la posición visual del transp asumiendo que está en loc_id.

    Reutiliza la convención de layout.transporter_position (offset a
    la izquierda del centro LÓGICO de la loc, apilado vertical si hay
    varios).
    """
    # Centro LÓGICO (rombo), CONSISTENTE con transporter_position.
    center = layout._location_ground_center(loc_id, theme)
    # Apilamos solo este transp en la loc; durante interpolación no nos
    # complicamos con co-localización de transp (caso raro).
    x_offset = -float(theme.location_radius) * 0.55
    return Point(center.x + x_offset, center.y)


# ---------------------------------------------------------------------------
# Interpolación: PackageMove
# ---------------------------------------------------------------------------


def _package_endpoint_pos(
    place,
    layout: WorldLayout,
    world,
    drone_pos: Point,
    theme: Theme,
) -> Point:
    """Resuelve un PackagePlace (arm/transp/loc) a un Point en pantalla.

    Usado por la ruta de Load/Unload para situar los extremos del trayecto
    de la caja:
      - "arm": ancla del brazo del drone (donde se dibuja una caja sostenida).
      - "transp": slot 0 del transporter (donde se apila una caja cargada).
      - "loc": posición de la caja en el suelo (defensivo; no ocurre en
        transferencias con transporter, pero lo cubrimos por robustez).

    drone_pos es la posición (estática) del drone protagonista; solo se usa
    para el extremo "arm".
    """
    if place.kind == "arm":
        drone = world.drones[place.drone_id]
        arm_ids = [a.id for a in drone.arms]
        idx = arm_ids.index(place.arm_id) if place.arm_id in arm_ids else 0
        return position_of_arm(drone_pos, idx, max(1, len(arm_ids)), theme)
    if place.kind == "transp":
        tpos = layout.transporter_position(place.transporter_id, theme)
        return position_in_transporter(tpos, 0, 1, theme)
    # kind == "loc": centro lógico de la loc (no se espera aquí).
    return layout._location_ground_center(place.loc_id, theme)


def _same_drone_package_move(action_pair, drone_id: str):
    """Devuelve la transición si `action_pair` (snap_a, snap_b) es un
    PackageMove del mismo dron; None si el par es None o no encadena.

    Dos PackageMoves consecutivos del MISMO dron están por construcción en la
    MISMA loc: cambiar de loc exige un Move (que sería un DroneMove, no un
    PackageMove). Por eso basta comparar el drone_id.
    """
    if action_pair is None:
        return None
    snap_x, snap_y = action_pair
    try:
        t = classify_transition(snap_x, snap_y)
    except Exception:
        return None
    if isinstance(t, TransitionPackageMove) and t.drone_id == drone_id:
        return t
    return None


def _choreo_entry_exit(
    layout_a, world_a, transition, center_pos, approach_pos,
    prev_action, next_action, viewbox, theme,
):
    """Anclas de entrada/salida de la coreografía, conscientes de las
    acciones vecinas (prev_action / next_action son pares de snapshots de la
    acción real anterior/siguiente, ya sin los Static "enter-interacting"
    que el runtime intercala; None si no hay).

    - entry: si la acción ANTERIOR es otra acción de paquete del mismo dron,
      el dron ENTRA desde la aproximación de aquella (no desde el centro);
      si no (llega a la loc por un Move, o es la primera acción), desde el
      centro.
    - exit: si la acción SIGUIENTE es otra acción de paquete del mismo dron,
      el dron SE QUEDA en su propia aproximación (no vuelve al centro) para
      encadenar directamente; si no (sale de la loc por un Move, o es la
      última acción), vuelve al centro.

    Resultado: el dron pasa por el centro al LLEGAR a la loc y al SALIR, pero
    NO entre acciones internas consecutivas — coge/deja una caja y va directo
    a la siguiente.
    """
    entry_pos = center_pos
    exit_pos = center_pos
    drone_id = transition.drone_id
    prev_t = _same_drone_package_move(prev_action, drone_id)
    if prev_t is not None:
        pa, pb = prev_action
        layout_prev_a = compute_layout(
            pa.world, viewbox, spread_factor=theme.location_spread_factor
        )
        layout_prev_b = compute_layout(
            pb.world, viewbox, spread_factor=theme.location_spread_factor
        )
        entry_pos = _box_approach_position(
            layout_prev_a, pa.world, prev_t, theme, layout_b=layout_prev_b
        )
    next_t = _same_drone_package_move(next_action, drone_id)
    if next_t is not None:
        exit_pos = approach_pos  # nos quedamos en nuestra aproximación
    return entry_pos, exit_pos


def _render_frame_package_move(
    surface: pygame.Surface,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    transition: TransitionPackageMove,
    eased_progress: float,
    theme: Theme,
    *,
    raw_progress: float = 0.0,
    sprite_manager: "SpriteManager | None" = None,
    prev_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
    next_action: tuple[WorldSnapshot, WorldSnapshot] | None = None,
) -> None:
    """Frame intermedio para PickUp/Deliver/Load/Unload (sin cámara).

    Procedimiento:
    1. Layout sobre snap_a.
    2. Dibujar fondo, aristas, locations.
    3. Transporters, personas y paquetes libres usando snap_a.
       EXCEPCIÓN: el paquete que se mueve NO se dibuja todavía (lo
       pintamos al final, en posición interpolada).
    4. Drones usando snap_a. EXCEPCIÓN: el drone protagonista se dibuja
       en INTERACTING (que en snap_a es MOVING/INTERACTING o IDLE).
       Para la mayoría de interpolaciones de paquete, snap_a tiene al
       drone ya en INTERACTING (es el snap_start de un Command durativo
       con efectos visuales sobre brazos). Usamos el state de snap_a tal
       cual.
    5. Dibujar el paquete en posición interpolada entre from_place y
       to_place (resolviendo cada extremo con WorldLayout y los helpers).

    El zoom/pan NO se tratan aquí: render_frame renderiza este frame a
    zoom=1 y escala la imagen resultante (zoom uniforme), así que ambos
    layouts comparten coordenadas canónicas y la caja no "salta".
    """
    world_a = snap_a.world
    world_b = snap_b.world
    viewbox = _scene_viewbox(surface.get_size(), theme)
    layout_a = compute_layout(
        world_a, viewbox, spread_factor=theme.location_spread_factor
    )
    # layout_b para resolver la posición de la caja en el destino
    # (Deliver: la caja está AtLocation en snap_b, no en snap_a).
    layout_b = compute_layout(
        world_b, viewbox, spread_factor=theme.location_spread_factor
    )

    # Para interpolar el paquete necesitamos también el layout de snap_b
    # cuando el destino "to_place" depende del estado en snap_b. Por
    # ejemplo, Deliver: el paquete acaba AtLocation(person.position); la
    # info de la persona está en ambos snaps (no cambia), así que basta
    # con layout_a. Para los otros casos, todos los referentes existen
    # en world_a. Por eso solo necesitamos layout_a.

    pkg_id = transition.package_id

    # 1-2. Fondo, aristas, locations.
    draw_background(surface, theme)
    _draw_edges(surface, layout_a, world_a, theme)
    for loc_id, position in layout_a.location_positions.items():
        draw_location(surface, position, loc_id, theme, sprite_manager=sprite_manager)

    # 3. Transporters. Su nivel de llenado (carrier_N) excluye el paquete
    #    que se mueve; los paquetes DENTRO no se dibujan (van ocultos en el
    #    carrier). El paquete en tránsito lo pinta la coreografía más abajo.
    for transporter_id, transporter in world_a.transporters.items():
        if transporter.position not in layout_a.location_positions:
            continue
        tpos = layout_a.transporter_position(transporter_id, theme)
        in_t = [
            p for p in world_a.packages_in_transporter(transporter_id)
            if p.id != pkg_id
        ]
        draw_transporter(
            surface,
            tpos,
            transporter_id,
            contents_count=len(in_t),
            capacity=transporter.capacity,
            theme=theme,
            shadow=True,  # transporters estáticos: apoyados en el suelo
            sprite_manager=sprite_manager,
        )

    # 4. Personas.
    _draw_persons(surface, world_a.persons, layout_a, theme, sprite_manager)

    # 5. Paquetes libres EXCEPTO el que se mueve (que se dibuja aparte,
    #    interpolado). El excluido sigue contando para el índice de slot.
    _draw_resting_packages(
        surface, world_a.packages, layout_a, theme, sprite_manager,
        persons=world_a.persons, exclude=(pkg_id,),
    )

    # 6-7. Drone protagonista + caja: COREOGRAFÍA DE 4 STEPS.
    #
    # Vale para PickUp/Deliver (caja en el SUELO) y para Load/Unload (caja
    # en un CARRIER co-localizado): la única diferencia es la posición
    # OBJETIVO a la que el dron se aproxima — el slot de la caja en el suelo,
    # o el carrier — que resuelve _box_approach_position. Para Load/Unload,
    # cuando la caja "está en el suelo" en realidad está DENTRO del carrier,
    # así que no se dibuja (queda oculta); el sprite carrier_N transmite el
    # llenado. El dron se acerca al carrier y hace la MISMA animación de
    # coger/dejar que en el suelo.
    #
    # La acción se divide en 4 pasos de tiempo equitativo (25% cada uno).
    # PickUp (coger del suelo):
    #   step 0 [0-25%]:  base, se desplaza del CENTRO de la loc hasta
    #                    alinearse con la caja (borde inferior del dron a
    #                    6px del borde inferior de la caja). Easing.
    #   step 1 [25-50%]: interacting (brazos extendidos), QUIETO en la
    #                    posición de aproximación. La caja sigue en suelo.
    #                    Al final de este step se hace el SNAP de propiedad.
    #   step 2 [50-75%]: base de nuevo, QUIETO, YA con la caja (capa box).
    #   step 3 [75-100%]: base, se desplaza de vuelta al CENTRO con la
    #                    caja. Easing.
    # Deliver (dejar en suelo): simétrico inverso —
    #   step 0: base con caja, centro → posición de depósito. Easing.
    #   step 1: interacting con caja, quieto. Snap (suelta) al final.
    #   step 2: base sin caja, quieto.
    #   step 3: base sin caja, posición → centro. Easing.
    #
    # Sprite discreto (sin transición): step 1 es interacting; resto base.
    # Snap de la caja en la frontera step1→step2 (progress=0.5), cuando el
    # dron está en interacting con los brazos abajo.
    protagonist = transition.drone_id
    picking_up = transition.from_place.kind != "arm"  # loc/transp→arm = coger

    center_pos = layout_a.drone_position(protagonist, theme)
    approach_pos = _box_approach_position(layout_a, world_a, transition, theme, layout_b=layout_b)
    # Anclas neighbor-aware: el dron encadena acciones consecutivas en la
    # misma loc sin volver al centro (entra desde la aproximación anterior y
    # se queda en la suya). En los bordes (Move de llegada/salida) usa el
    # centro. Sin vecinos (snap_prev/snap_next None) → centro en ambos, que
    # reproduce el comportamiento clásico.
    entry_pos, exit_pos = _choreo_entry_exit(
        layout_a, world_a, transition, center_pos, approach_pos,
        prev_action, next_action, viewbox, theme,
    )

    # Step index 0..3 y fracción local dentro del step.
    p = max(0.0, min(1.0, raw_progress))
    step = min(3, int(p * 4.0))
    local = (p * 4.0) - step  # 0..1 dentro del step
    local_eased = ease_in_out_cubic(local)

    # Posición del dron e interacting/caja según step y dirección.
    # Además calculamos direction_angle y tilt_angle SOLO durante los
    # steps de desplazamiento (0 y 3); en los steps de quietud (1 y 2)
    # el dron está parado en approach_pos y no debe rotar ni inclinarse.
    #
    # Para el tilt usamos la misma fórmula que en Move (envoltura tanh
    # tilt_envelope, 0→max→0), pero sobre `local` (fracción del
    # step, no del raw_progress global), porque el desplazamiento solo
    # ocurre durante ese 25% de la acción. Aplicamos un factor 0.5
    # sobre el pico de Move (`theme.drone_move_max_tilt_deg`): la
    # coreografía intra-loc cubre distancias mucho más cortas que un
    # Move entre locations y, con la misma constante, la inclinación
    # resultaba exagerada visualmente. Mitad del pico de Move da una
    # rotación notable pero sutil, coherente con la escala del
    # desplazamiento.
    direction_angle: float | None = None
    tilt_angle: float = 0.0
    if step in (0, 3):
        if step == 0:
            d_from, d_to = entry_pos, approach_pos
            drone_pos = lerp_point(d_from, d_to, local_eased)
        else:  # step == 3
            d_from, d_to = approach_pos, exit_pos
            drone_pos = lerp_point(d_from, d_to, local_eased)
        # Dirección y tilt: misma fórmula que en Move, proporcional a la
        # componente horizontal del desplazamiento de approach/retreat.
        ddx = d_to.x - d_from.x
        ddy = d_to.y - d_from.y
        if not (ddx == 0 and ddy == 0):
            direction_angle = math.degrees(math.atan2(-ddy, ddx)) % 360.0
        tilt_angle = lateral_tilt_deg(
            ddx, ddy,
            max(0.0, min(1.0, local)),
            theme.drone_move_max_tilt_deg,
            theme.drone_tilt_envelope_steepness,
        )
        interacting = False
        if step == 0:
            box_on_drone = not picking_up  # dejar: lleva la caja; coger: no
        else:
            box_on_drone = picking_up
    elif step == 1:
        drone_pos = approach_pos
        interacting = True
        box_on_drone = not picking_up
    else:  # step == 2
        drone_pos = approach_pos
        interacting = False
        box_on_drone = picking_up  # coger: ya la tiene; dejar: ya la soltó

    # Dibujar drones. El protagonista con su posición/sprite de coreografía;
    # el resto normal.
    for drone_id, drone in world_a.drones.items():
        if drone.position not in layout_a.location_positions:
            continue
        if drone_id != protagonist:
            _draw_drone_with_arms_and_packages(
                surface, layout_a, world_a, drone_id, theme,
                sprite_manager=sprite_manager,
            )
            continue
        # Protagonista: forzamos el estado visual (interacting o no) y la
        # posición de coreografía. Construimos el state visual.
        visual_state = DroneState.INTERACTING if interacting else DroneState.IDLE
        if box_on_drone:
            _draw_drone_choreo(
                surface, layout_a, world_a, drone_id, theme,
                position=drone_pos, visual_state=visual_state,
                force_box_for_package=pkg_id,
                sprite_manager=sprite_manager,
                direction_angle=direction_angle,
                tilt_angle=tilt_angle,
            )
        else:
            _draw_drone_choreo(
                surface, layout_a, world_a, drone_id, theme,
                position=drone_pos, visual_state=visual_state,
                skip_package=pkg_id,
                sprite_manager=sprite_manager,
                direction_angle=direction_angle,
                tilt_angle=tilt_angle,
            )

    # La caja, SOLO si no está sobre el dron (si lo está, la pinta el
    # compuesto). Se dibuja en su POSICIÓN FIJA del layout (la del rombo
    # del suelo donde está apilada), NO relativa al dron. El dron es
    # quien va a la caja; la caja no se mueve.
    #
    # EXCEPCIÓN Load/Unload: cuando la caja "no está sobre el dron" en una
    # transferencia con carrier, en realidad está DENTRO del carrier, así
    # que NO se dibuja (queda oculta; el sprite carrier_N indica el llenado).
    if not box_on_drone:
        target = transition.from_place if picking_up else transition.to_place
        if target.kind != "transp":
            pkg = world_a.packages[pkg_id]
            if picking_up:
                # PickUp: la caja está AtLocation en snap_a (suelo origen).
                box_pos = layout_a.package_position(pkg_id, theme)
            else:
                # Deliver: la caja va a la PERSONA (anima hacia ella; al
                # completarse la representa su sprite, no queda en el suelo).
                box_pos = _drop_box_pos(layout_b, transition, theme)
            draw_package(surface, box_pos, pkg.contains.id, theme, sprite_manager=sprite_manager)


def _box_approach_position(layout, world, transition, theme, *, layout_b=None) -> Point:
    """Posición a la que el dron se aproxima para coger/dejar la caja.

    Alineada en X con el OBJETIVO (su posición en el suelo o el carrier), y
    en Y tal que el borde inferior del dron base quede 6px por encima del
    borde inferior de la caja (para que, al extender brazos +6px, los bordes
    inferiores coincidan y el enganche sea exacto).

    El OBJETIVO es el extremo que NO es un brazo:
      - PickUp/Unload: el origen (from_place) — de donde coge la caja.
      - Deliver/Load:  el destino (to_place) — donde la deja.
    Si ese extremo es una loc, el objetivo es el slot de la caja en el
    suelo; si es un transporter (Load/Unload), el objetivo es el CARRIER:
    el dron se acerca al carrier y hace la misma animación que en el suelo.
    """
    box_ground = _approach_target_pos(layout, layout_b, world, transition, theme)
    # El dron se alinea en X con la caja; en Y, su CENTRO queda tal que el
    # borde inferior del dron base esté 6px sobre el borde inferior caja.
    a = theme.sprite_anchors
    body_w = a.body_size[0]
    body_base_h = a.body_size[1]
    box_h = a.box_size[1]
    scale = (2 * theme.drone_radius) / body_w
    box_bottom = box_ground.y + (box_h / 2.0) * scale
    drone_center_y = box_bottom - 6.0 * scale - (body_base_h / 2.0) * scale
    return Point(box_ground.x, drone_center_y)


def _approach_target_pos(layout, layout_b, world, transition, theme) -> Point:
    """Posición del OBJETIVO de la coreografía (suelo o carrier).

    El objetivo es el extremo no-brazo: el origen si se coge, el destino si
    se deja. Para loc → slot de la caja en el suelo; para transp → carrier.
    """
    pkg_id = transition.package_id
    if transition.from_place.kind != "arm":
        # Coger: objetivo = origen.
        src = transition.from_place
        if src.kind == "transp":
            return layout.transporter_position(src.transporter_id, theme)
        return layout.package_position(pkg_id, theme)  # loc: caja en suelo (snap_a)
    # Dejar: objetivo = destino.
    dst = transition.to_place
    if dst.kind == "transp":
        return layout.transporter_position(dst.transporter_id, theme)
    # Deliver: la caja va a la PERSONA (que cambia de sprite al recibir). El
    # dron se ACERCA a la persona y la caja anima hacia ella; no queda en el
    # suelo (la entrega la representa el sprite de la persona).
    if transition.person_id is not None:
        return layout.person_position(transition.person_id, theme)
    # loc: la caja queda AtLocation en snap_b; usamos su slot real.
    if layout_b is not None:
        try:
            return layout_b.package_position(pkg_id, theme)
        except (KeyError, ValueError):
            pass
    return layout._location_ground_center(dst.loc_id, theme)


def _drop_box_pos(layout_b, transition, theme) -> Point:
    """Posición de la caja que se DEJA (rama no-brazo de un drop).

    Refleja el destino real: la PERSONA en un Deliver (la caja anima hacia ella
    y, al completarse, la representa el sprite de la persona; no queda caja en
    el suelo) o el slot del suelo en una loc sin persona (defensivo)."""
    if transition.person_id is not None:
        return layout_b.person_position(transition.person_id, theme)
    return layout_b.package_position(transition.package_id, theme)


# ---------------------------------------------------------------------------
# Posiciones de la coreografía (compartidas con el seguimiento de cámara)
#
# Estos helpers ESPEJAN la geometría de _render_frame_package_move: el
# painter los usa implícitamente al dibujar (misma fórmula de steps) y
# locate_entity_interpolated los llama para seguir al drone o a la caja.
# Si cambia la coreografía en _render_frame_package_move, actualizar aquí.
# ---------------------------------------------------------------------------


def _package_move_drone_pos(
    layout_a: WorldLayout,
    layout_b: WorldLayout,
    world_a,
    transition,
    raw_progress: float,
    theme: Theme,
    entry_exit: tuple[Point, Point] | None = None,
) -> Point:
    """Posición del drone protagonista durante un PackageMove.

    Coreografía de 4 steps entry↔approach↔exit, igual para PickUp/Deliver
    (objetivo = suelo) y Load/Unload (objetivo = carrier): en ambos casos
    el dron se aproxima al objetivo que resuelve _box_approach_position.

    entry_exit: (entry_pos, exit_pos) neighbor-aware (encadenado de acciones
        consecutivas en la misma loc). Si None, ambos = centro de la loc
        (comportamiento clásico, sin contexto de vecinos).
    """
    protagonist = transition.drone_id
    center_pos = layout_a.drone_position(protagonist, theme)
    approach_pos = _box_approach_position(
        layout_a, world_a, transition, theme, layout_b=layout_b
    )
    entry_pos, exit_pos = entry_exit if entry_exit is not None else (center_pos, center_pos)
    p = max(0.0, min(1.0, raw_progress))
    step = min(3, int(p * 4.0))
    local_eased = ease_in_out_cubic((p * 4.0) - step)
    if step == 0:
        return lerp_point(entry_pos, approach_pos, local_eased)
    if step == 3:
        return lerp_point(approach_pos, exit_pos, local_eased)
    return approach_pos  # steps 1 y 2: quieto en approach


def _package_move_box_pos(
    layout_a: WorldLayout,
    layout_b: WorldLayout,
    world_a,
    world_b,
    transition,
    raw_progress: float,
    eased_progress: float,
    theme: Theme,
    entry_exit: tuple[Point, Point] | None = None,
) -> Point:
    """Posición de la caja que se mueve durante un PackageMove.

    Espeja la coreografía: si la caja va sobre el drone (según el step), su
    ancla de brazo siguiendo la posición coreográfica; si no, está en el
    OBJETIVO (slot de suelo para loc, o el carrier para Load/Unload — en
    este caso la caja no se dibuja, pero el seguimiento de cámara la sitúa
    sobre el carrier).
    """
    pkg_id = transition.package_id
    picking_up = transition.from_place.kind != "arm"
    p = max(0.0, min(1.0, raw_progress))
    step = min(3, int(p * 4.0))
    # box_on_drone por step (idéntico a _render_frame_package_move).
    if step in (0, 1):
        box_on_drone = not picking_up
    else:  # steps 2 y 3
        box_on_drone = picking_up

    if box_on_drone:
        drone_pos = _package_move_drone_pos(
            layout_a, layout_b, world_a, transition, raw_progress, theme, entry_exit
        )
        arm_place = transition.to_place if picking_up else transition.from_place
        arm_ids = [a.id for a in world_a.drones[transition.drone_id].arms]
        idx = arm_ids.index(arm_place.arm_id) if arm_place.arm_id in arm_ids else 0
        return position_of_arm(drone_pos, idx, max(1, len(arm_ids)), theme)
    # Caja en el objetivo (no sobre el dron): suelo (loc) o carrier (transp).
    target = transition.from_place if picking_up else transition.to_place
    if target.kind == "transp":
        return layout_a.transporter_position(target.transporter_id, theme)
    if picking_up:
        return layout_a.package_position(pkg_id, theme)
    return _drop_box_pos(layout_b, transition, theme)


def _draw_drone_choreo(
    surface, layout, world, drone_id, theme, *,
    position, visual_state, force_box_for_package=None,
    skip_package=None, sprite_manager=None,
    direction_angle: float | None = None,
    tilt_angle: float = 0.0,
) -> None:
    """Dibuja el dron protagonista durante la coreografía, con estado y
    posición visuales forzados (independientes del world de snap_a).

    visual_state: DroneState a representar (IDLE/base o INTERACTING).
    force_box_for_package: si la caja ya es del dron, su capa box.
    skip_package: si la caja aún no es del dron, no contarla.
    direction_angle: ángulo del vector de movimiento en grados (0=E, 90=N)
        durante los steps de desplazamiento de la coreografía (0 y 3);
        None en los steps de quietud (1, 2) o cuando no hay traslación.
        Determina la cara direccional (igual que en Move).
    tilt_angle: inclinación del dron en grados (positivo = horario yendo
        a la derecha), 0 si está quieto o sin componente horizontal. Se
        propaga tal cual al `draw_drone`. La fórmula la calcula
        `_render_frame_package_move` igual que en Move pero sobre la
        fracción local del step de desplazamiento, no sobre el progress
        global de la acción.
    """
    drone = world.drones[drone_id]
    box_count = 0
    forced_already_held = False
    for arm in drone.arms:
        pkg = _held_by(world, drone_id, arm.id)
        if pkg is not None and pkg.id != skip_package:
            box_count += 1
            if pkg.id == force_box_for_package:
                forced_already_held = True
    # force_box_for_package suma una caja SOLO si esa caja no está ya
    # contada en un brazo. En PickUp la caja aún no es del dron (hay que
    # forzarla); en Deliver la caja sigue HeldByArm en el world de la
    # coreografía, así que el bucle ya la contó y volver a sumarla pintaría
    # una segunda caja fantasma (una en cada brazo). Ese doble conteo era
    # el bug de la entrega.
    if force_box_for_package is not None and not forced_already_held:
        box_count += 1
    held_object = "box" if box_count > 0 else None

    draw_drone(
        surface, position, drone_id, visual_state, theme,
        sprite_manager=sprite_manager,
        direction_angle=direction_angle,
        finished=False,
        held_object=held_object,
        carrier_level=0,
        box_count=box_count,
        tilt_angle=tilt_angle,
    )
    if held_object == "box" and box_count >= 3:
        _draw_box_count_overlay(surface, position, box_count, theme)
# =====================================================================
# RENDER POR ENTIDAD (compositor uniforme)
# ---------------------------------------------------------------------
# render_frame anima UN protagonista por par de snapshots ADYACENTES.
# Ese modelo se rompe cuando:
#   - varias entidades cambian geometría a la vez (concurrencia), o
#   - la acción de una entidad abarca VARIAS transiciones porque otras
#     entidades intercalan sus snapshots (su "flip" de posición cae en
#     una transición de duración 0 → teletransporte).
#
# render_world_at resuelve ambos de forma uniforme: reconstruye, para CADA
# drone, sus segmentos de acción en tiempo virtual (MOVE / ACT / IDLE) y, en
# cada frame, dibuja cada drone interpolado sobre el span COMPLETO de su
# acción, con independencia de dónde caigan los snapshots de los demás. Los
# planes secuenciales de un solo drone producen exactamente la misma
# coreografía que render_frame (reutiliza sus mismos helpers de posición).
# =====================================================================

from droneplan_viz.commands import MoveWithTransporter as _MoveWithTransp  # noqa: E402

#: Periodo (s de tiempo virtual) del parpadeo de la cara de error
#: (alterna face_error1/face_error2). ~3 alternancias por segundo.
_ERROR_BLINK_PERIOD = 0.35

#: Duración FIJA (s de tiempo virtual) de la animación con la que un dron
#: parado cede/recupera su hueco del anillo cuando otro dron entra/sale de su
#: location. Se sitúa al FINAL del Move del otro dron: el dron parado mantiene
#: su sitio durante casi todo el Move y solo se reacomoda en estos últimos
#: segundos, justo cuando el otro llega/termina de irse.
_COLOCATION_EASE_DURATION = 1.2

#: Adelanto (s de tiempo virtual): cuántos segundos ANTES del fin del Move
#: termina la animación de reacomodo. 0.0 = termina justo en la llegada
#: (comportamiento por defecto). Subirlo hace que la animación EMPIECE (y
#: acabe) antes, manteniendo la misma duración _COLOCATION_EASE_DURATION.
_COLOCATION_EASE_LEAD = 1.8


#: Caché de segmentos por historial. Los segmentos son función pura del
#: historial (inmutable durante la reproducción): reconstruirlos en cada
#: frame era ~76% del coste del render en escenarios grandes. Guardamos solo
#: el historial activo (un único entry: se limpia al cambiar de historial),
#: con identidad confirmada por `is` y validada por longitud. Conservar una
#: referencia fuerte al historial cacheado impide además la reutilización de
#: id() mientras la entrada viva.
_SEGMENTS_CACHE: "dict[int, tuple]" = {}


#: Índice inverso (drone_id, arm_id) -> Package para el world dado. El método
#: World.package_held_by recorre TODOS los paquetes en cada consulta; el render
#: lo invoca varias veces por dron (una por brazo) y por frame, lo que en
#: escenarios con muchos paquetes es O(brazos·paquetes) por dron. Construir el
#: índice una vez por world (O(paquetes)) y resolver cada brazo en O(1) elimina
#: ese coste; además se reutiliza entre todos los drones que comparten world.
#: Caché acotada por id(packages) (identidad confirmada por `is`, con
#: referencia fuerte al mapping que impide la reutilización de id()), con
#: desalojo FIFO. NO se limpia por frame: los worlds de los snapshots (base y
#: los de los drones activos) son estables DENTRO de un segmento, así que el
#: índice se construye una vez por segmento y se reutiliza en todos los frames
#: de ese segmento, en vez de reconstruirse en cada frame.
_HELD_INDEX_CACHE: "dict[int, tuple]" = {}
_HELD_INDEX_CAP = 1024


def _held_by(world, drone_id: str, arm_id: str):
    """Paquete sostenido por (drone_id, arm_id) en `world`, o None.

    Equivalente a world.package_held_by pero con índice cacheado: O(1) tras
    construir el índice una vez por world. Uso exclusivo del render.
    """
    pkgs = world.packages
    key = id(pkgs)
    cached = _HELD_INDEX_CACHE.get(key)
    if cached is None or cached[0] is not pkgs:
        idx: dict[tuple[str, str], object] = {}
        for p in pkgs.values():
            at = p.at
            if isinstance(at, HeldByArm):
                idx[(at.drone_id, at.arm_id)] = p
        if len(_HELD_INDEX_CACHE) >= _HELD_INDEX_CAP:
            # Desaloja la entrada más antigua (FIFO sobre orden de inserción).
            oldest = next(iter(_HELD_INDEX_CACHE))
            del _HELD_INDEX_CACHE[oldest]
        cached = (pkgs, idx)
        _HELD_INDEX_CACHE[key] = cached
    return cached[1].get((drone_id, arm_id))


def _drone_segments(timeline) -> "dict[str, list[dict]]":
    """Reconstruye, por drone, sus segmentos de actividad en tiempo virtual.

    Un segmento es un tramo MAXIMAL de un mismo "modo" del drone:
      - 'move': el drone está en MOVING. origin = su loc durante el tramo,
        dest = su loc al volver a IDLE. transporter_id si arrastra carrier.
      - 'act':  el drone está en INTERACTING (PickUp/Deliver/Load/Unload).
      - 'idle': el drone está IDLE/ERROR (parado en su loc).
    Endpoints en tiempo virtual vía timeline.snapshot_times, de modo que la
    animación de cada acción cubre su DURACIÓN real aunque otros drones
    intercalen snapshots en medio (clave para la concurrencia).

    El resultado se cachea por historial: dentro de una misma reproducción el
    historial no cambia, así que se calcula una sola vez y los frames sucesivos
    reutilizan el mismo mapa.
    """
    history = timeline.history
    n = len(history)
    if n == 0:
        return {}

    cached = _SEGMENTS_CACHE.get(id(history))
    if cached is not None and cached[0] is history and cached[1] == n:
        return cached[2]

    times = timeline.snapshot_times
    # Filas de estado por snapshot: una pasada O(n) sobre el historial en vez
    # de O(drones·n) accesos a history.at() repetidos.
    rows = [history.at(i).world.drones for i in range(n)]
    drone_ids = list(rows[0].keys())
    out: dict[str, list[dict]] = {d: [] for d in drone_ids}
    MOV, INT = DroneState.MOVING, DroneState.INTERACTING

    def st(i, d):
        drn = rows[i].get(d)
        return drn.state if drn is not None else None

    for d in drone_ids:
        i = 0
        while i < n:
            s = st(i, d)
            if s == MOV:
                j = i
                while j < n and st(j, d) == MOV:
                    j += 1
                i_end = min(j, n - 1)
                prod = history.at(i).produced_by
                tid = (prod.transporter_id
                       if isinstance(prod, _MoveWithTransp) else None)
                out[d].append({
                    "kind": "move", "i_start": i, "i_end": i_end,
                    "v0": times[i], "v1": times[i_end],
                    "transporter_id": tid,
                })
                i = j
            elif s == INT:
                j = i
                while j < n and st(j, d) == INT:
                    j += 1
                i_end = min(j, n - 1)
                out[d].append({
                    "kind": "act", "i_start": i, "i_end": i_end,
                    "v0": times[i], "v1": times[i_end],
                })
                i = j
            else:  # IDLE / ERROR
                j = i
                while j < n and st(j, d) not in (MOV, INT):
                    j += 1
                i_end = min(j, n - 1) if j < n else n - 1
                out[d].append({
                    "kind": "idle", "i_start": i, "i_end": i_end,
                    "v0": times[i], "v1": times[i_end],
                })
                i = j
    _SEGMENTS_CACHE.clear()
    _SEGMENTS_CACHE[id(history)] = (history, n, out)
    return out


def _active_segment(segs: "list[dict]", t: float):
    """Segmento activo en t y su progreso local [0,1]."""
    if not segs:
        return None, 0.0
    seg = segs[0]
    for s in segs:
        if s["v0"] <= t + 1e-9:
            seg = s
        else:
            break
    v0, v1 = seg["v0"], seg["v1"]
    local = 0.0 if v1 <= v0 else max(0.0, min(1.0, (t - v0) / (v1 - v0)))
    return seg, local


def _adjacent_act(segs: "list[dict]", idx: int, step: int, history):
    """Par (snap_a, snap_b) de la acción de paquete vecina (mismo drone),
    saltando segmentos idle; None si no hay."""
    k = idx + step
    while 0 <= k < len(segs):
        if segs[k]["kind"] == "act":
            s = segs[k]
            return (history.at(s["i_start"]), history.at(s["i_end"]))
        if segs[k]["kind"] == "move":
            return None  # un Move corta el encadenado intra-loc
        k += step
    return None


def _adjacent_nonidle(segs: "list[dict]", idx: int, step: int):
    """Segmento NO-idle inmediatamente anterior/siguiente (saltando idles)."""
    k = idx + step
    while 0 <= k < len(segs):
        if segs[k]["kind"] != "idle":
            return segs[k]
        k += step
    return None


def _carrier_grab_pos(layout, loc_id, drone_id, tid, theme):
    """Posición del dron cuando agarra el carrier `tid` en `loc_id` (su
    reposo elevado `lift` px, idéntico a la fase de agarre de
    _carrier_drag_positions)."""
    a = theme.sprite_anchors
    lift = theme.drone_radius - a.carrier_overlap + a.carrier_size[1] / 2.0
    rest = _transporter_position_in_loc(layout, loc_id, tid, theme)
    return Point(rest.x, rest.y - lift)


def _relayout(base_layout, world):
    """WorldLayout para `world` reutilizando las posiciones de `base_layout`.

    Las localizaciones no cambian durante la reproducción, así que
    location_positions es idéntico para todos los snapshots del mismo
    escenario. Reutilizar el mapping ya calculado (y ya envuelto en
    MappingProxyType) evita recomputar el layout circular/afín y reasignar
    memoria por cada drone en movimiento de cada frame.
    """
    return WorldLayout(
        location_positions=base_layout.location_positions,
        viewbox=base_layout.viewbox,
        world_ref=world,
    )


def _settled_colocated_at(history, i_end: int, loc_id: str) -> "list[str]":
    """IDs de drones que estarán en `loc_id` una vez aplicados TODOS los
    eventos del MISMO instante simulado que el snapshot `i_end`.

    El runtime secuencia las llegadas simultáneas en snapshots consecutivos
    con el mismo `timestamp`. El grupo de co-localización "asentado" en destino
    es el del ÚLTIMO snapshot de esa racha. Usarlo (en vez del grupo del
    snapshot intermedio, que aún no incluye a los que llegan a la vez) hace
    que un dron que se mueve vuele DIRECTO a su hueco final del anillo, sin
    apuntar antes al centro/hueco de un grupo incompleto y saltar después.
    """
    t_end = history.at(i_end).timestamp
    j = i_end
    n = history.head_index
    while j + 1 <= n and history.at(j + 1).timestamp == t_end:
        j += 1
    world = history.at(j).world
    return sorted(dr.id for dr in world.drones.values() if dr.position == loc_id)


def _drone_spec_at(d, dsegs, t, history, viewbox, spread, theme,
                   base_world, base_layout, all_segs=None, settled_groups=None):
    """Resuelve cómo dibujar el drone `d` en el tiempo virtual `t`.

    Devuelve (spec, dragged, choreo_box, choreo_pkg):
      - spec: dict con mode 'idle'|'plain'|'choreo' y sus parámetros.
      - dragged: (tid, (carrier_pos, grounded)) si arrastra un carrier, o None.
      - choreo_box: (content_id, Point) si hay una caja suelta que dibujar.
      - choreo_pkg: id de paquete en coreografía activa, o None.

    Encadenado carrier↔acción: cuando una acción de paquete (Load/Unload) es
    adyacente a un MoveWithTransporter en la MISMA loc, el dron entra/sale por
    el punto de AGARRE del carrier (no por el centro), y el arrastre del
    carrier remapea su progreso para saltarse la fase de aproximación/soltar
    ya cubierta por la acción. Así no hay detour por el centro entre cargar y
    salir (ni entre llegar y descargar).
    """
    seg, local = _active_segment(dsegs, t)
    if seg is None:
        return dict(mode="idle"), None, None, None
    idx = dsegs.index(seg)

    if seg["kind"] == "move":
        snap_a = history.at(seg["i_start"])
        world_a = snap_a.world
        origin = world_a.drones[d].position
        dest = history.at(seg["i_end"]).world.drones[d].position
        tid = seg["transporter_id"]
        layout_a = _relayout(base_layout, world_a)
        tr = TransitionDroneMove(
            drone_id=d, from_loc_id=origin, to_loc_id=dest, transporter_id=tid,
        )
        if tid is not None:
            # Encadenado: ¿una acción (Load) en el ORIGEN justo antes, o una
            # acción (Unload) en el DESTINO justo después? Si es así, el dron
            # ya está / se queda junto al carrier → remapeamos el progreso
            # para saltarnos la aproximación (0–0.25) y/o el soltar (0.75–1).
            prev_ni = _adjacent_nonidle(dsegs, idx, -1)
            next_ni = _adjacent_nonidle(dsegs, idx, +1)
            from_chained = (
                prev_ni is not None and prev_ni["kind"] == "act"
                and history.at(prev_ni["i_start"]).world.drones[d].position == origin
            )
            to_chained = (
                next_ni is not None and next_ni["kind"] == "act"
                and history.at(next_ni["i_start"]).world.drones[d].position == dest
            )
            ga, gb = 0.25, 0.75
            lo = ga if from_chained else 0.0
            hi = gb if to_chained else 1.0
            eff = lo + local * (hi - lo)

            def drag(pp):
                return _carrier_drag_positions(layout_a, world_a, tr, pp, theme)
            drone_pos, carrier_pos = drag(eff)
            eps = 1e-3
            ah, _ = drag(min(1.0, eff + eps))
            bh, _ = drag(max(0.0, eff - eps))
            dx, dy = ah.x - bh.x, ah.y - bh.y
            dir_a = (None if (dx == 0 and dy == 0)
                     else math.degrees(math.atan2(-dy, dx)) % 360.0)
            tilt = lateral_tilt_deg(
                dx, dy, local, theme.drone_move_max_tilt_deg,
                theme.drone_tilt_envelope_steepness)
            carried = 0.25 < eff <= 0.75
            spec = dict(
                mode="plain", layout=layout_a, world=world_a, pos=drone_pos,
                dir=dir_a, tilt=tilt, dragged_transp=None,
            )
            return spec, (tid, (carrier_pos, not carried)), None, None

        d_from = _drone_position_in_loc(layout_a, world_a, origin, d, theme)
        # Destino: usar el grupo de co-localización YA ASENTADO en `dest`
        # (incluye los drones que llegan en el MISMO instante), para que el
        # dron vuele DIRECTO a su hueco final del anillo y no al centro/hueco
        # de un grupo incompleto que después provocaría un salto.
        settled = _settled_colocated_at(history, seg["i_end"], dest)
        if d not in settled:
            settled = sorted(settled + [d])
        d_to = layout_a.drone_position_for_colocation(dest, d, settled, theme)
        pos = lerp_point(d_from, d_to, ease_in_out_cubic(local))
        dx, dy = d_to.x - d_from.x, d_to.y - d_from.y
        dir_a = (None if (dx == 0 and dy == 0)
                 else math.degrees(math.atan2(-dy, dx)) % 360.0)
        tilt = lateral_tilt_deg(
            dx, dy, local, theme.drone_move_max_tilt_deg,
            theme.drone_tilt_envelope_steepness)
        spec = dict(
            mode="plain", layout=layout_a, world=world_a, pos=pos,
            dir=dir_a, tilt=tilt, dragged_transp=None,
        )
        return spec, None, None, None

    if seg["kind"] == "act":
        snap_a = history.at(seg["i_start"])
        snap_b = history.at(seg["i_end"])
        tr = classify_transition(snap_a, snap_b)
        if not isinstance(tr, TransitionPackageMove):
            return dict(mode="idle"), None, None, None
        wa, wb = snap_a.world, snap_b.world
        la = _relayout(base_layout, wa)
        lb = _relayout(base_layout, wb)
        pkg_id = tr.package_id
        picking_up = tr.from_place.kind != "arm"
        act_loc = wa.drones[d].position
        # Ancla de la coreografía en el hueco del grupo ASENTADO (igual que el
        # destino del Move): si en el MISMO instante van a llegar más drones a
        # act_loc, la acción (recoger/entregar) se ancla ya en el hueco final
        # del anillo, evitando el salto Move→acción del reparto masivo.
        sg = settled_groups.get(act_loc) if settled_groups is not None else None
        if sg is not None and d in sg and len(sg) > sum(
            1 for dr in wa.drones.values() if dr.position == act_loc
        ):
            center_pos = la.drone_position_for_colocation(act_loc, d, sg, theme)
        else:
            center_pos = la.drone_position(d, theme)
        approach_pos = _box_approach_position(la, wa, tr, theme, layout_b=lb)
        prev_act = _adjacent_act(dsegs, idx, -1, history)
        next_act = _adjacent_act(dsegs, idx, +1, history)
        entry_pos, exit_pos = _choreo_entry_exit(
            la, wa, tr, center_pos, approach_pos,
            prev_act, next_act, viewbox, theme,
        )
        # Encadenado con carrier: si la acción vecina es un MoveWithTransporter
        # en esta misma loc, entrar/salir por el AGARRE del carrier (no centro).
        prev_ni = _adjacent_nonidle(dsegs, idx, -1)
        next_ni = _adjacent_nonidle(dsegs, idx, +1)
        if (prev_ni is not None and prev_ni["kind"] == "move"
                and prev_ni["transporter_id"] is not None
                and history.at(prev_ni["i_end"]).world.drones[d].position == act_loc):
            entry_pos = _carrier_grab_pos(la, act_loc, d, prev_ni["transporter_id"], theme)
        if (next_ni is not None and next_ni["kind"] == "move"
                and next_ni["transporter_id"] is not None
                and history.at(next_ni["i_start"]).world.drones[d].position == act_loc):
            exit_pos = _carrier_grab_pos(la, act_loc, d, next_ni["transporter_id"], theme)

        p = max(0.0, min(1.0, local))
        step = min(3, int(p * 4.0))
        lo = (p * 4.0) - step
        lo_e = ease_in_out_cubic(lo)
        dir_a = None
        tilt = 0.0
        if step in (0, 3):
            if step == 0:
                d_from, d_to = entry_pos, approach_pos
            else:
                d_from, d_to = approach_pos, exit_pos
            drone_pos = lerp_point(d_from, d_to, lo_e)
            ddx, ddy = d_to.x - d_from.x, d_to.y - d_from.y
            if not (ddx == 0 and ddy == 0):
                dir_a = math.degrees(math.atan2(-ddy, ddx)) % 360.0
            tilt = lateral_tilt_deg(
                ddx, ddy, lo, theme.drone_move_max_tilt_deg,
                theme.drone_tilt_envelope_steepness)
            interacting = False
            box_on_drone = (not picking_up) if step == 0 else picking_up
        elif step == 1:
            drone_pos = approach_pos
            interacting = True
            box_on_drone = not picking_up
        else:
            drone_pos = approach_pos
            interacting = False
            box_on_drone = picking_up
        spec = dict(
            mode="choreo", layout=la, world=wa, pos=drone_pos,
            visual_state=(DroneState.INTERACTING if interacting else DroneState.IDLE),
            box_on_drone=box_on_drone, pkg_id=pkg_id, dir=dir_a, tilt=tilt,
        )
        choreo_box = None
        if not box_on_drone:
            target = tr.from_place if picking_up else tr.to_place
            if target.kind != "transp":
                box_pos = (la.package_position(pkg_id, theme) if picking_up
                           else _drop_box_pos(lb, tr, theme))
                choreo_box = (wa.packages[pkg_id].contains.id, box_pos)
        return spec, None, choreo_box, pkg_id

    # idle: por defecto, centro de la loc. Pero si el tramo idle está entre
    # dos segmentos que ENCADENAN en la misma loc (p.ej. último Load → Move
    # del carrier, o Move → primer Unload), el dron debe MANTENER la posición
    # de relevo (el agarre del carrier / la aproximación) en vez de rebotar al
    # centro. La posición de relevo es la ENTRADA del siguiente segmento (o,
    # en su defecto, la SALIDA del anterior); cuando no hay encadenado ambas
    # coinciden con el centro, así que el idle "real" no cambia.
    next_ni = _adjacent_nonidle(dsegs, idx, +1)
    prev_ni = _adjacent_nonidle(dsegs, idx, -1)
    hold = None
    if next_ni is not None:
        nsub, _d, _b, _p = _drone_spec_at(
            d, dsegs, next_ni["v0"], history, viewbox, spread, theme,
            base_world, base_layout, all_segs, settled_groups)
        if nsub.get("mode") in ("plain", "choreo"):
            hold = nsub["pos"]
    if hold is None and prev_ni is not None:
        psub, _d, _b, _p = _drone_spec_at(
            d, dsegs, max(prev_ni["v0"], prev_ni["v1"] - 1e-6),
            history, viewbox, spread, theme, base_world, base_layout, all_segs,
            settled_groups)
        if psub.get("mode") in ("plain", "choreo"):
            hold = psub["pos"]
    if hold is not None:
        # Dibujar en la posición de relevo PERO con el estado/geometría
        # ACTUAL en t (base_world): así un dron que falló (ERROR) durante el
        # tramo idle muestra su cara de fallo, no el estado del inicio del
        # segmento.
        spec = dict(
            mode="plain", layout=base_layout, world=base_world, pos=hold,
            dir=None, tilt=0.0, dragged_transp=None,
        )
        return spec, None, None, None
    # Co-localización: si OTRO dron entra/sale de la misma location, el
    # número de drones co-localizados cambia y con él el hueco del anillo de
    # este dron parado. En vez de saltar de golpe al cambiar el snapshot,
    # interpolamos su posición entre el reparto "antes" y "después" según el
    # progreso del Move del otro dron.
    # Grupo ASENTADO: si en el MISMO instante van a llegar más drones a la
    # location de `d` (llegadas simultáneas que el runtime secuencia en
    # snapshots consecutivos), colócalo YA en su hueco del grupo FINAL. Sin
    # esto, el dron parado se dibuja en el hueco del grupo parcial (p.ej. el
    # centro cuando aún está solo) y salta al hueco del anillo cuando llegan
    # los demás — el snapping del reparto masivo con varios drones.
    if settled_groups is not None:
        dloc_s = base_world.drones[d].position
        sg = settled_groups.get(dloc_s)
        if sg is not None and d in sg:
            present = sum(
                1 for dr in base_world.drones.values() if dr.position == dloc_s
            )
            if len(sg) > present:
                pos = base_layout.drone_position_for_colocation(
                    dloc_s, d, sg, theme
                )
                spec = dict(
                    mode="plain", layout=base_layout, world=base_world, pos=pos,
                    dir=None, tilt=0.0, dragged_transp=None,
                )
                return spec, None, None, None

    eased = _colocation_eased_pos(
        d, t, all_segs, history, base_world, base_layout, theme)
    if eased is not None:
        spec = dict(
            mode="plain", layout=base_layout, world=base_world, pos=eased,
            dir=None, tilt=0.0, dragged_transp=None,
        )
        return spec, None, None, None
    return dict(mode="idle"), None, None, None


def _colocation_eased_pos(d, t, all_segs, history, base_world, base_layout, theme):
    """Posición de reposo INTERPOLADA del dron parado `d` cuando otro dron
    entra o sale de su misma location.

    Devuelve un Point si hay exactamente UN dron en Move entrante/saliente de
    la location de `d` (mezcla el hueco de anillo "antes"/"después" según el
    progreso de ese Move), o None si no procede aplicar easing (ni un mover
    relevante, o varios a la vez → se deja el reparto base, sin suavizar).

    Continuidad: en progreso 0 devuelve el reparto que refleja base_world
    (el dron parado donde ya estaba); en progreso 1 devuelve el reparto al
    que base_world salta en el snapshot de fin de Move. Así empalma sin salto
    por ambos extremos.
    """
    dloc = base_world.drones[d].position
    if dloc is None or not all_segs:
        return None
    # `present`: drones en dloc según base_world. Durante un Move, el dron en
    # tránsito está en su ORIGEN en base_world: por eso un SALIENTE (origen =
    # dloc) sigue contándose como presente, y un ENTRANTE (origen != dloc)
    # aún no. Eso es justo el reparto "antes".
    present = sorted(
        did for did, dr in base_world.drones.items() if dr.position == dloc
    )
    relevant = []
    for other, osegs in all_segs.items():
        if other == d or not osegs:
            continue
        seg2, _local = _active_segment(osegs, t)
        if seg2 is None or seg2["kind"] != "move":
            continue
        origin = history.at(seg2["i_start"]).world.drones[other].position
        dest = history.at(seg2["i_end"]).world.drones[other].position
        v0, v1 = seg2["v0"], seg2["v1"]
        if dest == dloc and origin != dloc:
            relevant.append(("arrive", other, v0, v1))
        elif origin == dloc and dest != dloc:
            relevant.append(("leave", other, v0, v1))
    if len(relevant) != 1:
        return None
    kind, mover, v0, v1 = relevant[0]
    if kind == "arrive":
        before = present
        after = sorted(present + [mover])
    else:  # leave
        before = present
        after = sorted(x for x in present if x != mover)
    if d not in before or d not in after:
        return None
    # Ventana de DURACIÓN FIJA. Por defecto termina en v1 (la llegada); con
    # _COLOCATION_EASE_LEAD > 0 termina (y por tanto empieza) ese adelanto
    # antes. span se mantiene fijo en _COLOCATION_EASE_DURATION salvo que el
    # Move sea más corto. Para t >= ease_end, q se satura a 1 (reparto
    # "después"), y en v1 el snapshot ya coincide con él (empalma sin salto).
    ease_end = v1 - _COLOCATION_EASE_LEAD
    ease_start = max(v0, ease_end - _COLOCATION_EASE_DURATION)
    span = ease_end - ease_start
    q = 1.0 if span <= 0 else max(0.0, min(1.0, (t - ease_start) / span))
    e = q * q * (3.0 - 2.0 * q)  # smoothstep
    if e <= 0.0:
        # Aún en "antes": deja que la rama idle base lo dibuje (misma pos).
        return None
    pa = base_layout.drone_position_for_colocation(dloc, d, before, theme)
    pb = base_layout.drone_position_for_colocation(dloc, d, after, theme)
    return Point(pa.x + (pb.x - pa.x) * e, pa.y + (pb.y - pa.y) * e)


#: Caché de la CAPA ESTÁTICA del render: fondo + aristas del grafo +
#: locations. Esta capa es invariante durante toda la reproducción (la
#: topología del grafo no cambia: solo se mueven drones/paquetes), pero se
#: redibujaba en cada frame (61 locations + todas las aristas en escenarios
#: grandes ≈ varios ms/frame). Se rasteriza una vez por
#: (historial, tamaño, theme) a una Surface y se blittea de golpe; las
#: entidades dinámicas se dibujan encima, preservando el z-order exacto.
#: Clave por id(history) (estable durante toda la reproducción), tamaño y
#: id(theme); acotada con desalojo FIFO. Una nueva reproducción, un resize o
#: un cambio de theme invalidan la entrada de forma natural.
_STATIC_LAYER_CACHE: "dict[tuple, pygame.Surface]" = {}
_STATIC_LAYER_CAP = 8

def _static_layer(surface, history, base_layout, base_world, theme, sprite_manager):
    """Surface con fondo + aristas + locations, cacheada por reproducción.

    Devuelve una Surface del tamaño de `surface` lista para blittear como
    base de la escena.

    La visibilidad de las etiquetas (sprites._LABELS_VISIBLE, que depende
    del zoom) forma parte de la clave: las etiquetas de location se pintan
    en esta capa, así que una caché que ignore ese estado las dejaría
    "congeladas" al valor que tuvieran al crearse (bug: no desaparecían al
    hacer zoom-out mientras el resto de etiquetas sí lo hacían).
    """
    from droneplan_viz.render import sprites as _sprites

    labels_visible = _sprites._LABELS_VISIBLE
    key = (id(history), surface.get_size(), id(theme), labels_visible)
    cached = _STATIC_LAYER_CACHE.get(key)
    if cached is not None and cached[0] is history:
        return cached[1]
    layer = pygame.Surface(surface.get_size())
    draw_background(layer, theme)
    _draw_edges(layer, base_layout, base_world, theme)
    for loc_id, position in base_layout.location_positions.items():
        draw_location(layer, position, loc_id, theme, sprite_manager=sprite_manager)
    if len(_STATIC_LAYER_CACHE) >= _STATIC_LAYER_CAP:
        del _STATIC_LAYER_CACHE[next(iter(_STATIC_LAYER_CACHE))]
    _STATIC_LAYER_CACHE[key] = (history, layer)
    return layer


#: Factor máximo de supersampling del buffer de render cuando hay muchas
#: localizaciones. El buffer se dibuja a (viewport × SS) con los sprites a
#: resolución nativa y luego la cámara lo encaja en el viewport: así la vista
#: por defecto muestra TODO el grafo sin solape (es una reducción de una fuente
#: de alta resolución → nítida) y el zoom in recompone desde el buffer nítido,
#: recuperando la resolución de los sprites. Acota la RESOLUCIÓN del buffer (no
#: la escala de pantalla, que la fija `zoom`); su inverso 1/SS_MAX ≈ 0.33 es el
#: ajuste de las escenas más densas. Limita el coste/memoria con cientos de locs.
_SUPERSAMPLE_MAX: float = 3.0517578125

#: Buffer de render reutilizable (evita asignar una surface grande por frame).
_RENDER_BUFFER: "pygame.Surface | None" = None


def _render_buffer(width: int, height: int) -> pygame.Surface:
    """Surface reutilizable del tamaño pedido (se reasigna solo si cambia)."""
    global _RENDER_BUFFER
    if _RENDER_BUFFER is None or _RENDER_BUFFER.get_size() != (width, height):
        _RENDER_BUFFER = pygame.Surface((width, height))
    return _RENDER_BUFFER


def _supersample_factor(
    world, surface_size: tuple[int, int], theme: Theme
) -> float:
    """Factor de supersampling (>=1.0) para encajar el world en surface_size.

    Si a tamaño nativo las locations se solaparían en el viewport (el girasol
    de compute_layout reporta scale<1), se renderiza en un buffer mayor
    (viewport × factor) donde caben a resolución nativa. El factor es 1/scale,
    acotado a _SUPERSAMPLE_MAX para limitar coste/memoria (a partir de ahí el
    girasol del buffer admite un solape leve antes que encoger sprites).

    Es la ÚNICA fuente del factor: la usan tanto render_world_at (para
    dimensionar el buffer) como las funciones locate_* (para situar el foco
    donde realmente se dibuja), así que ambos quedan siempre coherentes.
    """
    vb = _scene_viewbox(surface_size, theme)
    fit = compute_layout(
        world, vb, spread_factor=theme.location_spread_factor
    ).scale
    if fit < 1.0:
        return min(_SUPERSAMPLE_MAX, 1.0 / fit)
    return 1.0


def supersample_factor(
    world, surface_size: tuple[int, int], theme: "Theme | None" = None
) -> float:
    """Factor de supersampling (>=1.0) de una escena en una surface dada.

    API pública (envoltorio de la lógica interna): la app la usa al cargar para
    convertir entre la escala ABSOLUTA del zoom (rejilla fija, relativa al sprite
    nativo) y la vista completa: el ajuste —todo visible— cae en escala 1/ss.
    También la usa el foco para centrar la cámara en coordenadas de pantalla.
    """
    if theme is None:
        theme = Theme.default()
    return _supersample_factor(world, surface_size, theme)


def render_world_at(
    surface: pygame.Surface,
    timeline,
    t: float,
    *,
    theme: Theme | None = None,
    sprite_manager: "SpriteManager | None" = None,
    zoom: float = 1.0,
    pan: tuple[int, int] = (0, 0),
    anim_time: float | None = None,
    labels_hidden: bool = False,
) -> None:
    """Render UNIFORME por entidad del mundo en el tiempo virtual t.

    Sustituye al par (sample_with_neighbors + render_frame) en demos y app:
    cada drone se interpola sobre el span completo de su acción, lo que da
    coreografía encadenada (sin pasar por el centro entre acciones de la
    misma loc) Y animación correcta de movimientos concurrentes (sin
    teletransporte), de forma uniforme para todos los casos de ejecución.

    zoom/pan: como en render_frame, se renderiza a zoom=1 y se escala la
    imagen resultante (zoom uniforme); zoom=1,pan=(0,0) es el camino rápido
    sin escalado.

    anim_time: reloj (s) que gobierna SOLO el parpadeo de la cara de error.
    Si se pasa un reloj de pared, la cara de error parpadea aunque la
    reproducción esté en pausa o al final del plan (donde t queda congelado).
    Si es None, se usa t (determinista, para tests y render estático).
    """
    if theme is None:
        theme = Theme.default()
    zoom = _clamp_zoom(zoom)
    set_labels_visible(zoom >= _LABEL_MIN_ZOOM and not labels_hidden)

    # ¿Tantas localizaciones que, a tamaño NATIVO, se solaparían en el
    # viewport? _supersample_factor lo indica con un factor SS>1. Si lo hay,
    # renderizamos en un buffer MAYOR (viewport × SS) donde el girasol las
    # separa a resolución nativa, y la cámara lo encaja: la vista por defecto
    # (zoom de usuario 1.0) muestra TODO el grafo como una reducción de una
    # fuente nítida (downscale, no upscale) y el zoom in recompone hacia el
    # píxel nativo 1:1 SIN perder resolución (a zoom = SS el buffer se ve 1:1).
    # Los sprites NO se encogen de forma permanente; el factor solo agranda el
    # lienzo de render. El foco (locate_*) usa el MISMO factor, así que el
    # seguimiento cae donde se dibuja.
    ss = 1.0
    if len(timeline.history) > 0:
        _base, _, _ = timeline.sample(t)
        ss = _supersample_factor(_base.world, surface.get_size(), theme)

    if ss > 1.0:
        bw = max(1, round(surface.get_width() * ss))
        bh = max(1, round(surface.get_height() * ss))
        buffer = _render_buffer(bw, bh)
        _render_world_at_plain(buffer, timeline, t, theme, sprite_manager,
                               anim_time=anim_time)
        # `zoom` es la escala ABSOLUTA del sprite respecto a su resolución
        # nativa (1.0 = píxel nativo 1:1), INDEPENDIENTE de la densidad de la
        # escena. El buffer está a resolución nativa (viewport×ss), así que
        # para que el sprite se vea a escala `zoom` en pantalla, la fuente se
        # compone a viewport×(zoom·ss): a zoom=1/ss se reduce el buffer al
        # viewport (vista completa) y a zoom=1 se muestra el sprite a tamaño
        # nativo 1:1 (zoom in recortando la escena).
        _blit_scaled_scene(surface, buffer, zoom * ss, pan, theme)
        return

    if not _camera_active(zoom, pan):
        _render_world_at_plain(surface, timeline, t, theme, sprite_manager,
                               anim_time=anim_time)
        return
    scene = pygame.Surface(surface.get_size())
    _render_world_at_plain(scene, timeline, t, theme, sprite_manager,
                           anim_time=anim_time)
    _blit_scaled_scene(surface, scene, zoom, pan, theme)


def _render_world_at_plain(
    surface: pygame.Surface,
    timeline,
    t: float,
    theme: Theme,
    sprite_manager: "SpriteManager | None",
    *,
    anim_time: float | None = None,
) -> None:
    history = timeline.history
    if len(history) == 0:
        draw_background(surface, theme)
        return

    viewbox = _scene_viewbox(surface.get_size(), theme)
    spread = theme.location_spread_factor

    # Snapshot base (estado activo en t): geometría estática (locations,
    # personas, paquetes libres) y estado "lógico" de cada entidad.
    base, _b, _p = timeline.sample(t)
    base_world = base.world
    base_layout = compute_layout(base_world, viewbox, spread_factor=spread)

    # NOTA: los sprites se dibujan SIEMPRE a tamaño nativo. Cuando hay tantas
    # locations que a tamaño nativo se solaparían en el viewport, el encaje NO
    # se hace encogiendo sprites (eso hornearía pérdida de resolución), sino
    # renderizando esta escena en un buffer MAYOR (supersampling, ver
    # render_world_at) donde el girasol las separa a resolución nativa; la
    # cámara reduce ese buffer para la vista completa y el zoom in recupera el
    # píxel 1:1. base_layout.scale es solo el factor de encaje que consume
    # render_world_at para dimensionar el buffer; aquí no se aplica.

    segs = _drone_segments(timeline)

    # Fase de parpadeo de la cara de error (alterna face_error1/face_error2).
    # Usa anim_time (reloj de pared) si se provee, para que parpadee incluso
    # en pausa o al final del plan; si no, el tiempo virtual t (determinista).
    blink_src = anim_time if anim_time is not None else t
    error_phase = int(blink_src / _ERROR_BLINK_PERIOD) % 2

    # --- 1) Resolver el estado de render de CADA drone en t -------------
    # Cada entrada: dict con cómo dibujar el drone y, si aplica, info de la
    # caja en coreografía y del carrier arrastrado.
    drone_draw: dict[str, dict] = {}
    dragged: dict[str, tuple[Point, bool]] = {}  # transporter_id -> (pos, grounded)
    choreo_boxes: list[tuple[str, Point]] = []   # (content_id, pos) cajas sueltas
    choreo_pkgs: set[str] = set()                 # paquetes en coreografía activa

    # Grupos de co-localización ASENTADOS del instante activo: el runtime
    # secuencia las llegadas/salidas simultáneas en snapshots consecutivos con
    # el mismo timestamp; para los drones PARADOS usamos el grupo del ÚLTIMO
    # snapshot de esa racha, así se colocan ya en su hueco final y no saltan
    # mientras van llegando los demás.
    _vt = timeline._virtual_times
    _bidx = min(len(_vt) - 1, max(0, bisect.bisect_right(_vt, t) - 1))
    _bts = base.timestamp
    _j = _bidx
    while _j + 1 <= history.head_index and history.at(_j + 1).timestamp == _bts:
        _j += 1
    settled_groups: dict[str, list[str]] = {}
    for _did, _dr in history.at(_j).world.drones.items():
        settled_groups.setdefault(_dr.position, []).append(_did)
    for _grp in settled_groups.values():
        _grp.sort()

    for d, dsegs in segs.items():
        if not dsegs:
            continue
        spec, drg, cbox, cpkg = _drone_spec_at(
            d, dsegs, t, history, viewbox, spread, theme, base_world,
            base_layout, segs, settled_groups
        )
        drone_draw[d] = spec
        if drg is not None:
            dragged[drg[0]] = drg[1]
        if cbox is not None:
            choreo_boxes.append(cbox)
        if cpkg is not None:
            choreo_pkgs.add(cpkg)

    # --- 2) Dibujar la escena en z-order --------------------------------
    # Capa estática (fondo + aristas + locations) cacheada por reproducción:
    # se blittea de una vez en lugar de redibujarla cada frame.
    surface.blit(
        _static_layer(surface, history, base_layout, base_world, theme,
                      sprite_manager),
        (0, 0),
    )

    # Transporters: arrastrados en posición interpolada; el resto en reposo.
    for tid, transporter in base_world.transporters.items():
        if transporter.position not in base_layout.location_positions and tid not in dragged:
            continue
        if tid in dragged:
            tpos, grounded = dragged[tid]
        else:
            tpos, grounded = base_layout.transporter_position(tid, theme), True
        in_t = base_world.packages_in_transporter(tid)
        draw_transporter(
            surface, tpos, tid, contents_count=len(in_t),
            capacity=transporter.capacity, theme=theme,
            shadow=grounded, sprite_manager=sprite_manager,
        )

    # Personas.
    _draw_persons(surface, base_world.persons, base_layout, theme, sprite_manager)

    # Paquetes libres (AtLocation), salvo los que están en coreografía
    #    (se dibujan aparte). Los excluidos siguen contando para el slot.
    _draw_resting_packages(
        surface, base_world.packages, base_layout, theme, sprite_manager,
        persons=base_world.persons, exclude=choreo_pkgs,
    )

    # Drones (cada uno según su estado de render).
    for d in base_world.drones:
        spec = drone_draw.get(d)
        if spec is None or spec["mode"] == "idle":
            drn = base_world.drones.get(d)
            if drn is None or drn.position not in base_layout.location_positions:
                continue
            _draw_drone_with_arms_and_packages(
                surface, base_layout, base_world, d, theme,
                sprite_manager=sprite_manager, error_phase=error_phase,
            )
        elif spec["mode"] == "plain":
            _draw_drone_with_arms_and_packages(
                surface, spec["layout"], spec["world"], d, theme,
                position_override=spec["pos"], sprite_manager=sprite_manager,
                direction_angle=spec["dir"], dragged_transporter_id=spec["dragged_transp"],
                tilt_angle=spec["tilt"], error_phase=error_phase,
            )
        else:  # choreo
            if spec["box_on_drone"]:
                _draw_drone_choreo(
                    surface, spec["layout"], spec["world"], d, theme,
                    position=spec["pos"], visual_state=spec["visual_state"],
                    force_box_for_package=spec["pkg_id"], sprite_manager=sprite_manager,
                    direction_angle=spec["dir"], tilt_angle=spec["tilt"],
                )
            else:
                _draw_drone_choreo(
                    surface, spec["layout"], spec["world"], d, theme,
                    position=spec["pos"], visual_state=spec["visual_state"],
                    skip_package=spec["pkg_id"], sprite_manager=sprite_manager,
                    direction_angle=spec["dir"], tilt_angle=spec["tilt"],
                )

    # Cajas sueltas de coreografía (encima de los drones, como render_frame).
    for cid, box_pos in choreo_boxes:
        draw_package(surface, box_pos, cid, theme, sprite_manager=sprite_manager)


def locate_entity_at(
    timeline,
    t: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
) -> tuple[int, int]:
    """Posición (px, py) donde render_world_at dibuja `entity_id` en t.

    Wrapper consciente del supersampling: si el render usa un buffer ampliado
    (viewport × SS) por exceso de locations, localiza en ese buffer y mapea de
    vuelta al viewport (÷ SS) para que el seguimiento de cámara caiga donde se
    dibuja. Sin supersampling delega directamente en el cuerpo.

    Raises:
        KeyError: si entity_id no existe en el world activo en t.
    """
    if theme is None:
        theme = Theme.default()
    base, _b, _p = timeline.sample(t)
    ss = _supersample_factor(base.world, surface_size, theme)
    if ss > 1.0:
        bw = max(1, round(surface_size[0] * ss))
        bh = max(1, round(surface_size[1] * ss))
        bx, by = _locate_entity_at_core(timeline, t, entity_id, (bw, bh), theme=theme)
        return (round(bx / ss), round(by / ss))
    return _locate_entity_at_core(timeline, t, entity_id, surface_size, theme=theme)


def _locate_entity_at_core(
    timeline,
    t: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
) -> tuple[int, int]:
    """Cuerpo de locate_entity_at SIN supersampling (localiza en surface_size).

    Raises:
        KeyError: si entity_id no existe en el world activo en t.
    """
    if theme is None:
        theme = Theme.default()
    history = timeline.history
    base, _b, _p = timeline.sample(t)
    world = base.world
    viewbox = _scene_viewbox(surface_size, theme)
    spread = theme.location_spread_factor
    base_layout = compute_layout(world, viewbox, spread_factor=spread)

    segs = _drone_segments(timeline)
    drone_pos: dict[str, Point] = {}
    dragged: dict[str, Point] = {}
    choreo_box_pos: dict[str, Point] = {}
    for d, dsegs in segs.items():
        if not dsegs:
            drone_pos[d] = base_layout.drone_position(d, theme)
            continue
        spec, drg, cbox, cpkg = _drone_spec_at(
            d, dsegs, t, history, viewbox, spread, theme, world, base_layout,
            segs
        )
        drone_pos[d] = (base_layout.drone_position(d, theme)
                        if spec["mode"] == "idle" else spec["pos"])
        if drg is not None:
            dragged[drg[0]] = drg[1][0]
        if cbox is not None and cpkg is not None:
            choreo_box_pos[cpkg] = cbox[1]

    if entity_id in world.drones:
        return drone_pos[entity_id].as_int_tuple()
    if entity_id in world.locations:
        return base_layout.location_position(entity_id).as_int_tuple()
    if entity_id in world.persons:
        return base_layout.person_position(entity_id, theme).as_int_tuple()
    if entity_id in world.transporters:
        if entity_id in dragged:
            return dragged[entity_id].as_int_tuple()
        return base_layout.transporter_position(entity_id, theme).as_int_tuple()
    if entity_id in world.packages:
        if entity_id in choreo_box_pos:
            return choreo_box_pos[entity_id].as_int_tuple()
        at = world.packages[entity_id].at
        if isinstance(at, HeldByArm):
            dpos = drone_pos.get(at.drone_id) or base_layout.drone_position(at.drone_id, theme)
            arm_ids = [a.id for a in world.drones[at.drone_id].arms]
            idx = arm_ids.index(at.arm_id) if at.arm_id in arm_ids else 0
            return position_of_arm(dpos, idx, max(1, len(arm_ids)), theme).as_int_tuple()
        if isinstance(at, InTransporter):
            tpos = dragged.get(at.transporter_id) or base_layout.transporter_position(at.transporter_id, theme)
            in_t = sorted(
                world.packages_in_transporter(at.transporter_id), key=lambda x: x.id
            )
            ids = [x.id for x in in_t]
            idx = ids.index(entity_id) if entity_id in ids else 0
            return position_in_transporter(tpos, idx, max(1, len(in_t)), theme).as_int_tuple()
        if isinstance(at, AtLocation):
            return base_layout.package_position(entity_id, theme).as_int_tuple()
    raise KeyError(f"entity_id {entity_id!r} no encontrado en world.")