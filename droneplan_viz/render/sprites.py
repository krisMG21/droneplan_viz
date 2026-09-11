"""
Sprites: funciones de dibujo de primitivas sobre pygame.Surface.

Cada función dibuja UNA entidad (drone, paquete, location, ...) en una
posición concreta. Son funciones puras: reciben todos los datos por
parámetro, no mantienen estado, no consultan al World, no consultan al
HistoryManager.

El painter.py es quien orquesta: pide al layout las posiciones, decide
qué dibujar en qué orden, y llama a estas funciones.

Decisiones de diseño:

1. NO se usa pygame.sprite.Sprite ni pygame.sprite.Group. Esas clases
   están pensadas para juegos con muchísimos objetos donde la
   actualización por delta-time vale la pena. Aquí solo dibujamos sobre
   una Surface por frame y el coste de las primitivas es despreciable.
   Funciones puras son más simples y más testables.

2. Generación programática (decisión 6.6 de la propuesta). Sin assets
   externos. Cada sprite se compone de primitivas pygame.draw: círculos
   para drones y personas, rectángulos para paquetes y transportadores,
   líneas para aristas y la X de error. Si en el futuro la sesión UI
   quiere sustituir por PNGs en assets/, basta con reimplementar estas
   funciones leyendo del disco. La firma se mantiene.

3. Texto: SysFont(None, size) cada vez. Pygame lo cachea internamente,
   así que el coste real es bajo. Si en pruebas de rendimiento detectamos
   overhead, añadimos un cache local en Theme o un Painter ligero.

4. Cada función recibe `Theme` aunque algunas no necesiten todos los
   campos. Justificación: una firma uniforme `(surface, position, ...,
   theme)` es más fácil de orquestar desde painter.py que firmas
   asimétricas. Coste cero en runtime.

5. Las posiciones se reciben como `Point` (float). La conversión a int
   para pygame.draw se hace internamente vía `Point.as_int_tuple()`.

6. El "borde" (contorno oscuro) se dibuja DESPUÉS del relleno (orden
   pygame habitual): se llama a `pygame.draw.circle(...)` con width=0
   (relleno) y luego con `width=theme.border_width` (contorno). Algunas
   versiones de pygame difieren ligeramente en cómo el border se centra
   sobre el perímetro; lo asumimos como detalle visual, no afecta a la
   testabilidad por píxel.

7. Las primitivas que dibujan texto usan `pygame.font.init()` perezoso:
   cualquier función que use texto verifica que el módulo font esté
   inicializado y lo arranca si no lo está. Esto desacopla el render
   del `pygame.init()` que invocará la UI envolvente.

Funciones públicas:

- draw_background(surface, theme)
- draw_edge(surface, from_pos, to_pos, theme)
- draw_location(surface, position, loc_id, theme, *, label=True)
- draw_drone(surface, position, drone_id, state, theme, *, label=True)
- draw_drone_arms(surface, drone_pos, arms_with_pkg, theme)
- draw_package(surface, position, content_id, theme, *, label=False)
- draw_transporter(surface, position, transporter_id, contents, theme, *, label=True)
- draw_person(surface, position, person_id, theme, *, label=True)
- draw_error_x(surface, position, theme)

Las funciones aceptan `label=False` para casos donde el painter quiera
omitir el texto (zoom out, render minimalista, tests específicos).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from droneplan_viz.render.geometry import Point
from droneplan_viz.render.theme import (
    RGB,
    Theme,
    color_for_content,
    color_for_drone_state,
    color_for_location,
)

if TYPE_CHECKING:
    # Solo para anotaciones de tipo. En runtime no se importa para evitar
    # un ciclo: sprite_manager.py importa de theme.py, y sprites.py recibe
    # el manager por parámetro (no lo construye), así que no necesita la
    # clase en runtime.
    from droneplan_viz.render.sprite_manager import SpriteManager


# ---------------------------------------------------------------------------
# Utilidades internas: gestión de fuente perezosa
# ---------------------------------------------------------------------------


def _ensure_font_ready() -> None:
    """Inicializa pygame.font si la UI envolvente no lo ha hecho aún.

    El render no quiere asumir que pygame.init() ya se llamó: la UI es
    quien tiene la display y debería haberlo invocado, pero defendemos
    en profundidad. Llamar a pygame.font.init() múltiples veces es
    inocuo (pygame lo detecta y no hace nada).

    Al (re)inicializar el módulo font, purgamos `_FONT_CACHE`: cualquier
    objeto Font creado en una "vida" anterior del módulo (antes de un
    pygame.font.quit() / pygame.quit()) queda inválido y reutilizarlo
    lanzaría "Invalid font". Purgar fuerza su recreación limpia.
    """
    if not pygame.font.get_init():
        pygame.font.init()
        _FONT_CACHE.clear()
        _TEXT_CACHE.clear()


#: Caché local de objetos de fuente, indexado por (font_name, font_path,
#: size). pygame.font.Font(path, size) NO se cachea internamente (a
#: diferencia de SysFont), y el painter renderiza varias etiquetas por
#: frame; recargar/parsear el .ttf en cada etiqueta sería un desperdicio.
#: El caché es de proceso y determinista (los objetos Font son reusables
#: entre renders). Esto materializa la nota 6 del docstring de theme.py.
_FONT_CACHE: dict[tuple[str | None, str | None, int], "pygame.font.Font"] = {}


#: Caché de Surfaces de texto ya rasterizadas, indexado por
#: (text, color, size, font_name, font_path). font.render() con antialiasing
#: rasteriza los glifos en cada llamada; el painter dibuja etiquetas
#: ESTÁTICAS por entidad (conteos "3/4", letras de contenido, nº de
#: necesidades) que son idénticas frame a frame. En escenarios con muchas
#: entidades, re-rasterizarlas 60 veces por segundo es coste puro. Las
#: Surfaces resultantes solo se blittean (lectura), así que reutilizar la
#: misma instancia es seguro. Se purga junto a _FONT_CACHE en reinicios del
#: módulo font.
_TEXT_CACHE: dict[tuple, "pygame.Surface"] = {}
_TEXT_CACHE_CAP = 512

#: Visibilidad global de las etiquetas de nombre de las entidades. El
#: pipeline de cámara la fija en cada frame según el zoom (ver painter):
#: a zoom < 1 el texto, rasterizado a zoom=1 y reducido por el escalado
#: nearest-neighbor, queda ilegible, así que se ocultan. Es una variable
#: de módulo (no un parámetro) para no propagar el zoom por las ~11
#: llamadas a draw_* y sus caminos de interpolación; el render es
#: secuencial (un hilo, un frame tras otro), así que el estado compartido
#: es seguro. Por defecto True (las etiquetas se ven).
_LABELS_VISIBLE: bool = True


def set_labels_visible(visible: bool) -> None:
    """Fija si las etiquetas de nombre se dibujan (ver _LABELS_VISIBLE)."""
    global _LABELS_VISIBLE
    _LABELS_VISIBLE = visible


def _get_font(
    size: int,
    font_name: str | None = None,
    font_path: str | None = None,
) -> "pygame.font.Font":
    """Devuelve un objeto de fuente, cacheado, para (font_name, font_path, size).

    Prioridad:
      1. font_path (un .ttf concreto, p.ej. Monogram que inyecta la app) →
         pygame.font.Font(font_path, size).
      2. font_name → pygame.font.SysFont(font_name, size) (None = fuente
         del sistema por defecto).

    Si la carga del .ttf falla (ruta inexistente, fichero corrupto), se
    cae a SysFont(font_name, size) en lugar de propagar la excepción: el
    render debe seguir dibujando aunque la fuente fina no esté disponible.
    """
    _ensure_font_ready()
    key = (font_name, font_path, size)
    font = _FONT_CACHE.get(key)
    if font is None:
        if font_path is not None:
            try:
                font = pygame.font.Font(font_path, size)
            except (OSError, FileNotFoundError, ValueError):
                font = pygame.font.SysFont(font_name, size)
        else:
            font = pygame.font.SysFont(font_name, size)
        _FONT_CACHE[key] = font
    return font


def _render_text(
    text: str,
    color: RGB,
    size: int,
    font_name: str | None,
    font_path: str | None = None,
) -> pygame.Surface:
    """Renderiza una cadena a una Surface pequeña con antialiasing.

    Usa _get_font (con caché). Si font_path está fijado, renderiza con esa
    fuente .ttf (p.ej. Monogram, la misma del inventario); si no, con
    SysFont(font_name, size). El antialiasing (True) coincide con el del
    inventario para una apariencia coherente.

    El resultado se cachea por (text, color, size, font_name, font_path): las
    etiquetas del render son estáticas y se repiten cada frame; rasterizar una
    sola vez y reutilizar la Surface (que solo se blittea) ahorra el coste de
    font.render por etiqueta y frame.
    """
    cache_key = (text, color, size, font_name, font_path)
    cached = _TEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    font = _get_font(size, font_name, font_path)
    try:
        surf = font.render(text, True, color)
    except pygame.error:
        # El Font cacheado quedó inválido (el módulo font se reinició tras
        # un pygame.quit() entre la creación y este uso, p.ej. entre tests
        # o tras un reset de display). Lo descartamos y reconstruimos una
        # vez con el módulo ya reinicializado.
        _FONT_CACHE.pop((font_name, font_path, size), None)
        font = _get_font(size, font_name, font_path)
        surf = font.render(text, True, color)
    if len(_TEXT_CACHE) >= _TEXT_CACHE_CAP:
        del _TEXT_CACHE[next(iter(_TEXT_CACHE))]
    _TEXT_CACHE[cache_key] = surf
    return surf


def _blit_centered(
    surface: pygame.Surface,
    rendered_text: pygame.Surface,
    center: Point,
) -> None:
    """Pega `rendered_text` sobre `surface` centrado en `center`."""
    tw, th = rendered_text.get_size()
    cx, cy = center.as_int_tuple()
    surface.blit(rendered_text, (cx - tw // 2, cy - th // 2))


def _blit_label_with_background(
    surface: pygame.Surface,
    rendered_text: pygame.Surface,
    center: Point,
    theme: Theme,
) -> None:
    """Pega el texto de una etiqueta con un recuadro de fondo detrás.

    Dibuja, centrado en `center`, un rectángulo de bordes redondeados del
    color de fondo de la escena (theme.background), ligeramente mayor que
    el texto, y encima el texto. El recuadro mejora la legibilidad de los
    nombres de las entidades cuando caen sobre sprites u otras zonas de
    contraste bajo. Es semitransparente para no tapar por completo lo que
    haya debajo.
    """
    tw, th = rendered_text.get_size()
    cx, cy = center.as_int_tuple()
    # Margen alrededor del texto (el recuadro es "ligeramente mayor").
    pad_x, pad_y = 5, 2
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
    r, g, b = theme.background
    pygame.draw.rect(
        bg,
        (r, g, b, 210),
        pygame.Rect(0, 0, bw, bh),
        border_radius=max(3, bh // 3),
    )
    surface.blit(bg, (cx - bw // 2, cy - bh // 2))
    surface.blit(rendered_text, (cx - tw // 2, cy - th // 2))


def _blit_sprite_centered(
    surface: pygame.Surface,
    sprite: pygame.Surface,
    center: Point,
) -> None:
    """Pega un sprite ya escalado centrado en `center`.

    Idéntico a _blit_centered pero con nombre semántico distinto para
    dejar claro en el código que lo que se pega es un sprite cargado de
    disco, no un texto renderizado. La operación es la misma (blit
    centrado), pero la separación documenta la intención.
    """
    sw, sh = sprite.get_size()
    cx, cy = center.as_int_tuple()
    surface.blit(sprite, (cx - sw // 2, cy - sh // 2))


def _draw_ground_shadow(
    surface: pygame.Surface,
    center: Point,
    object_size: tuple[int, int],
    theme: Theme,
) -> None:
    """Dibuja una sombra ovalada translúcida bajo un objeto apoyado.

    La sombra es una elipse PLANA (mucho más ancha que alta), oscura y
    semitransparente, centrada horizontalmente bajo el objeto y anclada a
    su BASE (borde inferior del lienzo). Debe pintarse ANTES del sprite del
    objeto para que éste quede encima y parezca apoyado sobre el suelo,
    separándose del sprite de location y ganando solidez.

    Args:
        center: centro del LIENZO del objeto (lo mismo que recibe la
            primitiva de dibujo: draw_package / draw_transporter).
        object_size: (w, h) del objeto en pantalla. La sombra se dimensiona
            a partir del ANCHO del objeto (no del tamaño de la location),
            de modo que objetos pequeños proyectan sombras pequeñas.

    Semitransparencia: pygame.draw.ellipse es OPACO sobre una Surface
    normal. Para la translucidez se dibuja la elipse con color RGBA sobre
    una Surface auxiliar con SRCALPHA y se pega con blit (alpha-blending
    estándar). El resultado es determinista.

    No hace nada si theme.shadow_enabled es False.
    """
    if not theme.shadow_enabled:
        return
    obj_w, obj_h = object_size
    shadow_w = max(2, round(obj_w * theme.shadow_width_factor))
    shadow_h = max(2, round(shadow_w * theme.shadow_height_ratio))
    cx, cy = center.as_int_tuple()
    # Centro de la sombra en la BASE del objeto, afinable con shadow_offset_y.
    shadow_cy = cy + obj_h // 2 + theme.shadow_offset_y

    shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
    alpha = max(0, min(255, theme.shadow_alpha))
    rgba = (theme.shadow_color[0], theme.shadow_color[1], theme.shadow_color[2], alpha)
    pygame.draw.ellipse(shadow_surf, rgba, pygame.Rect(0, 0, shadow_w, shadow_h))
    surface.blit(shadow_surf, (cx - shadow_w // 2, shadow_cy - shadow_h // 2))


def _drone_in_error(state) -> bool:
    """True si el state representa el estado ERROR de la FSM del drone."""
    return hasattr(state, "name") and state.name == "ERROR"


# Sectores de dirección → clave de cara. El ángulo se mide en grados
# matemáticos estándar (0°=Este, 90°=Norte, sentido antihorario) sobre el
# vector de desplazamiento (to - from) en coords de MUNDO (y hacia arriba).
# Como en pantalla la y crece hacia abajo, el painter invierte el signo de
# dy antes de llamar, de modo que aquí "arriba" = Norte.
#
# Solo hay 6 caras (N, S y las 4 diagonales). Repartimos los 360° en 6
# sectores de 60°, centrados en cada cara:
#   N  centrado en  90°   → [60, 120)
#   NW centrado en 135°   → [120, 180)
#   SW centrado en 225°   → [180, 240)  (y simétricos)
# Para los movimientos puramente E/O usamos las diagonales (NE/SE a la
# derecha, NW/SW a la izquierda), como pidió el usuario: las diagonales
# "valen también" para izquierda/derecha.
def _face_key_for_angle(angle_deg: float) -> str:
    """Devuelve la clave de cara direccional para un ángulo en grados.

    angle_deg en [0, 360). Convención matemática (0=E, 90=N, antihorario).
    """
    a = angle_deg % 360.0
    # 6 sectores de 60°, con frontera elegida para que E/O caigan en
    # diagonales (no hay cara E ni O propias).
    #   [60,120)   -> N
    #   [120,180)  -> NW
    #   [180,240)  -> SW
    #   [240,300)  -> S
    #   [300,360)  -> SE
    #   [0,60)     -> NE
    if 60.0 <= a < 120.0:
        return "face_N"
    if 120.0 <= a < 180.0:
        return "face_NW"
    if 180.0 <= a < 240.0:
        return "face_SW"
    if 240.0 <= a < 300.0:
        return "face_S"
    if 300.0 <= a < 360.0:
        return "face_SE"
    return "face_NE"  # [0, 60)


def _drone_face_key(
    state,
    *,
    direction_angle: float | None,
    finished: bool,
    error_phase: int = 0,
) -> str | None:
    """Decide la clave de cara para el cuerpo BASE del dron.

    Solo aplica al cuerpo base; el cuerpo interactuando trae su cara
    incrustada y el painter pasa face=None en ese caso.

    Reglas (acordadas con el usuario):
      - ERROR                       → alterna "face_error1"/"face_error2"
                                      según error_phase (0/1), para que la
                                      cara de fallo parpadee.
      - En movimiento (hay ángulo)  → cara direccional por sector.
      - Ejecución terminada         → "face_idle" (ojos cerrados).
      - Parado a la espera          → "face" (frontal neutra).
    """
    if _drone_in_error(state):
        return "face_error2" if (error_phase % 2) else "face_error1"
    if direction_angle is not None:
        return _face_key_for_angle(direction_angle)
    if finished:
        return "face_idle"
    return "face"


# ---------------------------------------------------------------------------
# Background y aristas
# ---------------------------------------------------------------------------


def draw_background(surface: pygame.Surface, theme: Theme) -> None:
    """Rellena la Surface con el color de fondo del theme.

    Es el primer paso de cualquier render_snapshot/render_frame: limpia
    la Surface antes de dibujar entidades.
    """
    surface.fill(theme.background)


def draw_edge(
    surface: pygame.Surface,
    from_pos: Point,
    to_pos: Point,
    theme: Theme,
) -> None:
    """Línea fina entre dos Locations conectadas por una arista del grafo.

    Pintada en theme.edge con grosor theme.edge_width. Pygame requiere
    coords int para draw.line.
    """
    pygame.draw.line(
        surface,
        theme.edge,
        from_pos.as_int_tuple(),
        to_pos.as_int_tuple(),
        theme.edge_width,
    )


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def draw_location(
    surface: pygame.Surface,
    position: Point,
    loc_id: str,
    theme: Theme,
    *,
    label: bool = True,
    sprite_manager: "SpriteManager | None" = None,
) -> None:
    """Dibuja una Location: sprite isométrico si disponible, si no elipse.

    Convención: `position` es el CENTRO DEL LIENZO del sprite (lo que la
    app pone como position_screen al diseñar el grafo). El centro LÓGICO
    de la location (donde aterrizan drones y se apilan cajas, sobre el
    rombo hábil del sprite isométrico) lo aplica el LAYOUT al calcular
    drone_position / package_position / etc., desplazando hacia arriba
    por theme.location_ground_offset_y. Aquí solo pintamos centrado.

    Sprite: tamaño nativo theme.location_sprite_native_size (p.ej. 110×88
    isométrico), escala entera theme.location_sprite_scale (2× → 220×176
    en pantalla, nearest-neighbor limpio).

    Fallback: elipse con proporción del lienzo del sprite (5:4 ancho:alto)
    para parecerse a la silueta isométrica.

    La etiqueta (loc_id) se pinta DEBAJO de la location.
    """
    cx, cy = position.as_int_tuple()

    # Dimensiones del sprite en pantalla (lienzo entero).
    native_w, native_h = theme.location_sprite_native_size
    scale = theme.location_sprite_scale
    screen_w, screen_h = native_w * scale, native_h * scale

    sprite = None
    if sprite_manager is not None:
        sprite = sprite_manager.get("location", size=(screen_w, screen_h))

    if sprite is not None:
        # Anclar el lienzo del sprite centrado en `position`. El offset
        # del rombo lo aplica el layout, no aquí.
        top_left = (cx - screen_w // 2, cy - screen_h // 2)
        surface.blit(sprite, top_left)
        visible_half_h = screen_h // 2
    else:
        # Fallback: elipse con la proporción del lienzo (5:4 ancho:alto),
        # centrada en `position` y ENCOGIDA por location_fallback_scale para
        # dejar separación entre localizaciones vecinas (evita el solape que
        # producía dibujarla al tamaño completo del lienzo del sprite).
        fill, border = color_for_location(loc_id, theme)
        ell_w = max(2, int(screen_w * theme.location_fallback_scale))
        ell_h = max(2, int(screen_h * theme.location_fallback_scale))
        rect = pygame.Rect(0, 0, ell_w, ell_h)
        rect.center = (cx, cy)
        pygame.draw.ellipse(surface, fill, rect)
        pygame.draw.ellipse(surface, border, rect, theme.border_width)
        visible_half_h = ell_h // 2

    if label and _LABELS_VISIBLE:
        # Etiqueta debajo del borde inferior VISIBLE (lienzo del sprite o
        # elipse encogida del fallback).
        label_y = cy + visible_half_h + 8
        label_pos = Point(position.x, float(label_y))
        text_surf = _render_text(
            loc_id,
            theme.text,
            theme.font_size_label,
            theme.font_name,
            theme.font_path,
        )
        _blit_label_with_background(surface, text_surf, label_pos, theme)


# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------


def draw_drone(
    surface: pygame.Surface,
    position: Point,
    drone_id: str,
    state,
    theme: Theme,
    *,
    label: bool = True,
    sprite_manager: "SpriteManager | None" = None,
    direction_angle: float | None = None,
    finished: bool = False,
    held_object: str | None = None,
    carrier_level: int = 0,
    box_count: int = 1,
    tilt_angle: float = 0.0,
    error_phase: int = 0,
) -> None:
    """Dibuja un drone componiendo capas si hay sprites, si no primitiva.

    MODELO DE CAPAS (rediseño): el dron ya no es un PNG entero por estado.
    Se compone apilando cuerpo + cara + objeto, y el compuesto se escala
    una vez. El SpriteManager hace la composición y la cachea; aquí solo
    decidimos QUÉ capas pedir.

    Selección de capas:
      - Cuerpo: "drone_interacting" si la FSM está en INTERACTING (brazos
        extendidos, cara incrustada → sin capa de cara), si no "drone"
        (base, con capa de cara).
      - Cara (solo sobre cuerpo base): _drone_face_key() según error /
        dirección de movimiento / terminado / espera.
      - Objeto: held_object ("box" o "carrier") con carrier_level (0..5).

    Args:
        direction_angle: ángulo de desplazamiento en grados (convención
            matemática, 0=E, 90=N) si el dron se está moviendo; None si
            está parado. Determina la cara direccional.
        finished: True si el dron terminó su ejecución (última posición
            del plan) → cara "idle" (ojos cerrados) en reposo.
        held_object: "box" si sostiene un paquete suelto, "carrier" si
            gestiona el carrier, None si va vacío.
        carrier_level: nivel de llenado del carrier (0..5) si
            held_object == "carrier".

    Fallback: si no hay sprite_manager o falta el cuerpo base, primitiva
    geométrica (círculo coloreado por estado + X si ERROR). Si el cuerpo
    existe pero falta alguna capa, se compone con lo disponible.

    El tamaño y las anclas salen del Theme; nada hardcodeado aquí.
    """
    cx, cy = position.as_int_tuple()
    r = theme.drone_radius

    composite = None
    if sprite_manager is not None:
        interacting = hasattr(state, "name") and state.name == "INTERACTING"
        body_key = "drone_interacting" if interacting else "drone"
        # La cara solo se superpone sobre el cuerpo base; el interacting
        # ya la trae incrustada.
        face_key = None
        if not interacting:
            face_key = _drone_face_key(
                state,
                direction_angle=direction_angle,
                finished=finished,
                error_phase=error_phase,
            )
        # Objeto agarrado.
        obj_key = None
        if held_object == "box":
            # Dos cajas (una por brazo) si lleva >=2; una si lleva 1.
            obj_key = "box2" if box_count >= 2 else "box"
        elif held_object == "carrier":
            lvl = max(0, min(5, carrier_level))
            obj_key = f"carrier_{lvl}"
        # Destino 1:1 con el cuerpo (diámetro 2*drone_radius).
        dest_size = (2 * r, 2 * r)
        composite = sprite_manager.get_drone_composite(
            body=body_key, face=face_key, obj=obj_key, dest_size=dest_size,
        )

    if composite is not None:
        # Anclar por la esquina superior izquierda del compuesto. El punto
        # de layout (position) es el CENTRO DEL CUERPO; convertimos a
        # top-left con el offset de registro del Theme, escalado al
        # tamaño destino. Así el dron no se mueve al añadir/quitar carrier
        # (lo que cuelga abajo no desplaza el cuerpo).
        anchors = theme.sprite_anchors
        body_w, body_h = anchors.body_size
        scale = (2 * r) / body_w  # factor cuerpo-nativo → destino
        reg_dx = anchors.registration_dx * scale
        reg_dy = anchors.registration_dy * scale

        if tilt_angle != 0.0:
            # Rotamos el compuesto sobre el CENTRO DEL CUERPO (no el centro
            # del lienzo del compuesto: el lienzo incluye lo que cuelga
            # debajo - carrier/cajas - y rotar por ese centro haría que el
            # cuerpo se desplazara).
            #
            # Convención: tilt POSITIVO = horario (drone yendo a la derecha
            # se inclina hacia adelante de la marcha). pygame.transform.rotate
            # usa convención antihoraria, así que invertimos el signo.
            #
            # Calculamos el centro del cuerpo en coords del lienzo SIN
            # rotar: top-left del cuerpo en el lienzo = (body_x, 0) =
            # ((canvas_w-body_w)//2, 0); centro = (canvas_w//2, body_h//2).
            # Tras escalar el lienzo entero por `scale`, esos píxeles del
            # centro del cuerpo en el compuesto resultante están en:
            comp_w, comp_h = composite.get_size()
            body_center_x_in_comp = comp_w / 2.0
            body_center_y_in_comp = (body_h / 2.0) * scale

            rotated = pygame.transform.rotate(composite, -tilt_angle)
            rw, rh = rotated.get_size()

            # Tras rotar, el "centro del cuerpo" en el compuesto original
            # se mueve. Calculamos su nueva posición usando la matriz de
            # rotación, con el centro de la imagen como pivote (eso es lo
            # que hace pygame.transform.rotate internamente).
            import math as _m
            theta = _m.radians(-tilt_angle)
            # Vector desde el centro del compuesto sin rotar al centro
            # del cuerpo:
            vx = body_center_x_in_comp - comp_w / 2.0
            vy = body_center_y_in_comp - comp_h / 2.0
            # Rotación: (x', y') = (x*cos - y*sin, x*sin + y*cos)
            vrx = vx * _m.cos(theta) - vy * _m.sin(theta)
            vry = vx * _m.sin(theta) + vy * _m.cos(theta)
            # Centro del compuesto rotado en pantalla → top-left del
            # rotado para que el centro del cuerpo quede en `position`.
            body_center_in_rotated_x = rw / 2.0 + vrx
            body_center_in_rotated_y = rh / 2.0 + vry
            top_left = (
                int(position.x - body_center_in_rotated_x),
                int(position.y - body_center_in_rotated_y),
            )
            surface.blit(rotated, top_left)
        else:
            top_left = (int(position.x + reg_dx), int(position.y + reg_dy))
            surface.blit(composite, top_left)
        # En ERROR, la cara face_error1 ya señala el fallo. Si por fallback
        # esa cara no existiera, superponemos la X como red de seguridad.
        if _drone_in_error(state):
            if sprite_manager._load_native_by_key("face_error1") is None:
                draw_error_x(surface, position, theme)
    else:
        # --- Primitiva (fallback) ---
        fill = color_for_drone_state(state, theme)
        pygame.draw.circle(surface, fill, (cx, cy), r)
        pygame.draw.circle(surface, theme.drone_border, (cx, cy), r, theme.border_width)
        if _drone_in_error(state):
            draw_error_x(surface, position, theme)

    if label and _LABELS_VISIBLE:
        # Etiqueta debajo del drone:
        label_pos = Point(position.x, position.y + r + 8)
        text_surf = _render_text(
            drone_id,
            theme.text,
            theme.font_size_id,
            theme.font_name,
            theme.font_path,
        )
        _blit_label_with_background(surface, text_surf, label_pos, theme)


def draw_drone_arms(
    surface: pygame.Surface,
    drone_pos: Point,
    arm_ids: list[str],
    holding_ids: list[str | None],
    theme: Theme,
) -> None:
    """Dibuja los brazos del drone como puntos en su perímetro.

    Cada brazo es un círculo pequeño en el borde del drone. Si el brazo
    sostiene un paquete (holding_ids[i] is not None), el brazo se dibuja
    relleno; si está libre, solo contorno.

    Los brazos se distribuyen equiespaciadamente alrededor del drone,
    empezando arriba (-π/2) en sentido horario.

    Args:
        drone_pos: centro del drone.
        arm_ids: lista de Arm.id en el orden en que aparecen en el Drone
            (estable por construcción del dominio, no se ordena alfabéticamente).
        holding_ids: paralela a arm_ids; cada elemento es el content_id
            del paquete que sostiene ese brazo, o None si está libre.
            Esta lista la calcula el painter consultando world.package_held_by.

    Notas:
    - No dibujamos el paquete sostenido aquí; eso lo hace draw_package
      directamente sobre la posición del brazo. Aquí solo dibujamos el
      MARCADOR del brazo.
    """
    n = len(arm_ids)
    if n == 0:
        return

    drone_r = theme.drone_radius
    arm_r = theme.arm_radius

    for i, _arm_id in enumerate(arm_ids):
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        ax = drone_pos.x + drone_r * math.cos(angle)
        ay = drone_pos.y + drone_r * math.sin(angle)
        arm_center = (int(round(ax)), int(round(ay)))

        if holding_ids[i] is not None:
            # Brazo ocupado: relleno con color del content que sostiene.
            content_color = color_for_content(holding_ids[i], theme)
            pygame.draw.circle(surface, content_color, arm_center, arm_r)
            pygame.draw.circle(
                surface, theme.drone_border, arm_center, arm_r, 1
            )
        else:
            # Brazo libre: solo contorno.
            pygame.draw.circle(
                surface, theme.drone_border, arm_center, arm_r, 1
            )


def position_of_arm(
    drone_pos: Point,
    arm_index: int,
    total_arms: int,
    theme: Theme,
) -> Point:
    """Posición visual del brazo `arm_index` de un drone.

    Útil para el painter cuando quiere dibujar un paquete sostenido por
    un brazo concreto: pide la posición del brazo y dibuja el paquete
    centrado ahí.

    Si total_arms == 0 devuelve drone_pos (caso degenerado, el drone no
    tiene brazos y nadie debería pedir esta función).
    """
    if total_arms == 0:
        return drone_pos
    angle = -math.pi / 2.0 + (2.0 * math.pi * arm_index / total_arms)
    return Point(
        drone_pos.x + theme.drone_radius * math.cos(angle),
        drone_pos.y + theme.drone_radius * math.sin(angle),
    )


def draw_error_x(
    surface: pygame.Surface,
    position: Point,
    theme: Theme,
) -> None:
    """Dibuja una X sobre la posición indicada. Para drones en ERROR.

    Centrada en `position`, con brazos de longitud ~drone_radius * 0.7
    y grosor theme.error_x_width.
    """
    cx, cy = position.as_int_tuple()
    arm_len = int(theme.drone_radius * 0.7)
    w = theme.error_x_width
    # Diagonal \
    pygame.draw.line(
        surface,
        theme.error_x,
        (cx - arm_len, cy - arm_len),
        (cx + arm_len, cy + arm_len),
        w,
    )
    # Diagonal /
    pygame.draw.line(
        surface,
        theme.error_x,
        (cx - arm_len, cy + arm_len),
        (cx + arm_len, cy - arm_len),
        w,
    )


# ---------------------------------------------------------------------------
# Paquetes
# ---------------------------------------------------------------------------


def draw_package(
    surface: pygame.Surface,
    position: Point,
    content_id: str,
    theme: Theme,
    *,
    label: bool = False,
    shadow: bool = False,
    sprite_manager: "SpriteManager | None" = None,
) -> None:
    """Dibuja un paquete: sprite si disponible, si no cuadrado por Content.

    Con sprite_manager: usa el asset "box" (box.png), la misma caja que
    el dron lleva como capa de objeto, de modo que una caja en el suelo y
    la misma caja agarrada por el dron son visualmente idénticas. Si no
    hay box.png, prueba "package" con variant=content_id (compatibilidad)
    y, en último término, cae al cuadrado coloreado por color_for_content.

    NOTA: con box.png se pierde el color por contenido en el suelo (box es
    monocromo). Es una decisión consciente (coherencia caja suelo↔dron);
    el color por contenido se conserva solo en el fallback primitiva.

    Etiqueta opcional con el content_id, pintada por debajo. Por defecto
    NO se dibuja: los paquetes son pequeños y la etiqueta puede saturar.

    shadow: si True (y theme.shadow_enabled), pinta una sombra ovalada
    bajo la caja ANTES del sprite. Solo debe activarse para cajas APOYADAS
    en el suelo; el painter lo deja en False para cajas dentro de un
    transporter o agarradas por un brazo (están en el aire, no proyectan
    sombra de suelo). Default False: la primitiva es agnóstica al contexto.
    """
    side = theme.package_size
    half = side // 2
    cx, cy = position.as_int_tuple()

    if shadow:
        _draw_ground_shadow(surface, position, (side, side), theme)

    sprite = None
    if sprite_manager is not None:
        # Preferir box.png (caja unificada suelo↔dron); si no, package.
        sprite = sprite_manager.get("box", size=(side, side))
        if sprite is None:
            sprite = sprite_manager.get("package", size=(side, side), variant=content_id)

    if sprite is not None:
        _blit_sprite_centered(surface, sprite, position)
        # Outline de color por contenido alrededor de la caja. box.png es
        # monocromo (misma caja suelo↔dron), así que el color por tipo se
        # perdía en el suelo; este borde lo recupera. En el fallback
        # primitiva NO hace falta: el cuadrado ya va coloreado por contenido.
        ow = theme.package_outline_width
        if ow > 0:
            color = color_for_content(content_id, theme)
            rect = pygame.Rect(cx - half, cy - half, side, side)
            pygame.draw.rect(surface, color, rect, ow)
    else:
        color = color_for_content(content_id, theme)
        rect = pygame.Rect(cx - half, cy - half, side, side)
        # Relleno
        pygame.draw.rect(surface, color, rect)
        # Borde oscuro fino
        pygame.draw.rect(surface, theme.drone_border, rect, 1)

    if label and _LABELS_VISIBLE:
        label_pos = Point(position.x, position.y + half + 7)
        text_surf = _render_text(
            content_id,
            theme.text_dim,
            theme.font_size_id - 1,  # un pelín más pequeño
            theme.font_name,
            theme.font_path,
        )
        _blit_label_with_background(surface, text_surf, label_pos, theme)


# ---------------------------------------------------------------------------
# Transportadores
# ---------------------------------------------------------------------------


def draw_transporter(
    surface: pygame.Surface,
    position: Point,
    transporter_id: str,
    contents_count: int,
    capacity: int,
    theme: Theme,
    *,
    label: bool = True,
    shadow: bool = False,
    sprite_manager: "SpriteManager | None" = None,
) -> None:
    """Dibuja un transportador: sprite carrier_N (mismo asset que el
    carrier que cuelga del dron, mismo conjunto de imágenes) si
    disponible, si no rectángulo. La N se determina por nivel de llenado
    (0..5 → carrier.png, carrier1.png, ..., carrier5.png), igual que la
    capa de objeto del dron arrastrándolo. Así, un transp parado en el
    suelo y el mismo transp arrastrado por un dron son visualmente
    idénticos (coherencia suelo↔dron, igual que con las cajas).

    El nivel de llenado lo comunica ENTERAMENTE el sprite carrier_N; no
    se dibuja ninguna barra de ocupación (el cambio de sprite ya simboliza
    cuántos paquetes lleva).

    Args:
        contents_count: paquetes actualmente dentro.
        capacity: capacidad declarada del transporter.
        shadow: si True (y theme.shadow_enabled), pinta una sombra ovalada
            bajo el transporter ANTES del sprite. Solo para transporters
            APOYADOS en el suelo; el painter lo deja en False cuando el
            transporter va arrastrado/colgando por el aire. Default False.
    """
    cx, cy = position.as_int_tuple()
    # Tamaño del transporter en pantalla. Si hay sprite, usamos el tamaño
    # nativo del carrier (38×30) SIN escalar — mismo tamaño físico que el
    # carrier que cuelga del dron. Si no hay sprite, usamos las
    # dimensiones primitiva del theme.
    if sprite_manager is not None:
        w = theme.sprite_anchors.carrier_size[0]
        h = theme.sprite_anchors.carrier_size[1]
    else:
        w = theme.transporter_width
        h = theme.transporter_height

    if shadow:
        _draw_ground_shadow(surface, position, (w, h), theme)

    sprite = None
    if sprite_manager is not None:
        # Nivel de llenado 0..5 → "carrier_0"..."carrier_5"
        if capacity > 0:
            level = round(5 * contents_count / capacity)
            level = max(0, min(5, level))
        else:
            level = 0
        sprite = sprite_manager.get(f"carrier_{level}", size=(w, h))

    if sprite is not None:
        _blit_sprite_centered(surface, sprite, position)
    else:
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        # Relleno
        pygame.draw.rect(surface, theme.transporter_fill, rect)
        # Borde
        pygame.draw.rect(surface, theme.transporter_border, rect, theme.border_width)

    if label and _LABELS_VISIBLE:
        # Etiqueta debajo del transp:
        label_pos = Point(position.x, position.y + h // 2 + 8)
        text_surf = _render_text(
            transporter_id,
            theme.text,
            theme.font_size_id,
            theme.font_name,
            theme.font_path,
        )
        _blit_label_with_background(surface, text_surf, label_pos, theme)


def position_in_transporter(
    transporter_pos: Point,
    pkg_index: int,
    total_pkgs: int,
    theme: Theme,
) -> Point:
    """Posición visual de un paquete dentro del transportador.

    Distribución horizontal a lo ancho del transp. Si solo hay 1 paquete,
    al centro. Si total_pkgs == 0, devuelve transporter_pos (no debería
    llamarse en esa situación, defensivo).
    """
    if total_pkgs <= 0:
        return transporter_pos
    if total_pkgs == 1:
        return transporter_pos
    # Distribuimos pkg_index en [-w/2 + margen, w/2 - margen]:
    w = theme.transporter_width
    margin = 6
    inner_w = w - 2 * margin
    step = inner_w / max(1, total_pkgs - 1)
    x_offset = -inner_w / 2.0 + pkg_index * step
    return Point(transporter_pos.x + x_offset, transporter_pos.y)


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


def draw_person(
    surface: pygame.Surface,
    position: Point,
    person_id: str,
    theme: Theme,
    *,
    label: bool = True,
    sprite_manager: "SpriteManager | None" = None,
    sprite_base: str | None = None,
    has_box: bool = False,
) -> None:
    """Dibuja una persona: sprite si disponible, si no círculo crema.

    sprite_base: nombre del set de sprite asignado a esta persona (p.ej.
        "person1"), elegido por el painter al repartir los sets detectados
        en assets entre las personas del problema. Si es None, se intenta el
        sprite genérico "person" (compatibilidad) y, si tampoco hay, círculo.
    has_box: True si la persona ya recibió su caja → variante "_box" del set
        (espera vs. entrega cumplida). Si el set no tiene variante _box, se
        usa la base (no cambia visualmente).

    El estado de espera vs. entrega cumplida lo transmite el propio sprite
    (base vs. _box): no se dibuja ningún anillo/indicador alrededor.
    """
    cx, cy = position.as_int_tuple()
    r = theme.person_radius

    sprite = None
    if sprite_manager is not None:
        if sprite_base is not None:
            sprite = sprite_manager.get_person(
                sprite_base, box=has_box, size=(2 * r, 2 * r)
            )
        if sprite is None:
            # Compatibilidad: sprite genérico "person" (person.png) si existe.
            sprite = sprite_manager.get("person", size=(2 * r, 2 * r))

    if sprite is not None:
        _blit_sprite_centered(surface, sprite, position)
    else:
        # Relleno
        pygame.draw.circle(surface, theme.person_fill, (cx, cy), r)
        # Borde fino
        pygame.draw.circle(surface, theme.person_border, (cx, cy), r, 1)

    if label and _LABELS_VISIBLE:
        # Etiqueta debajo:
        label_pos = Point(position.x, position.y + r + 10)
        text_surf = _render_text(
            person_id,
            theme.text,
            theme.font_size_id,
            theme.font_name,
            theme.font_path,
        )
        _blit_label_with_background(surface, text_surf, label_pos, theme)