"""
Layout: mapeo World → posiciones en pantalla (Point).

Este módulo decide DÓNDE se dibuja cada entidad del mundo. Es 100% puro:
no toca pygame, no muta nada del World, no depende de Theme para la
geometría central (sí para los offsets de co-localización, que son
parámetros de estilo).

Responsabilidades:

- A partir de un World y un ViewBox, calcula la posición en pantalla
  (Point) de CADA Location del grafo.
- Dadas esas posiciones, deriva la posición visual de cada drone (que
  vive en una Location, posiblemente compartiéndola con otros drones).
- Proporciona la posición visual de transportadores y personas (que
  también viven en una Location).
- Proporciona la posición visual de paquetes "libres" (AtLocation):
  apilados cerca de su localización.

Decisión central: dos estrategias de layout para Locations.

  1. Si TODAS las Location tienen position_screen != None, se hace un
     mapeo afín lineal del bounding box de esas coordenadas al ViewBox
     destino, preservando aspect ratio (escala uniforme).

  2. Si ALGUNA Location tiene position_screen == None, se descarta el
     campo y se aplica un layout circular determinista: ordenadas
     alfabéticamente por loc_id, distribuidas en un círculo centrado
     en el ViewBox.

Justificación del "todos o nada":

- Si solo respetamos las que tienen position_screen y las None las
  ponemos en otro sitio, el resultado es visualmente inconsistente
  (entidades sin coords podrían superponerse con las que sí, sin un
  criterio claro).
- Para escenarios docentes (≤10 locations), o están todas declaradas
  por el alumno o no lo está ninguna. La regla simple es la correcta.
- Documentado en el docstring de Location del dominio:
  "si el usuario no la proporciona, el render usará un layout
  automático."

Decisiones secundarias:

- Layout circular ordena por loc_id alfabético, NO por orden de
  inserción en el dict. Razones:
    * Determinismo: el orden de iteración del dict puede depender de
      cómo se construyó el World; el orden alfabético es estable.
    * Reproducibilidad de tests cross-run: misma entrada → mismo
      output siempre.
    * Si el alumno quiere otro orden, declara position_screen en cada
      Location.

- El radio del círculo es el 40% del menor de width/height del ViewBox,
  para garantizar margen alrededor (no roza los bordes). Si solo hay
  1 location, se coloca en el centro. Si hay 0, no devolvemos nada
  (caso degenerado del world vacío).

- Co-localización de drones: distribuidos en anillo a radio
  theme.drone_colocation_offset del centro de la Location. Orden
  determinista alfabético por drone.id. Si solo hay 1 drone, va al
  centro de la Location.

- Co-localización de transportadores y personas: ofrecemos posiciones
  simétricas pero diferenciadas. Transportador a la izquierda del
  centro, persona a la derecha. Si hay varios del mismo tipo
  co-localizados (caso poco frecuente), se apilan verticalmente.

- Paquetes libres (AtLocation): apilados en un cluster pequeño junto
  a la localización. Orden alfabético por package.id.

API pública del módulo:

    compute_layout(world, viewbox) -> WorldLayout
    WorldLayout (frozen) con métodos:
      - location_position(loc_id) -> Point
      - drone_position(drone_id, theme) -> Point
      - transporter_position(transporter_id, theme) -> Point
      - person_position(person_id, theme) -> Point
      - package_position(package_id, theme) -> Point  (solo si está AtLocation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from droneplan_viz.domain import (
    AtLocation,
    World,
)
from droneplan_viz.render.geometry import Point, ViewBox
from droneplan_viz.render.theme import Theme


#: Cachés de orden/co-localización por mapping de entidades. Los métodos de
#: posición resolvían en CADA llamada (y por entidad) la lista de entidades
#: co-localizadas (scan + sort O(N)) o el orden global de paquetes
#: (sorted(todas) + .index() O(K)). Como el render llama a estos métodos una
#: vez por entidad y por frame, eso era O(N²) por frame y crecía con la
#: cantidad de elementos y la co-localización — justo el coste que aparece
#: cuando los drones dejan de estar en MOVING (posición precalculada) y pasan
#: a IDLE/INTERACTING (que consultan drone_position). Precalcular una vez por
#: world (identidad confirmada por `is`) y resolver cada entidad en O(1)
#: elimina ese coste. Cachés acotadas con desalojo FIFO: en un frame pueden
#: convivir varios worlds (base + snapshots de drones activos).
_COLOC_CACHE: "dict[int, tuple]" = {}
_PKG_ORDER_CACHE: "dict[int, tuple]" = {}
_LAYOUT_CACHE_CAP = 256


def _colocation_groups(mapping):
    """{position: [ids ordenados]} para entidades con atributo .position.

    Cacheado por identidad del mapping. Una entidad en `mapping` está
    co-localizada con las demás de su misma `.position`; la lista ordenada
    reproduce exactamente el `sorted(... if e.position == X)` que hacían los
    métodos de posición, pero calculado una vez por world.
    """
    key = id(mapping)
    cached = _COLOC_CACHE.get(key)
    if cached is not None and cached[0] is mapping:
        return cached[1]
    groups: dict[str, list[str]] = {}
    for eid, ent in mapping.items():
        groups.setdefault(ent.position, []).append(eid)
    for ids in groups.values():
        ids.sort()
    if len(_COLOC_CACHE) >= _LAYOUT_CACHE_CAP:
        del _COLOC_CACHE[next(iter(_COLOC_CACHE))]
    _COLOC_CACHE[key] = (mapping, groups)
    return groups


def _packages_at_loc(packages):
    """{loc_id: [ids de cajas AtLocation allí, ordenados alfabéticamente]}.

    Índice PRESENTE por localización: a diferencia del rango global, el índice
    de una caja es su posición entre las que están AtLocation aquí AHORA, así
    que al extraer una, las siguientes la sustituyen (apilado tipo lista que
    pidió el rediseño compacto). Cacheado por identidad del mapping de paquetes
    (un único O(K log K) por snapshot).
    """
    key = id(packages)
    cached = _PKG_ORDER_CACHE.get(key)
    if cached is not None and cached[0] is packages:
        return cached[1]
    groups: "dict[str, list[str]]" = {}
    for pid, pkg in packages.items():
        at = pkg.at
        if isinstance(at, AtLocation):
            groups.setdefault(at.loc_id, []).append(pid)
    for loc in groups:
        groups[loc].sort()
    if len(_PKG_ORDER_CACHE) >= _LAYOUT_CACHE_CAP:
        del _PKG_ORDER_CACHE[next(iter(_PKG_ORDER_CACHE))]
    _PKG_ORDER_CACHE[key] = (packages, groups)
    return groups


# ---------------------------------------------------------------------------
# Apilado compacto de cajas libres y de transportadores
# ---------------------------------------------------------------------------
#
# Cajas (AtLocation): en la MITAD DERECHA de la zona hábil. Espejo de los
# transportadores. La capa frontal (la más cercana) es una pirámide truncada
# (fila de 3 abajo = BASE + fila de 2 encima, en los valles) pegada (0px). Se
# rellena base→encima. Al completarla (5 cajas), la siguiente capa se sitúa por
# DETRÁS, desplazada en la diagonal ARRIBA-DERECHA (el carrier inclina su
# diagonal intra-capa a la izquierda; la pila de cajas se inclina a la derecha
# = espejo). Como mucho _BOX_LAYERS capas (3×5 = 15 cajas) visibles; las que
# rebasan comparten la posición de la caja central de la última capa y NO se
# dibujan hasta que un dron interactúa con ellas (al irse extrayendo, la lista
# compacta y van aflorando). Sistema tipo lista: la caja n-ésima presente ocupa
# el slot n. El orden de pintado es POR CAPAS (igual que el carrier): capa de
# atrás entera primero, de delante después; dentro de capa, la fila de encima
# (más al fondo) antes que la base, para que la base tape lo que toca.
#
# Transportadores: como mucho _CARRIER_MAX visibles en 2 capas de
# _CARRIER_PER_LAYER. La 1ª capa en diagonal arriba-izquierda; la 2ª desplazada
# ARRIBA y por DETRÁS. Los que rebasan comparten la posición del último y no se
# dibujan hasta necesario.

_BOX_GAP = 0.0             # separación entre cajas (px) — pila pegada
_BOX_FRONT = 5             # cajas por capa (fila 3 base + fila 2 encima)
_BOX_LAYERS = 3            # capas visibles como máximo (3×5 = 15 cajas)
_BOX_LAYER_DX = 8.0        # desplazamiento por capa hacia la DERECHA (al fondo);
                           # espejo del carrier (cuya diagonal intra-capa va a la izq.)
_BOX_LAYER_DY = 8.0        # y hacia ARRIBA; DX = DY → pendiente 1 (45°), igual que el carrier
_BOX_RIGHT_FACTOR = 0.55   # ancla de la pila en la mitad derecha (×location_radius);
                           # igual magnitud que el carrier (a la izquierda) → simétrico
_BOX_BASE_Y = 12.0         # baja la pila para asentarla en el rombo

_CARRIER_PER_LAYER = 3     # transportadores por capa (diagonal arriba-izquierda)
_CARRIER_LAYERS = 2        # capas de transportadores
_CARRIER_MAX = _CARRIER_PER_LAYER * _CARRIER_LAYERS  # 6 visibles como mucho
_CARRIER_DX = -7.0         # diagonal dentro de una capa (izquierda)
_CARRIER_DY = -7.0         # (arriba)
_CARRIER_LAYER_DX = 12.0   # 2ª capa desplazada a la DERECHA
_CARRIER_LAYER_DY = -12.0  # y ARRIBA (por detrás)


def _box_capacity() -> int:
    """Cajas que caben en las capas visibles (el resto quedan ocultas)."""
    return _BOX_LAYERS * _BOX_FRONT


def _box_slot(idx: int, theme: Theme) -> "tuple[float, float]":
    """(dx, dy) del slot `idx` (0-based) respecto al ancla de la mitad derecha.

    Capa frontal: fila de 3 abajo (idx 0-2, la BASE) y fila de 2 encima en los
    valles (idx 3-4). Las capas siguientes se desplazan ARRIBA-DERECHA (van
    retrocediendo al fondo; espejo del carrier, cuya diagonal intra-capa va a la
    izquierda). A partir de _BOX_LAYERS capas, las cajas comparten la posición
    de la caja central (fila inferior, centro) de la última capa.
    """
    step = theme.package_size + _BOX_GAP
    if idx >= _box_capacity():
        # Rebose: caja central de la última capa (fila inferior, columna centro).
        idx = (_BOX_LAYERS - 1) * _BOX_FRONT + 1
    layer = idx // _BOX_FRONT
    within = idx % _BOX_FRONT
    lx = layer * _BOX_LAYER_DX        # capas hacia la DERECHA (espejo del carrier)
    ly = -layer * _BOX_LAYER_DY       # y hacia ARRIBA (al fondo)
    if within < 3:                    # fila de 3 (base)
        sx = (within - 1) * step      # -step, 0, +step
        sy = 0.0
    else:                             # fila de 2 (encima, en los valles)
        sx = (within - 3 - 0.5) * step  # -step/2, +step/2
        sy = -step
    return (lx + sx, ly + sy)


def _box_depth_key(idx: int) -> "tuple[int, int, int]":
    """Orden de dibujo (atrás→adelante) de cajas, POR CAPAS, igual que el
    carrier: la capa más al fondo (índice mayor) se pinta antes; dentro de una
    capa, la fila de encima (más al fondo) antes que la base (que queda como
    cara delantera y tapa lo que toca). Orden ASCENDENTE: la última clave es la
    cara delantera (capa 0, base).

    Se ordena por capa entera —no por la profundidad cruda del slot— porque la
    pila recede de forma uniforme: ninguna caja de una capa trasera debe quedar
    por delante de una de la capa frontal, así que pintar capas completas
    atrás→adelante da la oclusión correcta sin entrelazar cajas de capas
    distintas (que era lo que rompía el render)."""
    idx = min(idx, _box_capacity() - 1)
    layer = idx // _BOX_FRONT
    within = idx % _BOX_FRONT
    row_back = 0 if within >= 3 else 1   # 0 = fila de encima (fondo); 1 = base (frente)
    return (-layer, row_back, within)


def _carrier_slot(idx: int, theme: Theme) -> "tuple[float, float]":
    """(dx, dy) del transportador `idx` (0-based) respecto al centro del suelo.

    1ª capa (idx 0-2): diagonal arriba-izquierda desde la base. 2ª capa
    (idx 3-5): la misma diagonal desplazada ARRIBA y a la DERECHA (por detrás).
    A partir de _CARRIER_MAX, todos comparten la posición del último visible.
    """
    if idx >= _CARRIER_MAX:
        idx = _CARRIER_MAX - 1
    layer = idx // _CARRIER_PER_LAYER
    within = idx % _CARRIER_PER_LAYER
    base_x = -float(theme.location_radius) * 0.55
    dx = base_x + within * _CARRIER_DX + layer * _CARRIER_LAYER_DX
    dy = within * _CARRIER_DY + layer * _CARRIER_LAYER_DY
    return (dx, dy)


def _carrier_depth_key(idx: int) -> "tuple[int, int]":
    """Orden de dibujo (atrás→adelante) de transportadores: capa de arriba
    (detrás) primero y, dentro de una capa, el más alejado (arriba-izquierda)
    antes que el de delante. Orden ASCENDENTE."""
    layer = min(idx, _CARRIER_MAX - 1) // _CARRIER_PER_LAYER
    within = min(idx, _CARRIER_MAX - 1) % _CARRIER_PER_LAYER
    return (-layer, -within)


# ---------------------------------------------------------------------------
# WorldLayout: producto del cálculo, consumido por sprites/painter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorldLayout:
    """Posiciones calculadas para un World concreto sobre un ViewBox concreto.

    Datos primarios:
        location_positions: mapping inmutable loc_id → Point (centro de
            la Location en coordenadas de pantalla).
        viewbox: el ViewBox sobre el que se calculó. Útil para que el
            painter pueda dibujar el fondo y para asserts de tests.
        world_ref: referencia al World que se usó para calcular el
            layout. Necesaria para resolver dinámicamente las posiciones
            de drones/transportadores/personas/paquetes a partir de su
            position actual sin que el WorldLayout las almacene todas
            (que serían datos derivados redundantes).
        scale: factor de tamaño de dibujo en [0,1]. 1.0 = tamaño nativo. Es
            <1 solo cuando hay tantas locations que, repartidas por el
            viewbox, sus sprites a tamaño nativo se solaparían: entonces el
            painter dibuja las entidades a este factor (un "zoom out" de los
            tamaños, no de las posiciones, que ya caben en el frame). NO
            afecta a las posiciones (las resuelve el layout dentro del
            viewbox), así que locate_entity sigue siendo coherente.

    El WorldLayout NO almacena posiciones de drones/paquetes/etc.
    directamente porque:
    - Drone.position puede cambiar entre snapshots (un Move actualiza
      drone.position). Almacenarla aquí obligaría a recalcular el
      WorldLayout entero por cada snapshot. En vez de eso, el painter
      recalcula la posición de cada entidad al vuelo consultando al
      WorldLayout su método correspondiente.
    - Las posiciones derivadas dependen de theme (offsets de
      co-localización), que es un parámetro del render, no del layout.

    Inmutable. Las consultas son funciones puras de (world, theme).
    """

    location_positions: Mapping[str, Point]
    viewbox: ViewBox
    world_ref: World
    scale: float = 1.0

    def __post_init__(self) -> None:
        # Envolvemos en MappingProxyType para impedir mutaciones externas.
        # Coherente con el patrón del World.
        if not isinstance(self.location_positions, MappingProxyType):
            object.__setattr__(
                self,
                "location_positions",
                MappingProxyType(dict(self.location_positions)),
            )

    # -----------------------------------------------------------------
    # Consultas estructurales
    # -----------------------------------------------------------------

    def location_position(self, loc_id: str) -> Point:
        """Centro del LIENZO de la location en pantalla (lo que la app puso
        como position_screen). Es la referencia visual del nodo, donde se
        pinta el sprite del edificio centrado. Para anclar entidades que
        viven DENTRO de la location (drones, cajas, personas, transp), usar
        _location_ground_center para obtener el centro del rombo hábil."""
        return self.location_positions[loc_id]

    def _location_ground_center(self, loc_id: str, theme: Theme) -> Point:
        """Centro del SUELO HÁBIL (rombo isométrico) de la location.

        El sprite de location es isométrico (110×88 nativo, escala 2x →
        220×176 en pantalla). El rombo donde aterrizan drones y se apilan
        cajas NO coincide con el centro del lienzo: está desplazado
        verticalmente por theme.location_ground_offset_y (negativo = el
        rombo está arriba del centro del lienzo).

        Todas las entidades que viven DENTRO de la location (drones,
        cajas libres, personas, transporters) deben anclarse a este
        centro, no al de location_position (que es el centro visual del
        sprite, donde se pinta el edificio).

        Para layouts no-isométricos donde el "suelo" coincide con el
        centro del lienzo (location_ground_offset_y=0), esta función
        devuelve location_position tal cual: compatibilidad transparente.
        """
        center = self.location_positions[loc_id]
        return Point(center.x, center.y + theme.location_ground_offset_y)

    def drone_position(self, drone_id: str, theme: Theme) -> Point:
        """Posición visual del drone en su Location actual.

        Aplica DOS desplazamientos verticales sobre el centro del rombo:
          1. theme.drone_hover_y_offset SIEMPRE: el drone "vuela
             estacionario" sobre el rombo (negativo = arriba), liberando
             el rombo entero para cajas/carriers/objetos.
          2. theme.drone_colocation_y_lift SOLO si hay 2+ drones: lift
             adicional sobre el centro del anillo, deja sitio entre el
             anillo y la primera caja.

        El anillo de co-localización es determinista por orden
        alfabético de drone.id entre los drones co-localizados.

        Raises:
            KeyError: si drone_id no existe en el world.
        """
        drone = self.world_ref.drones[drone_id]
        # Drones co-localizados, ordenados alfabéticamente (cacheado por world):
        colocated = _colocation_groups(self.world_ref.drones).get(
            drone.position, [drone_id]
        )
        return self.drone_position_for_colocation(
            drone.position, drone_id, colocated, theme,
        )

    def drone_position_for_colocation(
        self,
        loc_id: str,
        drone_id: str,
        colocated_ids: "list[str]",
        theme: Theme,
    ) -> Point:
        """Posición de `drone_id` en `loc_id` para un conjunto EXPLÍCITO de
        drones co-localizados (no el del world actual).

        Misma geometría que `drone_position` (hover + lift de anillo +
        `_ring_offset`), pero con la lista de co-localizados dada por el
        caller. La usa el render para INTERPOLAR la posición de un dron
        parado cuando otro dron entra/sale de su location: la posición
        "antes" (con un conjunto) y "después" (con otro) se mezclan según
        el progreso del Move, evitando que el dron parado salte de golpe a
        su hueco del anillo al cambiar el número de co-localizados.

        `drone_id` debe pertenecer a `colocated_ids`.
        """
        ground = self._location_ground_center(loc_id, theme)
        # Hover SIEMPRE: dron flotando sobre el rombo.
        hover_center = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        colocated = sorted(colocated_ids)
        # Lift adicional solo si hay anillo (>=2 drones):
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

    def transporter_position(self, transporter_id: str, theme: Theme) -> Point:
        """Posición visual del transportador en su Location actual.

        Convención: el primer transportador a la izquierda del centro de la
        Location (deja sitio a drones y personas). Los co-localizados se apilan
        en 2 capas de _CARRIER_PER_LAYER: la 1ª en diagonal arriba-izquierda, la
        2ª desplazada arriba y por detrás (hasta _CARRIER_MAX visibles). Los que
        rebasan comparten la posición del último. Sistema tipo lista: la posición
        depende del índice entre los presentes (orden alfabético), así que al
        marcharse uno, los de detrás lo sustituyen.

        Raises:
            KeyError: si transporter_id no existe en el world.
        """
        transporter = self.world_ref.transporters[transporter_id]
        center = self._location_ground_center(transporter.position, theme)
        # Co-localizados ordenados alfabéticamente (cacheado por world):
        colocated = _colocation_groups(self.world_ref.transporters).get(
            transporter.position, [transporter_id]
        )
        idx = colocated.index(transporter_id)
        dx, dy = _carrier_slot(idx, theme)
        return Point(center.x + dx, center.y + dy)

    def person_position(self, person_id: str, theme: Theme) -> Point:
        """Posición visual de la persona en su Location.

        Convención simétrica a transportadores: las personas se dibujan
        ligeramente a la DERECHA del centro de la Location. Co-localizadas
        se apilan verticalmente, orden alfabético.

        Raises:
            KeyError: si person_id no existe en el world.
        """
        person = self.world_ref.persons[person_id]
        center = self._location_ground_center(person.position, theme)
        colocated = _colocation_groups(self.world_ref.persons).get(
            person.position, [person_id]
        )
        idx = colocated.index(person_id)
        x_offset = float(theme.location_radius) * 0.55
        y_offset = (idx - (len(colocated) - 1) / 2.0) * (theme.person_radius * 2 + 6)
        return Point(center.x + x_offset, center.y + y_offset)

    def package_position(self, package_id: str, theme: Theme) -> Point:
        """Posición visual de un paquete LIBRE (AtLocation).

        Solo aplicable si el paquete tiene at=AtLocation(...). Si está
        en HeldByArm o InTransporter, esta función lanza ValueError:
        la posición del paquete en esos casos depende del drone o del
        transportador que lo lleva, y la calcula el painter combinando
        drone_position/transporter_position con el offset del brazo/
        contenedor.

        Convención: paquetes libres se apilan justo debajo del centro
        de la Location, en una fila horizontal. Orden alfabético por
        package.id.

        Raises:
            KeyError: si package_id no existe en el world.
            ValueError: si el paquete no está AtLocation.
        """
        pkg = self.world_ref.packages[package_id]
        if not isinstance(pkg.at, AtLocation):
            raise ValueError(
                f"package_position solo aplica a paquetes AtLocation; "
                f"package {package_id!r} está en {type(pkg.at).__name__}. "
                f"Para HeldByArm/InTransporter, el painter calcula la "
                f"posición a partir del drone/transportador correspondiente."
            )
        center = self._location_ground_center(pkg.at.loc_id, theme)
        # Índice PRESENTE de la caja entre las que están AtLocation aquí
        # (orden alfabético): sistema tipo lista. El slot lo da _box_slot:
        # pirámide truncada (3+2) en la mitad derecha, con capas detrás para
        # el resto, sin salir de la zona hábil ni estorbar la vista.
        ids_here = _packages_at_loc(self.world_ref.packages).get(
            pkg.at.loc_id, [package_id]
        )
        idx = ids_here.index(package_id)
        dx, dy = _box_slot(idx, theme)
        anchor_x = theme.location_radius * _BOX_RIGHT_FACTOR
        return Point(center.x + anchor_x + dx, center.y + _BOX_BASE_Y + dy)


# ---------------------------------------------------------------------------
# Cálculo del layout
# ---------------------------------------------------------------------------


def compute_layout(
    world: World, viewbox: ViewBox, *, spread_factor: float = 0.40
) -> WorldLayout:
    """Construye un WorldLayout para un World y un ViewBox dados.

    Lógica:
    - Si TODAS las Locations tienen position_screen != None: mapeo afín
      lineal del bounding box de esas coordenadas al viewbox, preservando
      aspect ratio.
    - En cualquier otro caso (alguna o todas son None): layout circular
      determinista.
    - World sin locations: WorldLayout vacío (válido pero degenerado).

    Args:
        world: estado del mundo (cualquier snapshot vale, se usa solo
            world.locations).
        viewbox: rectángulo lógico donde encajar el world.
        spread_factor: factor de separación del layout CIRCULAR (semieje de
            la elipse como fracción de la dimensión del viewbox; se capa a
            0.50 dentro de _layout_circular). NO afecta al layout manual
            (position_screen). El default 0.40 preserva el comportamiento
            histórico para llamadas directas; el render pasa
            theme.location_spread_factor (0.50 por defecto).

    Returns:
        WorldLayout completo, listo para consumir por el painter.
    """
    locs = list(world.locations.values())

    if not locs:
        # Caso degenerado: World sin locations.
        return WorldLayout(
            location_positions={},
            viewbox=viewbox,
            world_ref=world,
        )

    all_have_screen = all(loc.position_screen is not None for loc in locs)

    scale = 1.0
    if all_have_screen:
        positions = _layout_from_position_screen(locs, viewbox)
    elif len(locs) >= _SUNFLOWER_MIN_N:
        # Muchas locations sin position_screen. Dos casos:
        #  - Hay grafo de costes (aristas entre locations): layout POR GRAFO
        #    (distancias ∝ coste, hubs al centro). Es lo correcto cuando la
        #    topología importa (la duración del Move es ∝ coste).
        #  - No hay aristas: reparto 2D por girasol (phyllotaxis), que usa todo
        #    el área y separa más que una sola elipse, decidido por NÚMERO para
        #    que el buffer ampliado del supersampling también reparta en 2D.
        # En ambos casos `scale` es el factor de encaje (separación real /
        # huella) que consume el supersampling; el dibujo es SIEMPRE nativo.
        if _has_graph_edges(locs, world):
            positions = _layout_force_directed(locs, world, viewbox, spread_factor)
            scale = _graph_draw_scale(positions)
        else:
            positions = _layout_sunflower(locs, viewbox, spread_factor)
            scale = _sunflower_draw_scale(len(locs), viewbox, spread_factor)
    else:
        positions = _layout_circular(locs, viewbox, spread_factor)

    return WorldLayout(
        location_positions=positions,
        viewbox=viewbox,
        world_ref=world,
        scale=scale,
    )


def _layout_from_position_screen(
    locs: list,
    viewbox: ViewBox,
) -> dict[str, Point]:
    """Mapeo afín del bounding box de position_screen al viewbox.

    Preserva aspect ratio (escala uniforme). El bounding box de origen
    se mapea centrado dentro del viewbox.

    Casos límite:
    - 1 sola location: se coloca en el centro del viewbox (el bounding
      box es un punto, escala indefinida).
    - Todas las locs con la MISMA position_screen: caso degenerado;
      todas en el centro del viewbox.
    - bbox de altura o anchura 0 (locs alineadas en una línea): se
      preserva la otra dimensión y se centra en la dimensión degenerada.
    """
    if len(locs) == 1:
        # Una sola location: al centro del viewbox.
        return {locs[0].id: viewbox.center}

    xs = [loc.position_screen[0] for loc in locs]
    ys = [loc.position_screen[1] for loc in locs]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    src_w = max_x - min_x
    src_h = max_y - min_y

    # Si bbox de origen es degenerado en ambas dimensiones (todas en el
    # mismo punto), todas al centro.
    if src_w == 0 and src_h == 0:
        return {loc.id: viewbox.center for loc in locs}

    # Margen del 10% dentro del viewbox para evitar entidades pegadas al borde.
    inner = viewbox.inset(min(viewbox.width, viewbox.height) * 0.08)
    if inner.width <= 0 or inner.height <= 0:
        # Viewbox tan pequeño que el inset es negativo: usa el viewbox tal cual.
        inner = viewbox

    # Escala uniforme: el bbox de origen entra entero en inner sin distorsión.
    # Si src_w o src_h es 0, asignamos esa escala como infinita (no limita).
    scale_x = inner.width / src_w if src_w > 0 else math.inf
    scale_y = inner.height / src_h if src_h > 0 else math.inf
    scale = min(scale_x, scale_y)

    # Tamaño efectivo del bbox proyectado:
    proj_w = src_w * scale if src_w > 0 else 0.0
    proj_h = src_h * scale if src_h > 0 else 0.0

    # Centramos el bbox proyectado dentro de inner.
    x0 = inner.left + (inner.width - proj_w) / 2.0
    y0 = inner.top + (inner.height - proj_h) / 2.0

    result: dict[str, Point] = {}
    for loc in locs:
        sx, sy = loc.position_screen
        # Posición dentro del bbox de origen, normalizada a [0, 1] en
        # cada eje (cuidando las dimensiones degeneradas):
        nx = (sx - min_x) / src_w if src_w > 0 else 0.5
        ny = (sy - min_y) / src_h if src_h > 0 else 0.5
        # Proyección al viewbox:
        x = x0 + nx * proj_w
        y = y0 + ny * proj_h
        result[loc.id] = Point(x, y)

    return result


#: Separación mínima (px) deseada entre centros de locations para que sus
#: sprites no se solapen. Sigue la huella del sprite de location en pantalla
#: (lienzo 220×176; rombo hábil 160×104): ~170 px deja las locations
#: contiguas casi sin solape de rombo. Es un parámetro de layout (cuánto
#: separar nodos), no un tamaño de dibujo: por eso vive aquí y no en Theme.
_LOCATION_FOOTPRINT = 170.0

#: A partir de cuántas locations se reparte en girasol (relleno 2D) en vez de
#: una sola elipse. Es un umbral por NÚMERO (no por si la elipse apelotona en
#: este viewbox concreto): así el layout es consistente tanto en el viewport
#: como en el buffer ampliado del supersampling (donde una elipse volvería a
#: dejar el interior vacío -un anillo- al "caber" en el viewbox mayor).
_SUNFLOWER_MIN_N = 13


def _ellipse_perimeter(rx: float, ry: float) -> float:
    """Perímetro de una elipse (aproximación de Ramanujan)."""
    h = ((rx - ry) ** 2) / ((rx + ry) ** 2) if (rx + ry) > 0 else 0.0
    return math.pi * (rx + ry) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _ellipse_radii(viewbox: ViewBox, spread_factor: float) -> tuple[float, float]:
    """Semiejes (rx, ry) de la elipse del layout, con los mismos topes que
    _layout_circular (proporción 5:4 y cap al viewbox)."""
    ratio = 5.0 / 4.0
    half_w = viewbox.width * spread_factor
    half_h = viewbox.height * spread_factor
    radius_y = min(half_h, half_w / ratio)
    max_radius_y = min(viewbox.height / 2.0, (viewbox.width / 2.0) / ratio)
    radius_y = min(radius_y, max_radius_y)
    return radius_y * ratio, radius_y


def _ellipse_would_crowd(n: int, viewbox: ViewBox, spread_factor: float) -> bool:
    """True si colocar n locations en UNA elipse las dejaría más juntas que
    _LOCATION_FOOTPRINT (separación de perímetro = perímetro/n)."""
    if n < 13:
        # Con pocas locations la elipse nunca apelotona; además preserva el
        # aspecto clásico (y los tests de layout de 3 locations).
        return False
    rx, ry = _ellipse_radii(viewbox, spread_factor)
    return _ellipse_perimeter(rx, ry) / n < _LOCATION_FOOTPRINT


#: Ángulo áureo (rad ≈ 137.5°): la rotación entre puntos consecutivos que
#: maximiza la separación mínima de N puntos en un disco (filotaxis de Vogel).
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

#: Memo del factor de escala del girasol, keyed por (n, rx, ry) redondeados.
#: La geometría del girasol depende solo de eso, así que la distancia mínima
#: (O(N²)) se calcula UNA vez por geometría, no por frame.
_SUNFLOWER_SCALE_CACHE: dict[tuple, float] = {}
_SUNFLOWER_SCALE_CAP = 64


def _sunflower_points(
    n: int, center: Point, radius_x: float, radius_y: float
) -> list[Point]:
    """Coordenadas crudas del girasol (sin ids): punto i a radio normalizado
    √((i+½)/n) y ángulo i·ángulo_áureo, escalado a la elipse."""
    pts: list[Point] = []
    for i in range(n):
        rn = math.sqrt((i + 0.5) / n)
        theta = i * _GOLDEN_ANGLE
        pts.append(Point(
            center.x + radius_x * rn * math.cos(theta),
            center.y + radius_y * rn * math.sin(theta),
        ))
    return pts


def _min_pairwise_distance(points: list[Point]) -> float:
    """Distancia mínima entre cualquier par de puntos (0 si <2 puntos)."""
    m = math.inf
    for i in range(len(points)):
        xi, yi = points[i].x, points[i].y
        for j in range(i + 1, len(points)):
            dx = xi - points[j].x
            dy = yi - points[j].y
            d = dx * dx + dy * dy
            if d < m:
                m = d
    return math.sqrt(m) if m != math.inf else 0.0


def _sunflower_draw_scale(n: int, viewbox: ViewBox, spread_factor: float) -> float:
    """Factor de tamaño de dibujo (∈ [0.18, 1.0]) para que los sprites de
    location no se solapen con el reparto girasol de n locations.

    Usa la distancia mínima REAL entre puntos del girasol y escala para que la
    huella del sprite (_LOCATION_FOOTPRINT) quepa en esa distancia (con un 10%
    de aire). Si ni así caben (muchísimas locations), satura a un mínimo para
    no hacerlos ilegibles. Se memoiza por geometría: el O(N²) ocurre una vez,
    no por frame.
    """
    rx, ry = _ellipse_radii(viewbox, spread_factor)
    key = (n, round(rx, 1), round(ry, 1))
    cached = _SUNFLOWER_SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    pts = _sunflower_points(n, viewbox.center, rx, ry)
    min_dist = _min_pairwise_distance(pts)
    scale = (min_dist * 0.90) / _LOCATION_FOOTPRINT if min_dist > 0 else 1.0
    scale = max(0.18, min(1.0, scale))
    if len(_SUNFLOWER_SCALE_CACHE) >= _SUNFLOWER_SCALE_CAP:
        del _SUNFLOWER_SCALE_CACHE[next(iter(_SUNFLOWER_SCALE_CACHE))]
    _SUNFLOWER_SCALE_CACHE[key] = scale
    return scale


def _layout_sunflower(
    locs: list, viewbox: ViewBox, spread_factor: float = 0.40
) -> dict[str, Point]:
    """Layout 2D determinista por phyllotaxis ("girasol"): reparte las
    locations por TODO el interior de la elipse del viewbox, no solo el
    perímetro, con separación casi uniforme (distancia entre vecinos
    ≈ radio·√(π/N)).

    Cada punto i se coloca a radio normalizado √((i+½)/N) (reparto de área
    uniforme) y ángulo i·ángulo_áureo (≈137.5°), que es la disposición que
    maximiza la separación mínima para N puntos en un disco. Orden por id
    (determinista). Los semiejes son los mismos que la elipse, así que el
    grafo ocupa la misma región segura del viewbox.
    """
    sorted_locs = sorted(locs, key=lambda L: L.id)
    n = len(sorted_locs)
    if n == 1:
        return {sorted_locs[0].id: viewbox.center}
    radius_x, radius_y = _ellipse_radii(viewbox, spread_factor)
    pts = _sunflower_points(n, viewbox.center, radius_x, radius_y)
    return {loc.id: pts[i] for i, loc in enumerate(sorted_locs)}


def _layout_circular(
    locs: list, viewbox: ViewBox, spread_factor: float = 0.40
) -> dict[str, Point]:
    """Layout elíptico determinista: locs en elipse 5:4, orden alfabético.

    La elipse tiene proporción 5:4 (ancho:alto), la misma que el LIENZO
    del sprite isométrico de las locations (110:88 nativo, 220:176 en
    pantalla). Así el grafo se ve "extendido" horizontalmente, evitando
    que los nodos queden comprimidos en altura cuando el sprite es más
    ancho que alto.

    `spread_factor` controla los semiejes como fracción de la dimensión del
    viewbox, PERO se capa para que las locations no rebasen el viewbox (es
    decir, radius_y ≤ viewbox.height/2 y radius_x ≤ viewbox.width/2). Como el
    viewbox que pasa el painter (_scene_viewbox) ya reserva la holgura de las
    decoraciones, este cap garantiza que NADA se recorte sea cual sea el
    tamaño de la surface ni el valor de spread_factor:
      - spread_factor ≤ 0.50 → el cap no actúa (radius = factor*dim).
      - spread_factor > 0.50 → el cap satura radius al borde del viewbox
        (no aumenta la separación pero tampoco recorta). El factor
        proporcional, sin cap, recortaría MÁS en surfaces grandes (la
        decoración es de tamaño fijo): el cap elimina esa fragilidad.

    Casos:
    - 1 location: al centro del viewbox.
    - N >= 2: en elipse, primera location en ángulo -90° (arriba) para
      resultado visualmente intuitivo, sentido horario.
    """
    sorted_locs = sorted(locs, key=lambda L: L.id)

    if len(sorted_locs) == 1:
        return {sorted_locs[0].id: viewbox.center}

    center = viewbox.center
    # Semieje horizontal limitado por (a) la fracción spread_factor del ancho,
    # y (b) la altura disponible escalada por 5:4. Proporción 5:4 → 1.25.
    ratio_x_over_y = 5.0 / 4.0
    half_w = viewbox.width * spread_factor
    half_h = viewbox.height * spread_factor
    # La elipse no debe salirse del semieje pedido: con rx = ry * ratio,
    # ry ≤ min(half_h, half_w/ratio).
    radius_y = min(half_h, half_w / ratio_x_over_y)
    # CAP de seguridad: las locations no deben rebasar el viewbox (cuyo borde
    # ya coincide con la frontera de la región segura de decoraciones). Tope:
    # radius_y ≤ viewbox.height/2 y radius_x = ratio*radius_y ≤ viewbox.width/2.
    max_radius_y = min(viewbox.height / 2.0, (viewbox.width / 2.0) / ratio_x_over_y)
    radius_y = min(radius_y, max_radius_y)
    radius_x = radius_y * ratio_x_over_y
    n = len(sorted_locs)

    result: dict[str, Point] = {}
    for i, loc in enumerate(sorted_locs):
        # Empezamos en -90° (arriba) y vamos en sentido horario.
        angle = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        x = center.x + radius_x * math.cos(angle)
        y = center.y + radius_y * math.sin(angle)
        result[loc.id] = Point(x, y)

    return result


# ---------------------------------------------------------------------------
# Layout por grafo (híbrido): semilla radial + relajación por fuerzas
# ---------------------------------------------------------------------------
#
# Cuando las locations no traen position_screen PERO el World tiene un grafo de
# costes (World.costs), repartirlas por girasol ignora la topología: el depósito
# (conectado a todas) acaba en una esquina y dos locations a igual coste pueden
# quedar a distancias arbitrarias, lo que hace que la velocidad aparente del
# dron varíe (la duración del Move es ∝ coste, así que distancia_en_pantalla
# debería ser ∝ coste para que la velocidad se vea constante).
#
# Este layout coloca las distancias ∝ coste y centra los nodos muy conectados:
#   1. SEMILLA RADIAL: el hub (mayor grado) al centro; el resto a radio =
#      distancia de coste al hub (Dijkstra) y ángulo áureo. Para una estrella
#      esto ya es la solución exacta (hoja a radio ∝ su coste).
#   2. RELAJACIÓN POR FUERZAS (estilo Obsidian): muelles en las aristas con
#      longitud de reposo = coste, repulsión entre todos los nodos, y leve
#      gravedad al centroide. Desenreda mallas/ciclos y múltiples hubs que la
#      semilla radial no resuelve, manteniéndose cerca de la semilla (pocas
#      iteraciones, enfriamiento) para no perturbar las estrellas.
#
# Determinista (orden por id, sin RNG, iteraciones fijas) y O(iter·N²): se
# calcula UNA vez por (grafo, viewbox) y se memoiza, no por frame.

#: A partir de cuántas locations se usa el layout por grafo (mismo umbral que
#: el girasol: por debajo se conserva la elipse clásica y los tests de N pequeño).
_GRAPH_MIN_N = _SUNFLOWER_MIN_N

#: Parámetros de la relajación por fuerzas. Calibrados (banco de grafos
#: sintéticos: estrella, malla, dos hubs) para que las distancias queden ∝ coste
#: —la semilla radial ya lo es para estrellas; la relajación apenas las toca— y
#: las mallas/ciclos se desenreden (error relativo de arista ~1–4%). Las fuerzas
#: se escalan por L0 (coste medio de arista): son adimensionales respecto a la
#: escala de costes del problema.
_FORCE_ATTRACTION = 0.40     # rigidez del muelle de arista (hacia longitud=coste)
_FORCE_REPULSION = 0.25      # intensidad de repulsión (× L0² / dist²)
_FORCE_GRAVITY = 0.012       # atracción al centroide (cohesión de componentes)
_FORCE_TEMP0_FACTOR = 0.20   # paso inicial como fracción de L0
_FORCE_COOLING = 0.990       # enfriamiento multiplicativo por iteración

#: Iteraciones ADAPTATIVAS: el bucle es O(iter·N²). Con un presupuesto fijo, el
#: nº de iteraciones baja al crecer N, acotando el coste (el grafo grande con
#: buena semilla converge en menos pasos). Mantiene el cálculo < ~0,5 s incluso
#: con ~100 locations, y es de una sola vez (memoizado).
_FORCE_ITER_BUDGET = 18000
_FORCE_MIN_ITERS = 240
_FORCE_MAX_ITERS = 480


def _force_iterations(n: int) -> int:
    """Nº de iteraciones de relajación para n nodos (presupuesto O(iter·N²))."""
    return max(_FORCE_MIN_ITERS, min(_FORCE_MAX_ITERS, round(_FORCE_ITER_BUDGET / n)))

#: Memo del layout por grafo, keyed por (ids, aristas, viewbox, spread).
_FORCE_LAYOUT_CACHE: dict[tuple, dict[str, Point]] = {}
_FORCE_LAYOUT_CAP = 16


def _has_graph_edges(locs: list, world) -> bool:
    """True si World.costs define al menos una arista entre dos locations
    distintas presentes (condición para usar el layout por grafo)."""
    ids = {L.id for L in locs}
    for (o, d) in world.costs.keys():
        if o != d and o in ids and d in ids:
            return True
    return False


def _graph_edges(loc_ids: set[str], world) -> dict[str, dict[str, float]]:
    """Adyacencia no dirigida {i: {j: coste}} a partir de World.costs.

    Ignora bucles y aristas a locations ausentes. Si hay duplicados o
    asimetría, toma el menor coste (determinista).
    """
    adj: dict[str, dict[str, float]] = {i: {} for i in loc_ids}
    for (o, d), c in world.costs.items():
        if o == d or o not in loc_ids or d not in loc_ids:
            continue
        cf = float(c)
        if d not in adj[o] or cf < adj[o][d]:
            adj[o][d] = cf
            adj[d][o] = cf
    return adj


def _dijkstra_costs(ids: list[str], adj: dict[str, dict[str, float]],
                    source: str) -> dict[str, float]:
    """Distancia de coste mínima de `source` a cada nodo (inf si inalcanzable)."""
    import heapq
    dist = {i: math.inf for i in ids}
    dist[source] = 0.0
    pq: list[tuple[float, str]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _graph_layout_key(loc_ids: list[str], world, viewbox: ViewBox,
                      spread_factor: float) -> tuple:
    """Clave de caché estable entre snapshots (mismas locs/costes/viewbox)."""
    edges = tuple(sorted(
        (o, d, round(float(c), 3)) for (o, d), c in world.costs.items()
        if o < d and o in loc_ids and d in loc_ids
    ))
    vb = (round(viewbox.left, 1), round(viewbox.top, 1),
          round(viewbox.width, 1), round(viewbox.height, 1))
    return (tuple(loc_ids), edges, vb, round(spread_factor, 3))


def _layout_force_directed(
    locs: list, world, viewbox: ViewBox, spread_factor: float = 0.40
) -> dict[str, Point]:
    """Layout por grafo: distancias ∝ coste, hubs al centro. Ver cabecera.

    Híbrido semilla-radial + relajación por fuerzas, determinista y memoizado.
    """
    ids = sorted(L.id for L in locs)
    n = len(ids)
    if n == 1:
        return {ids[0]: viewbox.center}

    key = _graph_layout_key(ids, world, viewbox, spread_factor)
    cached = _FORCE_LAYOUT_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    idset = set(ids)
    adj = _graph_edges(idset, world)

    # Longitud característica: coste medio de arista (escala de las fuerzas).
    all_costs = [c for i in ids for c in adj[i].values()]
    L0 = (sum(all_costs) / len(all_costs)) if all_costs else 1.0
    if L0 <= 0:
        L0 = 1.0

    # --- 1) Semilla radial: hub (mayor grado, desempate por id) al origen ---
    deg = {i: len(adj[i]) for i in ids}
    hub = min(ids, key=lambda i: (-deg[i], i))
    dist_hub = _dijkstra_costs(ids, adj, hub)
    finite = [d for d in dist_hub.values() if math.isfinite(d) and d > 0.0]
    rmax = max(finite) if finite else L0
    pos: dict[str, list] = {hub: [0.0, 0.0]}
    others = [i for i in ids if i != hub]
    for k, i in enumerate(others):
        r = dist_hub[i]
        if not math.isfinite(r) or r <= 0.0:
            r = rmax * 1.3  # componentes desconectadas: fuera, las junta la gravedad
        theta = k * _GOLDEN_ANGLE
        pos[i] = [r * math.cos(theta), r * math.sin(theta)]

    # --- 2) Relajación por fuerzas (muelles ∝ coste, repulsión, gravedad) ---
    k_rep = _FORCE_REPULSION * L0 * L0
    temp = _FORCE_TEMP0_FACTOR * L0
    eps = 1e-6
    for _ in range(_force_iterations(n)):
        disp = {i: [0.0, 0.0] for i in ids}
        # Repulsión entre todos los pares (O(N²)).
        for a in range(n):
            ia = ids[a]
            xa, ya = pos[ia]
            for b in range(a + 1, n):
                ib = ids[b]
                dx = xa - pos[ib][0]
                dy = ya - pos[ib][1]
                d2 = dx * dx + dy * dy
                if d2 < eps:
                    dx, dy, d2 = eps, 0.0, eps  # desempate determinista
                dist = math.sqrt(d2)
                f = k_rep / d2
                ux, uy = dx / dist, dy / dist
                disp[ia][0] += ux * f
                disp[ia][1] += uy * f
                disp[ib][0] -= ux * f
                disp[ib][1] -= uy * f
        # Atracción de aristas hacia su longitud de reposo (= coste).
        for i in ids:
            xi, yi = pos[i]
            for j, rest in adj[i].items():
                if j <= i:
                    continue  # cada arista una vez
                dx = xi - pos[j][0]
                dy = yi - pos[j][1]
                dist = math.sqrt(dx * dx + dy * dy) or eps
                f = _FORCE_ATTRACTION * (dist - rest)
                ux, uy = dx / dist, dy / dist
                disp[i][0] -= ux * f
                disp[i][1] -= uy * f
                disp[j][0] += ux * f
                disp[j][1] += uy * f
        # Gravedad al centroide (cohesión, sobre todo si hay componentes).
        cx = sum(pos[i][0] for i in ids) / n
        cy = sum(pos[i][1] for i in ids) / n
        for i in ids:
            disp[i][0] += (cx - pos[i][0]) * _FORCE_GRAVITY
            disp[i][1] += (cy - pos[i][1]) * _FORCE_GRAVITY
        # Aplicar con tope de temperatura (paso máximo) y enfriar.
        for i in ids:
            ddx, ddy = disp[i]
            dlen = math.hypot(ddx, ddy)
            if dlen > temp:
                ddx, ddy = ddx / dlen * temp, ddy / dlen * temp
            pos[i][0] += ddx
            pos[i][1] += ddy
        temp *= _FORCE_COOLING

    # --- 3) Encaje al viewbox: escala UNIFORME (preserva ∝ coste), centrado --
    xs = [pos[i][0] for i in ids]
    ys = [pos[i][1] for i in ids]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = (maxx - minx) or 1.0
    bh = (maxy - miny) or 1.0
    rx, ry = _ellipse_radii(viewbox, spread_factor)  # región segura del viewbox
    s = min((2.0 * rx) / bw, (2.0 * ry) / bh)  # uniforme: no distorsiona
    gcx = (minx + maxx) / 2.0
    gcy = (miny + maxy) / 2.0
    c = viewbox.center
    result = {
        i: Point(c.x + (pos[i][0] - gcx) * s, c.y + (pos[i][1] - gcy) * s)
        for i in ids
    }

    if len(_FORCE_LAYOUT_CACHE) >= _FORCE_LAYOUT_CAP:
        del _FORCE_LAYOUT_CACHE[next(iter(_FORCE_LAYOUT_CACHE))]
    _FORCE_LAYOUT_CACHE[key] = dict(result)
    return result


_GRAPH_SCALE_CACHE: "dict[tuple, float]" = {}
_GRAPH_SCALE_CAP = 256


def _graph_draw_scale(positions: dict[str, Point]) -> float:
    """Factor de encaje (∈ [0.18, 1.0]) para el supersampling: que la huella del
    sprite quepa en la distancia mínima real entre nodos del layout por grafo.

    Memoizado (gemelo de _sunflower_draw_scale): el layout por grafo es
    invariante entre snapshots —las locations no se mueven durante el plan— así
    que el O(n²) de _min_pairwise_distance se ejecuta una vez por geometría y los
    frames sucesivos reutilizan el resultado. Se usa clave de VALOR (posiciones
    redondeadas), no id(), porque _layout_force_directed devuelve una copia nueva
    del dict cacheado en cada llamada; construir la clave es O(n log n), muy por
    debajo del O(n²) que evita.
    """
    key = tuple(sorted(
        (round(p.x, 1), round(p.y, 1)) for p in positions.values()
    ))
    cached = _GRAPH_SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    min_dist = _min_pairwise_distance(list(positions.values()))
    scale = (min_dist * 0.90) / _LOCATION_FOOTPRINT if min_dist > 0 else 1.0
    result = max(0.18, min(1.0, scale))
    if len(_GRAPH_SCALE_CACHE) >= _GRAPH_SCALE_CAP:
        del _GRAPH_SCALE_CACHE[next(iter(_GRAPH_SCALE_CACHE))]
    _GRAPH_SCALE_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Utilidad interna: distribución en anillo para co-localización
# ---------------------------------------------------------------------------


def _ring_offset(
    center: Point,
    entity_id: str,
    colocated_ids: list[str],
    radius: float,
) -> Point:
    """Posición de una entidad dentro de un anillo de co-localizadas.

    Reglas:
    - Si solo hay 1 entidad co-localizada (esta misma): al centro.
    - Si hay N >= 2: distribuir en un anillo de `radius` alrededor del
      centro, ángulos equiespaciados, primera entidad en -90° (arriba).
      El orden lo dicta `colocated_ids` tal como se pasó (típicamente
      ya ordenado alfabéticamente por el caller).

    Args:
        center: posición central (típicamente el centro de la Location).
        entity_id: id de la entidad que queremos posicionar.
        colocated_ids: lista de TODOS los ids co-localizados (incluyendo
            el de entity_id), en el orden que define el reparto angular.
        radius: radio del anillo.

    Returns:
        Point con la posición de entity_id en el anillo.

    Raises:
        ValueError: si entity_id no está en colocated_ids (programación
            defectuosa; rompemos pronto en vez de devolver una posición
            silenciosamente incorrecta).
    """
    if entity_id not in colocated_ids:
        raise ValueError(
            f"entity_id {entity_id!r} no aparece en colocated_ids "
            f"{colocated_ids!r}; programación defectuosa del caller."
        )

    n = len(colocated_ids)
    if n == 1:
        return center

    idx = colocated_ids.index(entity_id)
    angle = -math.pi / 2.0 + (2.0 * math.pi * idx / n)
    return Point(
        center.x + radius * math.cos(angle),
        center.y + radius * math.sin(angle),
    )