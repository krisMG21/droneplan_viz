"""Formato de líneas del inventario dinámico del HUD.

El inventario muestra el estado actual del world snapshot en la tab
"Inventario" del panel lateral, agrupado por categoría:

    ── DRONES ──
      d1 · IDLE en deposito · brazos 2
      d2 · IDLE en casa1
    ── LOCATIONS ──
      casa1 · 0 pkg · 1 persona
      ...
    ── PERSONAS ──
      p1 en casa1 · pide 2 cosas
      ...
    ── PAQUETES ──
      pkg_med1 (medicina) en deposito
      ...
    ── TRANSPORTERS ──
      t1 en deposito · 0/4 paquetes

Las líneas se generan a partir de un World (snapshot actual). Cada línea
real (no las cabeceras) lleva asociado el id de la entidad que
representa, para que el click-to-focus pueda enfocar la cámara sobre ella.

Las funciones son PURAS: no consumen pygame, no mutan estado, no tienen
side effects. Son testeables aisladamente con datos del dominio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from droneplan_viz.domain import World
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.package import (
    AtLocation,
    HeldByArm,
    InTransporter,
)

#: Prefijos visuales para las cabeceras de sección. Usamos `·` (middot)
#: que sí está en Monogram extended (lo usamos también en otros textos
#: del HUD). Los box-drawing chars (U+2500) no están en Monogram y se
#: renderizan como cuadrados vacíos.
_HEADER_PREFIX = "· "
_HEADER_SUFFIX = " ·"

#: Tipos de item en el inventario: "header" (cabecera de sección, no
#: clicable) o "entity" (item real, click-to-focus enfoca su entity_id).
ItemKind = Literal["header", "entity"]


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """Una línea del inventario.

    Attributes:
        text: Texto a mostrar en el UISelectionList (legacy del diseño
            anterior). Es la concatenación de display_name + stats.
            Se mantiene por compatibilidad con tests existentes y por
            si se necesita una vista plana.
        kind: "header" para cabeceras de sección (no clicables);
            "entity" para items reales.
        entity_id: Id de la entidad asociada al item (None si kind=header).
            El controller usa este id para enfocar la cámara via
            focus_on_entity.
        display_name: Nombre principal de la entidad (e.g. "d1", "casa1").
            Se muestra como primera línea en el rediseño con cajas.
            None para headers.
        stats: Líneas de estadística (1 o 2 elementos) bajo el nombre.
            Cada línea se renderiza como un UILabel separado en la caja.
            Tupla vacía para headers.
        category: Cabecera bajo la que se agrupa el item (e.g. "DRONES",
            "LOCATIONS"). Para items kind="header" coincide con el nombre
            de su propia categoría; para entity, indica a qué sección
            pertenece (necesario para poder ocultar al plegar).
    """
    text: str
    kind: ItemKind
    entity_id: str | None = None
    display_name: str | None = None
    stats: tuple[str, ...] = ()
    category: str = ""


def build_inventory(world: World) -> tuple[InventoryItem, ...]:
    """Genera la lista completa de items del inventario para un World.

    El orden de las categorías es: Drones, Locations, Personas, Paquetes,
    Transporters. Dentro de cada categoría, los items se ordenan por id
    para que la lista sea estable (no salta al re-renderizar).

    Cada categoría empieza por una cabecera (kind="header") y continúa
    con los items entidad. Si una categoría está vacía, se omite la
    cabecera (no mostramos "· PAQUETES ·" si no hay paquetes).

    Cada item entity lleva `display_name` (nombre solo) y `stats` (1-2
    líneas de info adicional) para que el rediseño con cajas pueda
    renderizar el nombre destacado encima y las stats debajo. El campo
    `text` legacy se mantiene como concatenación de ambos por
    compatibilidad con el código anterior.

    Args:
        world: WorldSnapshot del que extraer las entidades.

    Returns:
        Tupla de InventoryItem en orden de visualización.
    """
    items: list[InventoryItem] = []

    def _add_section(
        category: str,
        ids: list[str],
        line_fn,
        structured_fn,
    ) -> None:
        if not ids:
            return
        items.append(InventoryItem(
            text=f"{_HEADER_PREFIX}{category}{_HEADER_SUFFIX}",
            kind="header",
            category=category,
        ))
        for eid in ids:
            name, stats = structured_fn(world, eid)
            items.append(InventoryItem(
                text=line_fn(world, eid),  # legacy plano
                kind="entity",
                entity_id=eid,
                display_name=name,
                stats=stats,
                category=category,
            ))

    _add_section("DRONES", sorted(world.drones),
                 format_drone_line, drone_structured)
    _add_section("LOCATIONS", sorted(world.locations),
                 format_location_line, location_structured)
    _add_section("PERSONAS", sorted(world.persons),
                 format_person_line, person_structured)
    _add_section("PAQUETES", sorted(world.packages),
                 format_package_line, package_structured)
    _add_section("TRANSPORTERS", sorted(world.transporters),
                 format_transporter_line, transporter_structured)

    return tuple(items)


def format_drone_line(world: World, drone_id: str) -> str:
    """Línea para un drone según su estado.

    - IDLE: `did · IDLE en loc · brazos N` (o si tiene carga, los muestra).
    - MOVING: `did · MOVING → destino` (durante una transición; aproximación).
    - INTERACTING: `did · INTERACTING en loc`.

    Como `DroneState` es un Enum, el state es un valor estático en el
    snapshot; no diferenciamos visualmente entre "drone en MOVING" y "drone
    parado momentáneamente con state=MOVING" — confiamos en el dato del
    snapshot.
    """
    drone = world.drones[drone_id]
    state_name = drone.state.name
    loc = drone.position
    # Para drones con carga, listamos los brazos con contenido en lugar
    # del recuento (más informativo, cabe holgadamente).
    arm_contents = []
    for arm in drone.arms:
        pkg = world.package_held_by(drone_id, arm.id)
        if pkg is not None:
            arm_contents.append((arm.id, pkg.id))
    if arm_contents:
        carga = ", ".join(f"{aid}:{pid}" for aid, pid in arm_contents)
        return f"{drone_id} · {state_name} en {loc} · {carga}"
    # Sin carga: cuenta brazos para informar capacidad disponible.
    n_arms = len(drone.arms)
    return f"{drone_id} · {state_name} en {loc} · brazos {n_arms}"


def format_location_line(world: World, loc_id: str) -> str:
    """Línea para una location con conteo agregado de contenido.

    `id · N pkg · M personas · K transp`. Omitimos categorías con cero
    contenido para reducir ruido (`casa1 · 1 persona` en lugar de
    `casa1 · 0 pkg · 1 personas · 0 transp`).
    """
    n_pkgs = sum(
        1 for p in world.packages.values()
        if isinstance(p.at, AtLocation) and p.at.loc_id == loc_id
    )
    n_persons = sum(1 for per in world.persons.values() if per.position == loc_id)
    n_transp = sum(1 for t in world.transporters.values() if t.position == loc_id)

    partes = []
    if n_pkgs > 0:
        partes.append(f"{n_pkgs} pkg")
    if n_persons > 0:
        # "1 persona" vs "2 personas". Heurística mínima.
        partes.append(f"{n_persons} persona" + ("" if n_persons == 1 else "s"))
    if n_transp > 0:
        partes.append(f"{n_transp} transp")

    if not partes:
        return f"{loc_id} · vacía"
    return f"{loc_id} · " + " · ".join(partes)


def format_person_line(world: World, person_id: str) -> str:
    """Línea para una persona: `id en loc · pide N cosas` (o "satisfecho")."""
    per = world.persons[person_id]
    needs_count = len(per.needs)
    if needs_count == 0:
        return f"{person_id} en {per.position} · satisfecho"
    # Listamos contenidos pedidos hasta cierto ancho razonable.
    if needs_count <= 2:
        needs_str = ", ".join(n.id for n in per.needs)
        return f"{person_id} en {per.position} · pide {needs_str}"
    # Para 3+, mostrar solo el recuento.
    return f"{person_id} en {per.position} · pide {needs_count} cosas"


def format_package_line(world: World, package_id: str) -> str:
    """Línea para un paquete según dónde esté.

    - AtLocation: `pkg_id (contenido) en loc`.
    - HeldByArm: `pkg_id → arm.X de drone_id`.
    - InTransporter: `pkg_id en transp_id`.
    """
    pkg = world.packages[package_id]
    contenido = pkg.contains.id
    at = pkg.at
    if isinstance(at, AtLocation):
        return f"{package_id} ({contenido}) en {at.loc_id}"
    if isinstance(at, HeldByArm):
        return f"{package_id} → arm.{at.arm_id} de {at.drone_id}"
    if isinstance(at, InTransporter):
        return f"{package_id} en {at.transporter_id}"
    # Defensivo: tipo desconocido.
    return f"{package_id} ({contenido}) · ?"


def format_transporter_line(world: World, transporter_id: str) -> str:
    """Línea para un transporter: `id en loc · X/Y paquetes`."""
    t = world.transporters[transporter_id]
    n_in = sum(1 for _ in world.packages_in_transporter(transporter_id))
    return f"{transporter_id} en {t.position} · {n_in}/{t.capacity} paquetes"


# ---------------------------------------------------------------------------
# Formatters "estructurados" para el rediseño con cajas
# ---------------------------------------------------------------------------
#
# Devuelven (display_name, stats) donde stats es una tupla de 1 o 2
# líneas. Estas funciones son las que consume el HUD rediseñado; las
# format_*_line legacy se mantienen para que el campo `text` de
# InventoryItem siga siendo válido y los tests anteriores no se rompan.


def drone_structured(world: World, drone_id: str) -> tuple[str, tuple[str, ...]]:
    """Drone: nombre = id, stats = ('STATE en loc', 'brazos N' o
    'izq:pkg, der:pkg' si lleva carga)."""
    drone = world.drones[drone_id]
    name = drone_id
    state_loc = f"{drone.state.name} en {drone.position}"
    # Si tiene carga, segunda línea muestra los brazos cargados.
    arm_contents = []
    for arm in drone.arms:
        pkg = world.package_held_by(drone_id, arm.id)
        if pkg is not None:
            arm_contents.append((arm.id, pkg.id))
    if arm_contents:
        carga = ", ".join(f"{aid}:{pid}" for aid, pid in arm_contents)
        return name, (state_loc, carga)
    return name, (state_loc, f"brazos {len(drone.arms)}")


def location_structured(world: World, loc_id: str) -> tuple[str, tuple[str, ...]]:
    """Location: nombre = id, stats = ('N pkg · M personas · K transp')
    o ('vacía') si no hay contenido."""
    name = loc_id
    n_pkgs = sum(
        1 for p in world.packages.values()
        if isinstance(p.at, AtLocation) and p.at.loc_id == loc_id
    )
    n_persons = sum(1 for per in world.persons.values() if per.position == loc_id)
    n_transp = sum(1 for t in world.transporters.values() if t.position == loc_id)

    partes = []
    if n_pkgs > 0:
        partes.append(f"{n_pkgs} pkg")
    if n_persons > 0:
        partes.append(f"{n_persons} persona" + ("" if n_persons == 1 else "s"))
    if n_transp > 0:
        partes.append(f"{n_transp} transp")

    if not partes:
        return name, ("vacía",)
    return name, (" · ".join(partes),)


def person_structured(world: World, person_id: str) -> tuple[str, tuple[str, ...]]:
    """Persona: nombre = id, stats = ('en loc', 'pide X' o 'satisfecho')."""
    per = world.persons[person_id]
    name = person_id
    line_loc = f"en {per.position}"
    if not per.needs:
        return name, (line_loc, "satisfecho")
    if len(per.needs) <= 2:
        needs_str = ", ".join(n.id for n in per.needs)
        return name, (line_loc, f"pide {needs_str}")
    return name, (line_loc, f"pide {len(per.needs)} cosas")


def package_structured(world: World, package_id: str) -> tuple[str, tuple[str, ...]]:
    """Paquete: nombre = id, stats = ('contenido', 'ubicación')."""
    pkg = world.packages[package_id]
    name = package_id
    contenido = f"({pkg.contains.id})"
    at = pkg.at
    if isinstance(at, AtLocation):
        ubic = f"en {at.loc_id}"
    elif isinstance(at, HeldByArm):
        ubic = f"arm.{at.arm_id} de {at.drone_id}"
    elif isinstance(at, InTransporter):
        ubic = f"en {at.transporter_id}"
    else:
        ubic = "?"
    return name, (contenido, ubic)


def transporter_structured(
    world: World, transporter_id: str
) -> tuple[str, tuple[str, ...]]:
    """Transporter: nombre = id, stats = ('en loc', 'X/Y paquetes')."""
    t = world.transporters[transporter_id]
    name = transporter_id
    n_in = sum(1 for _ in world.packages_in_transporter(transporter_id))
    return name, (f"en {t.position}", f"{n_in}/{t.capacity} paquetes")
