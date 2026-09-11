"""
Handlers de aplicación: traducen un Command genérico en una transición
inmutable (World, MetricsTracker) -> (World', MetricsTracker').

Las seis funciones apply_X aplican los efectos PDDL de su acción mediante
dataclasses.replace, en consonancia con la filosofía:
transiciones inmutables, sin mutación, una sola fuente de verdad para cada
relación.

La función pública apply(world, metrics, cmd) es el único punto de entrada;
dispatcha mediante match exhaustivo a la apply_X correspondiente.

Contrato general:
- Asumen Command pre-validado. La invocación con un Command inválido en
  el World dado tiene comportamiento indefinido (puede levantar KeyError,
  TypeError, etc.). El runtime garantiza que solo se llame
  apply() tras validate() correcto.
- Devuelven (World, MetricsTracker) nuevos. Los originales no se mutan.
- Actualizan las métricas que son responsabilidad del efecto PDDL en sí:
  total_cost (coste de vuelo, leído de world.costs) y action_count. Si el
  plan no declara costes, world.costs estará vacío y el delta de coste
  será 0.

Sobre total_time (makespan): NO es responsabilidad de los handlers. El
makespan depende del momento absoluto en que cada acción termina
(end_time), información temporal que vive en el plan y que solo el runtime
 conoce. El runner fija total_time = max(total_time, end_time)
en su evento END, tras llamar a apply(). Los handlers dejan total_time
intacto (lo heredan del MetricsTracker de entrada): en una transición
aislada no hay reloj y no tiene sentido inventar uno. Esto evita el código
muerto que existía, donde los handlers calculaban un total_time
que el runner sobreescribía siempre.

Sobre arm_id derivado en Deliver y LoadIntoTransporter: estos Commands no
llevan arm_id (decisión heredada). El handler localiza qué
brazo sostiene el paquete recorriendo world.packages. Como Validator
asegura que ALGÚN brazo del drone sostiene el paquete, la búsqueda nunca
falla cuando el Command está pre-validado.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from droneplan_viz.commands.base import Command
from droneplan_viz.commands.manipulation import Deliver, PickUp
from droneplan_viz.commands.movement import Move, MoveWithTransporter
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter
from droneplan_viz.domain import (
    AtLocation,
    HeldByArm,
    InTransporter,
    MetricsTracker,
    World,
)


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------
def _with_entity(world: World, collection: str, entity_id: str, entity: Any) -> World:
    """Devuelve un World nuevo con `entity` sustituyendo a la entrada
    `entity_id` de la colección `collection` (p.ej. "packages", "drones").

    Encapsula el patrón inmutable repetido por los handlers de entidad
    única: copiar el dict de la colección, reemplazar una entrada y
    reconstruir el World con dataclasses.replace. World.__post_init__
    re-envuelve el dict en MappingProxyType, así que el resultado mantiene
    la garantía de inmutabilidad del dominio.

    Se usa solo en los handlers que tocan una única colección. Los handlers
    que actualizan dos colecciones a la vez (apply_move_with_transporter,
    apply_deliver) mantienen su replace() explícito para dejar patente que
    ambas mutaciones son atómicas en una sola transición.
    """
    updated = dict(getattr(world, collection))
    updated[entity_id] = entity
    return replace(world, **{collection: updated})


# ---------------------------------------------------------------------------
# Movement handlers
# ---------------------------------------------------------------------------
def apply_move(
    world: World, metrics: MetricsTracker, cmd: Move
) -> tuple[World, MetricsTracker]:
    """Aplica los efectos de la acción 'volar'."""
    drone = world.drones[cmd.drone_id]
    origin = drone.position

    new_world = _with_entity(
        world, "drones", cmd.drone_id, replace(drone, position=cmd.destination_id)
    )

    cost = world.costs.get((origin, cmd.destination_id), 0.0)
    new_metrics = replace(
        metrics,
        total_cost=metrics.total_cost + cost,
        action_count=metrics.action_count + 1,
    )
    return new_world, new_metrics


def apply_move_with_transporter(
    world: World, metrics: MetricsTracker, cmd: MoveWithTransporter
) -> tuple[World, MetricsTracker]:
    """Aplica los efectos de 'mover-transportador': mueve drone y
    transportador a la nueva localización. Los paquetes dentro del
    transportador no se tocan; su localización efectiva sigue al
    transportador vía la relación InTransporter(t).
    """
    drone = world.drones[cmd.drone_id]
    transporter = world.transporters[cmd.transporter_id]
    origin = drone.position

    new_drones = dict(world.drones)
    new_drones[cmd.drone_id] = replace(drone, position=cmd.destination_id)
    new_transporters = dict(world.transporters)
    new_transporters[cmd.transporter_id] = replace(
        transporter, position=cmd.destination_id
    )

    new_world = replace(world, drones=new_drones, transporters=new_transporters)

    cost = world.costs.get((origin, cmd.destination_id), 0.0)
    new_metrics = replace(
        metrics,
        total_cost=metrics.total_cost + cost,
        action_count=metrics.action_count + 1,
    )
    return new_world, new_metrics


# ---------------------------------------------------------------------------
# Manipulation handlers
# ---------------------------------------------------------------------------
def apply_pick_up(
    world: World, metrics: MetricsTracker, cmd: PickUp
) -> tuple[World, MetricsTracker]:
    """Aplica 'recoger': paquete pasa a HeldByArm(drone_id, arm_id)."""
    pkg = world.packages[cmd.package_id]
    new_world = _with_entity(
        world, "packages", cmd.package_id,
        replace(pkg, at=HeldByArm(drone_id=cmd.drone_id, arm_id=cmd.arm_id)),
    )

    new_metrics = replace(metrics, action_count=metrics.action_count + 1)
    return new_world, new_metrics


def apply_deliver(
    world: World, metrics: MetricsTracker, cmd: Deliver
) -> tuple[World, MetricsTracker]:
    """Aplica 'entregar': la persona recibe el contenido, el paquete vuelve
    a estar libre en la localización (el PDDL clásico no destruye cajas).

    Como el Command no especifica arm_id (decisión el runtime: liberar el
    brazo que sea), aquí simplemente actualizamos el estado del paquete
    cambiando su 'at' de HeldByArm(...) a AtLocation(...). El "brazo
    liberado" se deriva automáticamente: al dejar de tener un Package
    con HeldByArm(drone_id=d, arm_id=a) apuntando a ese brazo, el brazo
    queda implícitamente libre para futuras consultas via
    world.package_held_by(d, a) -> None.
    """
    pkg = world.packages[cmd.package_id]
    person = world.persons[cmd.person_id]
    content = pkg.contains

    new_person = person.receive(content)
    new_pkg = replace(pkg, at=AtLocation(loc_id=person.position))

    new_packages = dict(world.packages)
    new_packages[cmd.package_id] = new_pkg
    new_persons = dict(world.persons)
    new_persons[cmd.person_id] = new_person

    new_world = replace(world, packages=new_packages, persons=new_persons)

    new_metrics = replace(metrics, action_count=metrics.action_count + 1)
    return new_world, new_metrics


# ---------------------------------------------------------------------------
# Transport handlers
# ---------------------------------------------------------------------------
def apply_load_into_transporter(
    world: World, metrics: MetricsTracker, cmd: LoadIntoTransporter
) -> tuple[World, MetricsTracker]:
    """Aplica 'poner-caja-en-transportador': paquete pasa de HeldByArm
    a InTransporter; el brazo queda libre implícitamente.

    Como el Command no especifica arm_id (decisión el runtime: liberar el
    brazo que sea), simplemente actualizamos el 'at' del paquete a
    InTransporter. El brazo del drone que lo sostenía queda libre
    automáticamente por ausencia de HeldByArm apuntando a él.
    """
    pkg = world.packages[cmd.package_id]
    new_world = _with_entity(
        world, "packages", cmd.package_id,
        replace(pkg, at=InTransporter(transporter_id=cmd.transporter_id)),
    )

    new_metrics = replace(metrics, action_count=metrics.action_count + 1)
    return new_world, new_metrics


def apply_unload_from_transporter(
    world: World, metrics: MetricsTracker, cmd: UnloadFromTransporter
) -> tuple[World, MetricsTracker]:
    """Aplica 'coger-caja-del-transportador': paquete pasa de InTransporter
    a HeldByArm(drone_id, arm_id).
    """
    pkg = world.packages[cmd.package_id]
    new_world = _with_entity(
        world, "packages", cmd.package_id,
        replace(pkg, at=HeldByArm(drone_id=cmd.drone_id, arm_id=cmd.arm_id)),
    )

    new_metrics = replace(metrics, action_count=metrics.action_count + 1)
    return new_world, new_metrics


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def apply(
    world: World, metrics: MetricsTracker, cmd: Command
) -> tuple[World, MetricsTracker]:
    """Dispatcha un Command a su handler concreto y devuelve la transición.

    Asume Command pre-validado. Si el Command no corresponde con el estado
    del World (p.ej. drone_id no existe), levantará KeyError u otra
    excepción de la stdlib; el runtime garantiza la pre-validación.

    Raises:
        TypeError: si cmd no es uno de los seis tipos conocidos. Branch
            defensivo análogo al de validate().
    """
    match cmd:
        case Move():                   return apply_move(world, metrics, cmd)
        case MoveWithTransporter():    return apply_move_with_transporter(world, metrics, cmd)
        case PickUp():                 return apply_pick_up(world, metrics, cmd)
        case Deliver():                return apply_deliver(world, metrics, cmd)
        case LoadIntoTransporter():    return apply_load_into_transporter(world, metrics, cmd)
        case UnloadFromTransporter():  return apply_unload_from_transporter(world, metrics, cmd)
        case _:
            raise TypeError(
                f"apply() recibió un objeto que no es un Command "
                f"conocido: {type(cmd).__name__}"
            )