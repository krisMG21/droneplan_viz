"""Drone: agregado de componentes con estado FSM.

Composición sobre herencia (decisión 1 del diseño): hay una sola clase
Drone, configurable mediante sus componentes. Las variantes "drone con
dos brazos" o "drone explorador sin brazos" no son subclases; son
combinaciones distintas de los mismos componentes.

Inmutable como el resto del dominio. Las transiciones de la FSM y los
movimientos se hacen con dataclasses.replace, creando una nueva instancia
y sustituyéndola en el World. Esta es la base sobre la que se apoyan los
snapshots del Memento.

Decisiones de modelado importantes:

    - position es un loc_id (str), no una Location. Las relaciones
      se modelan por id; el lookup contra el World es responsabilidad
      del consumidor.
    - arms es una tupla, no una lista. Coherente con la inmutabilidad.
      Una tupla vacía es estructuralmente válida (drone explorador);
      las reglas del dominio que requieran brazos las aplica el Validator.
    - El Drone NO almacena qué paquete sostiene cada brazo. Esa relación
      vive exclusivamente en Package.at vía HeldByArm(drone_id, arm_id).
      Una sola fuente de verdad.
    - state arranca en IDLE por defecto, lo natural al construir un
      drone que aún no ha hecho nada.
    - La unicidad de arm.id dentro del drone NO se valida aquí. Es
      regla del dominio y la verifica el Validator (o los builders del
      facade al ensamblar el mundo).

"""

from dataclasses import dataclass, field

from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone_state import DroneState


@dataclass(frozen=True, slots=True)
class Drone:
    """Agente del dominio. Se mueve, recoge, entrega, transita estados.

    Atributos:
        id: identificador simbólico único en el mundo.
        position: id de la Location actual del drone.
        arms: tupla de brazos. Puede estar vacía (drone explorador). La
            unicidad de los arm.id dentro de la tupla la verifica el
            Validator, no este tipo.
        state: estado actual de la FSM. Por defecto IDLE.
    """

    id: str
    position: str
    arms: tuple[Arm, ...] = field(default=())
    state: DroneState = field(default=DroneState.IDLE)
