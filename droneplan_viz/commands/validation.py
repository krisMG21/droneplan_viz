"""
Dispatch de validación: traduce un Command genérico a la función validate_X
correspondiente del Validador.

La función pública validate(world, cmd) es el único punto de entrada
para validar un Command. Internamente usa match exhaustivo sobre las seis
dataclasses concretas y delega en las funciones libres del paquete domain.

Decisiones de diseño:

- Match exhaustivo, no registry. Para seis acciones, un registry con
  decoradores o diccionarios sería overengineering. El match es exhaustivo
  por construcción: añadir una acción nueva implica un fallo de tipos en
  tiempo de comprobación si se olvida actualizar el dispatcher.

- El Validador del dominio NO se modifica. validate(world, cmd)
  es traducción pura entre la representación Command (dataclasses) y la
  representación interna del Validador (funciones libres con parámetros
  posicionales). El Validador no conoce qué es un Command.

- Las firmas de las validate_X son las consagradas. Las
  recordamos aquí porque el orden de parámetros varía entre funciones:
    validate_move(world, drone_id, to_loc_id)
    validate_move_with_transporter(world, drone_id, to_loc_id,
                                   transporter_id)
    validate_pick_up(world, drone_id, package_id, arm_id)
    validate_deliver(world, drone_id, package_id, person_id)
    validate_load_into_transporter(world, drone_id, package_id,
                                   transporter_id)
    validate_unload_from_transporter(world, drone_id, package_id,
                                     transporter_id, arm_id)

  Deliver y LoadIntoTransporter NO reciben arm_id por decisión heredada
  de el runtime: liberan el brazo que sea.

- El caso default del match (objeto que no es un Command conocido) lanza
  TypeError explícito. Defensa en profundidad.

- Tests aquí cubren EL DISPATCH, no las reglas del Validador. Las pruebas
  del Validador ya cubren la corrección de las reglas
  semánticas. Aquí solo verificamos que cada tipo de Command llega a su
  validate_X con los parámetros correctos.
"""
from __future__ import annotations

from droneplan_viz.commands.base import Command
from droneplan_viz.commands.manipulation import Deliver, PickUp
from droneplan_viz.commands.movement import Move, MoveWithTransporter
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter
from droneplan_viz.domain import (
    ValidationResult,
    World,
    validate_deliver,
    validate_load_into_transporter,
    validate_move,
    validate_move_with_transporter,
    validate_pick_up,
    validate_unload_from_transporter,
)


def validate(world: World, cmd: Command) -> ValidationResult:
    """Dispatcha un Command a la validate_X correspondiente del Validador.

    Args:
        world: estado actual del mundo contra el que se valida.
        cmd: Command a validar. Debe ser una de las seis dataclasses
            concretas del paquete commands.

    Returns:
        ValidationResult del Validador del dominio. ok=True si el comando
        es ejecutable en este World; ok=False con reason en caso contrario.

    Raises:
        TypeError: si cmd no es ninguno de los seis tipos conocidos. Este
            caso solo puede producirse si alguien introduce una clase nueva
            sin actualizar este dispatcher; en uso normal el sistema de
            tipos lo impide.
    """
    match cmd:
        case Move(drone_id=d, destination_id=dst):
            return validate_move(world, d, dst)
        case MoveWithTransporter(
            drone_id=d, transporter_id=t, destination_id=dst
        ):
            # Nota el orden: validate_move_with_transporter espera
            # (world, drone_id, to_loc_id, transporter_id), NO al revés.
            return validate_move_with_transporter(world, d, dst, t)
        case PickUp(drone_id=d, arm_id=a, package_id=p):
            # validate_pick_up: (world, drone_id, package_id, arm_id)
            return validate_pick_up(world, d, p, a)
        case Deliver(drone_id=d, package_id=p, person_id=pe):
            # validate_deliver: (world, drone_id, package_id, person_id).
            # NO recibe arm_id.
            return validate_deliver(world, d, p, pe)
        case LoadIntoTransporter(
            drone_id=d, package_id=p, transporter_id=t
        ):
            # validate_load_into_transporter:
            # (world, drone_id, package_id, transporter_id). NO recibe arm_id.
            return validate_load_into_transporter(world, d, p, t)
        case UnloadFromTransporter(
            drone_id=d, arm_id=a, package_id=p, transporter_id=t
        ):
            # validate_unload_from_transporter:
            # (world, drone_id, package_id, transporter_id, arm_id).
            return validate_unload_from_transporter(world, d, p, t, a)
        case _:
            raise TypeError(
                f"validate() recibió un objeto que no es un Command "
                f"conocido: {type(cmd).__name__}"
            )
