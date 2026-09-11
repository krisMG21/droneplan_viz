"""Paquete commands: patrón Command como objetos inmutables.

Exporta las seis dataclasses de acción, el Protocol estructural común,
la factory de identidades y los dos dispatchers (validación y aplicación
de efectos).

Las seis dataclasses concretas se construyen libremente:

    from droneplan_viz.commands import Move
    cmd = Move(drone_id="d1", destination_id="casa")

El Protocol se usa para tipados estructurales:

    from droneplan_viz.commands import Command
    def procesa(cmd: Command) -> None: ...

Los dispatchers son el único punto de entrada para validar y aplicar:

    from droneplan_viz.commands import validate, apply
    result = validate(world, cmd)
    if result:
        new_world, new_metrics = apply(world, metrics, cmd)
"""

from droneplan_viz.commands.base import Command, new_command_id
from droneplan_viz.commands.handlers import (
    apply,
    apply_deliver,
    apply_load_into_transporter,
    apply_move,
    apply_move_with_transporter,
    apply_pick_up,
    apply_unload_from_transporter,
)
from droneplan_viz.commands.manipulation import Deliver, PickUp
from droneplan_viz.commands.movement import Move, MoveWithTransporter
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter
from droneplan_viz.commands.validation import validate

__all__ = [
    # Protocol + factory
    "Command",
    "new_command_id",
    # Las seis dataclasses concretas
    "Move",
    "MoveWithTransporter",
    "PickUp",
    "Deliver",
    "LoadIntoTransporter",
    "UnloadFromTransporter",
    # Dispatchers
    "validate",
    "apply",
    # Handlers individuales (uso avanzado, p.ej. tests del runtime)
    "apply_move",
    "apply_move_with_transporter",
    "apply_pick_up",
    "apply_deliver",
    "apply_load_into_transporter",
    "apply_unload_from_transporter",
]
