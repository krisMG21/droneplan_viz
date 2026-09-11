"""
Commands de movimiento: Move y MoveWithTransporter.

Estos Commands representan las dos acciones del dominio PDDL que cambian
la posición del drone:

- Move: volar de la localización actual a otra (acción 'volar' del PDDL).
- MoveWithTransporter: volar arrastrando un transportador
  (acción 'mover-transportador' del PDDL, introducida en parte 2).

Ambos Commands son dataclasses inmutables (frozen + slots) que transportan
exclusivamente datos. La lógica de validación vive en commands/validation.py
delegando al Validador del dominio, y la lógica de aplicación
de efectos vive en commands/handlers.py.

Decisión de diseño: MoveWithTransporter NO lleva arm_id. La función
validate_move_with_transporter del Validador verifica "al menos
un brazo libre" sin especificar cuál, en consonancia con la acción PDDL
'mover-transportador' que no recibe el brazo como parámetro. Mantener
simetría perfecta entre Command y validate_X es la regla del proyecto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from droneplan_viz.commands.base import new_command_id


@dataclass(frozen=True, slots=True)
class Move:
    """Acción 'volar': el drone se desplaza a otra localización.

    Mapea uno-a-uno con validate_move(world, drone_id, destination_id) del
    Validador.

    Atributos:
        drone_id: identificador del drone que vuela.
        destination_id: identificador de la localización destino.
        duration: duración semántica de la acción. 0.0 en partes 1-2 del
            PDDL (sin tiempo) o en réplica de plan sin tiempos; en parte 3
            (durativas) recibe el valor de fly-cost reinterpretado como
            duración. El renderer puede aplicar una duración visual aparte.
        command_id: identidad única, auto-generada por defecto.
    """
    drone_id: str
    destination_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)


@dataclass(frozen=True, slots=True)
class MoveWithTransporter:
    """Acción 'mover-transportador': el drone vuela arrastrando un transportador.

    Mapea uno-a-uno con validate_move_with_transporter(world, drone_id,
    transporter_id, destination_id) del Validador.

    Atributos:
        drone_id: identificador del drone que vuela.
        transporter_id: identificador del transportador que arrastra.
        destination_id: identificador de la localización destino.
        duration: ver Move.duration.
        command_id: identidad única, auto-generada por defecto.

    Nota: no lleva arm_id. El Validador exige "al menos un brazo libre"
    en el drone pero no fija cuál; reflejamos esa misma indeterminación
    en el Command. Si en sesiones futuras el dominio decide modelar
    explícitamente qué brazo arrastra, este campo se añadirá aquí y al
    Validador en paralelo.
    """
    drone_id: str
    transporter_id: str
    destination_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)
