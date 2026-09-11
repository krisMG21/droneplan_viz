"""Núcleo lógico del dominio. Entidades, FSM, validador y métricas.

No importa nada de PyGame. Todo el subpaquete es testeable sin entorno gráfico.
"""

from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.content import Content
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.metrics import MetricsTracker
from droneplan_viz.domain.package import (
    AtLocation,
    HeldByArm,
    InTransporter,
    Package,
    PackageLocation,
)
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.domain.validator import (
    ValidationResult,
    validate_deliver,
    validate_load_into_transporter,
    validate_move,
    validate_move_with_transporter,
    validate_pick_up,
    validate_unload_from_transporter,
)
from droneplan_viz.domain.world import World

__all__ = [
    "Arm",
    "AtLocation",
    "Content",
    "Drone",
    "DroneState",
    "HeldByArm",
    "InTransporter",
    "Location",
    "MetricsTracker",
    "Package",
    "PackageLocation",
    "Person",
    "Transporter",
    "ValidationResult",
    "World",
    "validate_deliver",
    "validate_load_into_transporter",
    "validate_move",
    "validate_move_with_transporter",
    "validate_pick_up",
    "validate_unload_from_transporter",
]
