"""Paquete runtime: ejecutor de planes sobre el dominio.

Dado un World inicial y un Plan (secuencia de Commands con timestamps),
produce un historial de WorldSnapshots aplicando los Commands en el orden
correcto y respetando las restricciones de concurrencia del PDDL parte 3.

Estado actual: en construcción incremental. Expuesto por ahora:
- Plan, ScheduledCommand, PlanValidationError: modelado del plan.
- RunResult, CommandFailure, FailureKind: tipos de salida del runner
  (PlanRunner aún por implementar en pasos posteriores).
"""

from droneplan_viz.runtime.plan import (
    Plan,
    PlanValidationError,
    ScheduledCommand,
)
from droneplan_viz.runtime.result import (
    CommandFailure,
    FailureKind,
    RunResult,
)
from droneplan_viz.runtime.runner import PlanRunner

__all__ = [
    # Plan input types
    "Plan",
    "PlanValidationError",
    "ScheduledCommand",
    # Result output types
    "CommandFailure",
    "FailureKind",
    "RunResult",
    # Runner
    "PlanRunner",
]
