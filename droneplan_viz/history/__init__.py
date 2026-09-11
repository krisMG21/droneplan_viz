"""Paquete history: patrón Memento para navegación bidireccional del plan.

Exporta los tres componentes públicos:

- WorldSnapshot: foto inmutable de un estado en la línea temporal.
- HistoryManager: almacén lineal de snapshots con soporte de truncado.
- TimelineCursor: posición de lectura sobre un HistoryManager, con
  navegación (back/forward/goto/jumps) y commit_new() que encapsula la
  semántica Ctrl+Z.

Uso típico desde el runtime:

    from droneplan_viz.history import (
        WorldSnapshot, HistoryManager, TimelineCursor,
    )

    initial = WorldSnapshot(world=initial_world, metrics=MetricsTracker())
    history = HistoryManager(initial=initial)
    cursor = TimelineCursor(history)

    # Aplicar un Command produce un (World', MetricsTracker') que se
    # encapsula en un nuevo WorldSnapshot y se commit_new() en el cursor.
    new_world, new_metrics = apply(cursor.current.world,
                                   cursor.current.metrics,
                                   cmd)
    cursor.commit_new(
        WorldSnapshot(world=new_world, metrics=new_metrics,
                      produced_by=cmd, timestamp=now)
    )
"""

from droneplan_viz.history.cursor import TimelineCursor
from droneplan_viz.history.manager import HistoryManager
from droneplan_viz.history.snapshot import WorldSnapshot

__all__ = [
    "WorldSnapshot",
    "HistoryManager",
    "TimelineCursor",
]
