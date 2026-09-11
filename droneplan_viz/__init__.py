"""droneplan_viz: visualización y validación de planes PDDL para
logística humanitaria con drones.

Superficie pública principal: la fachada DronePlanViz, una API ergonómica
para construir el mundo y el plan y visualizarlos. El alumno traduce a mano
su dominio y su plan (generados con FF/OPTIC/LPG-TD) a llamadas de esta
fachada; no se parsea PDDL.

    from droneplan_viz import DronePlanViz

    viz = DronePlanViz()
    viz.world.location("deposito")
    ...
    viz.run()

Las capas internas (domain, commands, runtime, history, render) siguen
siendo importables por separado para uso avanzado o construcción directa de
World/Plan a mano, que continúa siendo API legítima de la librería.
"""
from droneplan_viz.facade import (
    DronePlanViz,
    DuplicateIdError,
    FacadeError,
    MixedTimingError,
    UnknownArmError,
    UnknownContentError,
    UnknownDroneError,
    UnknownLocationError,
    UnknownPackageError,
    UnknownPersonError,
    UnknownTransporterError,
    WrongReferenceKindError,
)

__all__ = [
    "DronePlanViz",
    "DuplicateIdError",
    "FacadeError",
    "MixedTimingError",
    "UnknownArmError",
    "UnknownContentError",
    "UnknownDroneError",
    "UnknownLocationError",
    "UnknownPackageError",
    "UnknownPersonError",
    "UnknownTransporterError",
    "WrongReferenceKindError",
]
