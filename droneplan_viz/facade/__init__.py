"""Paquete facade: la CARA VISIBLE de droneplan_viz.

API ergonómica que oculta la maquinaria de domain/commands/runtime/render
tras un objeto DronePlanViz con builders fluidos. Es la superficie pública
principal de la librería; el alumno traduce a mano su dominio y su plan
PDDL a llamadas de esta fachada.

Exporta DronePlanViz y la taxonomía de errores de construcción. Los
builders (WorldBuilder, AgentBuilder) son accesibles vía `viz.world` y
`viz.agents`; el resto del paquete (references, state) es interno.
"""
from droneplan_viz.facade.errors import (
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
from droneplan_viz.facade.facade import DronePlanViz

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
