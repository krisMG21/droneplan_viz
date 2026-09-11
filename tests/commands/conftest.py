"""
Fixtures y helpers compartidos para los tests de commands/.

Construye un World pequeño y consistente con dos localizaciones, un drone
con dos brazos, un transportador, un par de paquetes y una persona, todo
co-localizado de forma que cualquiera de las seis acciones del dominio sea
ejecutable cambiando algún detalle mínimo.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from droneplan_viz.domain import (
    Arm,
    AtLocation,
    Content,
    Drone,
    DroneState,
    HeldByArm,
    InTransporter,
    Location,
    MetricsTracker,
    Package,
    Person,
    Transporter,
    World,
)


@pytest.fixture
def base_world() -> World:
    """World con un drone (dos brazos) en 'deposito', un transportador,
    un paquete libre en 'deposito', un paquete sostenido por el brazo izq,
    un paquete dentro del transportador, una persona en 'casa1' que necesita
    medicina, y aristas en ambos sentidos entre deposito y casa1.
    """
    medicina = Content(id="medicina")
    comida = Content(id="comida")

    locations = {
        "deposito": Location(id="deposito", position_screen=(0, 0)),
        "casa1": Location(id="casa1", position_screen=(100, 0)),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="deposito",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        )
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }
    packages = {
        "libre1": Package(
            id="libre1", contains=medicina, at=AtLocation(loc_id="deposito")
        ),
        "sostenida": Package(
            id="sostenida", contains=comida,
            at=HeldByArm(drone_id="d1", arm_id="izq"),
        ),
        "en_trans": Package(
            id="en_trans", contains=medicina,
            at=InTransporter(transporter_id="t1"),
        ),
    }
    persons = {
        "ana": Person(
            id="ana", position="casa1",
            needs=(medicina,), has_received=(),
        ),
    }
    contents = {"medicina": medicina, "comida": comida}
    costs = {
        ("deposito", "casa1"): 10.0,
        ("casa1", "deposito"): 10.0,
    }
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        packages=packages,
        persons=persons,
        contents=contents,
        costs=costs,
        metric=None,
    )


@pytest.fixture
def initial_metrics() -> MetricsTracker:
    """MetricsTracker en estado inicial (todos los contadores a 0)."""
    return MetricsTracker()


def world_with_drone_in_error(world: World) -> World:
    """Helper: devuelve un World derivado donde el drone d1 está en ERROR."""
    new_drones = dict(world.drones)
    new_drones["d1"] = replace(world.drones["d1"], state=DroneState.ERROR)
    return replace(world, drones=new_drones)


def world_at(world: World, drone_id: str, location_id: str) -> World:
    """Helper: devuelve un World derivado con el drone en otra localización."""
    new_drones = dict(world.drones)
    new_drones[drone_id] = replace(world.drones[drone_id], position=location_id)
    return replace(world, drones=new_drones)
