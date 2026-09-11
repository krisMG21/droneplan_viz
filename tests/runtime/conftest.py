"""
Fixtures y helpers compartidos para los tests de runtime/.

Construye un World más rico que el `base_world` de tests/commands/:

- Dos drones (d1, d2), cada uno con dos brazos (izq, der). Permite
  tests de concurrencia de Sesión C que requieren multi-dron.
- Dos transportadores (t1, t2). Permite tests de regla "transportador
  exclusivo" sin tener que reconstruir worlds ad-hoc.
- Tres localizaciones (deposito, casa1, casa2) con costes simétricos
  entre todas. Costes no triviales (≠ 1.0) para detectar errores
  aritméticos en métricas.
- Dos personas (ana, bob), una por casa, con necesidades concretas.
- Tres paquetes en estados diversos (libre, sostenido, en transportador)
  para que la mayoría de las acciones sean ejecutables sin gimnasia
  preparatoria.

Además ofrece:

- Factorías de Commands con ids estables (make_move, make_pickup, etc.)
  para que los tests no se llenen de declaraciones largas.
- Helpers de aserción comunes.

Decisión: este conftest.py vive en tests/runtime/, NO en tests/. No
contamina los tests de otros paquetes. Sesión A y Sesión B tienen sus
propios conftests (o ninguno) y siguen funcionando exactamente igual.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
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


# ===========================================================================
# Contenidos (compartidos por entidades)
# ===========================================================================
MEDICINA = Content(id="medicina")
COMIDA = Content(id="comida")


# ===========================================================================
# Fixtures de World
# ===========================================================================
@pytest.fixture
def base_world() -> World:
    """World rico con dos drones, dos transportadores, tres localizaciones.

    Topología:
        deposito <--10--> casa1
        deposito <--20--> casa2
        casa1    <--15--> casa2
    (costes simétricos en ambas direcciones)

    Estado inicial:
        d1 en deposito, brazos izq+der vacíos
        d2 en deposito, brazos izq+der vacíos
        t1 en deposito, capacidad 4
        t2 en deposito, capacidad 4
        libre1 (medicina) en deposito
        libre2 (comida) en deposito
        libre3 (medicina) en casa1
        ana en casa1, necesita medicina
        bob en casa2, necesita comida y medicina

    Todos los drones empiezan en IDLE.
    """
    locations = {
        "deposito": Location(id="deposito", position_screen=(0, 0)),
        "casa1": Location(id="casa1", position_screen=(100, 0)),
        "casa2": Location(id="casa2", position_screen=(0, 100)),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="deposito",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
        "d2": Drone(
            id="d2",
            position="deposito",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
        "t2": Transporter(id="t2", position="deposito", capacity=4),
    }
    packages = {
        "libre1": Package(
            id="libre1", contains=MEDICINA,
            at=AtLocation(loc_id="deposito"),
        ),
        "libre2": Package(
            id="libre2", contains=COMIDA,
            at=AtLocation(loc_id="deposito"),
        ),
        "libre3": Package(
            id="libre3", contains=MEDICINA,
            at=AtLocation(loc_id="casa1"),
        ),
    }
    persons = {
        "ana": Person(
            id="ana", position="casa1",
            needs=(MEDICINA,), has_received=(),
        ),
        "bob": Person(
            id="bob", position="casa2",
            needs=(COMIDA, MEDICINA), has_received=(),
        ),
    }
    contents = {"medicina": MEDICINA, "comida": COMIDA}
    costs = {
        ("deposito", "casa1"): 10.0,
        ("casa1", "deposito"): 10.0,
        ("deposito", "casa2"): 20.0,
        ("casa2", "deposito"): 20.0,
        ("casa1", "casa2"): 15.0,
        ("casa2", "casa1"): 15.0,
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


# ===========================================================================
# Helpers de transformación de World (útiles en setup de tests)
# ===========================================================================
def world_with_drone_at(world: World, drone_id: str, loc_id: str) -> World:
    """Devuelve un World derivado con el drone reposicionado."""
    new_drones = dict(world.drones)
    new_drones[drone_id] = replace(world.drones[drone_id], position=loc_id)
    return replace(world, drones=new_drones)


def world_with_drone_in_error(world: World, drone_id: str) -> World:
    """Devuelve un World derivado con el drone forzado a ERROR.

    Útil para tests que verifican que el runtime respeta el sumidero
    forward heredado de Sesión A.
    """
    new_drones = dict(world.drones)
    new_drones[drone_id] = replace(
        world.drones[drone_id], state=DroneState.ERROR
    )
    return replace(world, drones=new_drones)


def world_with_package_held(
    world: World,
    package_id: str,
    drone_id: str,
    arm_id: str,
) -> World:
    """Devuelve un World derivado con un paquete pasado a HeldByArm."""
    new_packages = dict(world.packages)
    new_packages[package_id] = replace(
        world.packages[package_id],
        at=HeldByArm(drone_id=drone_id, arm_id=arm_id),
    )
    return replace(world, packages=new_packages)


def world_with_package_in_transporter(
    world: World, package_id: str, transporter_id: str
) -> World:
    """Devuelve un World derivado con un paquete dentro de un transportador."""
    new_packages = dict(world.packages)
    new_packages[package_id] = replace(
        world.packages[package_id],
        at=InTransporter(transporter_id=transporter_id),
    )
    return replace(world, packages=new_packages)


# ===========================================================================
# Factorías de Commands con ids legibles
#
# Cada factoría acepta los parámetros semánticos del Command + duration
# opcional + cmd_id opcional. Si no se da cmd_id, se deriva del tipo y
# de un sufijo automático para mantener unicidad (los tests pueden
# pasar cmd_id explícito cuando quieran trazabilidad explícita en los
# mensajes de error).
# ===========================================================================
def make_move(
    drone_id: str = "d1",
    destination_id: str = "casa1",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> Move:
    """Construye un Move con id estable si se da cmd_id."""
    if cmd_id is not None:
        return Move(
            drone_id=drone_id, destination_id=destination_id,
            duration=duration, command_id=cmd_id,
        )
    return Move(
        drone_id=drone_id, destination_id=destination_id,
        duration=duration,
    )


def make_move_with_transporter(
    drone_id: str = "d1",
    transporter_id: str = "t1",
    destination_id: str = "casa1",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> MoveWithTransporter:
    if cmd_id is not None:
        return MoveWithTransporter(
            drone_id=drone_id, transporter_id=transporter_id,
            destination_id=destination_id, duration=duration,
            command_id=cmd_id,
        )
    return MoveWithTransporter(
        drone_id=drone_id, transporter_id=transporter_id,
        destination_id=destination_id, duration=duration,
    )


def make_pickup(
    drone_id: str = "d1",
    arm_id: str = "izq",
    package_id: str = "libre1",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> PickUp:
    if cmd_id is not None:
        return PickUp(
            drone_id=drone_id, arm_id=arm_id,
            package_id=package_id, duration=duration,
            command_id=cmd_id,
        )
    return PickUp(
        drone_id=drone_id, arm_id=arm_id,
        package_id=package_id, duration=duration,
    )


def make_deliver(
    drone_id: str = "d1",
    package_id: str = "libre1",
    person_id: str = "ana",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> Deliver:
    if cmd_id is not None:
        return Deliver(
            drone_id=drone_id, package_id=package_id,
            person_id=person_id, duration=duration,
            command_id=cmd_id,
        )
    return Deliver(
        drone_id=drone_id, package_id=package_id,
        person_id=person_id, duration=duration,
    )


def make_load(
    drone_id: str = "d1",
    package_id: str = "libre1",
    transporter_id: str = "t1",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> LoadIntoTransporter:
    if cmd_id is not None:
        return LoadIntoTransporter(
            drone_id=drone_id, package_id=package_id,
            transporter_id=transporter_id, duration=duration,
            command_id=cmd_id,
        )
    return LoadIntoTransporter(
        drone_id=drone_id, package_id=package_id,
        transporter_id=transporter_id, duration=duration,
    )


def make_unload(
    drone_id: str = "d1",
    arm_id: str = "izq",
    package_id: str = "libre1",
    transporter_id: str = "t1",
    duration: float = 0.0,
    cmd_id: str | None = None,
) -> UnloadFromTransporter:
    if cmd_id is not None:
        return UnloadFromTransporter(
            drone_id=drone_id, arm_id=arm_id,
            package_id=package_id, transporter_id=transporter_id,
            duration=duration, command_id=cmd_id,
        )
    return UnloadFromTransporter(
        drone_id=drone_id, arm_id=arm_id,
        package_id=package_id, transporter_id=transporter_id,
        duration=duration,
    )
