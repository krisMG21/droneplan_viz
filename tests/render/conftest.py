"""Fixtures compartidas por los tests del paquete render.

Convención: cada fixture construye un World con una topología concreta
útil para un grupo de tests. Las fixtures son funciones puras: no usan
pygame, no mutan estado global.

Activamos SDL_VIDEODRIVER=dummy ANTES de cualquier `import pygame` para
que los tests que dibujen sobre Surface en memoria (sprites.py, painter.py)
funcionen en entornos sin servidor X (CI, sandbox de Claude, contenedores
docker headless). Hacemos esto en el conftest del paquete tests/render
porque pytest carga el conftest antes que cualquier import de los tests
del subdirectorio, garantizando que pygame se inicialice con el driver
dummy aunque algún test lo importe en sus globals.

Los tests del Paso 1 y 2 (theme, geometry, layout) NO tocan pygame, así
que esta línea es inocua para ellos.
"""
from __future__ import annotations

import os

# CRÍTICO: setdefault para no pisar la elección del usuario si quiere
# correr los tests con SDL real (raro, pero válido).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Sin sonido en sandbox: ahorra avisos de ALSA en stderr cuando pygame.init
# se llama desde tests.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

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


# ---------------------------------------------------------------------------
# Worlds básicos
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_world() -> World:
    """World sin ninguna entidad. Caso degenerado pero estructuralmente válido."""
    return World()


@pytest.fixture
def single_location_world() -> World:
    """World con exactamente una Location sin position_screen."""
    loc = Location(id="solo_aqui")
    return World(locations={loc.id: loc})


@pytest.fixture
def three_locations_circular() -> World:
    """Tres Locations sin position_screen: layout circular automático.

    Ids deliberadamente NO alfabéticos en el orden de inserción para
    verificar que el layout circular ordena por id, no por inserción.
    """
    locs = {
        "zorro": Location(id="zorro"),
        "casa1": Location(id="casa1"),
        "manzana": Location(id="manzana"),
    }
    return World(locations=locs)


@pytest.fixture
def three_locations_with_screen() -> World:
    """Tres Locations TODAS con position_screen explícita.

    Coords elegidas para formar un triángulo con bbox 100x100:
        a: (0, 0)
        b: (100, 0)
        c: (50, 100)
    """
    locs = {
        "a": Location(id="a", position_screen=(0, 0)),
        "b": Location(id="b", position_screen=(100, 0)),
        "c": Location(id="c", position_screen=(50, 100)),
    }
    return World(locations=locs)


@pytest.fixture
def mixed_screen_world() -> World:
    """Dos Locations con position_screen, una sin.

    Caso "alguna es None" → debe caer al layout circular automático
    (descartando los position_screen de las otras dos).
    """
    locs = {
        "con_coords_1": Location(id="con_coords_1", position_screen=(0, 0)),
        "con_coords_2": Location(id="con_coords_2", position_screen=(100, 100)),
        "sin_coords": Location(id="sin_coords"),  # position_screen = None
    }
    return World(locations=locs)


# ---------------------------------------------------------------------------
# World con entidades para tests de drone_position / transporter_position
# / person_position / package_position
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_world() -> World:
    """World con todas las clases de entidad presentes, escenario docente.

    Topología:
        Locations: casa1, casa2, deposito  (sin position_screen → circular)
        Drones: d1 (en deposito, IDLE, dos brazos), d2 (en casa1, IDLE)
        Transporter: t1 (en deposito, capacidad 4)
        Persons: p1 (en casa1, necesita medicina y comida),
                 p2 (en casa2, necesita agua)
        Contents: medicina, comida, agua
        Packages:
            pkg_med1 (medicina, AtLocation deposito)
            pkg_food1 (comida, AtLocation casa2)
            pkg_water1 (agua, AtLocation deposito)
    """
    contents = {
        "medicina": Content(id="medicina"),
        "comida": Content(id="comida"),
        "agua": Content(id="agua"),
    }

    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
        "deposito": Location(id="deposito"),
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
            position="casa1",
            arms=(Arm(id="izq"),),
            state=DroneState.IDLE,
        ),
    }

    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }

    persons = {
        "p1": Person(
            id="p1",
            position="casa1",
            needs=(contents["medicina"], contents["comida"]),
        ),
        "p2": Person(
            id="p2",
            position="casa2",
            needs=(contents["agua"],),
        ),
    }

    packages = {
        "pkg_med1": Package(
            id="pkg_med1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="deposito"),
        ),
        "pkg_food1": Package(
            id="pkg_food1",
            contains=contents["comida"],
            at=AtLocation(loc_id="casa2"),
        ),
        "pkg_water1": Package(
            id="pkg_water1",
            contains=contents["agua"],
            at=AtLocation(loc_id="deposito"),
        ),
    }

    return World(
        locations=locs,
        drones=drones,
        transporters=transporters,
        packages=packages,
        persons=persons,
        contents=contents,
    )


@pytest.fixture
def two_drones_same_location() -> World:
    """Dos drones EN LA MISMA Location. Tests de co-localización.

    Ambos en 'centro'. Tercera Location 'fuera' para que el layout
    circular tenga más de un nodo y el centro sea distinguible.
    """
    locs = {
        "centro": Location(id="centro"),
        "fuera": Location(id="fuera"),
    }
    drones = {
        "drone_a": Drone(id="drone_a", position="centro"),
        "drone_b": Drone(id="drone_b", position="centro"),
    }
    return World(locations=locs, drones=drones)


@pytest.fixture
def package_held_by_arm_world() -> World:
    """Un drone con un paquete en su brazo. Tests del error que lanza
    package_position cuando el paquete no está AtLocation.
    """
    contents = {"medicina": Content(id="medicina")}
    locs = {"casa1": Location(id="casa1")}
    drones = {
        "d1": Drone(id="d1", position="casa1", arms=(Arm(id="izq"),)),
    }
    packages = {
        "pkg1": Package(
            id="pkg1",
            contains=contents["medicina"],
            at=HeldByArm(drone_id="d1", arm_id="izq"),
        ),
    }
    return World(
        locations=locs,
        drones=drones,
        packages=packages,
        contents=contents,
    )


@pytest.fixture
def package_in_transporter_world() -> World:
    """Un paquete dentro de un transportador. Tests análogos al anterior."""
    contents = {"agua": Content(id="agua")}
    locs = {"deposito": Location(id="deposito")}
    transporters = {"t1": Transporter(id="t1", position="deposito", capacity=2)}
    packages = {
        "pkg1": Package(
            id="pkg1",
            contains=contents["agua"],
            at=InTransporter(transporter_id="t1"),
        ),
    }
    return World(
        locations=locs,
        transporters=transporters,
        packages=packages,
        contents=contents,
    )
