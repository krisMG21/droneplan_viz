"""Tests de history/snapshot.py: WorldSnapshot y test-canario.

El test-canario (TestSnapshotCanario) verifica las garantías de
inmutabilidad estructural establecidas en sesión A. Si alguno de esos
asserts se rompe en una sesión futura (sesión D o E), eso indica que
la asunción sobre la que se construyó la decisión "referencia directa
sin deepcopy" ya no se sostiene, y hay que meter deepcopy en
WorldSnapshot. El docstring del test es el documento operativo para
ese día.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from droneplan_viz.commands.movement import Move
from droneplan_viz.domain import (
    Arm,
    AtLocation,
    Content,
    Drone,
    DroneState,
    Location,
    MetricsTracker,
    Package,
    World,
)
from droneplan_viz.history.snapshot import WorldSnapshot


@pytest.fixture
def world() -> World:
    med = Content(id="medicina")
    return World(
        locations={"deposito": Location(id="deposito")},
        drones={
            "d1": Drone(
                id="d1", position="deposito",
                arms=(Arm(id="izq"),), state=DroneState.IDLE,
            )
        },
        packages={
            "c1": Package(id="c1", contains=med, at=AtLocation(loc_id="deposito"))
        },
        contents={"medicina": med},
    )


@pytest.fixture
def metrics() -> MetricsTracker:
    return MetricsTracker()


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------
class TestSnapshotConstruccion:
    def test_construccion_completa(self, world, metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        snap = WorldSnapshot(
            world=world, metrics=metrics, produced_by=cmd, timestamp=12.5
        )
        assert snap.world is world
        assert snap.metrics is metrics
        assert snap.produced_by is cmd
        assert snap.timestamp == 12.5

    def test_produced_by_none_es_legal(self, world, metrics):
        """El snapshot inicial no fue producido por ningún Command."""
        snap = WorldSnapshot(world=world, metrics=metrics)
        assert snap.produced_by is None

    def test_timestamp_default_cero(self, world, metrics):
        snap = WorldSnapshot(world=world, metrics=metrics)
        assert snap.timestamp == 0.0


# ---------------------------------------------------------------------------
# Inmutabilidad del propio Snapshot
# ---------------------------------------------------------------------------
class TestSnapshotInmutabilidad:
    def test_no_se_pueden_reasignar_campos(self, world, metrics):
        snap = WorldSnapshot(world=world, metrics=metrics)
        with pytest.raises(FrozenInstanceError):
            snap.timestamp = 99.0  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos(self, world, metrics):
        snap = WorldSnapshot(world=world, metrics=metrics)
        with pytest.raises((AttributeError, TypeError)):
            snap.extra = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Referencia directa: el snapshot apunta al MISMO objeto World
# ---------------------------------------------------------------------------
class TestSnapshotReferenciaDirecta:
    """Documentación operativa: WorldSnapshot guarda referencia, no copia.

    Esta propiedad es la base del rendimiento O(1) por snapshot. Si en
    alguna sesión futura se decide pasar a deepcopy (por ejemplo porque
    el test-canario de la siguiente sección se rompa), este test
    también habrá que cambiarlo a `is not` y reemplazar la justificación.
    """

    def test_snapshot_world_es_el_mismo_objeto(self, world, metrics):
        snap = WorldSnapshot(world=world, metrics=metrics)
        assert snap.world is world

    def test_snapshot_metrics_es_el_mismo_objeto(self, world, metrics):
        snap = WorldSnapshot(world=world, metrics=metrics)
        assert snap.metrics is metrics


# ---------------------------------------------------------------------------
# TEST-CANARIO: garantías de inmutabilidad de sesión A
# ---------------------------------------------------------------------------
class TestSnapshotCanario:
    """Test-canario: vigila las garantías de inmutabilidad estructural
    establecidas en sesión A, sobre las que descansa la decisión de
    NO hacer deepcopy en WorldSnapshot.

    Si CUALQUIERA de los asserts de esta sección se rompe en una sesión
    futura, eso indica que la propiedad ha dejado de ser cierta y los
    snapshots ya no son seguros sin deepcopy. La acción correctiva es
    localizada: editar history/snapshot.py para envolver World en
    copy.deepcopy() (o, si la regresión es solo de un componente, en
    el constructor de ese componente).

    Las propiedades vigiladas:

    1. World es frozen: no se pueden reasignar sus campos.
    2. Los dicts internos del World son MappingProxyType: no permiten
       __setitem__ ni __delitem__.
    3. Las entidades dentro del World son frozen: no se pueden reasignar
       sus campos.
    4. Las tuplas internas de las entidades son tuple, no list: no
       permiten append/extend/__setitem__.
    5. MetricsTracker es frozen.
    """

    def test_world_no_permite_reasignar_campos(self, world):
        with pytest.raises(FrozenInstanceError):
            world.metric = "hack"  # type: ignore[misc]

    def test_world_drones_es_mappingproxytype(self, world):
        assert isinstance(world.drones, MappingProxyType)

    def test_world_drones_no_permite_setitem(self, world):
        with pytest.raises(TypeError):
            world.drones["nuevo"] = world.drones["d1"]  # type: ignore[index]

    def test_world_drones_no_permite_delitem(self, world):
        with pytest.raises(TypeError):
            del world.drones["d1"]  # type: ignore[attr-defined]

    def test_world_packages_es_mappingproxytype(self, world):
        assert isinstance(world.packages, MappingProxyType)

    def test_world_locations_es_mappingproxytype(self, world):
        assert isinstance(world.locations, MappingProxyType)

    def test_world_persons_es_mappingproxytype(self, world):
        assert isinstance(world.persons, MappingProxyType)

    def test_world_transporters_es_mappingproxytype(self, world):
        assert isinstance(world.transporters, MappingProxyType)

    def test_world_contents_es_mappingproxytype(self, world):
        assert isinstance(world.contents, MappingProxyType)

    def test_world_costs_es_mappingproxytype(self, world):
        assert isinstance(world.costs, MappingProxyType)

    def test_drone_no_permite_reasignar_campos(self, world):
        drone = world.drones["d1"]
        with pytest.raises(FrozenInstanceError):
            drone.position = "otra"  # type: ignore[misc]

    def test_drone_arms_es_tuple(self, world):
        assert isinstance(world.drones["d1"].arms, tuple)

    def test_package_no_permite_reasignar_campos(self, world):
        pkg = world.packages["c1"]
        with pytest.raises(FrozenInstanceError):
            pkg.contains = Content(id="otra")  # type: ignore[misc]

    def test_metrics_no_permite_reasignar_campos(self, metrics):
        with pytest.raises(FrozenInstanceError):
            metrics.action_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Igualdad de snapshots
# ---------------------------------------------------------------------------
class TestSnapshotIgualdad:
    """La igualdad de snapshots es la de dataclass: campo a campo.

    Dos snapshots con mismos world, mismos metrics, mismo produced_by,
    mismo timestamp son iguales. Esto es útil para tests pero no se
    explota en el sistema en producción (los snapshots se identifican
    por su posición en la línea temporal del HistoryManager).
    """

    def test_dos_snapshots_con_mismos_campos_son_iguales(self, world, metrics):
        a = WorldSnapshot(world=world, metrics=metrics, timestamp=1.0)
        b = WorldSnapshot(world=world, metrics=metrics, timestamp=1.0)
        assert a == b

    def test_distintos_si_timestamp_distinto(self, world, metrics):
        a = WorldSnapshot(world=world, metrics=metrics, timestamp=1.0)
        b = WorldSnapshot(world=world, metrics=metrics, timestamp=2.0)
        assert a != b
