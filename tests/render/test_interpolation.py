"""Tests para droneplan_viz.render.interpolation.

Todos sin pygame. Estrategia: para no construir snapshots a mano
(frágil), ejecutamos planes con PlanRunner y clasificamos las parejas
de snapshots resultantes. Esto verifica también la integración
implícita con runtime/.

Cobertura por categoría:

- TransitionStatic: snapshot inicial, snap_a == snap_b, snap_a y snap_b
  de Commands distintos.
- TransitionDroneMove: Move y MoveWithTransporter (con y sin
  transporter).
- TransitionPackageMove: las cuatro acciones de manipulación
  (PickUp, Deliver, LoadIntoTransporter, UnloadFromTransporter).
- TransitionFailure: drone en ERROR tras fallo PDDL.
- Errores: Command desconocido lanza TypeError; inconsistencias
  semánticas lanzan ValueError.
"""
import dataclasses
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
    AtLocation,
    Content,
    HeldByArm,
    InTransporter,
    MetricsTracker,
    Package,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.history import WorldSnapshot
from droneplan_viz.render.interpolation import (
    PackagePlace,
    TransitionDroneMove,
    TransitionFailure,
    TransitionPackageMove,
    TransitionStatic,
    classify_transition,
)
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand


# ---------------------------------------------------------------------------
# Helpers y fixtures locales
# ---------------------------------------------------------------------------


def make_world_for_pickup_and_deliver() -> World:
    """World con d1 en casa1 (con paquete en suelo) y p1 que necesita medicina.

    Topología:
        Locations: casa1, casa2.
        Drone d1 en casa1, dos brazos (izq, der).
        Person p1 en casa1, necesita medicina.
        Package pkg_med1 (medicina) AtLocation casa1.
        Cost entre las locs definido para que Move sea válido.
    """
    contents = {"medicina": Content(id="medicina")}
    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="casa1",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
    }
    persons = {
        "p1": Person(
            id="p1",
            position="casa1",
            needs=(contents["medicina"],),
        ),
    }
    packages = {
        "pkg_med1": Package(
            id="pkg_med1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="casa1"),
        ),
    }
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(
        locations=locs,
        drones=drones,
        persons=persons,
        packages=packages,
        contents=contents,
        costs=costs,
    )


def make_world_with_transporter() -> World:
    """World con d1, transp t1 ambos en deposito, paquete en suelo y persona en casa1."""
    contents = {"comida": Content(id="comida")}
    locs = {
        "deposito": Location(id="deposito"),
        "casa1": Location(id="casa1"),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="deposito",
            arms=(Arm(id="izq"),),
            state=DroneState.IDLE,
        ),
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }
    packages = {
        "pkg_food1": Package(
            id="pkg_food1",
            contains=contents["comida"],
            at=AtLocation(loc_id="deposito"),
        ),
    }
    costs = {
        ("deposito", "casa1"): 10.0,
        ("casa1", "deposito"): 10.0,
    }
    return World(
        locations=locs,
        drones=drones,
        transporters=transporters,
        packages=packages,
        contents=contents,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# TransitionStatic
# ---------------------------------------------------------------------------


class TestTransitionStatic:
    """Casos sin animación: inicial, mismo snap, Commands distintos."""

    def test_snapshot_inicial_es_static(self):
        # Snapshot inicial: produced_by is None.
        world = make_world_for_pickup_and_deliver()
        snap = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=None,
            timestamp=0.0,
        )
        result = classify_transition(snap, snap)
        assert isinstance(result, TransitionStatic)

    def test_snap_a_igual_snap_b_es_static_aunque_tenga_command(self):
        # Caso degenerado: pasar el mismo snap dos veces.
        world = make_world_for_pickup_and_deliver()
        cmd = Move(drone_id="d1", destination_id="casa2", duration=5.0)
        snap = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=cmd,
            timestamp=5.0,
        )
        result = classify_transition(snap, snap)
        assert isinstance(result, TransitionStatic)

    def test_snap_start_durativo_es_static_por_fsm(self):
        # El discriminador del "no hay geometría que animar todavía" es la
        # FSM del drone en snap_b, NO la comparación de command_id (esa
        # regla rompía la concurrencia). Un snap_START real tiene al drone
        # protagonista en MOVING/INTERACTING: aún no ha cambiado geometría,
        # así que el tramo es Static aunque snap_a venga de OTRO Command.
        world = make_world_for_pickup_and_deliver()
        moving_d1 = dataclasses.replace(
            world.drones["d1"], state=DroneState.MOVING
        )
        world_moving = dataclasses.replace(
            world, drones={**world.drones, "d1": moving_d1}
        )
        cmd_a = Move(drone_id="d1", destination_id="casa2", duration=5.0, command_id="step1")
        cmd_b = Move(drone_id="d1", destination_id="casa1", duration=5.0, command_id="step2")
        # snap_a: end de step1 (drone IDLE). snap_b: start de step2 (MOVING).
        snap_a = WorldSnapshot(
            world=world, metrics=MetricsTracker(), produced_by=cmd_a, timestamp=5.0
        )
        snap_b = WorldSnapshot(
            world=world_moving, metrics=MetricsTracker(), produced_by=cmd_b, timestamp=5.0
        )
        result = classify_transition(snap_a, snap_b)
        assert isinstance(result, TransitionStatic)

    def test_inicial_a_snap_start_de_pickup_durativo_es_static(self):
        # Regresión del bug reportado por Sesión E (UI envolvente): el
        # tramo "snap inicial → snap_start del primer Command durativo"
        # debe clasificarse como Static.
        #
        # Razón semántica: entre ambos snapshots solo cambió drone.state
        # (IDLE → INTERACTING por el efecto at-start del PickUp); el
        # paquete sigue AtLocation y el drone sigue en su loc de origen.
        # La animación REAL del PickUp ocurre en el siguiente tramo
        # (snap_start → snap_end). Si esto se clasificara como
        # PackageMove, el painter intentaría animar el paquete dos
        # veces: una en el primer tramo (incorrecto, 0.6s por fallback)
        # y otra en el segundo (correcto, 5s de duración real).
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(
                drone_id="d1", arm_id="izq",
                package_id="pkg_med1", duration=5.0,
            ),
        ])
        runner.execute(plan)
        # Snapshots: [inicial, snap_start, snap_end] = 3.
        assert len(runner.history) == 3

        snap_initial = runner.history.at(0)
        snap_start = runner.history.at(1)
        # snap_start debe tener al drone en INTERACTING:
        assert snap_start.world.drones["d1"].state == DroneState.INTERACTING
        # El paquete sigue AtLocation en snap_start:
        from droneplan_viz.domain import AtLocation
        assert isinstance(
            snap_start.world.packages["pkg_med1"].at, AtLocation
        )

        result = classify_transition(snap_initial, snap_start)
        assert isinstance(result, TransitionStatic), (
            f"Se esperaba TransitionStatic, se obtuvo {type(result).__name__}. "
            f"Sesión E reportaba este bug: la animación del primer "
            f"Command durativo se ejecutaba dos veces."
        )

    def test_inicial_a_snap_start_de_move_durativo_es_static(self):
        # Análogo del anterior pero para Move (drone.state → MOVING).
        # Confirma que el fix funciona para los dos estados FSM no
        # terminales (MOVING e INTERACTING).
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=8.0),
        ])
        runner.execute(plan)
        assert len(runner.history) == 3

        snap_initial = runner.history.at(0)
        snap_start = runner.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.MOVING
        # El drone sigue en su loc de origen en snap_start:
        assert snap_start.world.drones["d1"].position == "casa1"

        result = classify_transition(snap_initial, snap_start)
        assert isinstance(result, TransitionStatic)

    def test_snap_start_a_snap_end_de_pickup_durativo_sigue_siendo_package_move(self):
        # Cara complementaria del test anterior: tras detectar el snap_start
        # como Static, el tramo siguiente (snap_start → snap_end) debe
        # seguir clasificándose como PackageMove. Confirma que el fix NO
        # rompe la animación real del Command durativo.
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(
                drone_id="d1", arm_id="izq",
                package_id="pkg_med1", duration=5.0,
            ),
        ])
        runner.execute(plan)

        snap_start = runner.history.at(1)
        snap_end = runner.history.at(2)
        result = classify_transition(snap_start, snap_end)
        assert isinstance(result, TransitionPackageMove)
        assert result.from_place.kind == "loc"
        assert result.to_place.kind == "arm"


# ---------------------------------------------------------------------------
# TransitionDroneMove: Move y MoveWithTransporter
# ---------------------------------------------------------------------------


class TestTransitionDroneMove:
    """Move y MoveWithTransporter producen TransitionDroneMove."""

    def test_move_simple(self):
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ])
        runner.execute(plan)
        # Snapshots: [inicial, start, end]
        assert len(runner.history) == 3
        snap_start = runner.history.at(1)
        snap_end = runner.history.at(2)
        result = classify_transition(snap_start, snap_end)

        assert isinstance(result, TransitionDroneMove)
        assert result.drone_id == "d1"
        assert result.from_loc_id == "casa1"
        assert result.to_loc_id == "casa2"
        assert result.transporter_id is None

    def test_move_con_duration_cero(self):
        # En modo secuencial (duration=0), solo hay un snapshot final
        # (no se emite snap_start, decisión 2.6 de Memoria C).
        # Inicial → end del Move.
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            Move(drone_id="d1", destination_id="casa2"),  # duration=0
        ])
        runner.execute(plan)
        assert len(runner.history) == 2
        snap_initial = runner.history.at(0)
        snap_end = runner.history.at(1)
        result = classify_transition(snap_initial, snap_end)

        assert isinstance(result, TransitionDroneMove)
        assert result.from_loc_id == "casa1"
        assert result.to_loc_id == "casa2"

    def test_move_with_transporter(self):
        world = make_world_with_transporter()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            MoveWithTransporter(
                drone_id="d1",
                transporter_id="t1",
                destination_id="casa1",
                duration=10.0,
            ),
        ])
        runner.execute(plan)
        snap_start = runner.history.at(1)
        snap_end = runner.history.at(2)
        result = classify_transition(snap_start, snap_end)

        assert isinstance(result, TransitionDroneMove)
        assert result.drone_id == "d1"
        assert result.from_loc_id == "deposito"
        assert result.to_loc_id == "casa1"
        assert result.transporter_id == "t1"


# ---------------------------------------------------------------------------
# TransitionPackageMove: PickUp, Deliver, Load, Unload
# ---------------------------------------------------------------------------


class TestTransitionPackageMove:
    """Los cuatro Commands de manipulación producen TransitionPackageMove."""

    def test_pickup(self):
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1", duration=5.0),
        ])
        runner.execute(plan)
        snap_start = runner.history.at(1)
        snap_end = runner.history.at(2)
        result = classify_transition(snap_start, snap_end)

        assert isinstance(result, TransitionPackageMove)
        assert result.drone_id == "d1"
        assert result.package_id == "pkg_med1"
        assert result.from_place == PackagePlace(kind="loc", loc_id="casa1")
        assert result.to_place == PackagePlace(
            kind="arm", drone_id="d1", arm_id="izq"
        )
        assert result.person_id is None

    def test_deliver(self):
        # Plan: PickUp luego Deliver.
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1", duration=5.0),
                start_time=0.0,
            ),
            ScheduledCommand(
                command=Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1", duration=5.0),
                start_time=5.0,
            ),
        ))
        runner.execute(plan)
        # Buscamos el par (start, end) del Deliver: hay que recorrer
        # los snapshots hasta encontrar el deliver.
        snap_start = None
        snap_end = None
        for i in range(len(runner.history)):
            snap = runner.history.at(i)
            if isinstance(snap.produced_by, Deliver):
                if snap_start is None:
                    snap_start = snap
                else:
                    snap_end = snap
                    break
        assert snap_start is not None and snap_end is not None

        result = classify_transition(snap_start, snap_end)
        assert isinstance(result, TransitionPackageMove)
        assert result.drone_id == "d1"
        assert result.package_id == "pkg_med1"
        # from_place debe ser el brazo izq de d1 (en snap_start, el
        # paquete está HeldByArm).
        assert result.from_place == PackagePlace(
            kind="arm", drone_id="d1", arm_id="izq"
        )
        # to_place debe ser la loc de p1 (casa1).
        assert result.to_place == PackagePlace(kind="loc", loc_id="casa1")
        assert result.person_id == "p1"

    def test_load_into_transporter(self):
        # Plan: PickUp seguido de Load.
        world = make_world_with_transporter()
        runner = PlanRunner(world)
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pkg_food1", duration=5.0),
                start_time=0.0,
            ),
            ScheduledCommand(
                command=LoadIntoTransporter(
                    drone_id="d1",
                    package_id="pkg_food1",
                    transporter_id="t1",
                    duration=5.0,
                ),
                start_time=5.0,
            ),
        ))
        runner.execute(plan)

        # Buscar par del Load.
        snap_start = None
        snap_end = None
        for i in range(len(runner.history)):
            snap = runner.history.at(i)
            if isinstance(snap.produced_by, LoadIntoTransporter):
                if snap_start is None:
                    snap_start = snap
                else:
                    snap_end = snap
                    break
        assert snap_start is not None and snap_end is not None

        result = classify_transition(snap_start, snap_end)
        assert isinstance(result, TransitionPackageMove)
        assert result.drone_id == "d1"
        assert result.package_id == "pkg_food1"
        assert result.from_place == PackagePlace(
            kind="arm", drone_id="d1", arm_id="izq"
        )
        assert result.to_place == PackagePlace(
            kind="transp", transporter_id="t1"
        )
        assert result.person_id is None

    def test_unload_from_transporter(self):
        # World inicial con el paquete ya InTransporter para simplificar.
        contents = {"agua": Content(id="agua")}
        locs = {"deposito": Location(id="deposito")}
        drones = {
            "d1": Drone(
                id="d1",
                position="deposito",
                arms=(Arm(id="izq"),),
                state=DroneState.IDLE,
            ),
        }
        transporters = {
            "t1": Transporter(id="t1", position="deposito", capacity=4),
        }
        packages = {
            "pkg_w1": Package(
                id="pkg_w1",
                contains=contents["agua"],
                at=InTransporter(transporter_id="t1"),
            ),
        }
        world = World(
            locations=locs,
            drones=drones,
            transporters=transporters,
            packages=packages,
            contents=contents,
        )

        runner = PlanRunner(world)
        plan = Plan.sequential([
            UnloadFromTransporter(
                drone_id="d1",
                arm_id="izq",
                package_id="pkg_w1",
                transporter_id="t1",
                duration=5.0,
            ),
        ])
        runner.execute(plan)

        snap_start = runner.history.at(1)
        snap_end = runner.history.at(2)
        result = classify_transition(snap_start, snap_end)

        assert isinstance(result, TransitionPackageMove)
        assert result.drone_id == "d1"
        assert result.package_id == "pkg_w1"
        assert result.from_place == PackagePlace(
            kind="transp", transporter_id="t1"
        )
        assert result.to_place == PackagePlace(
            kind="arm", drone_id="d1", arm_id="izq"
        )

    def test_deliver_sin_arm_id_en_command_lee_arm_de_snap_a(self):
        # Verificación específica de la decisión 2.5 de Memoria B:
        # Deliver NO lleva arm_id; el classifier lo recupera del world.
        # Probamos con un PickUp en brazo DER, y al Deliver debe leer DER.
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="der", package_id="pkg_med1", duration=5.0),
                start_time=0.0,
            ),
            ScheduledCommand(
                command=Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1", duration=5.0),
                start_time=5.0,
            ),
        ))
        runner.execute(plan)

        # Cogemos el snap_start del Deliver.
        snap_start = None
        snap_end = None
        for i in range(len(runner.history)):
            snap = runner.history.at(i)
            if isinstance(snap.produced_by, Deliver):
                if snap_start is None:
                    snap_start = snap
                else:
                    snap_end = snap
                    break
        assert snap_start is not None and snap_end is not None

        result = classify_transition(snap_start, snap_end)
        assert isinstance(result, TransitionPackageMove)
        # El arm_id se infirió del world, no del Command:
        assert result.from_place.arm_id == "der"


# ---------------------------------------------------------------------------
# TransitionFailure
# ---------------------------------------------------------------------------


class TestTransitionFailure:
    """Drone en ERROR tras fallo PDDL."""

    def test_pickup_fallido_genera_failure(self):
        # PickUp sin que el paquete esté en la loc del drone → falla PDDL.
        contents = {"medicina": Content(id="medicina")}
        locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
        drones = {
            "d1": Drone(
                id="d1",
                position="casa1",
                arms=(Arm(id="izq"),),
                state=DroneState.IDLE,
            ),
        }
        packages = {
            # Paquete está en CASA2, drone en CASA1: pickup va a fallar.
            "pkg_med1": Package(
                id="pkg_med1",
                contains=contents["medicina"],
                at=AtLocation(loc_id="casa2"),
            ),
        }
        world = World(
            locations=locs, drones=drones, packages=packages, contents=contents
        )

        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1"),
        ])
        result = runner.execute(plan)

        assert not result.succeeded
        assert len(result.failures) == 1

        # El snap del fallo:
        snap_initial = runner.history.at(0)
        snap_fail = runner.history.at(1)
        # Drone en ERROR:
        assert snap_fail.world.drones["d1"].state == DroneState.ERROR

        trans = classify_transition(snap_initial, snap_fail)
        assert isinstance(trans, TransitionFailure)
        assert trans.drone_id == "d1"
        # command_id viene del Command que falló:
        assert trans.command_id == result.failures[0].scheduled.command.command_id

    def test_failure_tiene_prioridad_sobre_tipo_de_command(self):
        # Si el drone está en ERROR, no devolvemos un TransitionDroneMove
        # aunque el Command sea un Move. Es esencial para que el painter
        # NO intente interpolar movimiento de un drone que falló.
        contents = {}
        locs = {"casa1": Location(id="casa1")}  # solo una loc → Move a "fantasma" va a fallar.
        drones = {
            "d1": Drone(
                id="d1",
                position="casa1",
                arms=(),
                state=DroneState.IDLE,
            ),
        }
        world = World(locations=locs, drones=drones, contents=contents)

        runner = PlanRunner(world)
        plan = Plan.sequential([
            Move(drone_id="d1", destination_id="fantasma"),
        ])
        result = runner.execute(plan)

        assert not result.succeeded
        snap_initial = runner.history.at(0)
        snap_fail = runner.history.at(1)
        trans = classify_transition(snap_initial, snap_fail)
        # Aunque el Command sea Move, esperamos TransitionFailure:
        assert isinstance(trans, TransitionFailure)
        # NO TransitionDroneMove:
        assert not isinstance(trans, TransitionDroneMove)


# ---------------------------------------------------------------------------
# Errores defensivos
# ---------------------------------------------------------------------------


class TestErroresDefensivos:
    """Inconsistencias en snapshots o Commands desconocidos lanzan."""

    def test_command_desconocido_lanza_typeerror(self):
        # Construimos un Command "falso" que no es ninguno de los seis.
        from dataclasses import dataclass, field
        from droneplan_viz.commands.base import new_command_id

        @dataclass(frozen=True, slots=True)
        class FakeCommand:
            drone_id: str = "d1"
            duration: float = 0.0
            command_id: str = field(default_factory=new_command_id)

        world = make_world_for_pickup_and_deliver()
        snap_a = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=None,
            timestamp=0.0,
        )
        snap_b = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=FakeCommand(),
            timestamp=1.0,
        )
        with pytest.raises(TypeError, match="FakeCommand"):
            classify_transition(snap_a, snap_b)

    def test_pickup_con_paquete_no_at_location_lanza(self):
        # Construimos un snap_a inconsistente: el Command es PickUp pero
        # el paquete YA está HeldByArm (estado imposible en runtime real,
        # pero defendemos contra bugs upstream).
        world = make_world_for_pickup_and_deliver()
        # Modificamos el paquete para ponerlo HeldByArm:
        world_modified = World(
            locations=world.locations,
            drones=world.drones,
            persons=world.persons,
            packages={
                "pkg_med1": Package(
                    id="pkg_med1",
                    contains=Content(id="medicina"),
                    at=HeldByArm(drone_id="d1", arm_id="izq"),
                ),
            },
            contents=world.contents,
            costs=world.costs,
        )
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1")
        snap_a = WorldSnapshot(
            world=world_modified,
            metrics=MetricsTracker(),
            produced_by=None,
            timestamp=0.0,
        )
        snap_b = WorldSnapshot(
            world=world_modified,
            metrics=MetricsTracker(),
            produced_by=cmd,
            timestamp=1.0,
        )
        with pytest.raises(ValueError, match="AtLocation"):
            classify_transition(snap_a, snap_b)

    def test_deliver_con_paquete_no_heldbyarm_lanza(self):
        # Snap_a inconsistente para Deliver.
        world = make_world_for_pickup_and_deliver()
        # pkg sigue AtLocation en snap_a (estado imposible cuando se
        # llama a Deliver, pero defendemos).
        cmd = Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1")
        snap_a = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=None,
            timestamp=0.0,
        )
        snap_b = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=cmd,
            timestamp=1.0,
        )
        with pytest.raises(ValueError, match="HeldByArm"):
            classify_transition(snap_a, snap_b)


# ---------------------------------------------------------------------------
# Sanity de inmutabilidad y estructura de las Transition
# ---------------------------------------------------------------------------


class TestTransitionInmutabilidad:
    """Las Transition son frozen+slots, comparables por valor."""

    def test_transition_static_singleton_estructural(self):
        a = TransitionStatic()
        b = TransitionStatic()
        # Igualdad estructural (frozen+slots → eq=True por defecto):
        assert a == b

    def test_transition_drone_move_igualdad_estructural(self):
        a = TransitionDroneMove(
            drone_id="d1",
            from_loc_id="casa1",
            to_loc_id="casa2",
            transporter_id=None,
        )
        b = TransitionDroneMove(
            drone_id="d1",
            from_loc_id="casa1",
            to_loc_id="casa2",
            transporter_id=None,
        )
        assert a == b

    def test_transition_drone_move_inmutable(self):
        t = TransitionDroneMove(
            drone_id="d1",
            from_loc_id="casa1",
            to_loc_id="casa2",
            transporter_id=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            t.drone_id = "d2"  # type: ignore[misc]

    def test_package_place_inmutable(self):
        p = PackagePlace(kind="loc", loc_id="casa1")
        with pytest.raises((AttributeError, TypeError)):
            p.loc_id = "casa2"  # type: ignore[misc]

    def test_package_place_igualdad_por_valor(self):
        a = PackagePlace(kind="loc", loc_id="casa1")
        b = PackagePlace(kind="loc", loc_id="casa1")
        c = PackagePlace(kind="loc", loc_id="casa2")
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Integración: clasificación a lo largo de un plan completo
# ---------------------------------------------------------------------------


class TestIntegracionPlanCompleto:
    """Sanity end-to-end: un plan multi-step se clasifica correctamente."""

    def test_plan_pickup_y_deliver_secuencia_completa(self):
        # Plan secuencial (todos los Commands con duration=0):
        # - PickUp medicina en casa1.
        # - Deliver medicina a p1 en casa1.
        # En modo secuencial NO hay snapshots start; un solo end por Command.
        world = make_world_for_pickup_and_deliver()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1"),
            Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1"),
        ])
        runner.execute(plan)

        # Snapshots: [inicial, after pickup, after deliver] = 3.
        assert len(runner.history) == 3

        # Par 1: inicial → pickup
        t1 = classify_transition(runner.history.at(0), runner.history.at(1))
        assert isinstance(t1, TransitionPackageMove)
        assert t1.from_place.kind == "loc"
        assert t1.to_place.kind == "arm"

        # Par 2: pickup → deliver. snap_a.produced_by es PickUp y
        # snap_b.produced_by es Deliver. En modo instantáneo cada snapshot
        # es el snap_END (drone IDLE) de su Command, con la geometría ya
        # aplicada. El discriminador FSM (drone IDLE en snap_b) hace que se
        # ANIME la geometría del Deliver (paquete brazo → loc), igual que se
        # animó el PickUp en el par 1. Así un plan instantáneo secuencial
        # anima TODAS sus acciones de forma uniforme (antes el Deliver se
        # teletransportaba porque se comparaba command_id).
        t2 = classify_transition(runner.history.at(1), runner.history.at(2))
        assert isinstance(t2, TransitionPackageMove)
        assert t2.from_place.kind == "arm"
        assert t2.to_place.kind == "loc"
