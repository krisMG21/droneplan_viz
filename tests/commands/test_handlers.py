"""Tests de commands/handlers.py: apply_X y dispatch apply()."""
from __future__ import annotations

import pytest

from droneplan_viz.commands.handlers import (
    apply,
    apply_deliver,
    apply_load_into_transporter,
    apply_move,
    apply_move_with_transporter,
    apply_pick_up,
    apply_unload_from_transporter,
)
from droneplan_viz.commands.manipulation import Deliver, PickUp
from droneplan_viz.commands.movement import Move, MoveWithTransporter
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter
from droneplan_viz.domain import (
    AtLocation,
    HeldByArm,
    InTransporter,
    MetricsTracker,
)


# ===========================================================================
# Move
# ===========================================================================
class TestApplyMove:
    def test_drone_cambia_de_posicion(self, base_world, initial_metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        new_world, _ = apply_move(base_world, initial_metrics, cmd)
        assert new_world.drones["d1"].position == "casa1"

    def test_origen_no_modificado(self, base_world, initial_metrics):
        """Inmutabilidad: el World original no se toca."""
        original_position = base_world.drones["d1"].position
        cmd = Move(drone_id="d1", destination_id="casa1")
        apply_move(base_world, initial_metrics, cmd)
        assert base_world.drones["d1"].position == original_position
        assert base_world.drones["d1"].position == "deposito"

    def test_metricas_originales_no_modificadas(self, base_world, initial_metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        apply_move(base_world, initial_metrics, cmd)
        assert initial_metrics.action_count == 0
        assert initial_metrics.total_cost == 0.0
        assert initial_metrics.total_time == 0.0

    def test_action_count_incrementa(self, base_world, initial_metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        _, new_metrics = apply_move(base_world, initial_metrics, cmd)
        assert new_metrics.action_count == 1

    def test_total_cost_suma_fly_cost(self, base_world, initial_metrics):
        # costs[(deposito, casa1)] = 10.0 en el base_world
        cmd = Move(drone_id="d1", destination_id="casa1")
        _, new_metrics = apply_move(base_world, initial_metrics, cmd)
        assert new_metrics.total_cost == 10.0

    def test_total_time_no_lo_toca_el_handler(self, base_world):
        """El makespan (total_time) es responsabilidad exclusiva del runner,
        no de los handlers. apply() debe dejar total_time intacto, heredando
        el valor del MetricsTracker de entrada, sea cual sea cmd.duration.

        Partimos de un total_time no-cero para que el test sea significativo:
        si el handler intentara recalcularlo (como hacía en Sesión B), el
        valor cambiaría; tras el fix, se conserva.
        """
        metrics_previo = MetricsTracker(total_time=42.0)
        cmd = Move(drone_id="d1", destination_id="casa1", duration=5.0)
        _, new_metrics = apply_move(base_world, metrics_previo, cmd)
        assert new_metrics.total_time == 42.0

    def test_total_time_intacto_con_duration_cero(self, base_world):
        """Igual que el anterior con duration=0.0: el handler nunca toca
        total_time."""
        metrics_previo = MetricsTracker(total_time=42.0)
        cmd = Move(drone_id="d1", destination_id="casa1", duration=0.0)
        _, new_metrics = apply_move(base_world, metrics_previo, cmd)
        assert new_metrics.total_time == 42.0

    def test_otras_entidades_no_se_tocan(self, base_world, initial_metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        new_world, _ = apply_move(base_world, initial_metrics, cmd)
        # paquetes, personas, transportadores intactos
        assert new_world.packages == base_world.packages
        assert new_world.persons == base_world.persons
        assert new_world.transporters == base_world.transporters


# ===========================================================================
# MoveWithTransporter
# ===========================================================================
class TestApplyMoveWithTransporter:
    def test_drone_y_transportador_se_mueven_juntos(self, base_world, initial_metrics):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        new_world, _ = apply_move_with_transporter(base_world, initial_metrics, cmd)
        assert new_world.drones["d1"].position == "casa1"
        assert new_world.transporters["t1"].position == "casa1"

    def test_paquetes_dentro_siguen_referenciando_el_transportador(
        self, base_world, initial_metrics
    ):
        """Los paquetes dentro del transportador NO cambian su 'at'.
        Siguen siendo InTransporter(t1); la localización efectiva se
        deriva de t1.position, que ahora es casa1.
        """
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        new_world, _ = apply_move_with_transporter(base_world, initial_metrics, cmd)

        pkg_en_trans = new_world.packages["en_trans"]
        assert isinstance(pkg_en_trans.at, InTransporter)
        assert pkg_en_trans.at.transporter_id == "t1"
        # Y el transportador ya está en casa1
        assert new_world.transporters["t1"].position == "casa1"

    def test_originales_no_se_tocan(self, base_world, initial_metrics):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        apply_move_with_transporter(base_world, initial_metrics, cmd)
        assert base_world.drones["d1"].position == "deposito"
        assert base_world.transporters["t1"].position == "deposito"

    def test_metricas_actualizadas(self, base_world, initial_metrics):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1",
            duration=12.0,
        )
        _, new_metrics = apply_move_with_transporter(base_world, initial_metrics, cmd)
        assert new_metrics.action_count == 1
        assert new_metrics.total_cost == 10.0
        # total_time NO lo toca el handler (lo fija el runner con end_time).
        assert new_metrics.total_time == initial_metrics.total_time


# ===========================================================================
# PickUp
# ===========================================================================
class TestApplyPickUp:
    def test_paquete_pasa_a_heldbyarm(self, base_world, initial_metrics):
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        new_world, _ = apply_pick_up(base_world, initial_metrics, cmd)

        pkg = new_world.packages["libre1"]
        assert isinstance(pkg.at, HeldByArm)
        assert pkg.at.drone_id == "d1"
        assert pkg.at.arm_id == "der"

    def test_helper_package_held_by_lo_encuentra(self, base_world, initial_metrics):
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        new_world, _ = apply_pick_up(base_world, initial_metrics, cmd)
        pkg = new_world.package_held_by("d1", "der")
        assert pkg is not None
        assert pkg.id == "libre1"

    def test_paquete_original_no_modificado(self, base_world, initial_metrics):
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        apply_pick_up(base_world, initial_metrics, cmd)
        orig = base_world.packages["libre1"]
        assert isinstance(orig.at, AtLocation)
        assert orig.at.loc_id == "deposito"

    def test_metricas_solo_incrementan_action_count(
        self, base_world, initial_metrics
    ):
        """PickUp no tiene coste de vuelo y no fija makespan: el único
        efecto sobre métricas es incrementar action_count. total_cost y
        total_time se heredan intactos del MetricsTracker de entrada.
        """
        cmd = PickUp(
            drone_id="d1", arm_id="der", package_id="libre1", duration=5.0
        )
        _, new_metrics = apply_pick_up(base_world, initial_metrics, cmd)
        assert new_metrics.action_count == 1
        assert new_metrics.total_cost == initial_metrics.total_cost
        assert new_metrics.total_time == initial_metrics.total_time


# ===========================================================================
# Deliver
# ===========================================================================
class TestApplyDeliver:
    def test_persona_recibe_contenido(self, base_world, initial_metrics):
        # Construyo un World ad-hoc donde el drone está en casa1 con
        # un paquete de medicina sostenido, ana también en casa1 esperando.
        from droneplan_viz.domain import (
            Arm, Content, Drone, DroneState, HeldByArm, Location, Package,
            Person, World,
        )
        med = Content(id="medicina")
        w = World(
            locations={"casa1": Location(id="casa1")},
            drones={
                "d1": Drone(
                    id="d1", position="casa1",
                    arms=(Arm(id="izq"),), state=DroneState.IDLE,
                )
            },
            packages={
                "p_med": Package(
                    id="p_med", contains=med,
                    at=HeldByArm(drone_id="d1", arm_id="izq"),
                )
            },
            persons={
                "ana": Person(
                    id="ana", position="casa1", needs=(med,), has_received=(),
                )
            },
            contents={"medicina": med},
        )
        cmd = Deliver(drone_id="d1", package_id="p_med", person_id="ana")
        new_world, _ = apply_deliver(w, initial_metrics, cmd)

        # La persona ahora ha recibido el contenido
        ana = new_world.persons["ana"]
        assert med in ana.has_received
        assert med not in ana.needs

        # El paquete queda libre en la localización de la persona
        pkg = new_world.packages["p_med"]
        assert isinstance(pkg.at, AtLocation)
        assert pkg.at.loc_id == "casa1"

    def test_originales_no_modificados(self, initial_metrics):
        from droneplan_viz.domain import (
            Arm, Content, Drone, DroneState, HeldByArm, Location, Package,
            Person, World,
        )
        med = Content(id="medicina")
        w = World(
            locations={"casa1": Location(id="casa1")},
            drones={
                "d1": Drone(
                    id="d1", position="casa1",
                    arms=(Arm(id="izq"),), state=DroneState.IDLE,
                )
            },
            packages={
                "p_med": Package(
                    id="p_med", contains=med,
                    at=HeldByArm(drone_id="d1", arm_id="izq"),
                )
            },
            persons={
                "ana": Person(
                    id="ana", position="casa1", needs=(med,), has_received=(),
                )
            },
            contents={"medicina": med},
        )
        cmd = Deliver(drone_id="d1", package_id="p_med", person_id="ana")
        apply_deliver(w, initial_metrics, cmd)
        # Inmutabilidad: ana original no recibió nada
        assert w.persons["ana"].has_received == ()
        assert med in w.persons["ana"].needs


# ===========================================================================
# LoadIntoTransporter
# ===========================================================================
class TestApplyLoad:
    def test_paquete_pasa_a_intransporter(self, base_world, initial_metrics):
        # 'sostenida' está siendo agarrada por izq -> la metemos en t1
        cmd = LoadIntoTransporter(drone_id="d1", package_id="sostenida", transporter_id="t1")
        new_world, _ = apply_load_into_transporter(base_world, initial_metrics, cmd)

        pkg = new_world.packages["sostenida"]
        assert isinstance(pkg.at, InTransporter)
        assert pkg.at.transporter_id == "t1"

    def test_brazo_queda_libre_implicitamente(self, base_world, initial_metrics):
        cmd = LoadIntoTransporter(drone_id="d1", package_id="sostenida", transporter_id="t1")
        new_world, _ = apply_load_into_transporter(base_world, initial_metrics, cmd)
        # No hay package_held_by para izq tras la carga
        assert new_world.package_held_by("d1", "izq") is None

    def test_paquetes_en_transportador_aumenta(self, base_world, initial_metrics):
        antes = len(base_world.packages_in_transporter("t1"))
        cmd = LoadIntoTransporter(drone_id="d1", package_id="sostenida", transporter_id="t1")
        new_world, _ = apply_load_into_transporter(base_world, initial_metrics, cmd)
        despues = len(new_world.packages_in_transporter("t1"))
        assert despues == antes + 1


# ===========================================================================
# UnloadFromTransporter
# ===========================================================================
class TestApplyUnload:
    def test_paquete_pasa_a_heldbyarm(self, base_world, initial_metrics):
        # 'en_trans' está en t1, lo cogemos con der
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        new_world, _ = apply_unload_from_transporter(
            base_world, initial_metrics, cmd
        )

        pkg = new_world.packages["en_trans"]
        assert isinstance(pkg.at, HeldByArm)
        assert pkg.at.drone_id == "d1"
        assert pkg.at.arm_id == "der"

    def test_paquetes_en_transportador_disminuye(self, base_world, initial_metrics):
        antes = len(base_world.packages_in_transporter("t1"))
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        new_world, _ = apply_unload_from_transporter(
            base_world, initial_metrics, cmd
        )
        despues = len(new_world.packages_in_transporter("t1"))
        assert despues == antes - 1


# ===========================================================================
# Dispatcher apply()
# ===========================================================================
class TestApplyDispatcher:
    """El dispatch enruta cada tipo a su handler. Verificamos cada uno
    comparando el resultado del dispatcher con el del handler directo.
    """

    def test_move(self, base_world, initial_metrics):
        cmd = Move(drone_id="d1", destination_id="casa1")
        via_dispatch = apply(base_world, initial_metrics, cmd)
        directo = apply_move(base_world, initial_metrics, cmd)
        # Comparar Worlds: como World no implementa __eq__ campo a campo
        # útil aquí, comparamos un campo característico
        assert via_dispatch[0].drones["d1"].position == directo[0].drones["d1"].position
        assert via_dispatch[1] == directo[1]

    def test_pick_up(self, base_world, initial_metrics):
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        new_world, _ = apply(base_world, initial_metrics, cmd)
        assert isinstance(new_world.packages["libre1"].at, HeldByArm)

    def test_load(self, base_world, initial_metrics):
        cmd = LoadIntoTransporter(drone_id="d1", package_id="sostenida", transporter_id="t1")
        new_world, _ = apply(base_world, initial_metrics, cmd)
        assert isinstance(new_world.packages["sostenida"].at, InTransporter)

    def test_unload(self, base_world, initial_metrics):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        new_world, _ = apply(base_world, initial_metrics, cmd)
        assert isinstance(new_world.packages["en_trans"].at, HeldByArm)

    def test_dispatch_distingue_load_de_unload(
        self, base_world, initial_metrics
    ):
        """El dispatch enruta Load y Unload a handlers radicalmente
        distintos: uno deja el paquete dentro del transportador, el
        otro lo saca. Tras la reconciliación con Sesión A, además
        tienen formas distintas (Load sin arm_id, Unload con).
        """
        load = LoadIntoTransporter(drone_id="d1", package_id="sostenida", transporter_id="t1")
        new_world_load, _ = apply(base_world, initial_metrics, load)
        assert isinstance(new_world_load.packages["sostenida"].at, InTransporter)

        unload = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        new_world_unload, _ = apply(base_world, initial_metrics, unload)
        assert isinstance(new_world_unload.packages["en_trans"].at, HeldByArm)


class TestApplyDispatcherDefensivo:
    def test_objeto_arbitrario_lanza_typeerror(self, base_world, initial_metrics):
        with pytest.raises(TypeError, match="no es un Command"):
            apply(base_world, initial_metrics, "no_es_un_command")  # type: ignore[arg-type]
