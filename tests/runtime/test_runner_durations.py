"""
Tests de PlanRunner para Commands con duraciones > 0.

Cubre el comportamiento del runner cuando las acciones tienen extensión
temporal (modo PDDL parte 3 sin concurrencia todavía). En particular:

1. Snapshots duales: start con drone.state := MOVING/INTERACTING,
   end con efectos PDDL y drone.state := IDLE.
2. Mapeo Command -> DroneState: MOVING para los dos Move-*; INTERACTING
   para los cuatro restantes (PickUp, Deliver, Load, Unload).
3. Tiebreak END-antes-que-START entre Commands distintos en el mismo
   timestamp (caso back-to-back).
4. Tiebreak START-antes-que-END dentro del mismo Command (caso
   duration=0, ya cubierto en runner_sequential).
5. Corrección de total_time con end_time real.
6. Cancelación de END cuando el START falla.
7. Comportamiento de los snapshots para el render (Sesión D): timestamps
   correctos, produced_by correcto.
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import DroneState
from droneplan_viz.runtime import (
    Plan,
    PlanRunner,
    ScheduledCommand,
)

from tests.runtime.conftest import (
    make_deliver,
    make_load,
    make_move,
    make_move_with_transporter,
    make_pickup,
    make_unload,
    world_with_drone_at,
    world_with_package_held,
    world_with_package_in_transporter,
)


# ===========================================================================
# 1. Snapshots duales: estructura básica
# ===========================================================================
class TestSnapshotsDualesEstructura:
    """Un Command con duration>0 produce DOS snapshots: start y end."""

    def test_move_con_duracion_produce_dos_snapshots(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0, cmd_id="M")
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # 1 inicial + 2 (start + end) = 3
        assert len(r.history) == 3

    def test_pickup_con_duracion_produce_dos_snapshots(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="P"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        assert len(r.history) == 3

    def test_deliver_con_duracion_produce_dos_snapshots(self, base_world):
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        cmd = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="D"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        assert len(r.history) == 3

    def test_duracion_cero_sigue_dando_un_solo_snapshot(self, base_world):
        """Regresión: el caso duration=0 sigue produciendo solo el
        snapshot end (no se reintroduce el start optimizable)."""
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=0.0, cmd_id="M0")
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # 1 inicial + 1 end = 2
        assert len(r.history) == 2


# ===========================================================================
# 2. Snapshot start: contenido
# ===========================================================================
class TestSnapshotStartContenido:
    """El snapshot start solo cambia drone.state; los efectos PDDL
    NO se aplican hasta el snapshot end."""

    def test_start_pone_drone_en_moving_para_move(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # snapshot[1] = start, snapshot[2] = end
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.MOVING

    def test_start_pone_drone_en_moving_para_move_with_transporter(
        self, base_world
    ):
        r = PlanRunner(base_world)
        cmd = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.MOVING

    def test_start_pone_drone_en_interacting_para_pickup(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_pickup("d1", "izq", "libre1", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.INTERACTING

    def test_start_pone_drone_en_interacting_para_deliver(self, base_world):
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        cmd = make_deliver("d1", "libre1", "ana", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.INTERACTING

    def test_start_pone_drone_en_interacting_para_load(self, base_world):
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        cmd = make_load("d1", "libre1", "t1", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.INTERACTING

    def test_start_pone_drone_en_interacting_para_unload(self, base_world):
        world = world_with_package_in_transporter(
            base_world, "libre1", "t1"
        )
        r = PlanRunner(world)
        cmd = make_unload("d1", "izq", "libre1", "t1", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].state == DroneState.INTERACTING

    def test_start_no_aplica_efectos_pddl(self, base_world):
        """Móvil en marcha en snapshot start: drone aún en deposito,
        NO en casa1."""
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.world.drones["d1"].position == "deposito"

    def test_start_no_aplica_pickup(self, base_world):
        """Snapshot start de un PickUp: el paquete NO está aún en el brazo."""
        from droneplan_viz.domain import AtLocation
        r = PlanRunner(base_world)
        cmd = make_pickup("d1", "izq", "libre1", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        # libre1 sigue en el suelo del depósito.
        assert snap_start.world.packages["libre1"].at == AtLocation(
            loc_id="deposito"
        )

    def test_start_no_modifica_metricas(self, base_world):
        """El snapshot start lleva las metrics SIN cambios respecto al
        snapshot anterior."""
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_inicial = r.history.at(0)
        snap_start = r.history.at(1)
        assert snap_start.metrics is snap_inicial.metrics

    def test_start_produced_by_es_el_command(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0, cmd_id="el_mov")
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.produced_by is cmd

    def test_start_timestamp_es_start_time_del_scheduled(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=3.5),
        ))
        r.execute(plan)
        snap_start = r.history.at(1)
        assert snap_start.timestamp == 3.5


# ===========================================================================
# 3. Snapshot end: contenido
# ===========================================================================
class TestSnapshotEndContenido:
    def test_end_aplica_efectos_pddl(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        assert snap_end.world.drones["d1"].position == "casa1"

    def test_end_restaura_drone_a_idle(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_pickup("d1", "izq", "libre1", duration=5.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        assert snap_end.world.drones["d1"].state == DroneState.IDLE

    def test_end_timestamp_es_end_time(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=3.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        assert snap_end.timestamp == 13.0

    def test_end_produced_by_es_el_command(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0, cmd_id="X")
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        assert snap_end.produced_by is cmd

    def test_end_actualiza_action_count(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        assert snap_end.metrics.action_count == 1

    def test_end_actualiza_total_cost(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        snap_end = r.history.at(2)
        # fly-cost(deposito, casa1) = 10.0 en base_world.
        assert snap_end.metrics.total_cost == 10.0


# ===========================================================================
# 4. Corrección de total_time con end_time
# ===========================================================================
class TestCorreccionTotalTime:
    """El bug heredado de Sesión B: los handlers usan
    max(total_time, total_time + duration), que en uso aislado da
    valor correcto solo para una acción. Para varias acciones en
    paralelo (en el sentido: total_time crece monótonamente con la
    suma de durations), la fórmula está mal. El runner corrige.

    Aún sin concurrencia, podemos demostrar la diferencia de
    comportamiento entre 'handler crudo' y 'runner con fix' usando un
    plan con start_time creciente NO contiguo (huecos en el tiempo).
    """

    def test_total_time_es_end_time_de_un_solo_command(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "casa1", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # end_time = 0 + 10 = 10
        assert r.current_metrics.total_time == 10.0

    def test_total_time_no_suma_si_hay_hueco(self, base_world):
        """Plan con dos Commands: el primero acaba en t=10, el segundo
        empieza en t=100. El total_time del segundo debe ser 110 (su
        end_time), NO 20 (10 + 10 del primero + 10 del segundo)."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),    # end=10
            ScheduledCommand(command=m2, start_time=100.0),  # end=110
        ))
        r.execute(plan)
        # makespan = 110, no 20.
        assert r.current_metrics.total_time == 110.0

    def test_total_time_es_maximo_de_end_times(self, base_world):
        """Plan secuencial back-to-back: m1 [0,10), m2 [10,20)."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        r.execute(plan)
        # end_time del último = 20.
        assert r.current_metrics.total_time == 20.0


# ===========================================================================
# 5. Tiebreak END-antes-que-START entre Commands distintos
# ===========================================================================
class TestTiebreakBackToBack:
    """Caso back-to-back: m1 acaba en t=10, m2 empieza en t=10. Sin el
    tiebreak END-antes-que-START, el START de m2 vería al drone aún en
    MOVING y validate fallaría. Con el tiebreak correcto, primero
    procesa END(m1) (drone vuelve a IDLE), luego START(m2)."""

    def test_dos_moves_back_to_back_exitosos(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        # d1 al final está en deposito.
        assert r.current_world.drones["d1"].position == "deposito"

    def test_pickup_y_move_back_to_back(self, base_world):
        """PickUp [0,5) y Move [5,15) del mismo drone."""
        r = PlanRunner(base_world)
        p = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p"
        )
        m = make_move("d1", "casa1", duration=10.0, cmd_id="m")
        plan = Plan(scheduled=(
            ScheduledCommand(command=p, start_time=0.0),
            ScheduledCommand(command=m, start_time=5.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        # libre1 en brazo izq, drone en casa1.
        from droneplan_viz.domain import HeldByArm
        assert r.current_world.packages["libre1"].at == HeldByArm(
            drone_id="d1", arm_id="izq"
        )
        assert r.current_world.drones["d1"].position == "casa1"

    def test_history_estructura_de_back_to_back(self, base_world):
        """Verificamos la secuencia exacta de snapshots producidos."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        r.execute(plan)
        # snap 0: inicial; snap 1: start m1; snap 2: end m1;
        # snap 3: start m2; snap 4: end m2.
        assert len(r.history) == 5
        assert r.history.at(0).produced_by is None
        assert r.history.at(1).produced_by is m1
        assert r.history.at(1).timestamp == 0.0
        assert r.history.at(2).produced_by is m1
        assert r.history.at(2).timestamp == 10.0
        assert r.history.at(3).produced_by is m2
        assert r.history.at(3).timestamp == 10.0  # mismo timestamp que el end de m1
        assert r.history.at(4).produced_by is m2
        assert r.history.at(4).timestamp == 20.0

    def test_drone_state_transiciones_correctas_en_back_to_back(
        self, base_world
    ):
        """A lo largo del historial:
            snap 0 (inicial):   IDLE
            snap 1 (start m1):  MOVING
            snap 2 (end m1):    IDLE
            snap 3 (start m2):  MOVING
            snap 4 (end m2):    IDLE
        """
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        r.execute(plan)
        expected = [
            DroneState.IDLE,
            DroneState.MOVING,
            DroneState.IDLE,
            DroneState.MOVING,
            DroneState.IDLE,
        ]
        for i, st in enumerate(expected):
            assert r.history.at(i).world.drones["d1"].state == st, (
                f"snapshot {i}: esperaba {st.name}"
            )


# ===========================================================================
# 6. Cancelación del END cuando el START falla
# ===========================================================================
class TestCancelacionEND:
    """Si el START de un Command produce un fallo PDDL, su END nunca
    debe procesarse: ni snapshot end, ni apply(), ni transición a IDLE."""

    def test_start_falla_no_se_genera_snapshot_end(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "narnia", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # 1 inicial + 1 snapshot del fallo = 2.
        # NO debe haber un tercer snapshot (end).
        assert len(r.history) == 2

    def test_start_falla_drone_se_queda_en_error(self, base_world):
        r = PlanRunner(base_world)
        cmd = make_move("d1", "narnia", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        # El drone está en ERROR; el END nunca lo restauró a IDLE.
        assert r.current_world.drones["d1"].state == DroneState.ERROR

    def test_start_falla_no_se_suma_coste(self, base_world):
        """Si el END no se procesa, apply() no se llama, total_cost no sube."""
        r = PlanRunner(base_world)
        cmd = make_move("d1", "narnia", duration=10.0)
        plan = Plan(scheduled=(
            ScheduledCommand(command=cmd, start_time=0.0),
        ))
        r.execute(plan)
        assert r.current_metrics.total_cost == 0.0
        assert r.current_metrics.action_count == 0
        assert r.current_metrics.failed_commands == 1

    def test_command_posterior_se_ejecuta_aunque_anterior_fallara(
        self, base_world
    ):
        """Si m1 falla, su END se descarta. m2 (de otro drone) debe
        ejecutarse normalmente."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "narnia", duration=10.0, cmd_id="m1_fail")
        m2 = make_move("d2", "casa1", duration=10.0, cmd_id="m2_ok")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert len(result.failures) == 1
        assert r.current_world.drones["d2"].position == "casa1"


# ===========================================================================
# 7. Makespan con duraciones reales
# ===========================================================================
class TestMakespan:
    def test_makespan_es_end_time_del_ultimo_command(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        result = r.execute(plan)
        assert result.makespan == 20.0

    def test_makespan_no_cuenta_command_fallado(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="ok")
        m2 = make_move("d1", "narnia", duration=10.0, cmd_id="fail")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        result = r.execute(plan)
        # m1 exitoso end_time=10; m2 falla, no cuenta.
        # makespan = 10.
        assert result.makespan == 10.0


# ===========================================================================
# 8. Plan completo con duraciones reales (PDDL parte 3 sin concurrencia)
# ===========================================================================
class TestPlanCompletoConDuraciones:
    def test_pickup_move_deliver_con_duraciones_pdf(self, base_world):
        """Reproduce el patrón canónico PDDL parte 3:
            - PickUp duration=5.0 (regla del PDF: no-vuelo = 5s)
            - Move duration=10.0 (fly-cost = 10 en base_world)
            - Deliver duration=5.0
        Todo del mismo drone, secuencial.
        """
        r = PlanRunner(base_world)
        p = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p"
        )
        m = make_move("d1", "casa1", duration=10.0, cmd_id="m")
        d = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="d"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p, start_time=0.0),   # [0, 5)
            ScheduledCommand(command=m, start_time=5.0),   # [5, 15)
            ScheduledCommand(command=d, start_time=15.0),  # [15, 20)
        ))
        result = r.execute(plan)
        assert result.succeeded
        # Estado final: drone en casa1, ana sin necesidades.
        assert r.current_world.drones["d1"].position == "casa1"
        assert r.current_world.persons["ana"].needs == ()
        # Métricas: 3 acciones, makespan=20, cost=10 (solo el move).
        assert result.final_metrics.action_count == 3
        assert result.final_metrics.total_cost == 10.0
        assert result.makespan == 20.0

    def test_history_size_para_plan_de_3_acciones_con_duracion(
        self, base_world
    ):
        """3 acciones, todas con duration>0, sin solapamiento.
        Cada una produce 2 snapshots. Total: 1 inicial + 6 = 7."""
        r = PlanRunner(base_world)
        p = make_pickup("d1", "izq", "libre1", duration=5.0, cmd_id="p")
        m = make_move("d1", "casa1", duration=10.0, cmd_id="m")
        d = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="d"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p, start_time=0.0),
            ScheduledCommand(command=m, start_time=5.0),
            ScheduledCommand(command=d, start_time=15.0),
        ))
        r.execute(plan)
        assert len(r.history) == 7


# ===========================================================================
# 9. Plan mixto: algunas duraciones cero, otras no
# ===========================================================================
class TestPlanMixto:
    """Plan con algunos Commands atemporales (duration=0) y otros
    durativos (duration>0). El runner los procesa cada uno con su
    semántica."""

    def test_plan_mixto_funciona_correctamente(self, base_world):
        r = PlanRunner(base_world)
        p = make_pickup(
            "d1", "izq", "libre1", duration=0.0, cmd_id="p_atemp"
        )
        m = make_move("d1", "casa1", duration=10.0, cmd_id="m_dur")
        d = make_deliver(
            "d1", "libre1", "ana", duration=0.0, cmd_id="d_atemp"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p, start_time=0.0),    # [0, 0)
            ScheduledCommand(command=m, start_time=0.0),    # [0, 10)
            ScheduledCommand(command=d, start_time=10.0),   # [10, 10)
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.persons["ana"].needs == ()
        assert r.current_world.drones["d1"].position == "casa1"

    def test_plan_mixto_history_size_correcto(self, base_world):
        """Plan mixto: 2 con duration=0 (1 snap cada uno) + 1 con
        duration>0 (2 snaps). Total: 1 inicial + 2 + 2 = 5."""
        r = PlanRunner(base_world)
        p = make_pickup(
            "d1", "izq", "libre1", duration=0.0, cmd_id="p"
        )
        m = make_move("d1", "casa1", duration=10.0, cmd_id="m")
        d = make_deliver(
            "d1", "libre1", "ana", duration=0.0, cmd_id="d"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p, start_time=0.0),
            ScheduledCommand(command=m, start_time=0.0),
            ScheduledCommand(command=d, start_time=10.0),
        ))
        r.execute(plan)
        assert len(r.history) == 5


# ===========================================================================
# 10. Drone en MOVING/INTERACTING rechaza siguientes Commands
# ===========================================================================
class TestDroneOcupadoRechazaComandos:
    """Sin necesidad de ResourceTable: el Validator de Sesión A
    (vía _check_drone_ready) rechaza Commands sobre drones MOVING o
    INTERACTING. Esto es el mecanismo natural por el que el runner
    bloquea acciones simultáneas sobre el mismo drone.

    En este paso, sin concurrencia, el bug ocurre cuando un usuario
    construye un Plan que pide al mismo drone hacer dos cosas en
    intervalos solapados. Sin ResourceTable, el runner solo lo
    detectará gracias a _check_drone_ready cuando llegue el START
    del segundo Command y vea al drone en MOVING.
    """

    def test_segundo_command_sobre_drone_moving_falla(self, base_world):
        """m1 ocupa d1 en [0,10); m2 sobre d1 intenta empezar en t=5."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "casa2", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=5.0),
        ))
        result = r.execute(plan)
        # m2 falla porque cuando intenta empezar (t=5), d1 está MOVING.
        assert not result.succeeded
        assert len(result.failures) == 1
        assert result.failures[0].scheduled.command is m2
        assert "MOVING" in result.failures[0].reason or "ocupado" in result.failures[0].reason

    def test_segundo_command_sobre_drone_interacting_falla(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d1", "der", "libre2", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),  # [0,5)
            ScheduledCommand(command=p2, start_time=2.0),  # [2,7) -- solapado
        ))
        result = r.execute(plan)
        # p2 falla porque d1 está INTERACTING en t=2.
        assert not result.succeeded
        assert result.failures[0].scheduled.command is p2

    def test_back_to_back_no_falla_aunque_state_cambia(self, base_world):
        """Caso límite: m2 empieza exactamente cuando m1 acaba. El
        tiebreak END-antes-que-START garantiza que d1 está IDLE cuando
        m2 valida."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "deposito", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=10.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# 11. Acciones concurrentes en drones distintos
# ===========================================================================
class TestAccionesConcurrentesEnDronesDistintos:
    """Sin ResourceTable explícita aún, pero el runner ya soporta dos
    drones actuando 'en paralelo' (intervalos solapados) sobre recursos
    disjuntos. Esto funciona porque la validación PDDL solo rechaza
    al MISMO drone ocupado; drones distintos no interfieren.

    Estos tests fijan el comportamiento esperado de Sesión C para casos
    concurrentes simples. Los casos donde DOS drones interfieren (caja
    compartida, persona compartida, transportador compartido) son los
    que pide la ResourceTable, paso 6.
    """

    def test_dos_moves_drones_distintos_en_paralelo_exitosos(self, base_world):
        r = PlanRunner(base_world)
        m_a = make_move("d1", "casa1", duration=10.0, cmd_id="A")
        m_b = make_move("d2", "casa2", duration=20.0, cmd_id="B")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m_a, start_time=0.0),  # [0,10)
            ScheduledCommand(command=m_b, start_time=0.0),  # [0,20)
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.drones["d1"].position == "casa1"
        assert r.current_world.drones["d2"].position == "casa2"

    def test_makespan_es_max_de_ambos(self, base_world):
        r = PlanRunner(base_world)
        m_a = make_move("d1", "casa1", duration=10.0, cmd_id="A")
        m_b = make_move("d2", "casa2", duration=20.0, cmd_id="B")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m_a, start_time=0.0),
            ScheduledCommand(command=m_b, start_time=0.0),
        ))
        result = r.execute(plan)
        # makespan = max(10, 20) = 20
        assert result.makespan == 20.0

    def test_total_time_es_makespan_no_suma(self, base_world):
        """Test CRÍTICO del bug fix: si los handlers de Sesión B fueran
        consultados ingenuamente, total_time sería 30 (10+20). Con la
        corrección, es 20 (max)."""
        r = PlanRunner(base_world)
        m_a = make_move("d1", "casa1", duration=10.0, cmd_id="A")
        m_b = make_move("d2", "casa2", duration=20.0, cmd_id="B")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m_a, start_time=0.0),
            ScheduledCommand(command=m_b, start_time=0.0),
        ))
        r.execute(plan)
        assert r.current_metrics.total_time == 20.0
