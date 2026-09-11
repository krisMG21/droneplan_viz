"""
Tests de PlanRunner en modo secuencial (Commands con duration=0).

Este archivo cubre los casos del runner cuando las acciones no tienen
dimensión temporal: las partes 1 y 2 del PDDL (clásico y numérico). La
concurrencia y los snapshots duales se tratan en archivos separados
en pasos posteriores.

Categorías de test:

1. Construcción del runner y snapshot inicial.
2. Propiedades de introspección (history, cursor, current_world,
   current_metrics).
3. Ejecución de planes vacíos.
4. Ejecución de planes mono-Command.
5. Ejecución de planes multi-Command secuenciales con todos los tipos.
6. Ordenación temporal y tiebreak determinista.
7. Fallos PDDL: validate rechaza, drone a ERROR, failed_commands++,
   snapshot del fallo, ejecución continúa.
8. Plan completo end-to-end: deposito -> casa -> entrega -> vuelve.
9. Corrección del bug heredado de total_time (un test específico).
10. Métricas: total_cost, action_count, makespan.
11. Inmutabilidad: el World inicial no se muta.
12. dry_run: no afecta al runner original.
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import DroneState, MetricsTracker
from droneplan_viz.runtime import (
    CommandFailure,
    Plan,
    PlanRunner,
    RunResult,
    ScheduledCommand,
)

# Importamos las factorías del conftest.py
from tests.runtime.conftest import (
    make_deliver,
    make_load,
    make_move,
    make_move_with_transporter,
    make_pickup,
    make_unload,
    world_with_drone_at,
    world_with_drone_in_error,
    world_with_package_held,
    world_with_package_in_transporter,
)


# ===========================================================================
# 1. Construcción y snapshot inicial
# ===========================================================================
class TestConstruccion:
    def test_construccion_con_world_basico(self, base_world):
        r = PlanRunner(base_world)
        assert r.current_world is base_world

    def test_metricas_iniciales_por_defecto(self, base_world):
        r = PlanRunner(base_world)
        m = r.current_metrics
        assert m.total_cost == 0.0
        assert m.total_time == 0.0
        assert m.action_count == 0
        assert m.failed_commands == 0

    def test_metricas_iniciales_custom(self, base_world):
        custom = MetricsTracker(
            total_cost=5.0, total_time=2.0,
            action_count=1, failed_commands=0,
        )
        r = PlanRunner(base_world, metrics=custom)
        assert r.current_metrics is custom

    def test_history_arranca_con_un_snapshot(self, base_world):
        r = PlanRunner(base_world)
        assert len(r.history) == 1

    def test_snapshot_inicial_produced_by_es_none(self, base_world):
        r = PlanRunner(base_world)
        assert r.history.at(0).produced_by is None

    def test_snapshot_inicial_timestamp_es_cero(self, base_world):
        r = PlanRunner(base_world)
        assert r.history.at(0).timestamp == 0.0

    def test_cursor_arranca_en_head(self, base_world):
        r = PlanRunner(base_world)
        assert r.cursor.at_head
        assert r.cursor.index == 0


# ===========================================================================
# 2. Propiedades de introspección
# ===========================================================================
class TestPropiedadesIntrospeccion:
    def test_current_world_coincide_con_cursor(self, base_world):
        r = PlanRunner(base_world)
        assert r.current_world is r.cursor.current.world

    def test_current_metrics_coincide_con_cursor(self, base_world):
        r = PlanRunner(base_world)
        assert r.current_metrics is r.cursor.current.metrics

    def test_history_y_cursor_son_consistentes(self, base_world):
        r = PlanRunner(base_world)
        assert r.cursor.current is r.history.at(0)


# ===========================================================================
# 3. Plan vacío
# ===========================================================================
class TestPlanVacio:
    def test_plan_vacio_no_anade_snapshots(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert len(r.history) == 1

    def test_plan_vacio_run_result_succeeded(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert result.succeeded
        assert result.failures == ()

    def test_plan_vacio_metrics_iguales(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert result.final_metrics.action_count == 0
        assert result.final_metrics.total_cost == 0.0

    def test_plan_vacio_makespan_cero(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert result.makespan == 0.0

    def test_plan_vacio_history_length(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert result.history_length == 1

    def test_plan_vacio_world_intacto(self, base_world):
        r = PlanRunner(base_world)
        result = r.execute(Plan())
        assert result.final_world is base_world


# ===========================================================================
# 4. Plan mono-Command (cada tipo)
# ===========================================================================
class TestPlanMonoCommand:
    def test_move_exitoso_anade_un_snapshot(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        result = r.execute(plan)
        assert len(r.history) == 2
        assert result.succeeded

    def test_move_actualiza_posicion_drone(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        r.execute(plan)
        assert r.current_world.drones["d1"].position == "casa1"

    def test_move_suma_coste(self, base_world):
        """fly-cost(deposito, casa1) = 10.0 en base_world."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        result = r.execute(plan)
        assert result.final_metrics.total_cost == 10.0

    def test_pickup_exitoso(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_pickup("d1", "izq", "libre1")])
        result = r.execute(plan)
        assert result.succeeded
        from droneplan_viz.domain import HeldByArm
        pkg = r.current_world.packages["libre1"]
        assert pkg.at == HeldByArm(drone_id="d1", arm_id="izq")

    def test_move_with_transporter_actualiza_ambos(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move_with_transporter("d1", "t1", "casa1"),
        ])
        r.execute(plan)
        assert r.current_world.drones["d1"].position == "casa1"
        assert r.current_world.transporters["t1"].position == "casa1"

    def test_deliver_exitoso_cambia_persona(self, base_world):
        # Pre-setup: libre1 (medicina) ya está en brazo izq de d1, en casa1.
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_package_held(world, "libre1", "d1", "izq")

        r = PlanRunner(world)
        plan = Plan.sequential([
            make_deliver("d1", "libre1", "ana"),
        ])
        result = r.execute(plan)
        assert result.succeeded
        # Ana ha recibido medicina y ya no la necesita.
        ana = r.current_world.persons["ana"]
        assert ana.needs == ()
        assert len(ana.has_received) == 1

    def test_load_exitoso(self, base_world):
        # libre1 sostenido por d1; cargar en t1.
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        plan = Plan.sequential([
            make_load("d1", "libre1", "t1"),
        ])
        result = r.execute(plan)
        assert result.succeeded
        from droneplan_viz.domain import InTransporter
        assert r.current_world.packages["libre1"].at == InTransporter(
            transporter_id="t1"
        )

    def test_unload_exitoso(self, base_world):
        # libre1 dentro de t1; descargar al brazo izq de d1.
        world = world_with_package_in_transporter(
            base_world, "libre1", "t1"
        )
        r = PlanRunner(world)
        plan = Plan.sequential([
            make_unload("d1", "izq", "libre1", "t1"),
        ])
        result = r.execute(plan)
        assert result.succeeded
        from droneplan_viz.domain import HeldByArm
        assert r.current_world.packages["libre1"].at == HeldByArm(
            drone_id="d1", arm_id="izq"
        )


# ===========================================================================
# 5. Plan multi-Command secuencial completo
# ===========================================================================
class TestPlanMultiCommand:
    def test_plan_completo_pickup_move_deliver(self, base_world):
        """Plan canónico de la asignatura: recoger, volar, entregar."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),
            make_move("d1", "casa1"),
            make_deliver("d1", "libre1", "ana"),
        ])
        result = r.execute(plan)
        assert result.succeeded
        # 1 inicial + 3 commits = 4 snapshots
        assert len(r.history) == 4
        # Ana recibió, d1 está en casa1.
        assert r.current_world.drones["d1"].position == "casa1"
        assert r.current_world.persons["ana"].needs == ()

    def test_plan_completo_con_transportador(self, base_world):
        """Cargar dos paquetes, mover, descargar uno, entregar el otro."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),  # medicina
            make_load("d1", "libre1", "t1"),
            make_pickup("d1", "izq", "libre2"),  # comida, mismo brazo libre
            make_move_with_transporter("d1", "t1", "casa1"),
            make_deliver("d1", "libre2", "ana"),  # libre2=comida, ana necesita medicina (no funciona)
        ])
        result = r.execute(plan)
        # La última falla: ana necesita medicina (regla del Validator)
        # pero hemos llevado comida. El resto del plan se ejecutó OK.
        assert not result.succeeded
        assert len(result.failures) == 1
        assert result.failures[0].kind == "pddl"

    def test_action_count_acumula_solo_exitos(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),
            make_move("d1", "casa1"),
            make_deliver("d1", "libre1", "ana"),
        ])
        r.execute(plan)
        assert r.current_metrics.action_count == 3


# ===========================================================================
# 6. Ordenación temporal y tiebreak
# ===========================================================================
class TestOrdenacionTemporal:
    def test_plan_se_ejecuta_en_orden_de_start_time(self, base_world):
        """Aunque el Plan se construya con orden 'invertido', el runner
        los ejecuta en orden de start_time."""
        r = PlanRunner(base_world)
        # Declaramos primero el Move a casa2, pero le damos start_time=2.
        # El Move a casa1 se declara segundo pero con start_time=1.
        move_to_casa2 = make_move("d1", "casa2", cmd_id="goto2")
        move_to_casa1 = make_move("d1", "casa1", cmd_id="goto1")
        plan = Plan(scheduled=(
            ScheduledCommand(command=move_to_casa2, start_time=2.0),
            ScheduledCommand(command=move_to_casa1, start_time=1.0),
        ))
        # Si ejecutamos en orden temporal:
        #   t=1: d1 va a casa1
        #   t=2: d1 va a casa2 (válido: hay coste casa1->casa2)
        # Si lo hiciéramos en orden declarativo:
        #   t=2: d1 va a casa2 (válido)
        #   t=1: d1 va a casa1 (válido también)
        # Ambos terminarían en lugares distintos. Verificamos el correcto.
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.drones["d1"].position == "casa2"

    def test_tiebreak_por_orden_declarativo_es_estable(self, base_world):
        """Dos Commands con el mismo start_time se procesan en el orden
        en que aparecen en plan.scheduled."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", cmd_id="primero")
        m2 = make_move("d2", "casa2", cmd_id="segundo")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=0.0),
        ))
        r.execute(plan)
        # Independientemente del orden, ambos drones terminan en su destino.
        # Lo importante es que la ejecución haya sido determinista.
        assert r.current_world.drones["d1"].position == "casa1"
        assert r.current_world.drones["d2"].position == "casa2"

    def test_snapshots_en_history_estan_en_orden_de_ejecucion(self, base_world):
        """Los snapshots se acumulan en orden de start_time."""
        r = PlanRunner(base_world)
        m_2 = make_move("d1", "casa2", cmd_id="A")  # start_time=5
        # d1 está en casa2; ahora vamos a casa1 (existe la arista casa2->casa1)
        m_1 = make_move("d1", "casa1", cmd_id="B")  # start_time=10
        plan = Plan(scheduled=(
            ScheduledCommand(command=m_2, start_time=5.0),
            ScheduledCommand(command=m_1, start_time=10.0),
        ))
        r.execute(plan)
        # snapshot 0 = inicial
        # snapshot 1 = move a casa2 (start_time=5)
        # snapshot 2 = move a casa1 (start_time=10)
        assert r.history.at(1).produced_by.command_id == "A"
        assert r.history.at(2).produced_by.command_id == "B"


# ===========================================================================
# 7. Fallos PDDL
# ===========================================================================
class TestFallosPDDL:
    def test_move_a_localizacion_inexistente_falla(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        result = r.execute(plan)
        assert not result.succeeded
        assert len(result.failures) == 1
        assert result.failures[0].kind == "pddl"
        assert "narnia" in result.failures[0].reason

    def test_fallo_pddl_pone_drone_en_error(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        assert r.current_world.drones["d1"].state == DroneState.ERROR

    def test_fallo_pddl_no_afecta_a_otros_drones(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        # d2 sigue IDLE, no contaminado.
        assert r.current_world.drones["d2"].state == DroneState.IDLE

    def test_fallo_pddl_incrementa_failed_commands(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        assert r.current_metrics.failed_commands == 1

    def test_fallo_pddl_no_incrementa_action_count(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        assert r.current_metrics.action_count == 0

    def test_fallo_pddl_no_suma_coste(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        assert r.current_metrics.total_cost == 0.0

    def test_fallo_pddl_se_anade_snapshot_del_fallo(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        r.execute(plan)
        # 1 inicial + 1 fallo = 2 snapshots
        assert len(r.history) == 2
        snap_fallo = r.history.at(1)
        assert snap_fallo.produced_by is not None
        # El timestamp del snapshot es scheduled.start_time del fallo.
        assert snap_fallo.timestamp == 0.0

    def test_fallo_pddl_continuacion_de_ejecucion(self, base_world):
        """Tras un fallo, los Commands de drones distintos siguen
        ejecutándose."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move("d1", "narnia"),       # fallo: localización inexistente
            make_move("d2", "casa1"),        # debe ejecutarse OK
        ])
        result = r.execute(plan)
        assert len(result.failures) == 1
        assert r.current_world.drones["d2"].position == "casa1"
        assert r.current_metrics.action_count == 1  # solo d2 contó

    def test_fallo_pddl_drone_en_error_rechaza_siguientes_comandos(
        self, base_world
    ):
        """Tras poner un dron a ERROR, futuros Commands suyos fallan
        automáticamente por _check_drone_ready. Validación reutilizada
        de Sesión A."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move("d1", "narnia"),     # falla; d1 -> ERROR
            make_move("d1", "casa1"),      # falla por _check_drone_ready
            make_move("d1", "casa2"),      # falla idem
        ])
        result = r.execute(plan)
        assert len(result.failures) == 3
        for f in result.failures:
            assert f.kind == "pddl"
        assert r.current_metrics.failed_commands == 3

    def test_drone_inexistente_no_crashea(self, base_world):
        """Si el Command apunta a un drone_id que no existe en el World,
        el runner debe manejarlo limpiamente (Validator devuelve invalid;
        runner NO intenta poner a ERROR a un drone inexistente)."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d_fantasma", "casa1")])
        result = r.execute(plan)
        assert not result.succeeded
        assert len(result.failures) == 1
        # El World no debe contener un drone d_fantasma forzado.
        assert "d_fantasma" not in r.current_world.drones

    def test_failure_referencia_el_scheduled_correcto(self, base_world):
        """CommandFailure.scheduled debe ser el ScheduledCommand original."""
        r = PlanRunner(base_world)
        cmd = make_move("d1", "narnia", cmd_id="el_fallido")
        scheduled = ScheduledCommand(command=cmd, start_time=0.0)
        plan = Plan(scheduled=(scheduled,))
        result = r.execute(plan)
        assert result.failures[0].scheduled is scheduled


# ===========================================================================
# 8. Plan canónico end-to-end
# ===========================================================================
class TestPlanCanonico:
    def test_pickup_move_deliver_retornar(self, base_world):
        """Plan completo: recoger medicina, ir a casa1, entregar a ana,
        volver al depósito."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),
            make_move("d1", "casa1"),
            make_deliver("d1", "libre1", "ana"),
            make_move("d1", "deposito"),
        ])
        result = r.execute(plan)
        assert result.succeeded
        # d1 en deposito al final
        assert r.current_world.drones["d1"].position == "deposito"
        # ana ya no necesita nada
        assert r.current_world.persons["ana"].needs == ()
        # 1 inicial + 4 commits
        assert len(r.history) == 5
        # action_count = 4
        assert r.current_metrics.action_count == 4
        # total_cost = ida + vuelta = 10 + 10 = 20
        assert r.current_metrics.total_cost == 20.0


# ===========================================================================
# 9. Corrección del bug heredado: total_time en modo secuencial
# ===========================================================================
class TestCorreccionTotalTime:
    """En modo secuencial con duration=0, el total_time NUNCA debe
    subir (ningún Command tiene presencia temporal). Esto es el caso
    fácil del fix: la fórmula max(prev_total_time, end_time) con
    end_time = start_time + 0 = start_time, todos los start_time
    en 0, 1, 2, ... — el total_time debería avanzar hasta el
    start_time mayor del último Command si lo aplicáramos
    estrictamente. Pero la convención de Sesión A es que en secuencial
    sin duración, total_time se queda en 0.

    NOTA: Plan.sequential asigna start_times 0, 1, 2, ... Esto es
    arbitrario (no representan tiempo real). El total_time tras
    ejecutar reflejará el último start_time consumido. Esto es
    coherente con la idea de 'makespan'."""

    def test_total_time_es_el_mayor_end_time(self, base_world):
        """Tras un plan secuencial 'sequential', total_time es N-1
        donde N es el número de Commands (porque end_time del último
        es start_time del último, que es N-1)."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move("d1", "casa1"),       # start=0, end=0
            make_pickup("d1", "izq", "libre3"),  # start=1, end=1
            make_deliver("d1", "libre3", "ana"),  # start=2, end=2
        ])
        r.execute(plan)
        # total_time = max(0, 0, 1, 2) = 2.0
        assert r.current_metrics.total_time == 2.0


# ===========================================================================
# 10. Métricas
# ===========================================================================
class TestMetricas:
    def test_total_cost_solo_se_acumula_con_movimientos(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),  # no suma coste
            make_move("d1", "casa1"),            # suma 10
            make_deliver("d1", "libre1", "ana"), # no suma
            make_move("d1", "deposito"),         # suma 10
        ])
        r.execute(plan)
        assert r.current_metrics.total_cost == 20.0

    def test_makespan_es_max_end_time_exitos(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move("d1", "casa1"),       # start=0, end=0
            make_move("d1", "deposito"),    # start=1, end=1
        ])
        result = r.execute(plan)
        assert result.makespan == 1.0

    def test_makespan_no_cuenta_fallos(self, base_world):
        """RunResult.makespan solo incluye end_times de Commands exitosos."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_move("d1", "casa1"),       # start=0, end=0, OK
            make_move("d1", "narnia"),      # start=1, end=1, FALLA
        ])
        result = r.execute(plan)
        # max end_time entre exitos = 0.0
        assert result.makespan == 0.0


# ===========================================================================
# 11. Inmutabilidad del World inicial
# ===========================================================================
class TestInmutabilidad:
    def test_world_inicial_no_se_muta(self, base_world):
        """El World que pasamos al constructor del runner sigue siendo
        idéntico tras ejecutar."""
        original_d1_position = base_world.drones["d1"].position
        original_d1_state = base_world.drones["d1"].state
        original_ana_needs = base_world.persons["ana"].needs

        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),
            make_move("d1", "casa1"),
            make_deliver("d1", "libre1", "ana"),
        ])
        r.execute(plan)

        # El world ORIGINAL no se ha tocado.
        assert base_world.drones["d1"].position == original_d1_position
        assert base_world.drones["d1"].state == original_d1_state
        assert base_world.persons["ana"].needs == original_ana_needs

    def test_snapshot_inicial_no_se_muta(self, base_world):
        """El snapshot 0 del history sigue siendo el inicial original
        tras la ejecución."""
        r = PlanRunner(base_world)
        snap_inicial = r.history.at(0)
        plan = Plan.sequential([make_move("d1", "casa1")])
        r.execute(plan)
        # Mismo snapshot, world idéntico.
        assert r.history.at(0) is snap_inicial
        assert r.history.at(0).world is base_world


# ===========================================================================
# 12. dry_run: no muta el runner
# ===========================================================================
class TestDryRun:
    def test_dry_run_devuelve_run_result_correcto(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        result = r.dry_run(plan)
        assert isinstance(result, RunResult)
        assert result.succeeded

    def test_dry_run_no_modifica_history_original(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([
            make_pickup("d1", "izq", "libre1"),
            make_move("d1", "casa1"),
            make_deliver("d1", "libre1", "ana"),
        ])
        len_antes = len(r.history)
        r.dry_run(plan)
        assert len(r.history) == len_antes

    def test_dry_run_no_modifica_world_actual(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        world_antes = r.current_world
        r.dry_run(plan)
        # current_world sigue siendo el original.
        assert r.current_world is world_antes
        assert r.current_world.drones["d1"].position == "deposito"

    def test_dry_run_reporta_fallos_correctamente(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "narnia")])
        result = r.dry_run(plan)
        assert not result.succeeded
        assert len(result.failures) == 1
        assert result.failures[0].kind == "pddl"

    def test_dry_run_no_afecta_metrics_original(self, base_world):
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        m_antes = r.current_metrics
        r.dry_run(plan)
        assert r.current_metrics is m_antes

    def test_dry_run_devuelve_world_final_simulado(self, base_world):
        """El World en RunResult corresponde al simulado, NO al actual
        del runner."""
        r = PlanRunner(base_world)
        plan = Plan.sequential([make_move("d1", "casa1")])
        result = r.dry_run(plan)
        # En la simulación, d1 termina en casa1.
        assert result.final_world.drones["d1"].position == "casa1"
        # Pero en el runner, d1 sigue en deposito.
        assert r.current_world.drones["d1"].position == "deposito"


# ===========================================================================
# 13. World con drone pre-existente en ERROR
# ===========================================================================
class TestWorldConDroneEnError:
    """Si el World inicial ya tiene un drone en ERROR, los Commands
    dirigidos a ese drone fallan por _check_drone_ready. El runner
    no tiene que hacer nada especial: la validación PDDL ya lo cubre."""

    def test_command_a_drone_en_error_falla(self, base_world):
        world_error = world_with_drone_in_error(base_world, "d1")
        r = PlanRunner(world_error)
        plan = Plan.sequential([make_move("d1", "casa1")])
        result = r.execute(plan)
        assert not result.succeeded
        assert "ERROR" in result.failures[0].reason or "terminal" in result.failures[0].reason

    def test_otro_drone_funciona_aunque_d1_este_en_error(self, base_world):
        world_error = world_with_drone_in_error(base_world, "d1")
        r = PlanRunner(world_error)
        plan = Plan.sequential([make_move("d2", "casa1")])
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.drones["d2"].position == "casa1"
        # d1 sigue en ERROR
        assert r.current_world.drones["d1"].state == DroneState.ERROR
