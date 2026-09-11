"""Tests del módulo droneplan_viz_app.scenarios.

Estos tests verifican el CONTRATO de las factorías de escenarios:
ejecutables sin excepciones por PlanRunner, con la topología documentada,
y produciendo el resultado esperado (éxito o fallo según corresponda).

Los detalles internos (qué edges hay en el grafo, qué duración tiene cada
Command) los verifico de manera mínima — el escenario es material docente
ajustable, no API pública estable. Lo crítico es:

  1. El world está bien construido (validate() no lanza).
  2. El plan es estructuralmente válido (Plan(...) no lanza en __post_init__).
  3. PlanRunner.execute() lo procesa sin excepción.
  4. El RunResult cumple las invariantes del escenario (succeeded sí/no,
     número de fallos esperado, makespan razonable).
  5. El SCENARIOS dict expone los nombres esperados.

Sin pygame: estos tests son lógica pura sobre domain/commands/runtime.
"""
from __future__ import annotations

import pytest

from droneplan_viz.runtime import PlanRunner
from droneplan_viz_app.scenarios import (
    SCENARIOS,
    _build_demo_world,
    build_demo_scenario,
    build_failure_demo_scenario,
)


# ---------------------------------------------------------------------------
# build_demo_scenario: escenario feliz
# ---------------------------------------------------------------------------


class TestBuildDemoScenario:
    def test_devuelve_world_y_plan(self):
        result = build_demo_scenario()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_world_tiene_topologia_documentada(self):
        world, _plan = build_demo_scenario()
        # 4 locations, 2 drones, 1 transporter, 2 persons, 3 packages, 3 contents.
        assert set(world.locations.keys()) == {"casa1", "casa2", "deposito", "hospital"}
        assert set(world.drones.keys()) == {"d1", "d2"}
        assert set(world.transporters.keys()) == {"t1"}
        assert set(world.persons.keys()) == {"p1", "p2"}
        assert set(world.packages.keys()) == {"pkg_med1", "pkg_food1", "pkg_water1"}
        assert set(world.contents.keys()) == {"medicina", "comida", "agua"}

    def test_d1_dos_brazos_d2_un_brazo(self):
        world, _plan = build_demo_scenario()
        assert len(world.drones["d1"].arms) == 2
        assert len(world.drones["d2"].arms) == 1

    def test_plan_tiene_tres_commands(self):
        _world, plan = build_demo_scenario()
        assert len(plan.scheduled) == 3

    def test_planrunner_ejecuta_sin_fallos(self):
        world, plan = build_demo_scenario()
        result = PlanRunner(world).execute(plan)
        assert result.succeeded is True
        assert result.failures == ()

    def test_makespan_18s(self):
        """PickUp 5s a t=0, Move 8s a t=5 (end=13), Deliver 5s a t=13
        (end=18). Makespan: 18.0."""
        world, plan = build_demo_scenario()
        result = PlanRunner(world).execute(plan)
        assert result.makespan == pytest.approx(18.0)

    def test_historial_no_vacio_tras_ejecutar(self):
        """No fijo el número exacto (depende de la política de snapshots
        duales del runner, decisión §2.7 de Sesión D); solo que ejecutar
        el plan deja una serie de snapshots cuyo número crece con la
        cantidad de Commands.
        """
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        # snapshot inicial + algunos más.
        assert len(runner.history) > 1

    def test_funcion_es_pura_invocaciones_repetidas_dan_estado_igual(self):
        """Llamar varias veces devuelve worlds estructuralmente iguales.
        Cierra un canario implícito: si alguien añadiese estado de
        módulo o aleatoriedad, este test caería.
        """
        world_a, plan_a = build_demo_scenario()
        world_b, plan_b = build_demo_scenario()
        # No comparamos identidad (son objetos nuevos) pero sí el
        # contenido relevante.
        assert set(world_a.drones.keys()) == set(world_b.drones.keys())
        assert len(plan_a.scheduled) == len(plan_b.scheduled)
        for sa, sb in zip(plan_a.scheduled, plan_b.scheduled):
            assert sa.start_time == sb.start_time
            assert sa.command.command_id == sb.command.command_id


# ---------------------------------------------------------------------------
# build_failure_demo_scenario: escenario didáctico mixto
# ---------------------------------------------------------------------------


class TestBuildFailureDemoScenario:
    def test_devuelve_world_y_plan(self):
        result = build_failure_demo_scenario()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mismas_locations_que_escenario_feliz(self):
        """Misma topología, distinto plan."""
        world_ok, _ = build_demo_scenario()
        world_fail, _ = build_failure_demo_scenario()
        assert set(world_ok.locations.keys()) == set(world_fail.locations.keys())
        assert set(world_ok.drones.keys()) == set(world_fail.drones.keys())

    def test_plan_tiene_cuatro_commands(self):
        """Un éxito, un fallo, dos éxitos posteriores."""
        _world, plan = build_failure_demo_scenario()
        assert len(plan.scheduled) == 4

    def test_planrunner_ejecuta_y_devuelve_failed(self):
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        assert result.succeeded is False
        assert len(result.failures) == 1

    def test_el_fallo_es_de_kind_pddl(self):
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        (failure,) = result.failures  # exactamente uno
        assert failure.kind == "pddl"

    def test_el_fallo_apunta_al_command_correcto(self):
        """El fallo identificado debe ser el PickUp d2/pkg_water1, no
        otro. Defiende contra reordenaciones accidentales del plan que
        rompan la lección didáctica."""
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        (failure,) = result.failures
        assert failure.scheduled.command.command_id == "fail_pickup_water"
        assert failure.scheduled.start_time == pytest.approx(2.0)

    def test_el_fallo_menciona_co_localizacion(self):
        """El mensaje del Validator es 'no están co-localizados'.
        Probarlo blindamos que el escenario sigue produciendo ESE fallo
        concreto (y no, por ejemplo, 'paquete no existe' que sería un
        bug de construcción del plan)."""
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        (failure,) = result.failures
        assert "co-localizados" in failure.reason

    def test_runner_no_aborta_los_commands_posteriores(self):
        """Tras el fallo de t=2, los Commands de d1 a t=5 y t=13 siguen
        ejecutándose y dejan p1 con la medicina. Documenta la propiedad
        no-cancelativa del runner (relevante para la UI: el usuario verá
        animación tras el fallo)."""
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        # action_count incluye los Commands aplicados con éxito.
        # 3 éxitos (PickUp d1, Move d1, Deliver d1).
        assert result.final_metrics.action_count == 3
        assert result.final_metrics.failed_commands == 1


# ---------------------------------------------------------------------------
# Registro SCENARIOS
# ---------------------------------------------------------------------------


class TestScenariosRegistry:
    def test_expone_demo_y_failure_demo(self):
        assert "demo" in SCENARIOS
        assert "failure_demo" in SCENARIOS

    def test_callables_devuelven_world_plan(self):
        for name, factory in SCENARIOS.items():
            world, plan = factory()
            # smoke check de los tipos: si esto falla, el registro está mal.
            assert hasattr(world, "drones"), f"factoría '{name}' no devuelve World"
            assert hasattr(plan, "scheduled"), f"factoría '{name}' no devuelve Plan"

    def test_no_factorias_inesperadas(self):
        """Cierra el registro: añadir una factoría requiere modificar
        esta lista, lo que actúa de checklist para que también se añadan
        sus tests y se documente en la memoria."""
        assert set(SCENARIOS.keys()) == {"demo", "failure_demo"}


# ---------------------------------------------------------------------------
# Migración de build_demo_scenario a la fachada (Sesión F)
# ---------------------------------------------------------------------------


class TestDemoConstruidoConFachada:
    """build_demo_scenario se construye ahora vía DronePlanViz. Estos tests
    blindan que la migración reproduce EXACTAMENTE el escenario anterior:
    cualquier divergencia entre la versión por fachada y la manual de
    referencia (_build_demo_world) cae aquí."""

    def test_demo_world_igual_a_construccion_manual(self):
        """El World montado por la fachada es estructuralmente idéntico al
        de _build_demo_world() (la referencia hecha a mano que conserva el
        camino antiguo)."""
        world_fachada, _plan = build_demo_scenario()
        assert world_fachada == _build_demo_world()

    def test_demo_plan_conserva_ids_originales(self):
        _world, plan = build_demo_scenario()
        ids = [sc.command.command_id for sc in plan.scheduled]
        assert ids == ["step_pickup", "step_move", "step_deliver"]

    def test_demo_plan_conserva_tiempos_y_duraciones(self):
        _world, plan = build_demo_scenario()
        starts = [sc.start_time for sc in plan.scheduled]
        durs = [sc.command.duration for sc in plan.scheduled]
        assert starts == [0.0, 5.0, 13.0]
        assert durs == [5.0, 8.0, 5.0]

    def test_failure_demo_sigue_montado_a_mano_misma_topologia(self):
        """El escenario de fallo conserva la misma topología (comparte la
        construcción manual _build_demo_world)."""
        world_fail, _plan = build_failure_demo_scenario()
        assert world_fail == _build_demo_world()
