"""Tests de runtime/result.py: CommandFailure y RunResult.

Estas son dataclasses pasivas; los tests cubren construcción,
inmutabilidad, igualdad estructural, propiedades derivadas y exportes
del paquete. No hay lógica de runtime aquí: el PlanRunner que las
produce se testea en sus propios archivos en pasos posteriores.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from droneplan_viz.commands import Move, PickUp
from droneplan_viz.domain import MetricsTracker, World
from droneplan_viz.runtime import (
    CommandFailure,
    FailureKind,
    Plan,
    RunResult,
    ScheduledCommand,
)


# ===========================================================================
# CommandFailure: construcción y inmutabilidad
# ===========================================================================
class TestCommandFailureConstruccion:
    def test_construccion_basica_pddl(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(
            scheduled=sched, reason="el drone no existe", kind="pddl"
        )
        assert f.scheduled is sched
        assert f.reason == "el drone no existe"
        assert f.kind == "pddl"

    def test_construccion_concurrency(self):
        cmd = Move(drone_id="d1", destination_id="casa1", duration=10.0)
        sched = ScheduledCommand(command=cmd, start_time=5.0)
        f = CommandFailure(
            scheduled=sched,
            reason="drone d1 ocupado en [0, 8)",
            kind="concurrency",
        )
        assert f.kind == "concurrency"

    def test_construccion_structural(self):
        """La categoría structural existe pero no la usa el runner por
        ahora; está reservada para usos futuros y debe construirse."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(
            scheduled=sched, reason="dummy", kind="structural"
        )
        assert f.kind == "structural"

    def test_referencia_directa_al_scheduled(self):
        """No deepcopy. CommandFailure mantiene referencia al objeto."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        assert f.scheduled is sched
        assert f.scheduled.command is cmd


class TestCommandFailureInmutabilidad:
    def test_no_se_puede_reasignar_kind(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        with pytest.raises(FrozenInstanceError):
            f.kind = "concurrency"  # type: ignore[misc]

    def test_no_se_puede_reasignar_reason(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        with pytest.raises(FrozenInstanceError):
            f.reason = "otra cosa"  # type: ignore[misc]

    def test_slots_rechaza_nuevos_atributos(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        with pytest.raises((AttributeError, TypeError)):
            f.extra = "z"  # type: ignore[attr-defined]


class TestCommandFailureIgualdad:
    def test_igualdad_estructural(self):
        cmd = Move(
            drone_id="d1", destination_id="casa1", command_id="fixed"
        )
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f1 = CommandFailure(scheduled=sched, reason="r", kind="pddl")
        f2 = CommandFailure(scheduled=sched, reason="r", kind="pddl")
        assert f1 == f2

    def test_desigualdad_por_kind(self):
        cmd = Move(drone_id="d1", destination_id="casa1", command_id="fix")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f1 = CommandFailure(scheduled=sched, reason="r", kind="pddl")
        f2 = CommandFailure(scheduled=sched, reason="r", kind="concurrency")
        assert f1 != f2

    def test_desigualdad_por_reason(self):
        cmd = Move(drone_id="d1", destination_id="casa1", command_id="fix")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        f1 = CommandFailure(scheduled=sched, reason="a", kind="pddl")
        f2 = CommandFailure(scheduled=sched, reason="b", kind="pddl")
        assert f1 != f2


# ===========================================================================
# RunResult: construcción
# ===========================================================================
class TestRunResultConstruccion:
    def test_construccion_minima_plan_vacio(self):
        """Plan vacío, world vacío, sin fallos. Caso degenerado pero
        constructible."""
        plan = Plan()
        world = World()
        metrics = MetricsTracker()
        r = RunResult(plan=plan, final_world=world, final_metrics=metrics)
        assert r.plan is plan
        assert r.final_world is world
        assert r.final_metrics is metrics
        assert r.failures == ()
        assert r.history_length == 1
        assert r.makespan == 0.0

    def test_construccion_con_failures(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        failure = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            failures=(failure,),
        )
        assert r.failures == (failure,)
        assert len(r.failures) == 1

    def test_construccion_con_history_length_custom(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            history_length=10,
        )
        assert r.history_length == 10

    def test_construccion_con_makespan_custom(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            makespan=42.5,
        )
        assert r.makespan == 42.5


# ===========================================================================
# RunResult.succeeded (propiedad derivada)
# ===========================================================================
class TestRunResultSucceeded:
    def test_succeeded_true_cuando_no_hay_failures(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
        )
        assert r.succeeded is True

    def test_succeeded_false_con_un_failure(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        failure = CommandFailure(scheduled=sched, reason="x", kind="pddl")
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            failures=(failure,),
        )
        assert r.succeeded is False

    def test_succeeded_false_con_varios_failures(self):
        cmd1 = Move(
            drone_id="d1", destination_id="casa1", command_id="a"
        )
        cmd2 = PickUp(
            drone_id="d1", arm_id="izq", package_id="p1", command_id="b"
        )
        s1 = ScheduledCommand(command=cmd1, start_time=0.0)
        s2 = ScheduledCommand(command=cmd2, start_time=1.0)
        f1 = CommandFailure(scheduled=s1, reason="x", kind="pddl")
        f2 = CommandFailure(scheduled=s2, reason="y", kind="concurrency")
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            failures=(f1, f2),
        )
        assert r.succeeded is False


# ===========================================================================
# RunResult: inmutabilidad
# ===========================================================================
class TestRunResultInmutabilidad:
    def test_no_se_puede_reasignar_failures(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
        )
        with pytest.raises(FrozenInstanceError):
            r.failures = ()  # type: ignore[misc]

    def test_no_se_puede_reasignar_makespan(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
        )
        with pytest.raises(FrozenInstanceError):
            r.makespan = 99.0  # type: ignore[misc]

    def test_slots_rechaza_nuevos_atributos(self):
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
        )
        with pytest.raises((AttributeError, TypeError)):
            r.extra = 1  # type: ignore[attr-defined]

    def test_replace_produce_nueva_instancia(self):
        """dataclasses.replace funciona sobre frozen+slots y deja la
        original intacta. Patrón idiomático para 'modificar' inmutables."""
        r = RunResult(
            plan=Plan(),
            final_world=World(),
            final_metrics=MetricsTracker(),
            makespan=5.0,
        )
        r2 = replace(r, makespan=10.0)
        assert r.makespan == 5.0  # original intacto
        assert r2.makespan == 10.0
        assert r is not r2


# ===========================================================================
# RunResult: igualdad estructural
# ===========================================================================
class TestRunResultIgualdad:
    def test_igualdad_dos_resultados_equivalentes(self):
        plan = Plan()
        world = World()
        metrics = MetricsTracker()
        r1 = RunResult(plan=plan, final_world=world, final_metrics=metrics)
        r2 = RunResult(plan=plan, final_world=world, final_metrics=metrics)
        assert r1 == r2

    def test_desigualdad_por_makespan(self):
        plan = Plan()
        world = World()
        metrics = MetricsTracker()
        r1 = RunResult(
            plan=plan, final_world=world,
            final_metrics=metrics, makespan=0.0,
        )
        r2 = RunResult(
            plan=plan, final_world=world,
            final_metrics=metrics, makespan=1.0,
        )
        assert r1 != r2


# ===========================================================================
# Tipos del módulo (FailureKind)
# ===========================================================================
class TestFailureKind:
    """FailureKind es un Literal expuesto para tipado externo.

    Verificamos que los tres valores son utilizables como kind="..."
    en CommandFailure. (En runtime, FailureKind no aporta más que
    documentación; pyright/mypy es quien verifica la corrección.)
    """

    def test_pddl_es_kind_valido(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        kind: FailureKind = "pddl"
        f = CommandFailure(scheduled=sched, reason="x", kind=kind)
        assert f.kind == "pddl"

    def test_concurrency_es_kind_valido(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        kind: FailureKind = "concurrency"
        f = CommandFailure(scheduled=sched, reason="x", kind=kind)
        assert f.kind == "concurrency"

    def test_structural_es_kind_valido(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        kind: FailureKind = "structural"
        f = CommandFailure(scheduled=sched, reason="x", kind=kind)
        assert f.kind == "structural"


# ===========================================================================
# Exportes del paquete runtime
# ===========================================================================
class TestExportes:
    """Verifica que los nuevos tipos están en la superficie pública."""

    def test_command_failure_se_importa(self):
        from droneplan_viz.runtime import CommandFailure as Imported
        assert Imported is CommandFailure

    def test_run_result_se_importa(self):
        from droneplan_viz.runtime import RunResult as Imported
        assert Imported is RunResult

    def test_failure_kind_se_importa(self):
        from droneplan_viz.runtime import FailureKind as Imported
        assert Imported is FailureKind

    def test_estan_en_all(self):
        import droneplan_viz.runtime as runtime_pkg
        assert "CommandFailure" in runtime_pkg.__all__
        assert "RunResult" in runtime_pkg.__all__
        assert "FailureKind" in runtime_pkg.__all__
