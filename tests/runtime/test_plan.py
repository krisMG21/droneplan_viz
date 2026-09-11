"""Tests del modelado del plan: ScheduledCommand, Plan, PlanValidationError."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.runtime.plan import (
    Plan,
    PlanValidationError,
    ScheduledCommand,
)


# ===========================================================================
# ScheduledCommand
# ===========================================================================
class TestScheduledCommand:
    """ScheduledCommand: dataclass frozen inmutable con end_time derivado."""

    def test_construccion_basica(self):
        cmd = Move(drone_id="d1", destination_id="casa1", duration=10.0)
        sched = ScheduledCommand(command=cmd, start_time=5.0)
        assert sched.command is cmd
        assert sched.start_time == 5.0

    def test_end_time_derivado(self):
        cmd = Move(drone_id="d1", destination_id="casa1", duration=10.0)
        sched = ScheduledCommand(command=cmd, start_time=5.0)
        assert sched.end_time == 15.0

    def test_end_time_con_duration_cero(self):
        """Acción atemporal: end_time == start_time. Con la convención
        cerrado-abierto, la acción nunca está 'activa' (intervalo vacío)
        pero está completada desde start_time."""
        cmd = Move(drone_id="d1", destination_id="casa1", duration=0.0)
        sched = ScheduledCommand(command=cmd, start_time=3.0)
        assert sched.end_time == 3.0

    def test_end_time_en_t_cero(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="p1", duration=5.0)
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        assert sched.end_time == 5.0

    def test_inmutabilidad_start_time(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        with pytest.raises(FrozenInstanceError):
            sched.start_time = 99.0  # type: ignore[misc]

    def test_inmutabilidad_command(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        with pytest.raises(FrozenInstanceError):
            sched.command = cmd  # type: ignore[misc]

    def test_slots_no_admiten_nuevos_atributos(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        # frozen + slots: TypeError o AttributeError, ambos aceptables.
        with pytest.raises((AttributeError, TypeError)):
            sched.nuevo_campo = "foo"  # type: ignore[attr-defined]

    def test_igualdad_estructural(self):
        cmd = Move(
            drone_id="d1", destination_id="casa1",
            command_id="fixed_id",  # importante: id fijo para igualdad
        )
        s1 = ScheduledCommand(command=cmd, start_time=2.0)
        s2 = ScheduledCommand(command=cmd, start_time=2.0)
        assert s1 == s2

    def test_desigualdad_por_start_time(self):
        cmd = Move(drone_id="d1", destination_id="casa1", command_id="x")
        s1 = ScheduledCommand(command=cmd, start_time=2.0)
        s2 = ScheduledCommand(command=cmd, start_time=3.0)
        assert s1 != s2

    def test_referencia_directa_al_command_no_copy(self):
        """El ScheduledCommand referencia el Command, no lo copia."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        assert sched.command is cmd

    def test_funciona_con_los_seis_tipos_de_command(self):
        """ScheduledCommand acepta cualquier Command del Protocol."""
        commands = [
            Move(drone_id="d1", destination_id="casa1"),
            MoveWithTransporter(
                drone_id="d1", transporter_id="t1", destination_id="casa1"
            ),
            PickUp(drone_id="d1", arm_id="izq", package_id="p1"),
            Deliver(drone_id="d1", package_id="p1", person_id="ana"),
            LoadIntoTransporter(
                drone_id="d1", package_id="p1", transporter_id="t1"
            ),
            UnloadFromTransporter(
                drone_id="d1", arm_id="izq",
                package_id="p1", transporter_id="t1",
            ),
        ]
        for c in commands:
            sched = ScheduledCommand(command=c, start_time=1.0)
            assert sched.command is c
            assert sched.end_time == 1.0 + c.duration


# ===========================================================================
# Plan - construcción válida
# ===========================================================================
class TestPlanConstruccionValida:
    """Casos en los que Plan se construye sin error."""

    def test_plan_vacio(self):
        p = Plan()
        assert len(p) == 0
        assert tuple(p) == ()

    def test_plan_vacio_explicito(self):
        p = Plan(scheduled=())
        assert len(p) == 0

    def test_plan_un_command(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        p = Plan(scheduled=(sched,))
        assert len(p) == 1
        assert p.scheduled[0] is sched

    def test_plan_varios_commands(self):
        cmds = [
            Move(drone_id="d1", destination_id="casa1"),
            PickUp(drone_id="d1", arm_id="izq", package_id="p1"),
            Deliver(drone_id="d1", package_id="p1", person_id="ana"),
        ]
        scheds = tuple(
            ScheduledCommand(command=c, start_time=float(i))
            for i, c in enumerate(cmds)
        )
        p = Plan(scheduled=scheds)
        assert len(p) == 3

    def test_plan_admite_start_time_cero(self):
        """t=0.0 es válido (frontera inferior inclusiva)."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        Plan(scheduled=(sched,))  # no lanza

    def test_plan_admite_duration_cero(self):
        """duration=0.0 es válido (parte 1-2 del PDDL)."""
        cmd = Move(drone_id="d1", destination_id="casa1", duration=0.0)
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        Plan(scheduled=(sched,))  # no lanza

    def test_plan_admite_timestamps_no_ordenados(self):
        """El Plan no exige que los ScheduledCommand estén ordenados por
        start_time; el runtime se encargará de ordenar al ejecutar."""
        c1 = Move(drone_id="d1", destination_id="casa1", command_id="m1")
        c2 = Move(drone_id="d1", destination_id="casa1", command_id="m2")
        scheds = (
            ScheduledCommand(command=c1, start_time=5.0),
            ScheduledCommand(command=c2, start_time=2.0),
        )
        Plan(scheduled=scheds)  # no lanza


# ===========================================================================
# Plan - validación estructural: start_time negativo
# ===========================================================================
class TestPlanStartTimeNegativo:
    def test_un_command_start_negativo(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=-0.001)
        with pytest.raises(PlanValidationError, match="start_time"):
            Plan(scheduled=(sched,))

    def test_un_command_start_muy_negativo(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        sched = ScheduledCommand(command=cmd, start_time=-100.0)
        with pytest.raises(PlanValidationError, match="negativ"):
            Plan(scheduled=(sched,))

    def test_segundo_command_negativo_se_detecta(self):
        """El error indica la posición del Command problemático."""
        c1 = Move(drone_id="d1", destination_id="casa1", command_id="m1")
        c2 = Move(drone_id="d1", destination_id="casa1", command_id="m2")
        scheds = (
            ScheduledCommand(command=c1, start_time=0.0),
            ScheduledCommand(command=c2, start_time=-1.0),
        )
        with pytest.raises(PlanValidationError, match="posición 1"):
            Plan(scheduled=scheds)


# ===========================================================================
# Plan - validación estructural: duration negativa
# ===========================================================================
class TestPlanDurationNegativa:
    def test_command_con_duration_negativa(self):
        cmd = Move(drone_id="d1", destination_id="casa1", duration=-5.0)
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        with pytest.raises(PlanValidationError, match="duration"):
            Plan(scheduled=(sched,))

    def test_command_con_duration_negativa_pequena(self):
        cmd = Move(drone_id="d1", destination_id="casa1", duration=-0.0001)
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        with pytest.raises(PlanValidationError, match="duration"):
            Plan(scheduled=(sched,))

    def test_duration_cero_no_se_rechaza(self):
        """Frontera inferior inclusiva: duration=0.0 es legal."""
        cmd = Move(drone_id="d1", destination_id="casa1", duration=0.0)
        sched = ScheduledCommand(command=cmd, start_time=0.0)
        Plan(scheduled=(sched,))  # no lanza


# ===========================================================================
# Plan - validación estructural: command_id únicos
# ===========================================================================
class TestPlanCommandIdUnicos:
    def test_ids_duplicados_explicitos_fallan(self):
        c1 = Move(drone_id="d1", destination_id="casa1", command_id="repe")
        c2 = Move(drone_id="d1", destination_id="casa2", command_id="repe")
        scheds = (
            ScheduledCommand(command=c1, start_time=0.0),
            ScheduledCommand(command=c2, start_time=1.0),
        )
        with pytest.raises(PlanValidationError, match="duplicado"):
            Plan(scheduled=scheds)

    def test_dos_commands_iguales_estructuralmente_pero_distintos_ids_pasan(self):
        """Por defecto, dos Move(d1, casa1) reciben ids distintos (uuid4),
        así que el plan es válido aunque estructuralmente sean 'idénticos'.
        Esto es deliberado: dos pasos de plan que aplican la misma acción
        no deben colapsar en el historial."""
        c1 = Move(drone_id="d1", destination_id="casa1")
        c2 = Move(drone_id="d1", destination_id="casa1")
        assert c1.command_id != c2.command_id  # uuid4 por defecto
        scheds = (
            ScheduledCommand(command=c1, start_time=0.0),
            ScheduledCommand(command=c2, start_time=1.0),
        )
        Plan(scheduled=scheds)  # no lanza

    def test_id_duplicado_en_tercer_command_se_detecta_con_posicion(self):
        c1 = Move(drone_id="d1", destination_id="casa1", command_id="a")
        c2 = Move(drone_id="d1", destination_id="casa1", command_id="b")
        c3 = Move(drone_id="d1", destination_id="casa1", command_id="a")
        scheds = (
            ScheduledCommand(command=c1, start_time=0.0),
            ScheduledCommand(command=c2, start_time=1.0),
            ScheduledCommand(command=c3, start_time=2.0),
        )
        with pytest.raises(PlanValidationError, match="posición 2"):
            Plan(scheduled=scheds)


# ===========================================================================
# Plan - dunder methods
# ===========================================================================
class TestPlanDunderMethods:
    def test_len_vacio(self):
        assert len(Plan()) == 0

    def test_len_no_vacio(self):
        cmds = [
            Move(drone_id="d1", destination_id="casa1"),
            Move(drone_id="d1", destination_id="casa2"),
        ]
        scheds = tuple(
            ScheduledCommand(command=c, start_time=float(i))
            for i, c in enumerate(cmds)
        )
        assert len(Plan(scheduled=scheds)) == 2

    def test_iter_preserva_orden_de_declaracion(self):
        """Plan itera en orden de declaración, NO de start_time. El
        runtime es quien reordena temporalmente al ejecutar."""
        c1 = Move(drone_id="d1", destination_id="casa1", command_id="primero")
        c2 = Move(drone_id="d1", destination_id="casa2", command_id="segundo")
        scheds = (
            # Aposta: primero en orden declarativo está más adelante en t.
            ScheduledCommand(command=c1, start_time=10.0),
            ScheduledCommand(command=c2, start_time=2.0),
        )
        p = Plan(scheduled=scheds)
        result = list(p)
        assert result[0].command.command_id == "primero"
        assert result[1].command.command_id == "segundo"

    def test_es_iterable_dos_veces(self):
        """No es un iterador consumible; cada iteración empieza de nuevo."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        p = Plan(scheduled=(ScheduledCommand(command=cmd, start_time=0.0),))
        assert list(p) == list(p)  # dos pasadas iguales


# ===========================================================================
# Plan - inmutabilidad
# ===========================================================================
class TestPlanInmutabilidad:
    def test_plan_es_frozen(self):
        p = Plan()
        with pytest.raises(FrozenInstanceError):
            p.scheduled = ()  # type: ignore[misc]

    def test_plan_no_admite_nuevos_atributos(self):
        p = Plan()
        with pytest.raises((AttributeError, TypeError)):
            p.nuevo = "x"  # type: ignore[attr-defined]


# ===========================================================================
# Plan.sequential factory
# ===========================================================================
class TestPlanSequential:
    def test_sequential_vacia(self):
        p = Plan.sequential([])
        assert len(p) == 0

    def test_sequential_un_command(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        p = Plan.sequential([cmd])
        assert len(p) == 1
        assert p.scheduled[0].command is cmd
        assert p.scheduled[0].start_time == 0.0

    def test_sequential_varios_commands_timestamps_consecutivos(self):
        cmds = [
            Move(drone_id="d1", destination_id="casa1"),
            PickUp(drone_id="d1", arm_id="izq", package_id="p1"),
            Deliver(drone_id="d1", package_id="p1", person_id="ana"),
        ]
        p = Plan.sequential(cmds)
        assert [s.start_time for s in p.scheduled] == [0.0, 1.0, 2.0]

    def test_sequential_preserva_orden_de_iterable(self):
        cmds = [
            Move(drone_id="d1", destination_id="casa1", command_id="a"),
            Move(drone_id="d1", destination_id="casa2", command_id="b"),
            Move(drone_id="d1", destination_id="casa3", command_id="c"),
        ]
        p = Plan.sequential(cmds)
        ids = [s.command.command_id for s in p.scheduled]
        assert ids == ["a", "b", "c"]

    def test_sequential_acepta_generador(self):
        """El argumento es Iterable, no list. Acepta cualquier iterable."""
        def gen():
            yield Move(drone_id="d1", destination_id="casa1")
            yield Move(drone_id="d1", destination_id="casa2")
        p = Plan.sequential(gen())
        assert len(p) == 2

    def test_sequential_falla_con_ids_duplicados(self):
        """Si el caller pasa Commands con ids manuales duplicados, el
        plan sigue aplicando su validación estructural."""
        cmds = [
            Move(drone_id="d1", destination_id="casa1", command_id="dup"),
            Move(drone_id="d1", destination_id="casa2", command_id="dup"),
        ]
        with pytest.raises(PlanValidationError, match="duplicado"):
            Plan.sequential(cmds)

    def test_sequential_devuelve_plan_inmutable(self):
        p = Plan.sequential([Move(drone_id="d1", destination_id="casa1")])
        with pytest.raises(FrozenInstanceError):
            p.scheduled = ()  # type: ignore[misc]

    def test_sequential_timestamps_son_float_no_int(self):
        """Asegura que start_time es float incluso cuando vienen de un
        enumerate (que produce ints). Coherente con la firma del tipo."""
        cmds = [Move(drone_id="d1", destination_id="casa1")]
        p = Plan.sequential(cmds)
        assert isinstance(p.scheduled[0].start_time, float)


# ===========================================================================
# PlanValidationError
# ===========================================================================
class TestPlanValidationError:
    def test_es_subclase_de_value_error(self):
        """Permite a callers genéricos que capturan ValueError reaccionar
        correctamente."""
        assert issubclass(PlanValidationError, ValueError)

    def test_se_puede_lanzar_con_mensaje(self):
        with pytest.raises(PlanValidationError, match="mi mensaje"):
            raise PlanValidationError("mi mensaje")

    def test_capturable_como_value_error(self):
        with pytest.raises(ValueError):
            raise PlanValidationError("foo")


# ===========================================================================
# Integración mínima: Plan con todos los tipos de Command
# ===========================================================================
class TestPlanConTodosLosTipos:
    """Verifica que Plan no discrimina entre tipos de Command."""

    def test_plan_mezcla_los_seis_tipos(self):
        commands = [
            Move(drone_id="d1", destination_id="casa1", duration=10.0),
            MoveWithTransporter(
                drone_id="d1", transporter_id="t1",
                destination_id="casa2", duration=15.0,
            ),
            PickUp(
                drone_id="d1", arm_id="izq",
                package_id="p1", duration=5.0,
            ),
            Deliver(
                drone_id="d1", package_id="p1",
                person_id="ana", duration=5.0,
            ),
            LoadIntoTransporter(
                drone_id="d1", package_id="p2",
                transporter_id="t1", duration=5.0,
            ),
            UnloadFromTransporter(
                drone_id="d1", arm_id="der",
                package_id="p2", transporter_id="t1",
                duration=5.0,
            ),
        ]
        p = Plan.sequential(commands)
        assert len(p) == 6
        # Cada ScheduledCommand mantiene la referencia al Command original.
        for sched, original in zip(p.scheduled, commands):
            assert sched.command is original
