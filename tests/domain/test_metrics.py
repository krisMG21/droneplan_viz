"""Tests de MetricsTracker.

Cubren: construcción y defaults, inmutabilidad estricta, los cuatro
métodos record_* (no mutan self, suman/maximizan correctamente,
incrementan contadores apropiados sin duplicar), igualdad por valor,
y un par de tests de escenarios realistas con múltiples registros
encadenados.
"""

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.domain.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Construcción y defaults
# ---------------------------------------------------------------------------


class TestConstruccion:
    def test_todos_los_defaults_a_cero(self):
        m = MetricsTracker()
        assert m.total_cost == 0.0
        assert m.total_time == 0.0
        assert m.action_count == 0
        assert m.failed_commands == 0

    def test_construccion_con_valores_iniciales(self):
        # Útil para reconstruir desde un snapshot guardado.
        m = MetricsTracker(
            total_cost=100.0,
            total_time=50.0,
            action_count=7,
            failed_commands=2,
        )
        assert m.total_cost == 100.0
        assert m.total_time == 50.0
        assert m.action_count == 7
        assert m.failed_commands == 2


# ---------------------------------------------------------------------------
# Inmutabilidad
# ---------------------------------------------------------------------------


class TestInmutabilidad:
    def test_no_se_pueden_reasignar_campos(self):
        m = MetricsTracker()
        with pytest.raises(FrozenInstanceError):
            m.total_cost = 99.0  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            m.action_count = 5  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self):
        m = MetricsTracker()
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            m.cantidad_de_paquetes_entregados = 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# record_move
# ---------------------------------------------------------------------------


class TestRecordMove:
    def test_devuelve_nueva_instancia(self):
        m = MetricsTracker()
        nuevo = m.record_move(cost=10.0)
        assert nuevo is not m

    def test_no_muta_el_original(self):
        m = MetricsTracker()
        _ = m.record_move(cost=10.0)
        assert m.total_cost == 0.0
        assert m.action_count == 0

    def test_incrementa_total_cost(self):
        m = MetricsTracker().record_move(cost=66.0)
        assert m.total_cost == 66.0

    def test_incrementa_action_count(self):
        m = MetricsTracker().record_move(cost=10.0)
        assert m.action_count == 1

    def test_no_toca_total_time(self):
        # record_move solo registra el coste; el tiempo lo registra
        # record_action_completed o record_move_completed.
        m = MetricsTracker().record_move(cost=10.0)
        assert m.total_time == 0.0

    def test_no_toca_failed_commands(self):
        m = MetricsTracker().record_move(cost=10.0)
        assert m.failed_commands == 0

    def test_costes_se_acumulan(self):
        m = MetricsTracker()
        m = m.record_move(cost=10.0)
        m = m.record_move(cost=20.0)
        m = m.record_move(cost=5.5)
        assert m.total_cost == 35.5
        assert m.action_count == 3


# ---------------------------------------------------------------------------
# record_action_completed
# ---------------------------------------------------------------------------


class TestRecordActionCompleted:
    def test_devuelve_nueva_instancia(self):
        m = MetricsTracker()
        nuevo = m.record_action_completed(end_time=15.0)
        assert nuevo is not m

    def test_no_muta_el_original(self):
        m = MetricsTracker()
        _ = m.record_action_completed(end_time=15.0)
        assert m.total_time == 0.0
        assert m.action_count == 0

    def test_actualiza_total_time(self):
        m = MetricsTracker().record_action_completed(end_time=15.0)
        assert m.total_time == 15.0

    def test_incrementa_action_count(self):
        m = MetricsTracker().record_action_completed(end_time=15.0)
        assert m.action_count == 1

    def test_no_toca_total_cost(self):
        m = MetricsTracker().record_action_completed(end_time=15.0)
        assert m.total_cost == 0.0

    def test_total_time_toma_el_maximo_no_la_suma(self):
        # Esta es la semántica de makespan: lo que importa es cuándo
        # termina la última, no la suma de duraciones.
        m = MetricsTracker()
        m = m.record_action_completed(end_time=10.0)
        m = m.record_action_completed(end_time=5.0)  # más temprano
        # Total_time sigue siendo 10, no 15.
        assert m.total_time == 10.0

    def test_total_time_avanza_solo_si_el_nuevo_end_es_mayor(self):
        m = MetricsTracker()
        m = m.record_action_completed(end_time=20.0)
        m = m.record_action_completed(end_time=12.0)
        m = m.record_action_completed(end_time=25.0)
        assert m.total_time == 25.0

    def test_action_count_se_acumula_aun_con_end_times_iguales(self):
        m = MetricsTracker()
        m = m.record_action_completed(end_time=10.0)
        m = m.record_action_completed(end_time=10.0)
        # Dos acciones que terminan a la vez: total_time no avanza,
        # pero ambas se cuentan.
        assert m.total_time == 10.0
        assert m.action_count == 2


# ---------------------------------------------------------------------------
# record_move_completed: el atajo
# ---------------------------------------------------------------------------


class TestRecordMoveCompleted:
    """Movimientos: coste + end_time + action_count en un solo cambio.
    El motivo del atajo está documentado en el docstring del método:
    invocar record_move + record_action_completed contaría la acción
    dos veces."""

    def test_actualiza_los_tres_campos_a_la_vez(self):
        m = MetricsTracker().record_move_completed(cost=66.0, end_time=66.0)
        assert m.total_cost == 66.0
        assert m.total_time == 66.0
        assert m.action_count == 1

    def test_action_count_se_incrementa_una_sola_vez(self):
        m = MetricsTracker().record_move_completed(cost=10.0, end_time=5.0)
        # No 2 (lo que pasaría si llamáramos record_move +
        # record_action_completed encadenados).
        assert m.action_count == 1

    def test_makespan_max_aplica_aqui_tambien(self):
        m = MetricsTracker()
        m = m.record_move_completed(cost=10.0, end_time=20.0)
        m = m.record_move_completed(cost=5.0, end_time=15.0)
        # Segundo end_time es menor: no avanza total_time.
        assert m.total_time == 20.0
        # Pero los costes sí se acumulan.
        assert m.total_cost == 15.0
        assert m.action_count == 2


# ---------------------------------------------------------------------------
# record_failed_command
# ---------------------------------------------------------------------------


class TestRecordFailedCommand:
    def test_devuelve_nueva_instancia(self):
        m = MetricsTracker()
        nuevo = m.record_failed_command()
        assert nuevo is not m

    def test_incrementa_failed_commands(self):
        m = MetricsTracker().record_failed_command()
        assert m.failed_commands == 1

    def test_no_toca_metricas_pddl(self):
        # Los comandos fallidos no aparecen en total-cost, total-time
        # ni en action_count. Quedan como información didáctica
        # separada del rendimiento del plan.
        m = MetricsTracker(
            total_cost=100.0, total_time=50.0, action_count=5
        ).record_failed_command()
        assert m.total_cost == 100.0
        assert m.total_time == 50.0
        assert m.action_count == 5

    def test_fallos_se_acumulan(self):
        m = MetricsTracker()
        for _ in range(7):
            m = m.record_failed_command()
        assert m.failed_commands == 7


# ---------------------------------------------------------------------------
# Igualdad
# ---------------------------------------------------------------------------


class TestIgualdad:
    def test_mismos_valores_iguales(self):
        a = MetricsTracker(
            total_cost=10.0, total_time=5.0, action_count=2, failed_commands=1
        )
        b = MetricsTracker(
            total_cost=10.0, total_time=5.0, action_count=2, failed_commands=1
        )
        assert a == b

    def test_distintos_valores_no_iguales(self):
        a = MetricsTracker(total_cost=10.0)
        b = MetricsTracker(total_cost=20.0)
        assert a != b


# ---------------------------------------------------------------------------
# Escenarios realistas: traza de un plan
# ---------------------------------------------------------------------------


class TestEscenarioPlanCompleto:
    """Simulamos cómo un plan real iría registrando métricas a lo largo
    de su ejecución, encadenando llamadas a los métodos record_*."""

    def test_plan_secuencial_de_un_drone(self):
        # Un drone hace: mover(66), recoger, mover(66), entregar.
        # Tiempos: 0-66 move, 66-67 recoger, 67-133 move, 133-134 entrega.
        m = MetricsTracker()
        m = m.record_move_completed(cost=66.0, end_time=66.0)
        m = m.record_action_completed(end_time=67.0)  # recoger
        m = m.record_move_completed(cost=66.0, end_time=133.0)
        m = m.record_action_completed(end_time=134.0)  # entregar

        assert m.total_cost == 132.0  # solo los movimientos
        assert m.total_time == 134.0  # makespan = última acción
        assert m.action_count == 4
        assert m.failed_commands == 0

    def test_plan_con_errores(self):
        # Tres comandos válidos, dos rechazados por el Validator.
        m = MetricsTracker()
        m = m.record_move_completed(cost=10.0, end_time=10.0)
        m = m.record_failed_command()
        m = m.record_action_completed(end_time=11.0)
        m = m.record_failed_command()
        m = m.record_move_completed(cost=5.0, end_time=16.0)

        assert m.total_cost == 15.0
        assert m.total_time == 16.0
        assert m.action_count == 3  # los rechazados no cuentan
        assert m.failed_commands == 2

    def test_plan_concurrente_dos_drones(self):
        # Dos drones se mueven en paralelo: dron1 termina en t=50,
        # dron2 en t=80. El makespan es 80, no 50+80.
        m = MetricsTracker()
        m = m.record_move_completed(cost=50.0, end_time=50.0)  # dron1
        m = m.record_move_completed(cost=80.0, end_time=80.0)  # dron2
        assert m.total_cost == 130.0  # los costes sí suman
        assert m.total_time == 80.0   # el tiempo es makespan, no suma
        assert m.action_count == 2
