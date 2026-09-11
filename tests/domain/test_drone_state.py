"""Tests de DroneState.

Cubre los cuatro miembros, los dos métodos de preguntas estáticas
(is_busy, is_terminal_forward), e ilustra cómo se usará el enum en
match exhaustivos desde el Validator y los handlers.

Los tests están parametrizados por estado de forma que añadir un futuro
estado al enum sin actualizar los tests rompa la suite. Esto cazaría
olvidos como "añadí PAUSED pero no decidí si está busy o no".
"""

import pytest

from droneplan_viz.domain.drone_state import DroneState


# ---------------------------------------------------------------------------
# Estructura del enum
# ---------------------------------------------------------------------------


class TestDroneStateMiembros:
    def test_existen_los_cuatro_estados(self):
        assert DroneState.IDLE
        assert DroneState.MOVING
        assert DroneState.INTERACTING
        assert DroneState.ERROR

    def test_son_distintos_entre_si(self):
        miembros = {
            DroneState.IDLE,
            DroneState.MOVING,
            DroneState.INTERACTING,
            DroneState.ERROR,
        }
        assert len(miembros) == 4

    def test_numero_total_de_estados(self):
        # Si alguien añade un estado nuevo, este test obliga a actualizar
        # explícitamente todos los métodos y todos los tests parametrizados
        # de abajo. Sirve como ancla de "decisión deliberada al ampliar".
        assert len(DroneState) == 4


# ---------------------------------------------------------------------------
# is_busy: ¿el drone está ocupado ahora mismo?
# ---------------------------------------------------------------------------


class TestIsBusy:
    @pytest.mark.parametrize(
        "estado, esperado",
        [
            (DroneState.IDLE, False),
            (DroneState.MOVING, True),
            (DroneState.INTERACTING, True),
            (DroneState.ERROR, False),  # paralizado, no ocupado
        ],
    )
    def test_is_busy_por_estado(self, estado, esperado):
        assert estado.is_busy() is esperado

    def test_busy_y_terminal_son_dimensiones_independientes(self):
        # Documentación ejecutable: ERROR no es busy, pero sí es terminal.
        # MOVING es busy pero no terminal. No hay solapamiento conceptual.
        assert DroneState.ERROR.is_terminal_forward() is True
        assert DroneState.ERROR.is_busy() is False
        assert DroneState.MOVING.is_busy() is True
        assert DroneState.MOVING.is_terminal_forward() is False


# ---------------------------------------------------------------------------
# is_terminal_forward: ¿este estado es sumidero en la rama forward?
# ---------------------------------------------------------------------------


class TestIsTerminalForward:
    @pytest.mark.parametrize(
        "estado, esperado",
        [
            (DroneState.IDLE, False),
            (DroneState.MOVING, False),
            (DroneState.INTERACTING, False),
            (DroneState.ERROR, True),
        ],
    )
    def test_solo_error_es_terminal_forward(self, estado, esperado):
        assert estado.is_terminal_forward() is esperado

    def test_hay_exactamente_un_estado_terminal(self):
        # Garantía estructural: el sumidero es único. Si en el futuro
        # hubiera más de uno, conviene que esa decisión se tome de forma
        # explícita y este test se actualice deliberadamente.
        terminales = [s for s in DroneState if s.is_terminal_forward()]
        assert terminales == [DroneState.ERROR]


# ---------------------------------------------------------------------------
# Comparación: igualdad e identidad
# ---------------------------------------------------------------------------


class TestComparacion:
    """Los miembros de un Enum son singletons. Tanto `==` como `is`
    funcionan; preferimos `is` cuando comparamos contra un literal del
    enum (más explícito), pero `==` también es correcto. Los tests
    verifican ambas vías porque ambos patrones aparecerán en el código."""

    def test_igualdad_con_si_mismo(self):
        assert DroneState.IDLE == DroneState.IDLE
        assert DroneState.IDLE is DroneState.IDLE

    def test_distintos_no_son_iguales(self):
        assert DroneState.IDLE != DroneState.MOVING
        assert DroneState.IDLE is not DroneState.MOVING


# ---------------------------------------------------------------------------
# Uso desde el Validator y handlers: match exhaustivo
# ---------------------------------------------------------------------------


class TestUsoEnMatch:
    """El Validator y los handlers usarán match/case sobre DroneState.
    Estos tests documentan el patrón y comprueban que cubrirlo de forma
    exhaustiva es sencillo."""

    def test_match_cubre_los_cuatro_estados(self):
        def describir(s: DroneState) -> str:
            match s:
                case DroneState.IDLE:
                    return "disponible"
                case DroneState.MOVING:
                    return "en tránsito"
                case DroneState.INTERACTING:
                    return "interactuando"
                case DroneState.ERROR:
                    return "paralizado por error"

        assert describir(DroneState.IDLE) == "disponible"
        assert describir(DroneState.MOVING) == "en tránsito"
        assert describir(DroneState.INTERACTING) == "interactuando"
        assert describir(DroneState.ERROR) == "paralizado por error"

    def test_iteracion_sobre_todos_los_estados(self):
        # Patrón útil para tests parametrizados externos o para construir
        # tablas de mapeo (p. ej. colores en el renderer).
        todos = list(DroneState)
        assert len(todos) == 4
        assert DroneState.IDLE in todos
        assert DroneState.ERROR in todos
