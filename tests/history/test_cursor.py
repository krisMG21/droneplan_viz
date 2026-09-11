"""Tests de history/cursor.py: TimelineCursor."""
from __future__ import annotations

import pytest

from droneplan_viz.domain import (
    Arm, Content, Drone, DroneState, Location, MetricsTracker, World,
)
from droneplan_viz.history.cursor import TimelineCursor
from droneplan_viz.history.manager import HistoryManager
from droneplan_viz.history.snapshot import WorldSnapshot


@pytest.fixture
def make_snapshot():
    """Factoría de snapshots distinguibles por su timestamp."""
    def _make(timestamp: float = 0.0) -> WorldSnapshot:
        med = Content(id="medicina")
        w = World(
            locations={"deposito": Location(id="deposito")},
            drones={
                "d1": Drone(
                    id="d1", position="deposito",
                    arms=(Arm(id="izq"),), state=DroneState.IDLE,
                )
            },
            contents={"medicina": med},
        )
        return WorldSnapshot(world=w, metrics=MetricsTracker(), timestamp=timestamp)
    return _make


@pytest.fixture
def populated_history(make_snapshot):
    """Historial con 4 snapshots: timestamps 0.0, 1.0, 2.0, 3.0."""
    hm = HistoryManager(initial=make_snapshot(0.0))
    hm.commit(make_snapshot(1.0))
    hm.commit(make_snapshot(2.0))
    hm.commit(make_snapshot(3.0))
    return hm


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------
class TestCursorInicializacion:
    def test_arranca_en_head(self, populated_history):
        cur = TimelineCursor(populated_history)
        assert cur.index == populated_history.head_index
        assert cur.index == 3

    def test_at_head_true_al_inicio(self, populated_history):
        cur = TimelineCursor(populated_history)
        assert cur.at_head is True

    def test_at_start_false_si_hay_mas_de_un_snapshot(self, populated_history):
        cur = TimelineCursor(populated_history)
        assert cur.at_start is False

    def test_historial_unitario_arranca_at_start_y_at_head(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        cur = TimelineCursor(hm)
        assert cur.at_head is True
        assert cur.at_start is True

    def test_current_devuelve_head(self, populated_history):
        cur = TimelineCursor(populated_history)
        assert cur.current is populated_history.at(populated_history.head_index)
        assert cur.current.timestamp == 3.0


# ---------------------------------------------------------------------------
# back / forward
# ---------------------------------------------------------------------------
class TestCursorBack:
    def test_back_decrementa_indice(self, populated_history):
        cur = TimelineCursor(populated_history)
        snap = cur.back()
        assert cur.index == 2
        assert snap is populated_history.at(2)

    def test_back_devuelve_none_en_inicio(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        assert cur.back() is None
        assert cur.index == 0

    def test_back_multiple_llega_al_inicio(self, populated_history):
        cur = TimelineCursor(populated_history)
        for _ in range(3):
            cur.back()
        assert cur.index == 0
        assert cur.at_start is True

    def test_back_en_inicio_no_mueve(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        cur.back()
        cur.back()
        cur.back()
        assert cur.index == 0


class TestCursorForward:
    def test_forward_incrementa_indice(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        snap = cur.forward()
        assert cur.index == 1
        assert snap is populated_history.at(1)

    def test_forward_devuelve_none_en_head(self, populated_history):
        cur = TimelineCursor(populated_history)
        assert cur.forward() is None
        assert cur.at_head is True

    def test_forward_en_head_no_mueve(self, populated_history):
        cur = TimelineCursor(populated_history)
        original = cur.index
        cur.forward()
        cur.forward()
        assert cur.index == original


# ---------------------------------------------------------------------------
# goto
# ---------------------------------------------------------------------------
class TestCursorGoto:
    def test_goto_valido(self, populated_history):
        cur = TimelineCursor(populated_history)
        snap = cur.goto(1)
        assert cur.index == 1
        assert snap is populated_history.at(1)

    def test_goto_inicio(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.goto(0)
        assert cur.at_start is True

    def test_goto_head(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        cur.goto(3)
        assert cur.at_head is True

    def test_goto_indice_negativo_lanza(self, populated_history):
        cur = TimelineCursor(populated_history)
        with pytest.raises(IndexError):
            cur.goto(-1)

    def test_goto_fuera_de_rango_lanza(self, populated_history):
        cur = TimelineCursor(populated_history)
        with pytest.raises(IndexError):
            cur.goto(99)

    def test_goto_falla_no_mueve_cursor(self, populated_history):
        """Si goto falla, el cursor mantiene su posición previa."""
        cur = TimelineCursor(populated_history)
        cur.goto(1)
        with pytest.raises(IndexError):
            cur.goto(99)
        assert cur.index == 1


# ---------------------------------------------------------------------------
# jump_to_head / jump_to_start
# ---------------------------------------------------------------------------
class TestCursorJumps:
    def test_jump_to_head(self, populated_history):
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        snap = cur.jump_to_head()
        assert cur.at_head is True
        assert snap is populated_history.at(3)

    def test_jump_to_start(self, populated_history):
        cur = TimelineCursor(populated_history)
        snap = cur.jump_to_start()
        assert cur.at_start is True
        assert snap is populated_history.at(0)


# ---------------------------------------------------------------------------
# commit_new: caso normal (cursor en head)
# ---------------------------------------------------------------------------
class TestCursorCommitNewEnHead:
    def test_commit_new_en_head_anade_y_avanza(
        self, populated_history, make_snapshot
    ):
        cur = TimelineCursor(populated_history)
        assert cur.at_head is True
        antes = len(populated_history)
        nuevo = make_snapshot(99.0)
        idx = cur.commit_new(nuevo)
        # Se añadió al final
        assert len(populated_history) == antes + 1
        # El cursor avanzó al nuevo head
        assert cur.index == idx
        assert cur.at_head is True
        assert cur.current is nuevo

    def test_commit_new_no_descarta_si_estaba_en_head(
        self, populated_history, make_snapshot
    ):
        cur = TimelineCursor(populated_history)
        anteriores = [populated_history.at(i) for i in range(len(populated_history))]
        cur.commit_new(make_snapshot(99.0))
        # Todos los snapshots previos siguen ahí
        for i, original in enumerate(anteriores):
            assert populated_history.at(i) is original


# ---------------------------------------------------------------------------
# commit_new: caso Ctrl+Z (cursor en medio)
# ---------------------------------------------------------------------------
class TestCursorCommitNewEnMedio:
    def test_commit_new_en_medio_trunca_y_anade(
        self, populated_history, make_snapshot
    ):
        cur = TimelineCursor(populated_history)
        cur.goto(1)  # historial: [0, 1, 2, 3], cursor en 1
        # Snapshots 2 y 3 deberían descartarse
        nuevo = make_snapshot(99.0)
        idx = cur.commit_new(nuevo)
        # Historial esperado: [0, 1, nuevo]
        assert len(populated_history) == 3
        assert idx == 2
        assert cur.index == 2
        assert cur.at_head is True
        assert cur.current is nuevo

    def test_commit_new_en_medio_preserva_predecesores(
        self, populated_history, make_snapshot
    ):
        snap0 = populated_history.at(0)
        snap1 = populated_history.at(1)
        cur = TimelineCursor(populated_history)
        cur.goto(1)
        cur.commit_new(make_snapshot(99.0))
        # snap0 y snap1 siguen accesibles
        assert populated_history.at(0) is snap0
        assert populated_history.at(1) is snap1

    def test_commit_new_desde_inicio_descarta_todo_lo_demas(
        self, populated_history, make_snapshot
    ):
        snap0 = populated_history.at(0)
        cur = TimelineCursor(populated_history)
        cur.jump_to_start()
        cur.commit_new(make_snapshot(99.0))
        assert len(populated_history) == 2
        assert populated_history.at(0) is snap0
        assert cur.index == 1

    def test_back_tras_commit_new_lleva_al_predecesor(
        self, populated_history, make_snapshot
    ):
        """Tras commit_new desde cursor en 1, back desde el nuevo head
        debe llevar al snapshot 1, no a uno de los descartados.
        """
        snap1 = populated_history.at(1)
        cur = TimelineCursor(populated_history)
        cur.goto(1)
        cur.commit_new(make_snapshot(99.0))
        # cursor en el nuevo head (índice 2)
        snap = cur.back()
        assert snap is snap1
        assert cur.index == 1


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------
class TestCursorSlots:
    def test_no_se_pueden_anadir_atributos(self, populated_history):
        cur = TimelineCursor(populated_history)
        with pytest.raises(AttributeError):
            cur.extra = 1  # type: ignore[attr-defined]
