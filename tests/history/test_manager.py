"""Tests de history/manager.py: HistoryManager."""
from __future__ import annotations

import pytest

from droneplan_viz.domain import (
    Arm, Content, Drone, DroneState, Location, MetricsTracker, World,
)
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


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------
class TestHistoryManagerInicializacion:
    def test_inicial_longitud_uno(self, make_snapshot):
        snap = make_snapshot(timestamp=0.0)
        hm = HistoryManager(initial=snap)
        assert len(hm) == 1

    def test_inicial_accesible_en_posicion_cero(self, make_snapshot):
        snap = make_snapshot(timestamp=0.0)
        hm = HistoryManager(initial=snap)
        assert hm.at(0) is snap

    def test_head_index_es_cero_al_inicio(self, make_snapshot):
        snap = make_snapshot()
        hm = HistoryManager(initial=snap)
        assert hm.head_index == 0


# ---------------------------------------------------------------------------
# commit sin truncar
# ---------------------------------------------------------------------------
class TestHistoryManagerCommitSimple:
    def test_commit_aumenta_longitud(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        idx = hm.commit(make_snapshot(1.0))
        assert len(hm) == 2
        assert idx == 1

    def test_commit_devuelve_indice_correcto(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        for i in range(5):
            idx = hm.commit(make_snapshot(float(i + 1)))
            assert idx == i + 1

    def test_commit_actualiza_head_index(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        hm.commit(make_snapshot(1.0))
        hm.commit(make_snapshot(2.0))
        assert hm.head_index == 2

    def test_snapshots_accesibles_por_indice(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        snap1 = make_snapshot(1.0)
        snap2 = make_snapshot(2.0)
        hm.commit(snap1)
        hm.commit(snap2)
        assert hm.at(1) is snap1
        assert hm.at(2) is snap2


# ---------------------------------------------------------------------------
# at: acceso fuera de rango
# ---------------------------------------------------------------------------
class TestHistoryManagerAt:
    def test_at_indice_negativo_lanza_indexerror(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(IndexError):
            hm.at(-1)

    def test_at_indice_fuera_de_rango_lanza_indexerror(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(IndexError):
            hm.at(1)

    def test_at_lanza_aunque_estemos_en_head_si_pedimos_mas(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        hm.commit(make_snapshot(1.0))
        # head_index = 1, pedir 2 falla
        with pytest.raises(IndexError):
            hm.at(2)


# ---------------------------------------------------------------------------
# commit con truncado
# ---------------------------------------------------------------------------
class TestHistoryManagerCommitTruncado:
    def test_truncado_descarta_desde_indice(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot(0.0))
        hm.commit(make_snapshot(1.0))
        hm.commit(make_snapshot(2.0))
        hm.commit(make_snapshot(3.0))
        # historial: [0, 1, 2, 3]
        nuevo = make_snapshot(99.0)
        idx = hm.commit(nuevo, truncate_from=2)
        # historial esperado: [0, 1, nuevo]
        assert len(hm) == 3
        assert idx == 2
        assert hm.at(2) is nuevo

    def test_truncado_preserva_snapshots_anteriores(self, make_snapshot):
        snap0 = make_snapshot(0.0)
        snap1 = make_snapshot(1.0)
        hm = HistoryManager(initial=snap0)
        hm.commit(snap1)
        hm.commit(make_snapshot(2.0))
        hm.commit(make_snapshot(3.0))
        hm.commit(make_snapshot(99.0), truncate_from=2)
        # snap0 y snap1 deben seguir accesibles
        assert hm.at(0) is snap0
        assert hm.at(1) is snap1

    def test_truncado_desde_head_index_equivale_a_sin_truncar(self, make_snapshot):
        """Truncar desde head_index + 1 (= len) no descarta nada y añade."""
        hm = HistoryManager(initial=make_snapshot(0.0))
        hm.commit(make_snapshot(1.0))
        # head_index = 1, len = 2
        idx = hm.commit(make_snapshot(2.0), truncate_from=2)
        assert len(hm) == 3
        assert idx == 2

    def test_truncado_cero_prohibido(self, make_snapshot):
        """truncate_from=0 borraría el inicial; está prohibido."""
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(ValueError, match="inicial"):
            hm.commit(make_snapshot(1.0), truncate_from=0)

    def test_truncado_negativo_lanza_indexerror(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(IndexError):
            hm.commit(make_snapshot(1.0), truncate_from=-1)

    def test_truncado_fuera_de_rango_lanza_valueerror(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(ValueError, match="fuera de rango"):
            hm.commit(make_snapshot(1.0), truncate_from=99)

    def test_truncado_desde_uno_deja_solo_inicial(self, make_snapshot):
        snap0 = make_snapshot(0.0)
        hm = HistoryManager(initial=snap0)
        hm.commit(make_snapshot(1.0))
        hm.commit(make_snapshot(2.0))
        nuevo = make_snapshot(99.0)
        idx = hm.commit(nuevo, truncate_from=1)
        # Quedan [snap0, nuevo]
        assert len(hm) == 2
        assert hm.at(0) is snap0
        assert hm.at(1) is nuevo
        assert idx == 1


# ---------------------------------------------------------------------------
# Slots: el HistoryManager mismo no acepta atributos arbitrarios
# ---------------------------------------------------------------------------
class TestHistoryManagerSlots:
    def test_no_se_pueden_anadir_atributos(self, make_snapshot):
        hm = HistoryManager(initial=make_snapshot())
        with pytest.raises(AttributeError):
            hm.algo_nuevo = 1  # type: ignore[attr-defined]
