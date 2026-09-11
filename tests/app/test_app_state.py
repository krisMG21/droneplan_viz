"""Tests del módulo droneplan_viz_app.app_state.

Verifican el contrato de AppState: defaults, mutabilidad acotada (los
slots evitan typos), e invariantes detectables por validate().

Sin pygame: AppState es lógica pura, no toca SDL.
"""
from __future__ import annotations

import pytest

from droneplan_viz_app.app_state import (
    ALLOWED_SPEEDS,
    DEFAULT_SPEED,
    DEFAULT_WINDOW_SIZE,
    AppState,
)


# ---------------------------------------------------------------------------
# Construcción y defaults
# ---------------------------------------------------------------------------


class TestAppStateConstruction:
    def test_defaults_documentados(self):
        s = AppState()
        assert s.playback_time == 0.0
        assert s.playback_speed == DEFAULT_SPEED
        assert s.paused is True  # decisión documentada: arranca pausada.
        assert s.window_size == DEFAULT_WINDOW_SIZE
        assert s.selected_failure is None

    def test_default_speed_es_uno(self):
        assert DEFAULT_SPEED == 1.0
        assert DEFAULT_SPEED in ALLOWED_SPEEDS

    def test_allowed_speeds_es_la_lista_consensuada(self):
        """Decisión 3.4 confirmada en Paso E.1: cinco velocidades discretas."""
        assert ALLOWED_SPEEDS == (0.25, 0.5, 1.0, 2.0, 4.0)

    def test_default_window_size_propuesto(self):
        """Tamaño consensuado en §2.3 de la propuesta."""
        assert DEFAULT_WINDOW_SIZE == (1280, 800)

    def test_constructor_acepta_overrides(self):
        s = AppState(playback_time=5.5, playback_speed=2.0, paused=False)
        assert s.playback_time == 5.5
        assert s.playback_speed == 2.0
        assert s.paused is False
        # Lo no especificado conserva su default.
        assert s.window_size == DEFAULT_WINDOW_SIZE
        assert s.selected_failure is None


# ---------------------------------------------------------------------------
# Mutabilidad y slots
# ---------------------------------------------------------------------------


class TestAppStateMutability:
    def test_es_mutable_en_sitio(self):
        """AppState NO es frozen: el controller muta in situ."""
        s = AppState()
        s.playback_time = 3.0
        s.paused = False
        s.playback_speed = 2.0
        assert s.playback_time == 3.0
        assert s.paused is False
        assert s.playback_speed == 2.0

    def test_slots_previenen_typos(self):
        """Asignar a un campo inexistente debe fallar (gracias a slots).
        Esto previene una clase entera de bugs típicos en código
        mutable: 'self.paused' vs 'self.pause' por descuido."""
        s = AppState()
        with pytest.raises(AttributeError):
            s.pause = True  # noqa: típico typo, debe fallar.

    def test_no_tiene_dict(self):
        """Slots: AppState no debe tener __dict__."""
        s = AppState()
        assert not hasattr(s, "__dict__")


# ---------------------------------------------------------------------------
# Invariantes vía validate()
# ---------------------------------------------------------------------------


class TestAppStateValidate:
    def test_estado_inicial_es_valido(self):
        AppState().validate()  # no lanza.

    def test_estado_tras_mutacion_legitima_es_valido(self):
        s = AppState()
        s.playback_time = 9.5
        s.playback_speed = 4.0
        s.paused = False
        s.window_size = (1024, 768)
        s.selected_failure = 2
        s.validate()  # no lanza.

    def test_playback_time_negativo_falla(self):
        s = AppState(playback_time=-1.0)
        with pytest.raises(AssertionError, match="playback_time"):
            s.validate()

    def test_speed_no_permitido_falla(self):
        """0.7× no está en ALLOWED_SPEEDS: el controller solo asigna
        valores del set discreto. Pasar 0.7 a mano fuera del controller
        viola el contrato."""
        s = AppState(playback_speed=0.7)
        with pytest.raises(AssertionError, match="playback_speed"):
            s.validate()

    @pytest.mark.parametrize("speed", list(ALLOWED_SPEEDS))
    def test_todas_las_speeds_permitidas_son_validas(self, speed):
        s = AppState(playback_speed=speed)
        s.validate()  # no lanza para ninguno.

    def test_window_size_cero_falla(self):
        s = AppState(window_size=(0, 100))
        with pytest.raises(AssertionError, match="window_size"):
            s.validate()
        s2 = AppState(window_size=(100, 0))
        with pytest.raises(AssertionError, match="window_size"):
            s2.validate()

    def test_window_size_no_tupla_falla(self):
        s = AppState()
        s.window_size = [1280, 800]  # type: ignore[assignment]
        # lista en vez de tupla.
        with pytest.raises(AssertionError, match="window_size"):
            s.validate()

    def test_selected_failure_none_es_valido(self):
        s = AppState(selected_failure=None)
        s.validate()

    def test_selected_failure_no_negativo(self):
        s = AppState(selected_failure=0)
        s.validate()  # 0 es un índice válido.
        s.selected_failure = 17
        s.validate()
        s.selected_failure = -1
        with pytest.raises(AssertionError, match="selected_failure"):
            s.validate()

    # ---- Cámara (zoom + pan) ----
    #
    # Los campos zoom/pan añadidos en E2 para la cámara entregada por D2.
    # Defaults: zoom=1.0, pan=(0, 0). Equivalen a "sin cámara": el render
    # con esos valores es píxel-idéntico al render sin esos kwargs
    # (invariante blindado en tests/render/test_camera.py).

    def test_camara_defaults_valor_neutro(self):
        """Los defaults equivalen a 'sin cámara'."""
        s = AppState()
        assert s.zoom == 1.0
        assert s.pan == (0, 0)
        s.validate()

    def test_zoom_dentro_del_rango_es_valido(self):
        from droneplan_viz_app.app_state import ZOOM_MIN, ZOOM_MAX
        for z in (ZOOM_MIN, 1.0, 1.25, 1.5, ZOOM_MAX):
            s = AppState(zoom=z)
            s.validate()

    def test_zoom_por_debajo_del_minimo_falla(self):
        from droneplan_viz_app.app_state import ZOOM_MIN
        s = AppState(zoom=ZOOM_MIN / 2)
        with pytest.raises(AssertionError, match="zoom"):
            s.validate()

    def test_zoom_por_encima_del_maximo_falla(self):
        from droneplan_viz_app.app_state import ZOOM_MAX
        s = AppState(zoom=ZOOM_MAX * 2)
        with pytest.raises(AssertionError, match="zoom"):
            s.validate()

    def test_pan_acepta_enteros_negativos_y_positivos(self):
        """No hay clamp a un rango concreto del pan; cualquier (int,int)
        es válido. Es responsabilidad del usuario decidir si arrastrar
        muy lejos saca el world del marco visible."""
        for pan in ((0, 0), (100, -50), (-1000, 2000)):
            s = AppState(pan=pan)
            s.validate()

    def test_pan_con_float_falla(self):
        s = AppState()
        s.pan = (1.5, 0)  # type: ignore[assignment]
        with pytest.raises(AssertionError, match="pan"):
            s.validate()

    def test_pan_de_longitud_distinta_de_dos_falla(self):
        s = AppState()
        s.pan = (1, 2, 3)  # type: ignore[assignment]
        with pytest.raises(AssertionError, match="pan"):
            s.validate()

    # ---- Inventory collapse state ----
    #
    # set[str] de categorías plegadas en la tab Inventario. Persiste a
    # través de redimensionados (vive en AppState). Default vacío =
    # todas las secciones desplegadas.

    def test_inventory_collapsed_default_es_vacio(self):
        s = AppState()
        assert s.inventory_collapsed == set()
        s.validate()

    def test_dos_appstates_tienen_sets_independientes(self):
        """Regresión clásica del default mutable compartido: si el
        default fuera set() directo en lugar de field(default_factory=set),
        todas las AppState compartirían la misma instancia."""
        s1 = AppState()
        s2 = AppState()
        s1.inventory_collapsed.add("DRONES")
        assert "DRONES" not in s2.inventory_collapsed

    def test_inventory_collapsed_acepta_categorias_validas(self):
        s = AppState()
        s.inventory_collapsed.add("DRONES")
        s.inventory_collapsed.add("LOCATIONS")
        s.validate()

    def test_inventory_collapsed_no_es_set_falla(self):
        s = AppState()
        s.inventory_collapsed = ["DRONES"]  # type: ignore[assignment]
        with pytest.raises(AssertionError, match="inventory_collapsed"):
            s.validate()

    def test_inventory_collapsed_con_no_string_falla(self):
        s = AppState()
        s.inventory_collapsed.add(123)  # type: ignore[arg-type]
        with pytest.raises(AssertionError, match="inventory_collapsed"):
            s.validate()


class TestFollowedEntity:
    def test_default_es_none(self):
        s = AppState()
        assert s.followed_entity is None
        s.validate()

    def test_acepta_string(self):
        s = AppState()
        s.followed_entity = "d1"
        s.validate()

    def test_no_string_ni_none_falla(self):
        s = AppState()
        s.followed_entity = 123  # type: ignore[assignment]
        with pytest.raises(AssertionError, match="followed_entity"):
            s.validate()
