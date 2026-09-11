"""Tests del módulo droneplan_viz_app.app.

Smoke tests con SDL dummy. Verifican:

  - DroneplanVizApp se construye con cada escenario disponible.
  - run_one_frame() ejecuta sin crashear y avanza el estado coherentemente.
  - El loop responde correctamente a eventos sintéticos (QUIT, KEYDOWN,
    botón del HUD vía evento de pygame_gui).
  - save_screenshot produce un PNG inspeccionable sin avanzar el estado.

NO se testea que el loop continuo de run() respete el FPS cap (sería
test de tiempo real, frágil).
"""
from __future__ import annotations

import os
import tempfile

import pygame
import pygame_gui
import pytest

from droneplan_viz_app.app import (
    DroneplanVizApp,
    _clamp_to_min,
    _desktop_size,
)
from droneplan_viz_app.app_state import ALLOWED_SPEEDS
from droneplan_viz_app.scenarios import SCENARIOS


@pytest.fixture(autouse=True)
def cleanup_pygame():
    """Garantiza que pygame.quit() se ejecuta tras cada test para no
    arrastrar estado entre ellos (cada DroneplanVizApp hace
    pygame.init() pero el quit lo gestiona run()/teardown)."""
    yield
    pygame.quit()


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------


class TestAppConstruction:
    def test_construye_con_demo(self):
        app = DroneplanVizApp(scenario_name="demo")
        assert app.state.playback_time == 0.0
        assert app.state.paused is True
        assert app.timeline.duration > 0

    def test_construye_con_failure_demo(self):
        app = DroneplanVizApp(scenario_name="failure_demo")
        assert app.timeline.duration > 0

    def test_construye_con_tamano_personalizado(self):
        app = DroneplanVizApp(scenario_name="demo", window_size=(1024, 640))
        assert app.state.window_size == (1024, 640)

    def test_escenario_desconocido_lanza(self):
        with pytest.raises(ValueError, match="desconocido"):
            DroneplanVizApp(scenario_name="no_existe")

    def test_todos_los_escenarios_construyen(self):
        """Validación que cubre todos los SCENARIOS registrados."""
        for name in SCENARIOS:
            app = DroneplanVizApp(scenario_name=name)
            assert app.timeline is not None


# ---------------------------------------------------------------------------
# run_one_frame
# ---------------------------------------------------------------------------


class TestRunOneFrame:
    def test_no_crashea(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.run_one_frame(dt_override=0.016)

    def test_pausado_no_avanza(self):
        app = DroneplanVizApp(scenario_name="demo")
        # Arranca pausado.
        for _ in range(10):
            app.run_one_frame(dt_override=0.5)
        assert app.state.playback_time == 0.0

    def test_playing_avanza(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.paused = False
        for _ in range(10):
            app.run_one_frame(dt_override=0.1)
        # Debe haber avanzado ≈ 10 × 0.1 = 1.0s a speed=1.
        assert app.state.playback_time == pytest.approx(1.0, abs=0.01)

    def test_playing_a_velocidad_4x(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.paused = False
        app.state.playback_speed = 4.0
        for _ in range(5):
            app.run_one_frame(dt_override=0.1)
        # 5 × 0.1 × 4 = 2.0s.
        assert app.state.playback_time == pytest.approx(2.0, abs=0.01)

    def test_auto_pausa_al_final(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.paused = False
        # Saltar cerca del final.
        app.state.playback_time = app.timeline.duration - 0.05
        app.run_one_frame(dt_override=1.0)
        assert app.state.paused is True
        assert app.state.playback_time == pytest.approx(app.timeline.duration)

    def test_mantener_punto_avanza_frame_a_frame(self):
        # Mantener '.' pulsada avanza de forma sostenida, un fotograma por
        # frame (sondeo con pygame.key.get_pressed en run_one_frame).
        from unittest.mock import patch
        from droneplan_viz_app.app_state import FRAME_STEP

        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 5.0
        app.state.paused = True

        class _Pressed:
            def __init__(self, keys):
                self.keys = keys

            def __getitem__(self, k):
                return k in self.keys

        with patch("pygame.key.get_pressed",
                   return_value=_Pressed({pygame.K_PERIOD})):
            for _ in range(10):
                app.run_one_frame(dt_override=0.0)
        assert app.state.playback_time == pytest.approx(5.0 + 10 * FRAME_STEP)

    def test_mantener_coma_y_punto_no_avanza(self):
        # Ambas teclas pulsadas se cancelan: no hay avance.
        from unittest.mock import patch

        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 5.0
        app.state.paused = True

        class _Pressed:
            def __init__(self, keys):
                self.keys = keys

            def __getitem__(self, k):
                return k in self.keys

        with patch("pygame.key.get_pressed",
                   return_value=_Pressed({pygame.K_COMMA, pygame.K_PERIOD})):
            app.run_one_frame(dt_override=0.0)
        assert app.state.playback_time == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Eventos sintéticos
# ---------------------------------------------------------------------------


class TestAppEvents:
    def test_quit_termina_el_loop(self):
        app = DroneplanVizApp(scenario_name="demo")
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        app.run_one_frame(dt_override=0.016)
        assert app._running is False

    def test_space_alterna_pause(self):
        app = DroneplanVizApp(scenario_name="demo")
        assert app.state.paused is True
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        app.run_one_frame(dt_override=0.016)
        assert app.state.paused is False

    def test_left_arrow_es_step_back(self):
        app = DroneplanVizApp(scenario_name="demo")
        # Moverse a un snapshot intermedio primero.
        times = app.timeline.snapshot_times
        app.state.playback_time = times[3]
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        app.run_one_frame(dt_override=0.016)
        assert app.state.playback_time == pytest.approx(times[2])

    def test_right_arrow_es_step_forward(self):
        app = DroneplanVizApp(scenario_name="demo")
        times = app.timeline.snapshot_times
        app.state.playback_time = times[1]
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        app.run_one_frame(dt_override=0.016)
        assert app.state.playback_time == pytest.approx(times[2])

    def test_home(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 10.0
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME))
        app.run_one_frame(dt_override=0.016)
        assert app.state.playback_time == 0.0

    def test_end(self):
        app = DroneplanVizApp(scenario_name="demo")
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END))
        app.run_one_frame(dt_override=0.016)
        assert app.state.playback_time == pytest.approx(app.timeline.duration)

    def test_videoresize_actualiza_window_size(self):
        app = DroneplanVizApp(scenario_name="demo")
        pygame.event.post(pygame.event.Event(
            pygame.VIDEORESIZE, w=1100, h=700, size=(1100, 700)
        ))
        app.run_one_frame(dt_override=0.016)
        # Por encima del mínimo (1024x640) → respeta el tamaño.
        assert app.state.window_size == (1100, 700)

    def test_videoresize_demasiado_pequeno_clampea_al_minimo(self):
        app = DroneplanVizApp(scenario_name="demo")
        pygame.event.post(pygame.event.Event(
            pygame.VIDEORESIZE, w=300, h=300, size=(300, 300)
        ))
        app.run_one_frame(dt_override=0.016)
        # Clampeado al mínimo 1024x640.
        assert app.state.window_size == (1024, 640)

    def test_videoresize_invoca_relayout_del_hud(self):
        """Cuando la app recibe VIDEORESIZE, el HUD se recompone para
        que sus widgets cuadren con las nuevas regions. Verificamos
        sondeando el ancho de la progress bar antes y después."""
        app = DroneplanVizApp(scenario_name="demo")
        bar_w_before = app._hud._progress_bar.rect.width
        pygame.event.post(pygame.event.Event(
            pygame.VIDEORESIZE, w=1600, h=900, size=(1600, 900)
        ))
        app.run_one_frame(dt_override=0.016)
        bar_w_after = app._hud._progress_bar.rect.width
        assert bar_w_after > bar_w_before

    def test_resize_preserva_playback_time(self):
        """Resize NO debe alterar el estado del playback."""
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 7.5
        pygame.event.post(pygame.event.Event(
            pygame.VIDEORESIZE, w=1600, h=900, size=(1600, 900)
        ))
        app.run_one_frame(dt_override=0.0)
        assert app.state.playback_time == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Pantalla completa (Alt+Enter / F11)
# ---------------------------------------------------------------------------


def _expected_fullscreen_size() -> tuple[int, int]:
    """Tamaño con el que la app entra en pantalla completa en ESTE equipo.

    No se puede fijar un literal: depende de la resolución del escritorio
    (con SDL dummy, la que reporte el backend), clampeada al mínimo del
    layout.
    """
    return _clamp_to_min(_desktop_size())


def _post_key(key: int, mod: int = 0) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))


class TestAppFullscreen:
    def test_f11_entra_en_pantalla_completa(self):
        app = DroneplanVizApp(scenario_name="demo")
        assert app.fullscreen is False
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is True

    def test_f11_dos_veces_vuelve_a_ventana(self):
        app = DroneplanVizApp(scenario_name="demo", window_size=(1100, 700))
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is False
        # Se restaura el tamaño de ventana previo, no el de la pantalla.
        assert app.state.window_size == (1100, 700)

    def test_alt_enter_alterna(self):
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_RETURN, pygame.KMOD_LALT)
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is True

    def test_enter_sin_alt_no_alterna(self):
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_RETURN)
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is False

    def test_escape_sale_de_pantalla_completa(self):
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        _post_key(pygame.K_ESCAPE)
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is False

    def test_escape_con_ayuda_abierta_solo_cierra_la_ayuda(self):
        """Esc cierra la ayuda ANTES de tocar la pantalla completa."""
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_F11)
        _post_key(pygame.K_h)
        app.run_one_frame(dt_override=0.016)
        assert app.state.show_help is True
        _post_key(pygame.K_ESCAPE)
        app.run_one_frame(dt_override=0.016)
        assert app.state.show_help is False
        assert app.fullscreen is True

    def test_pantalla_completa_recompone_el_layout(self):
        """window_size y regions pasan al tamaño real de la superficie."""
        app = DroneplanVizApp(scenario_name="demo", window_size=(1100, 700))
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        expected = _expected_fullscreen_size()
        assert app.state.window_size == expected
        # Las regiones caben dentro de la superficie de dibujo (si no,
        # subsurface() reventaría al pintar el mundo).
        assert app._window.get_size() == expected
        assert app._regions.world.right <= expected[0]
        assert app._regions.hud_bottom.bottom <= expected[1]

    def test_videoresize_se_ignora_en_pantalla_completa(self):
        """El aviso de redimensionado del gestor de ventanas no debe
        devolver la app a modo ventana."""
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.016)
        w, h = _expected_fullscreen_size()
        pygame.event.post(pygame.event.Event(
            pygame.VIDEORESIZE, w=1280, h=800, size=(1280, 800)
        ))
        app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is True
        assert app.state.window_size == (w, h)

    def test_pantalla_completa_preserva_playback_y_zoom(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 7.5
        zoom_antes = app.state.zoom
        _post_key(pygame.K_F11)
        app.run_one_frame(dt_override=0.0)
        assert app.state.playback_time == pytest.approx(7.5)
        assert app.state.zoom == pytest.approx(zoom_antes)

    def test_render_en_pantalla_completa_no_revienta(self):
        """Varios frames completos (render + HUD) con la ventana en
        pantalla completa y de vuelta."""
        app = DroneplanVizApp(scenario_name="demo")
        _post_key(pygame.K_F11)
        for _ in range(3):
            app.run_one_frame(dt_override=0.016)
        _post_key(pygame.K_F11)
        for _ in range(3):
            app.run_one_frame(dt_override=0.016)
        assert app.fullscreen is False


# ---------------------------------------------------------------------------
# HUD events (vía pygame_gui)
# ---------------------------------------------------------------------------


class TestAppHudInteraction:
    def test_play_pause_button_via_evento_gui(self):
        """Simular un click en el botón Play/Pause emitiendo un evento
        sintético UI_BUTTON_PRESSED."""
        app = DroneplanVizApp(scenario_name="demo")
        assert app.state.paused is True
        pygame.event.post(pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": app._hud._btn_play_pause},
        ))
        app.run_one_frame(dt_override=0.016)
        assert app.state.paused is False

    def test_speed_button_cambia_velocidad(self):
        app = DroneplanVizApp(scenario_name="demo")
        # Encontrar el botón de speed 4×.
        idx_4x = ALLOWED_SPEEDS.index(4.0)
        btn = app._hud._btn_speeds[idx_4x]
        pygame.event.post(pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": btn},
        ))
        app.run_one_frame(dt_override=0.016)
        assert app.state.playback_speed == 4.0

    def test_failure_select_salta_al_instante_del_fallo(self):
        """Simular click sobre la tarjeta de fallo: state debe saltar a
        t=2.0 y marcar selected_failure=0."""
        app = DroneplanVizApp(scenario_name="failure_demo")
        app._hud.show_metrics_failures_tab()
        app.run_one_frame(dt_override=0.0)
        assert app._hud._fail_cards
        card_img = app._hud._fail_cards[0][0]
        cx, cy = card_img.rect.center
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (cx, cy)},
        ))
        app.run_one_frame(dt_override=0.016)

        assert app.state.playback_time == pytest.approx(2.0)
        assert app.state.selected_failure == 0
        assert app.state.paused is True


# ---------------------------------------------------------------------------
# Cámara (paso 2 de integración: AppState.zoom/pan llegan a render_frame)
# ---------------------------------------------------------------------------


class TestAppCamera:
    """Blindan que la app pasa state.zoom y state.pan a render_frame, y
    que los defaults reproducen el render previo a la integración de
    cámara (invariante de no-regresión).

    Comparan frames vía hash MD5 del buffer RGBA del window: dos frames
    iguales producen hashes idénticos. Es la herramienta más simple
    posible para un test píxel-a-píxel y no requiere numpy.
    """

    @staticmethod
    def _hash_window(window) -> str:
        import hashlib
        import pygame
        # tobytes() reemplaza al deprecado tostring().
        return hashlib.md5(pygame.image.tobytes(window, "RGBA")).hexdigest()

    def test_defaults_son_neutros_no_alteran_el_render(self):
        """Con state.zoom=1.0 y state.pan=(0,0) la app debe producir un
        frame idéntico al que producía antes de la integración de cámara.
        Esto blinda el invariante crítico de D2: zoom=1, pan=0 es
        píxel-idéntico al render sin esos kwargs."""
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 9.0
        # Defaults explícitos: documentamos que esto representa "sin cámara".
        app.state.zoom = 1.0
        app.state.pan = (0, 0)
        app.run_one_frame(dt_override=0.0)
        h_default = self._hash_window(app._window)

        # Restaurar defaults tras tocarlos debe volver al mismo hash.
        app.state.zoom = 2.0
        app.run_one_frame(dt_override=0.0)
        app.state.zoom = 1.0
        app.state.pan = (0, 0)
        app.run_one_frame(dt_override=0.0)
        h_restored = self._hash_window(app._window)
        assert h_default == h_restored

    def test_mutar_zoom_produce_frame_distinto(self):
        """Cambiar state.zoom y volver a renderizar debe alterar el frame.
        Si este test falla, la app no está pasando zoom a render_frame."""
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 9.0
        app.run_one_frame(dt_override=0.0)
        h_base = self._hash_window(app._window)

        app.state.zoom = 2.0
        app.run_one_frame(dt_override=0.0)
        h_zoom = self._hash_window(app._window)
        assert h_base != h_zoom, (
            "el zoom no llegó a render_frame: hashes iguales"
        )

    def test_mutar_pan_produce_frame_distinto(self):
        """Cambiar state.pan y volver a renderizar debe alterar el frame.
        Si este test falla, la app no está pasando pan a render_frame."""
        app = DroneplanVizApp(scenario_name="demo")
        app.state.playback_time = 9.0
        app.run_one_frame(dt_override=0.0)
        h_base = self._hash_window(app._window)

        app.state.pan = (100, 0)
        app.run_one_frame(dt_override=0.0)
        h_pan = self._hash_window(app._window)
        assert h_base != h_pan, (
            "el pan no llegó a render_frame: hashes iguales"
        )


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------


class TestScreenshot:
    def test_save_screenshot_crea_png(self):
        app = DroneplanVizApp(scenario_name="demo")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            app.save_screenshot(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000, "PNG sospechosamente pequeño"
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_screenshot_no_avanza_el_estado(self):
        """Verifica que save_screenshot usa dt=0 y no altera el playback."""
        app = DroneplanVizApp(scenario_name="demo")
        app.state.paused = False
        app.state.playback_time = 3.0
        before = app.state.playback_time
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            app.save_screenshot(path)
            assert app.state.playback_time == pytest.approx(before)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestAppCameraFollow:
    """Seguimiento de cámara: con state.followed_entity, run_one_frame
    recentra cada frame sobre la posición INTERPOLADA de la entidad."""

    def _find_move_time(self, app):
        from droneplan_viz.render import locate_entity, locate_entity_interpolated
        size = app._regions.world.size
        for i in range(1, 200):
            t = app.timeline.duration * i / 200.0
            a, b, p = app.timeline.sample(t)
            stat = locate_entity(a.world, "d1", size, theme=app._theme)
            interp = locate_entity_interpolated(a, b, p, "d1", size, theme=app._theme)
            if stat != interp:
                return t
        return None

    def test_seguimiento_centra_en_dron_en_movimiento(self):
        from droneplan_viz.render import locate_entity_interpolated
        app = DroneplanVizApp(scenario_name="demo")
        t_move = self._find_move_time(app)
        assert t_move is not None, "el demo no tiene un DroneMove de d1"

        app.state.followed_entity = "d1"
        app.state.playback_time = t_move
        app.run_one_frame(dt_override=0.0)  # pausado: el tiempo no avanza

        a, b, p = app.timeline.sample(app.state.playback_time)
        size = app._regions.world.size
        interp = locate_entity_interpolated(a, b, p, "d1", size, theme=app._theme)
        sw, sh = size
        z = app.state.zoom
        exp = (
            int(sw // 2 - (sw / 2 + (interp[0] - sw / 2) * z)),
            int(sh // 2 - (sh / 2 + (interp[1] - sh / 2) * z)),
        )
        assert app.state.pan == exp

    def test_seguimiento_a_id_inexistente_se_cancela(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.followed_entity = "fantasma"
        app.run_one_frame(dt_override=0.0)
        assert app.state.followed_entity is None

    def test_sin_seguimiento_no_recentra(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.state.followed_entity = None
        app.state.pan = (0, 0)
        app.run_one_frame(dt_override=0.0)
        assert app.state.pan == (0, 0)


class TestPanelTabs:
    """API para revelar la pestaña Métricas+Fallos del panel lateral
    (usada por demo_app para que las capturas de fallo muestren el panel)."""

    def test_app_expone_hud(self):
        app = DroneplanVizApp(scenario_name="demo")
        from droneplan_viz_app.hud import Hud
        assert isinstance(app.hud, Hud)

    def test_mostrar_pestana_metricas_no_lanza(self):
        app = DroneplanVizApp(scenario_name="failure_demo")
        # No debe lanzar; tras llamarla, un frame se dibuja sin error.
        app.hud.show_metrics_failures_tab()
        app.state.selected_failure = 0
        app.run_one_frame(dt_override=0.0)

    def test_alternar_pestanas_no_lanza(self):
        app = DroneplanVizApp(scenario_name="demo")
        app.hud.show_metrics_failures_tab()
        app.hud.show_inventory_tab()
        app.run_one_frame(dt_override=0.0)