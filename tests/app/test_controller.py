"""Tests del módulo droneplan_viz_app.controller.

Lógica pura sin pygame.display ni pygame_gui. Verifican:
  - tick() avanza correctamente y aplica auto-pausa.
  - handle_pygame_event() despacha por event.type+key correctamente.
  - Cada handler individual (on_play_pause, on_step_back, etc.) cumple
    su contrato documentado.
  - handle_hud_intent() despacha por intent.kind.
  - HudIntent dataclass es frozen.

Construye un Timeline real usando build_demo_scenario() de scenarios.py
para no duplicar setup. Los snapshot_times resultantes (7 snapshots del
demo feliz) cubren los casos interesantes de navegación.
"""
from __future__ import annotations

import pygame
import pytest

from droneplan_viz.render import Theme, Timeline
from droneplan_viz.runtime import PlanRunner
from droneplan_viz_app.app_state import (
    ALLOWED_SPEEDS,
    AppState,
    DEFAULT_PAN,
    DEFAULT_ZOOM,
    FRAME_STEP,
    ZOOM_MAX,
    ZOOM_MIN,
)
from droneplan_viz_app.controller import (
    HudIntent,
    ZOOM_STEP,
    focus_on_entity,
    focus_on_entity_frame,
    handle_hud_action_select,
    handle_hud_failure_select,
    handle_hud_intent,
    handle_hud_inventory_focus,
    handle_hud_inventory_toggle_section,
    handle_pygame_event,
    on_action_clicked,
    on_camera_reset,
    on_end,
    on_failure_clicked,
    on_home,
    on_pan,
    on_play_pause,
    on_speed_button,
    on_frame_back,
    on_frame_forward,
    on_step_back,
    on_step_forward,
    on_zoom,
    tick,
)
from droneplan_viz_app.scenarios import (
    build_demo_scenario,
    build_failure_demo_scenario,
)


# ---------------------------------------------------------------------------
# Fixtures: timeline real de los escenarios
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_timeline() -> Timeline:
    """Timeline del escenario feliz (7 snapshots, duration 18.0s)."""
    world, plan = build_demo_scenario()
    runner = PlanRunner(world)
    runner.execute(plan)
    return Timeline(runner.history, theme=Theme.default())


@pytest.fixture
def failure_timeline_and_starts() -> tuple[Timeline, tuple[float, ...]]:
    """Timeline del escenario de fallo + tupla de start_times de fallos."""
    world, plan = build_failure_demo_scenario()
    runner = PlanRunner(world)
    result = runner.execute(plan)
    timeline = Timeline(runner.history, theme=Theme.default())
    starts = tuple(f.scheduled.start_time for f in result.failures)
    return timeline, starts


@pytest.fixture
def demo_world():
    """World inicial del escenario demo + Theme por defecto, para tests
    de focus_on_entity que no necesitan ejecutar el plan."""
    world, _ = build_demo_scenario()
    return world, Theme.default()


# ---------------------------------------------------------------------------
# HudIntent
# ---------------------------------------------------------------------------


class TestHudIntent:
    def test_construccion_basica(self):
        i = HudIntent(kind="play_pause")
        assert i.kind == "play_pause"
        assert i.payload is None

    def test_payload_opcional(self):
        i = HudIntent(kind="speed", payload=2.0)
        assert i.payload == 2.0

    def test_es_frozen(self):
        i = HudIntent(kind="home")
        with pytest.raises(Exception):  # FrozenInstanceError
            i.kind = "end"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# tick()
# ---------------------------------------------------------------------------


class TestTick:
    def test_pausado_no_avanza(self, demo_timeline):
        s = AppState(paused=True, playback_time=2.0)
        tick(s, dt=0.1, timeline=demo_timeline)
        assert s.playback_time == 2.0

    def test_avanza_por_dt_a_speed_uno(self, demo_timeline):
        s = AppState(paused=False, playback_speed=1.0, playback_time=2.0)
        tick(s, dt=0.5, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(2.5)

    def test_avanza_segun_speed(self, demo_timeline):
        s = AppState(paused=False, playback_speed=4.0, playback_time=0.0)
        tick(s, dt=0.5, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(2.0)

    def test_speed_lento_avanza_menos(self, demo_timeline):
        s = AppState(paused=False, playback_speed=0.25, playback_time=0.0)
        tick(s, dt=1.0, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(0.25)

    def test_auto_pausa_al_alcanzar_duration(self, demo_timeline):
        """Cruzar la duration: clampea Y pausa."""
        d = demo_timeline.duration
        s = AppState(paused=False, playback_speed=1.0, playback_time=d - 0.1)
        tick(s, dt=1.0, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(d)
        assert s.paused is True

    def test_exactamente_en_duration_pausa(self, demo_timeline):
        """Caso borde: dt te deja justo en duration."""
        d = demo_timeline.duration
        s = AppState(paused=False, playback_speed=1.0, playback_time=d - 1.0)
        tick(s, dt=1.0, timeline=demo_timeline)
        assert s.paused is True
        assert s.playback_time == pytest.approx(d)

    def test_no_se_pausa_si_no_alcanza_duration(self, demo_timeline):
        s = AppState(paused=False, playback_speed=1.0, playback_time=2.0)
        tick(s, dt=0.5, timeline=demo_timeline)
        assert s.paused is False


# ---------------------------------------------------------------------------
# Handlers individuales
# ---------------------------------------------------------------------------


class TestOnPlayPause:
    def test_toggle_de_pausado_a_play(self):
        s = AppState(paused=True)
        on_play_pause(s)
        assert s.paused is False

    def test_toggle_de_play_a_pausado(self):
        s = AppState(paused=False)
        on_play_pause(s)
        assert s.paused is True

    def test_borra_selected_failure(self):
        s = AppState(paused=True, selected_failure=2)
        on_play_pause(s)
        assert s.selected_failure is None

    def test_no_modifica_playback_time(self):
        s = AppState(paused=True, playback_time=5.5)
        on_play_pause(s)
        assert s.playback_time == 5.5


class TestOnHome:
    def test_resetea_playback_time_a_cero(self, demo_timeline):
        s = AppState(playback_time=10.0, paused=False)
        on_home(s)
        assert s.playback_time == 0.0

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False, playback_time=5.0)
        on_home(s)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(selected_failure=1)
        on_home(s)
        assert s.selected_failure is None


class TestOnEnd:
    def test_salta_a_duration(self, demo_timeline):
        s = AppState(playback_time=0.0, paused=False)
        on_end(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(demo_timeline.duration)

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False)
        on_end(s, timeline=demo_timeline)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(selected_failure=0)
        on_end(s, timeline=demo_timeline)
        assert s.selected_failure is None


class TestOnStepBack:
    def test_desde_un_snapshot_intermedio_va_al_anterior(self, demo_timeline):
        """playback_time = snapshot_times[3] → debería ir a snapshot_times[2]."""
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[3], paused=False)
        on_step_back(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_desde_entre_snapshots_va_al_anterior(self, demo_timeline):
        """playback_time entre times[2] y times[3] → va a times[2]."""
        times = demo_timeline.snapshot_times
        midpoint = (times[2] + times[3]) / 2
        s = AppState(playback_time=midpoint)
        on_step_back(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_en_el_primer_snapshot_no_se_mueve(self, demo_timeline):
        """playback_time = 0 → no hay snapshot anterior, se queda en 0."""
        s = AppState(playback_time=0.0)
        on_step_back(s, timeline=demo_timeline)
        assert s.playback_time == 0.0

    def test_antes_del_inicio_no_se_mueve(self, demo_timeline):
        """playback_time negativo (caso patológico) → bisect_left devuelve
        0, idx-1 < 0, no se mueve. Defensivo."""
        s = AppState(playback_time=0.0)
        # No usamos s = AppState(playback_time=-1) porque viola el invariante.
        # Caso real: pulsar step_back varias veces estando en 0.
        on_step_back(s, timeline=demo_timeline)
        on_step_back(s, timeline=demo_timeline)
        assert s.playback_time == 0.0

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False, playback_time=10.0)
        on_step_back(s, timeline=demo_timeline)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(playback_time=10.0, selected_failure=0)
        on_step_back(s, timeline=demo_timeline)
        assert s.selected_failure is None


class TestOnStepForward:
    def test_desde_un_snapshot_va_al_siguiente(self, demo_timeline):
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[2])
        on_step_forward(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[3])

    def test_desde_entre_snapshots_va_al_siguiente(self, demo_timeline):
        times = demo_timeline.snapshot_times
        midpoint = (times[2] + times[3]) / 2
        s = AppState(playback_time=midpoint)
        on_step_forward(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[3])

    def test_en_el_ultimo_snapshot_no_se_mueve(self, demo_timeline):
        """playback_time = duration → no hay siguiente."""
        s = AppState(playback_time=demo_timeline.duration)
        on_step_forward(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(demo_timeline.duration)

    def test_step_forward_multiple_recorre_todos_los_snapshots(self, demo_timeline):
        """Stepping forward N veces desde 0 nos lleva al snapshot N (o al final)."""
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=0.0)
        for i in range(1, len(times)):
            on_step_forward(s, timeline=demo_timeline)
            assert s.playback_time == pytest.approx(times[i]), (
                f"tras {i} pasos esperaba times[{i}]={times[i]}, "
                f"obtuve {s.playback_time}"
            )

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False, playback_time=5.0)
        on_step_forward(s, timeline=demo_timeline)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(playback_time=5.0, selected_failure=1)
        on_step_forward(s, timeline=demo_timeline)
        assert s.selected_failure is None


class TestOnFrameForward:
    def test_avanza_un_frame(self, demo_timeline):
        s = AppState(playback_time=5.0, paused=False)
        on_frame_forward(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(5.0 + FRAME_STEP)

    def test_clamp_a_duration(self, demo_timeline):
        s = AppState(playback_time=demo_timeline.duration)
        on_frame_forward(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(demo_timeline.duration)

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False, playback_time=5.0)
        on_frame_forward(s, timeline=demo_timeline)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(playback_time=5.0, selected_failure=0)
        on_frame_forward(s, timeline=demo_timeline)
        assert s.selected_failure is None


class TestOnFrameBack:
    def test_retrocede_un_frame(self, demo_timeline):
        s = AppState(playback_time=5.0, paused=False)
        on_frame_back(s, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(5.0 - FRAME_STEP)

    def test_clamp_a_cero(self, demo_timeline):
        s = AppState(playback_time=0.0)
        on_frame_back(s, timeline=demo_timeline)
        assert s.playback_time == 0.0

    def test_pausa(self, demo_timeline):
        s = AppState(paused=False, playback_time=5.0)
        on_frame_back(s, timeline=demo_timeline)
        assert s.paused is True

    def test_borra_selected_failure(self, demo_timeline):
        s = AppState(playback_time=5.0, selected_failure=0)
        on_frame_back(s, timeline=demo_timeline)
        assert s.selected_failure is None


class TestOnSpeedButton:
    @pytest.mark.parametrize("speed", list(ALLOWED_SPEEDS))
    def test_asigna_velocidad_permitida(self, speed):
        s = AppState(playback_speed=1.0)
        on_speed_button(s, speed)
        assert s.playback_speed == speed

    def test_velocidad_no_permitida_lanza_value_error(self):
        s = AppState()
        with pytest.raises(ValueError, match="no permitida"):
            on_speed_button(s, 0.7)

    def test_no_modifica_paused(self):
        s = AppState(paused=False)
        on_speed_button(s, 2.0)
        assert s.paused is False
        s2 = AppState(paused=True)
        on_speed_button(s2, 2.0)
        assert s2.paused is True

    def test_no_modifica_selected_failure(self):
        """Cambiar velocidad mirando un fallo no debe deselecionarlo."""
        s = AppState(selected_failure=1)
        on_speed_button(s, 2.0)
        assert s.selected_failure == 1


class TestOnFailureClicked:
    def test_salta_al_start_time_y_pausa(self, failure_timeline_and_starts):
        tl, starts = failure_timeline_and_starts
        # 1 fallo a t=2.0.
        s = AppState(playback_time=0.0, paused=False)
        on_failure_clicked(s, 0, starts[0], timeline=tl, failures_count=1)
        assert s.playback_time == pytest.approx(starts[0])
        assert s.paused is True

    def test_marca_selected_failure(self, failure_timeline_and_starts):
        tl, starts = failure_timeline_and_starts
        s = AppState()
        on_failure_clicked(s, 0, starts[0], timeline=tl, failures_count=1)
        assert s.selected_failure == 0

    def test_indice_fuera_de_rango_lanza(self, failure_timeline_and_starts):
        tl, _ = failure_timeline_and_starts
        s = AppState()
        with pytest.raises(ValueError, match="fuera de rango"):
            on_failure_clicked(s, 5, 2.0, timeline=tl, failures_count=1)
        with pytest.raises(ValueError, match="fuera de rango"):
            on_failure_clicked(s, -1, 2.0, timeline=tl, failures_count=1)

    def test_start_time_negativo_lanza(self, failure_timeline_and_starts):
        tl, _ = failure_timeline_and_starts
        s = AppState()
        with pytest.raises(ValueError, match="start_time"):
            on_failure_clicked(s, 0, -1.0, timeline=tl, failures_count=1)

    def test_start_time_excediendo_duration_se_clampea(
        self, failure_timeline_and_starts
    ):
        """Defensivo: si por alguna razón el start_time excede duration
        (improbable pero teóricamente posible si el plan tenía Commands
        a futuro y todos fallaron), clampeamos."""
        tl, _ = failure_timeline_and_starts
        s = AppState()
        huge = tl.duration + 100.0
        on_failure_clicked(s, 0, huge, timeline=tl, failures_count=1)
        assert s.playback_time == pytest.approx(tl.duration)


# ---------------------------------------------------------------------------
# Despachador handle_pygame_event
# ---------------------------------------------------------------------------


class TestHandlePygameEvent:
    def test_space_alterna_pausado(self, demo_timeline):
        s = AppState(paused=True)
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.paused is False

    def test_left_es_step_back(self, demo_timeline):
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[3])
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_right_es_step_forward(self, demo_timeline):
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[1])
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_home(self, demo_timeline):
        s = AppState(playback_time=10.0)
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == 0.0
        assert s.paused is True

    def test_end(self, demo_timeline):
        s = AppState(playback_time=0.0)
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == pytest.approx(demo_timeline.duration)

    def test_videoresize_actualiza_window_size(self, demo_timeline):
        s = AppState()
        ev = pygame.event.Event(pygame.VIDEORESIZE, w=1024, h=768, size=(1024, 768))
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.window_size == (1024, 768)

    def test_otras_teclas_ignoradas(self, demo_timeline):
        """Pulsar una tecla no mapeada no debe hacer nada."""
        s_before = AppState(playback_time=5.0, paused=True, playback_speed=2.0)
        s = AppState(playback_time=5.0, paused=True, playback_speed=2.0)
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == s_before.playback_time
        assert s.paused == s_before.paused
        assert s.playback_speed == s_before.playback_speed

    def test_otros_event_types_ignorados(self, demo_timeline):
        """Eventos no manejados (MOUSEMOTION, etc.) no deben tocar nada."""
        s = AppState(playback_time=5.0, paused=False)
        ev = pygame.event.Event(pygame.MOUSEMOTION, pos=(100, 100), rel=(1, 1), buttons=(0, 0, 0))
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.playback_time == 5.0
        assert s.paused is False


# ---------------------------------------------------------------------------
# Despachador handle_hud_intent
# ---------------------------------------------------------------------------


class TestHandleHudIntent:
    def test_play_pause_intent(self, demo_timeline):
        s = AppState(paused=True)
        handle_hud_intent(s, HudIntent(kind="play_pause"), timeline=demo_timeline)
        assert s.paused is False

    def test_step_back_intent(self, demo_timeline):
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[3])
        handle_hud_intent(s, HudIntent(kind="step_back"), timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_step_forward_intent(self, demo_timeline):
        times = demo_timeline.snapshot_times
        s = AppState(playback_time=times[1])
        handle_hud_intent(s, HudIntent(kind="step_forward"), timeline=demo_timeline)
        assert s.playback_time == pytest.approx(times[2])

    def test_home_intent(self, demo_timeline):
        s = AppState(playback_time=10.0)
        handle_hud_intent(s, HudIntent(kind="home"), timeline=demo_timeline)
        assert s.playback_time == 0.0

    def test_end_intent(self, demo_timeline):
        s = AppState(playback_time=0.0)
        handle_hud_intent(s, HudIntent(kind="end"), timeline=demo_timeline)
        assert s.playback_time == pytest.approx(demo_timeline.duration)

    def test_speed_intent(self, demo_timeline):
        s = AppState(playback_speed=1.0)
        handle_hud_intent(s, HudIntent(kind="speed", payload=2.0), timeline=demo_timeline)
        assert s.playback_speed == 2.0

    def test_speed_intent_sin_payload_falla(self, demo_timeline):
        s = AppState()
        with pytest.raises(AssertionError, match="speed"):
            handle_hud_intent(
                s, HudIntent(kind="speed", payload=None), timeline=demo_timeline
            )

    def test_failure_select_no_se_procesa_aqui(self, demo_timeline):
        """failure_select requiere contexto extra; debe lanzar."""
        s = AppState()
        with pytest.raises(ValueError, match="failure_select"):
            handle_hud_intent(
                s,
                HudIntent(kind="failure_select", payload=0),
                timeline=demo_timeline,
            )


class TestHandleHudFailureSelect:
    def test_aplica_correctamente(self, failure_timeline_and_starts):
        tl, starts = failure_timeline_and_starts
        s = AppState()
        intent = HudIntent(kind="failure_select", payload=0)
        handle_hud_failure_select(s, intent, starts, timeline=tl)
        assert s.playback_time == pytest.approx(starts[0])
        assert s.paused is True
        assert s.selected_failure == 0


# ---------------------------------------------------------------------------
# Cámara: handlers puros
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pan_drag_state():
    """Garantiza aislamiento entre tests del arrastre con botón central.

    El controller mantiene un módulo-local `_pan_drag_last` que vive
    entre MOUSEBUTTONDOWN(2) y MOUSEBUTTONUP(2). Si un test deja un
    arrastre activo y otro test ejecuta MOUSEMOTION, el segundo
    mutaría state.pan inesperadamente. Este fixture autouse resetea esa
    variable al inicio (y por simetría al final) de cada test.
    """
    import droneplan_viz_app.controller as ctrl
    ctrl._pan_drag_last = None
    yield
    ctrl._pan_drag_last = None


class TestOnZoom:
    """Handler puro: multiplica state.zoom por factor, satura al rango."""

    def test_factor_mayor_que_uno_amplia(self):
        s = AppState()
        on_zoom(s, ZOOM_STEP)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_factor_menor_que_uno_reduce(self):
        s = AppState()
        on_zoom(s, 1.0 / ZOOM_STEP)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM / ZOOM_STEP)

    def test_satura_al_maximo_no_lanza(self):
        s = AppState()
        on_zoom(s, 100.0)
        assert s.zoom == ZOOM_MAX

    def test_satura_al_minimo_no_lanza(self):
        s = AppState()
        on_zoom(s, 0.001)
        assert s.zoom == ZOOM_MIN

    def test_aplicaciones_repetidas_son_acumulativas(self):
        """5 ticks de zoom in deben coincidir con ZOOM_STEP**5 (o saturar)."""
        s = AppState()
        for _ in range(5):
            on_zoom(s, ZOOM_STEP)
        esperado = min(ZOOM_MAX, DEFAULT_ZOOM * ZOOM_STEP ** 5)
        assert s.zoom == pytest.approx(esperado)

    def test_estado_resultante_pasa_validate(self):
        """Tras cualquier secuencia de zooms el state debe seguir válido."""
        s = AppState()
        for _ in range(20):
            on_zoom(s, ZOOM_STEP)
        s.validate()
        for _ in range(40):
            on_zoom(s, 1.0 / ZOOM_STEP)
        s.validate()


class TestRejillaZoom:
    """Rejilla FIJA de zoom como ESCALA ABSOLUTA del sprite respecto al nativo
    (1.0 = nativo 1:1): 10 niveles 1.25^k, k=-5..+4, idénticos en toda escena.
    El zoom de ARRANQUE lo elige fitting_zoom(ss) para que todo entre en
    pantalla; el reset de cámara vuelve a ese home_zoom."""

    def test_rejilla_es_escala_absoluta_fija(self):
        from droneplan_viz_app.app_state import ZOOM_LEVELS
        esperado = tuple(ZOOM_STEP ** k for k in range(-5, 5))
        assert tuple(pytest.approx(v) for v in ZOOM_LEVELS) == esperado
        assert len(ZOOM_LEVELS) == 10
        assert ZOOM_MIN == pytest.approx(ZOOM_STEP ** -5)   # zoom-out máximo
        assert ZOOM_MAX == pytest.approx(ZOOM_STEP ** 4)    # zoom-in máximo
        assert DEFAULT_ZOOM == pytest.approx(1.0)           # sprite nativo 1:1

    def test_zoom_out_satura_en_minimo(self):
        s = AppState()
        for _ in range(20):
            on_zoom(s, 1.0 / ZOOM_STEP)
        assert s.zoom == pytest.approx(ZOOM_MIN)

    def test_zoom_in_satura_en_maximo(self):
        s = AppState()
        for _ in range(20):
            on_zoom(s, ZOOM_STEP)
        assert s.zoom == pytest.approx(ZOOM_MAX)

    def test_niveles_alcanzables_son_la_rejilla(self):
        from droneplan_viz_app.app_state import ZOOM_LEVELS
        # Recorriendo de mínimo a máximo, los valores visitados = ZOOM_LEVELS.
        s = AppState()
        for _ in range(20):
            on_zoom(s, 1.0 / ZOOM_STEP)  # al mínimo
        visitados = [s.zoom]
        for _ in range(20):
            on_zoom(s, ZOOM_STEP)
            if visitados[-1] != s.zoom:
                visitados.append(s.zoom)
        assert tuple(pytest.approx(v) for v in visitados) == tuple(ZOOM_LEVELS)

    def test_fitting_zoom_elige_nivel_que_encaja(self):
        from droneplan_viz_app.app_state import fitting_zoom, ZOOM_LEVELS
        # ss=1 (escena dispersa): cabe a resolución nativa.
        assert fitting_zoom(1.0) == pytest.approx(1.0)
        # ss=2 (densa): la vista completa cae en 0.5 → mayor nivel <= 0.5.
        f = fitting_zoom(2.0)
        assert f <= 0.5 + 1e-9 and f == pytest.approx(ZOOM_STEP ** -4)
        # ss enorme: satura al mínimo (zoom-out máximo).
        assert fitting_zoom(100.0) == pytest.approx(ZOOM_MIN)
        # El resultado SIEMPRE es un nivel de la rejilla fija.
        for ss in (1.0, 1.5, 2.02, 3.05, 10.0):
            assert any(abs(fitting_zoom(ss) - L) < 1e-9 for L in ZOOM_LEVELS)

    def test_reset_vuelve_al_home_zoom(self):
        # El reset de cámara vuelve al zoom de arranque de la escena (home_zoom),
        # no a un 1.0 fijo (que en escena densa estaría muy acercado).
        s = AppState()
        s.home_zoom = ZOOM_STEP ** -3  # como si la escena fuese densa
        s.zoom = ZOOM_MAX
        s.pan = (50, 50)
        on_camera_reset(s)
        assert s.zoom == pytest.approx(s.home_zoom)
        assert s.pan == (0, 0)


class TestOnPan:
    """Handler puro: suma a state.pan, sin clamp."""

    def test_pan_acumula(self):
        s = AppState()
        on_pan(s, 50, -30)
        assert s.pan == (50, -30)
        on_pan(s, 10, 100)
        assert s.pan == (60, 70)

    def test_pan_permite_valores_grandes(self):
        """No hay clamp: arrastrar muy lejos es legal (se recupera con reset)."""
        s = AppState()
        on_pan(s, 100_000, -100_000)
        assert s.pan == (100_000, -100_000)
        s.validate()

    def test_pan_convierte_floats_a_int(self):
        """Si llega un dx/dy float (algunos drivers lo permiten), int()
        evita corromper la tupla."""
        s = AppState()
        on_pan(s, 1.7, -2.9)  # type: ignore[arg-type]
        assert s.pan == (1, -2)


class TestOnCameraReset:
    def test_restaura_defaults(self):
        s = AppState(zoom=1.5, pan=(120, -80))
        on_camera_reset(s)
        assert s.zoom == DEFAULT_ZOOM
        assert s.pan == DEFAULT_PAN


# ---------------------------------------------------------------------------
# Cámara: despacho desde eventos pygame
# ---------------------------------------------------------------------------


class TestCameraPygameEvents:
    """Verifica que handle_pygame_event traduce eventos pygame en
    mutaciones correctas del state."""

    def test_mousewheel_arriba_zoom_in(self, demo_timeline):
        s = AppState()
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_mousewheel_abajo_zoom_out(self, demo_timeline):
        s = AppState()
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=-1, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM / ZOOM_STEP)

    def test_mousewheel_y_cero_no_hace_nada(self, demo_timeline):
        """Defensivo: algunos drivers reportan y=0 en eventos espurios."""
        s = AppState()
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=0, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == DEFAULT_ZOOM

    def test_mousewheel_sobre_world_hace_zoom(self, demo_timeline, monkeypatch):
        """Con world_region pasado, la rueda hace zoom solo si el cursor
        está dentro de esa región."""
        s = AppState()
        world = pygame.Rect(0, 56, 900, 644)
        # Mock de get_pos: cursor dentro del world. No usamos set_pos
        # porque requiere video inicializado (estos tests son puros).
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (450, 378))
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline, world_region=world)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_mousewheel_fuera_de_world_no_hace_zoom(self, demo_timeline, monkeypatch):
        """Con world_region pasado, la rueda NO hace zoom si el cursor
        está fuera de esa región (p. ej. sobre el panel lateral). El
        scroll del inventario lo gestiona pygame_gui por separado."""
        s = AppState()
        world = pygame.Rect(0, 56, 900, 644)
        # Cursor sobre el panel lateral (x=1090, fuera del world).
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (1090, 400))
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline, world_region=world)
        assert s.zoom == DEFAULT_ZOOM  # sin cambios.

    def test_mousewheel_sin_world_region_siempre_zoom(self, demo_timeline):
        """Retrocompatibilidad: sin world_region, la rueda hace zoom
        independientemente de la posición del cursor (comportamiento
        anterior, usado por tests y posibles llamadas legacy). No
        consulta get_pos, así que no necesita video ni mock."""
        s = AppState()
        ev = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_tecla_mas_zoom_in(self, demo_timeline):
        s = AppState()
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PLUS)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_tecla_igual_tambien_zoom_in(self, demo_timeline):
        """K_EQUALS (la tecla física de +/= sin Shift) también amplía,
        para no obligar al usuario a pulsar Shift."""
        s = AppState()
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM * ZOOM_STEP)

    def test_tecla_menos_zoom_out(self, demo_timeline):
        s = AppState()
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == pytest.approx(DEFAULT_ZOOM / ZOOM_STEP)

    def test_tecla_cero_resetea(self, demo_timeline):
        s = AppState(zoom=1.5, pan=(100, -50))
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0)
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.zoom == DEFAULT_ZOOM
        assert s.pan == DEFAULT_PAN

    def test_arrastre_boton_central_aplica_pan(self, demo_timeline):
        """Secuencia realista: down → motion → motion → up.
        El delta acumulado debe llegar al state.pan."""
        s = AppState()
        # Inicio del arrastre en (200, 150).
        down = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=2, pos=(200, 150)
        )
        handle_pygame_event(s, down, timeline=demo_timeline)
        assert s.pan == (0, 0)  # solo el down no mueve nada.

        # Primer motion: el cursor se va a (250, 120). Delta (+50, -30).
        m1 = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(250, 120), rel=(50, -30),
            buttons=(0, 1, 0),
        )
        handle_pygame_event(s, m1, timeline=demo_timeline)
        assert s.pan == (50, -30)

        # Segundo motion: cursor a (260, 110). Delta (+10, -10).
        m2 = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(260, 110), rel=(10, -10),
            buttons=(0, 1, 0),
        )
        handle_pygame_event(s, m2, timeline=demo_timeline)
        assert s.pan == (60, -40)

        # Up termina el arrastre.
        up = pygame.event.Event(
            pygame.MOUSEBUTTONUP, button=2, pos=(260, 110)
        )
        handle_pygame_event(s, up, timeline=demo_timeline)

        # Tras el up, nuevos MOUSEMOTION no deben mutar el pan.
        m3 = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(500, 500), rel=(240, 390),
            buttons=(0, 0, 0),
        )
        handle_pygame_event(s, m3, timeline=demo_timeline)
        assert s.pan == (60, -40)  # sin cambio.

    def test_mousemotion_sin_arrastre_activo_no_muta_pan(self, demo_timeline):
        """Mover el ratón con el botón central NO pulsado no afecta a pan."""
        s = AppState()
        ev = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(300, 300), rel=(100, 100),
            buttons=(0, 0, 0),
        )
        handle_pygame_event(s, ev, timeline=demo_timeline)
        assert s.pan == (0, 0)

    def test_boton_izquierdo_no_inicia_arrastre(self, demo_timeline):
        """Solo botón 2 (central) inicia pan. El izquierdo (1) se reserva
        para selección/click-to-focus futuro."""
        s = AppState()
        down = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(200, 150)
        )
        handle_pygame_event(s, down, timeline=demo_timeline)
        motion = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(300, 250), rel=(100, 100),
            buttons=(1, 0, 0),
        )
        handle_pygame_event(s, motion, timeline=demo_timeline)
        assert s.pan == (0, 0)


# ---------------------------------------------------------------------------
# Cámara: focus_on_entity (click-to-focus, base para el inventario futuro)
# ---------------------------------------------------------------------------


class TestFocusOnEntity:
    """El helper combina locate_entity (D2) con la fórmula de centrado
    documentada en la nota de D2 para producir el (zoom, pan) que deja
    a la entidad en el centro de la subsurface.
    """

    def test_zoom_1_centra_drone_en_la_surface(self, demo_world):
        """Con zoom=1, el pan compensa exactamente el offset entre la
        posición de la entidad y el centro de la surface."""
        world, theme = demo_world
        from droneplan_viz.render import locate_entity
        sw, sh = 900, 600
        lx, ly = locate_entity(world, "d1", (sw, sh), theme=theme)

        s = AppState()
        focus_on_entity(s, world, "d1", (sw, sh), theme=theme, zoom=1.0)
        assert s.zoom == 1.0
        assert s.pan == (sw // 2 - lx, sh // 2 - ly)

    def test_zoom_2_dobla_la_correccion(self, demo_world):
        """A zoom 2, el zoom expande desde el centro por lo que la
        entidad queda al doble de distancia del centro. El pan debe
        compensar el doble."""
        world, theme = demo_world
        from droneplan_viz.render import locate_entity
        sw, sh = 900, 600
        lx, ly = locate_entity(world, "d1", (sw, sh), theme=theme)

        s = AppState()
        focus_on_entity(s, world, "d1", (sw, sh), theme=theme, zoom=1.5)
        # Fórmula explícita de D2.
        expected_pan = (
            int(sw // 2 - (sw / 2 + (lx - sw / 2) * 1.5)),
            int(sh // 2 - (sh / 2 + (ly - sh / 2) * 1.5)),
        )
        assert s.zoom == 1.5
        assert s.pan == expected_pan

    def test_zoom_none_mantiene_el_actual_del_state(self, demo_world):
        """Si no se pasa zoom, focus usa el state.zoom existente."""
        world, theme = demo_world
        s = AppState(zoom=1.5)
        focus_on_entity(s, world, "d1", (900, 600), theme=theme)
        assert s.zoom == 1.5  # sin cambios al zoom.
        # Y el pan se calcula con z=1.5.

    def test_zoom_explicito_satura(self, demo_world):
        """zoom muy alto se satura a ZOOM_MAX antes de calcular el pan."""
        world, theme = demo_world
        s = AppState()
        focus_on_entity(s, world, "d1", (900, 600), theme=theme, zoom=100.0)
        assert s.zoom == ZOOM_MAX

    def test_id_inexistente_propaga_keyerror(self, demo_world):
        """locate_entity lanza KeyError; focus debe propagarlo sin envolver."""
        world, theme = demo_world
        s = AppState()
        with pytest.raises(KeyError):
            focus_on_entity(s, world, "drone_inventado", (900, 600), theme=theme)

    def test_package_held_by_arm_propaga_valueerror(self):
        """Cuando el package está HeldByArm o InTransporter, locate_entity
        lanza ValueError; focus lo propaga. El llamante debe enfocar el
        drone/transporter que sostiene el paquete (decisión documentada
        en la nota de D2)."""
        # Construimos un snapshot donde un paquete está siendo cargado.
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        # Snapshot 3 = snap_start de Move(d1,casa1): d1 lleva pkg_med1.
        snap = runner.history.at(3)
        theme = Theme.default()

        s = AppState()
        with pytest.raises(ValueError, match="pkg_med1"):
            focus_on_entity(s, snap.world, "pkg_med1", (900, 600), theme=theme)

    def test_estado_resultante_pasa_validate(self, demo_world):
        """Tras un focus arbitrario, AppState debe seguir siendo válido
        (invariantes de zoom y pan preservados)."""
        world, theme = demo_world
        for entity in ("d1", "d2", "casa1", "hospital", "deposito"):
            s = AppState()
            focus_on_entity(s, world, entity, (900, 600), theme=theme, zoom=2.5)
            s.validate()

    def test_locations_son_focuseables(self, demo_world):
        """Las 4 locations del demo (casa1, casa2, hospital, deposito)
        deben ser todas focuseables sin error."""
        world, theme = demo_world
        for loc in world.locations:
            s = AppState()
            focus_on_entity(s, world, loc, (900, 600), theme=theme)
            # Sanity: el pan resultante debe ser distinto de (0,0) para
            # locations que no estén en el centro inicial. Como mínimo
            # alguno de los dos componentes es no-cero.
            assert s.pan != (0, 0) or loc == "centro_imaginario", (
                f"focus a {loc} dio pan=(0,0); probablemente no se aplicó"
            )


# ---------------------------------------------------------------------------
# Cámara: handle_hud_inventory_focus (click sobre item del inventario)
# ---------------------------------------------------------------------------


class TestHandleHudInventoryFocus:
    """handle_hud_inventory_focus es el helper que la app llama cuando
    recibe un HudIntent kind='inventory_focus'. Aplica focus_on_entity
    sobre la entidad identificada en el payload."""

    def test_focus_a_entidad_valida_muta_pan(self, demo_world):
        """Click en una location existente debe mover el pan al centrarla
        en la subsurface."""
        world, theme = demo_world
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="casa1")
        handle_hud_inventory_focus(s, intent, world, (900, 600), theme=theme)
        # pan != defaults (alguna componente no-cero porque casa1 no está
        # centrada en (450, 300)).
        assert s.pan != (0, 0)

    def test_focus_a_drone_muta_pan(self, demo_world):
        world, theme = demo_world
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="d1")
        handle_hud_inventory_focus(s, intent, world, (900, 600), theme=theme)
        assert s.pan != (0, 0)

    def test_zoom_no_cambia(self, demo_world):
        """El helper SOLO muta pan; zoom queda como estaba en state."""
        world, theme = demo_world
        s = AppState(zoom=1.5)
        intent = HudIntent(kind="inventory_focus", payload="casa1")
        handle_hud_inventory_focus(s, intent, world, (900, 600), theme=theme)
        assert s.zoom == 1.5

    def test_id_inexistente_no_lanza(self, demo_world):
        """Click sobre entidad que ya no existe en el snapshot actual
        (condición de carrera al refrescar): silenciar el error, NO mutar."""
        world, theme = demo_world
        s = AppState()
        pan_inicial = s.pan
        intent = HudIntent(kind="inventory_focus", payload="entidad_inventada")
        # No debe lanzar KeyError.
        handle_hud_inventory_focus(s, intent, world, (900, 600), theme=theme)
        assert s.pan == pan_inicial  # sin cambios.

    def test_paquete_held_by_arm_no_lanza_y_no_muta(self):
        """Click sobre un paquete HeldByArm: locate_entity lanza
        ValueError; el helper lo captura y NO muta. El usuario debe
        enfocar el drone que lo sostiene."""
        from droneplan_viz_app.scenarios import build_demo_scenario
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        snap = runner.history.at(3)  # d1 lleva pkg_med1.
        theme = Theme.default()

        s = AppState()
        pan_inicial = s.pan
        intent = HudIntent(kind="inventory_focus", payload="pkg_med1")
        handle_hud_inventory_focus(s, intent, snap.world, (900, 600), theme=theme)
        assert s.pan == pan_inicial

    def test_estado_resultante_valida_invariantes(self, demo_world):
        """Tras un focus, AppState debe seguir siendo válido."""
        world, theme = demo_world
        s = AppState(zoom=1.5)
        intent = HudIntent(kind="inventory_focus", payload="hospital")
        handle_hud_inventory_focus(s, intent, world, (900, 600), theme=theme)
        s.validate()


class TestHandleHudIntentRechazaInventoryFocus:
    """handle_hud_intent (despachador genérico) NO debe procesar
    inventory_focus; debe lanzar ValueError con mensaje útil."""

    def test_lanza_value_error(self, demo_timeline):
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="d1")
        with pytest.raises(ValueError, match="inventory_focus"):
            handle_hud_intent(s, intent, timeline=demo_timeline)


# ---------------------------------------------------------------------------
# Inventario: plegado de secciones (handle_hud_inventory_toggle_section)
# ---------------------------------------------------------------------------


class TestHandleHudInventoryToggleSection:
    """El click sobre una cabecera del inventario alterna el estado
    plegado/desplegado de esa categoría en state.inventory_collapsed."""

    def test_plegar_categoria_la_anade_al_set(self):
        s = AppState()
        assert "DRONES" not in s.inventory_collapsed
        intent = HudIntent(kind="inventory_toggle_section", payload="DRONES")
        handle_hud_inventory_toggle_section(s, intent)
        assert "DRONES" in s.inventory_collapsed

    def test_desplegar_categoria_la_quita_del_set(self):
        s = AppState()
        s.inventory_collapsed.add("LOCATIONS")
        intent = HudIntent(
            kind="inventory_toggle_section", payload="LOCATIONS"
        )
        handle_hud_inventory_toggle_section(s, intent)
        assert "LOCATIONS" not in s.inventory_collapsed

    def test_toggle_es_idempotente_en_pares(self):
        """Dos toggles seguidos dejan el estado igual que al principio."""
        s = AppState()
        intent = HudIntent(kind="inventory_toggle_section", payload="PERSONAS")
        handle_hud_inventory_toggle_section(s, intent)
        handle_hud_inventory_toggle_section(s, intent)
        assert "PERSONAS" not in s.inventory_collapsed

    def test_toggle_no_afecta_otras_categorias(self):
        s = AppState()
        s.inventory_collapsed.add("DRONES")
        intent = HudIntent(kind="inventory_toggle_section", payload="PAQUETES")
        handle_hud_inventory_toggle_section(s, intent)
        # DRONES sigue plegado, PAQUETES recién plegado.
        assert s.inventory_collapsed == {"DRONES", "PAQUETES"}

    def test_estado_resultante_es_valido(self):
        s = AppState()
        intent = HudIntent(kind="inventory_toggle_section", payload="DRONES")
        handle_hud_inventory_toggle_section(s, intent)
        s.validate()


class TestHandleHudIntentRechazaInventoryToggle:
    """handle_hud_intent genérico NO procesa inventory_toggle_section."""

    def test_lanza_value_error(self, demo_timeline):
        s = AppState()
        intent = HudIntent(kind="inventory_toggle_section", payload="DRONES")
        with pytest.raises(ValueError, match="inventory_toggle_section"):
            handle_hud_intent(s, intent, timeline=demo_timeline)


# ---------------------------------------------------------------------------
# Seguimiento de cámara: followed_entity + focus_on_entity_frame
# ---------------------------------------------------------------------------


def _two_snaps_drone_move():
    """(snap_a, snap_b, theme): DroneMove real d1 casa1→casa2.

    Ambos snapshots producidos por el mismo Move, para que
    classify_transition lo reconozca como TransitionDroneMove (un
    produced_by=None se clasificaría como estático y no interpolaría).
    """
    from droneplan_viz.commands import Move
    from droneplan_viz.domain import Drone, Location, World
    from droneplan_viz.domain.drone_state import DroneState
    from droneplan_viz.history import WorldSnapshot
    from droneplan_viz.domain.metrics import MetricsTracker

    def w(loc):
        return World(
            locations={"casa1": Location(id="casa1"), "casa2": Location(id="casa2")},
            drones={"d1": Drone(id="d1", position=loc, arms=(), state=DroneState.IDLE)},
            costs={("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0},
        )

    move = Move(drone_id="d1", destination_id="casa2", duration=5.0, command_id="mv")
    snap_a = WorldSnapshot(world=w("casa1"), metrics=MetricsTracker(),
                           produced_by=move, timestamp=0.0)
    snap_b = WorldSnapshot(world=w("casa2"), metrics=MetricsTracker(),
                           produced_by=move, timestamp=5.0)
    return snap_a, snap_b, Theme.default()


class TestCameraFollow:
    SIZE = (900, 600)

    def test_inventory_focus_activa_seguimiento(self, demo_world):
        world, theme = demo_world
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="d1")
        handle_hud_inventory_focus(s, intent, world, self.SIZE, theme=theme)
        assert s.followed_entity == "d1"

    def test_focus_a_id_inexistente_no_activa_seguimiento(self, demo_world):
        world, theme = demo_world
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="fantasma")
        handle_hud_inventory_focus(s, intent, world, self.SIZE, theme=theme)
        assert s.followed_entity is None

    def test_paquete_sostenido_si_activa_seguimiento(self):
        # Un paquete sostenido por un drone YA es seguible: el seguimiento
        # por frame lo sitúa en el brazo del drone. Enfocarlo desde el
        # inventario activa followed_entity aunque el encuadre estático
        # inicial no sea resoluble (locate_entity lanza ValueError).
        from droneplan_viz_app.scenarios import build_demo_scenario
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        snap = runner.history.at(3)  # d1 lleva pkg_med1
        s = AppState()
        intent = HudIntent(kind="inventory_focus", payload="pkg_med1")
        handle_hud_inventory_focus(s, intent, snap.world, self.SIZE, theme=Theme.default())
        assert s.followed_entity == "pkg_med1"

    def test_pan_manual_cancela_seguimiento(self, demo_world):
        world, theme = demo_world
        s = AppState()
        s.followed_entity = "d1"
        on_pan(s, 10, 5)
        assert s.followed_entity is None

    def test_reset_camara_cancela_seguimiento(self):
        s = AppState()
        s.followed_entity = "d1"
        on_camera_reset(s)
        assert s.followed_entity is None

    def test_focus_frame_centra_en_posicion_interpolada(self):
        snap_a, snap_b, theme = _two_snaps_drone_move()
        from droneplan_viz.render import locate_entity, locate_entity_interpolated
        estatica = locate_entity(snap_a.world, "d1", self.SIZE, theme=theme)
        interp = locate_entity_interpolated(snap_a, snap_b, 0.5, "d1", self.SIZE, theme=theme)
        assert estatica != interp  # precondición: hay movimiento

        s = AppState()
        s.zoom = 1.0
        focus_on_entity_frame(s, snap_a, snap_b, 0.5, "d1", self.SIZE, theme=theme)
        # Con zoom 1, pan centra la posición INTERPOLADA (no la estática).
        sw, sh = self.SIZE
        assert s.pan == (sw // 2 - interp[0], sh // 2 - interp[1])

    def test_focus_frame_estado_valido(self):
        snap_a, snap_b, theme = _two_snaps_drone_move()
        s = AppState(zoom=1.5)
        focus_on_entity_frame(s, snap_a, snap_b, 0.3, "d1", self.SIZE, theme=theme)
        s.validate()


# ---------------------------------------------------------------------------
# Salto a una acción (panel de Acciones): NO altera play/pausa
# ---------------------------------------------------------------------------


class TestAccionSelect:
    """Clicar una acción mueve playback_time al inicio de la acción SIN
    cambiar el estado de reproducción (a diferencia del salto a un fallo,
    que sí pausa). Igual que el seguimiento de entidades."""

    def _starts(self):
        from droneplan_viz_app.hud import build_hud_plan_info
        world, plan = build_failure_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        timeline = Timeline(runner.history, theme=Theme.default())
        info = build_hud_plan_info(
            timeline.duration, timeline.snapshot_times,
            result.failures, result.plan, runner.history,
        )
        return timeline, info

    def test_action_starts_orden_cronologico_y_marca_fallo(self):
        timeline, info = self._starts()
        # Hay al menos una acción y los inicios son crecientes.
        assert len(info.action_starts) >= 1
        assert list(info.action_starts) == sorted(info.action_starts)
        # En el escenario de fallo, alguna etiqueta lleva la marca de fallo.
        assert any("FALLÓ" in lab for lab in info.action_labels)

    def test_salto_en_play_mantiene_play(self):
        timeline, info = self._starts()
        s = AppState(paused=False, playback_speed=1.0, playback_time=99.0)
        on_action_clicked(
            s, 1, info.action_starts[1],
            timeline=timeline, actions_count=len(info.action_starts),
        )
        assert s.playback_time == min(info.action_starts[1], timeline.duration)
        assert s.paused is False  # sigue en play

    def test_salto_en_pausa_mantiene_pausa(self):
        timeline, info = self._starts()
        s = AppState(paused=True, playback_time=0.0)
        on_action_clicked(
            s, 0, info.action_starts[0],
            timeline=timeline, actions_count=len(info.action_starts),
        )
        assert s.playback_time == min(info.action_starts[0], timeline.duration)
        assert s.paused is True  # sigue en pausa

    def test_no_toca_selected_failure(self):
        timeline, info = self._starts()
        s = AppState(paused=False, playback_time=0.0)
        s.selected_failure = 0
        on_action_clicked(
            s, 0, info.action_starts[0],
            timeline=timeline, actions_count=len(info.action_starts),
        )
        assert s.selected_failure == 0  # el salto a acción no lo toca

    def test_clamp_a_duration(self):
        timeline, info = self._starts()
        s = AppState(paused=False, playback_time=0.0)
        on_action_clicked(
            s, 0, timeline.duration + 100.0,
            timeline=timeline, actions_count=len(info.action_starts),
        )
        assert s.playback_time == timeline.duration

    def test_indice_fuera_de_rango_es_noop(self):
        timeline, info = self._starts()
        s = AppState(paused=False, playback_time=3.0)
        handle_hud_action_select(
            s, HudIntent(kind="action_select", payload=999),
            info.action_starts, timeline=timeline,
        )
        assert s.playback_time == 3.0  # sin cambios

    def test_handle_hud_action_select_salta(self):
        timeline, info = self._starts()
        s = AppState(paused=False, playback_time=0.0)
        idx = len(info.action_starts) - 1
        handle_hud_action_select(
            s, HudIntent(kind="action_select", payload=idx),
            info.action_starts, timeline=timeline,
        )
        assert s.playback_time == min(info.action_starts[idx], timeline.duration)
        assert s.paused is False
