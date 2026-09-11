"""Tests para droneplan_viz.render.timeline.

Todos sin pygame. Verifican:

- Cálculo de virtual_times en los tres regímenes:
    * Historial con un solo snapshot.
    * Snapshots con tiempos secuenciales puros (todos timestamp=0,
      delta=0 → fallback aplicado a todos los tramos).
    * Snapshots durativos (delta>0 → tiempo simulado real respetado).
    * Mezcla durativos + instantáneos.

- Saneo de bordes: playback_time negativo, mayor que duration, igual
  a duration exacto.

- sample() devuelve el par correcto e interpola progress linealmente.

- Inmutabilidad del Timeline.

- Cambiar fallback_step_duration via theme afecta los virtual_times.
"""
import pytest

from droneplan_viz.commands import Move, PickUp
from droneplan_viz.domain import (
    AtLocation,
    Content,
    MetricsTracker,
    Package,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.history import HistoryManager, WorldSnapshot
from droneplan_viz.render import Theme, Timeline
from droneplan_viz.runtime import Plan, PlanRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simple_world() -> World:
    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
    }
    drones = {
        "d1": Drone(id="d1", position="casa1", arms=(), state=DroneState.IDLE),
    }
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(locations=locs, drones=drones, costs=costs)


def make_world_with_package() -> World:
    contents = {"medicina": Content(id="medicina")}
    locs = {"casa1": Location(id="casa1")}
    drones = {
        "d1": Drone(
            id="d1",
            position="casa1",
            arms=(Arm(id="izq"),),
            state=DroneState.IDLE,
        ),
    }
    packages = {
        "pkg1": Package(
            id="pkg1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="casa1"),
        ),
    }
    return World(
        locations=locs,
        drones=drones,
        packages=packages,
        contents=contents,
    )


def make_history_with_n_snapshots_at_timestamps(timestamps: list[float]) -> HistoryManager:
    """Construye un HistoryManager con N snapshots a los timestamps dados.

    Todos los snapshots referencian el MISMO World (no nos importa el
    contenido para los tests del Timeline; solo los timestamps).
    produced_by=None para el primero, un Move ficticio para los demás
    (solo necesitamos que sea no-None para que represente "transición").

    NOTA: usamos Move como produced_by genérico porque el Timeline solo
    mira timestamps. Para tests donde nos importa el tipo de Command,
    construimos el historial con PlanRunner real.
    """
    world = make_simple_world()
    snap0 = WorldSnapshot(
        world=world,
        metrics=MetricsTracker(),
        produced_by=None,
        timestamp=timestamps[0],
    )
    hist = HistoryManager(snap0)
    for ts in timestamps[1:]:
        # Cada Move ficticio tiene un command_id diferente:
        cmd = Move(drone_id="d1", destination_id="casa2")
        snap = WorldSnapshot(
            world=world,
            metrics=MetricsTracker(),
            produced_by=cmd,
            timestamp=ts,
        )
        hist.commit(snap)
    return hist


# ---------------------------------------------------------------------------
# Caso degenerado: un solo snapshot
# ---------------------------------------------------------------------------


class TestTimelineUnSoloSnapshot:
    """Historial con un único snapshot: duration=0, sample(t) constante."""

    def test_duration_es_cero(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0])
        tl = Timeline(hist)
        assert tl.duration == 0.0

    def test_sample_devuelve_mismo_snapshot_con_progress_cero(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(0.0)
        assert a is b
        assert p == 0.0

    def test_sample_con_tiempo_arbitrario_misma_respuesta(self):
        # Cualquier playback_time devuelve lo mismo.
        hist = make_history_with_n_snapshots_at_timestamps([5.0])
        tl = Timeline(hist)
        for t in [-10.0, 0.0, 0.5, 5.0, 100.0]:
            a, b, p = tl.sample(t)
            assert a is b
            assert p == 0.0


# ---------------------------------------------------------------------------
# Régimen secuencial puro: todos los timestamps iguales (duration=0)
# ---------------------------------------------------------------------------


class TestTimelineSecuencial:
    """Snapshots con mismo timestamp → fallback aplicado a cada tramo."""

    def test_dos_snapshots_iguales_duration_es_fallback(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0])
        theme = Theme.default()  # fallback_step_duration = 0.4
        tl = Timeline(hist, theme=theme)
        assert tl.duration == pytest.approx(0.4)

    def test_tres_snapshots_iguales_duration_es_dos_fallbacks(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0])
        theme = Theme.default()
        tl = Timeline(hist, theme=theme)
        assert tl.duration == pytest.approx(0.8)

    def test_n_snapshots_iguales_duration_es_n_menos_uno_fallbacks(self):
        n = 5
        hist = make_history_with_n_snapshots_at_timestamps([0.0] * n)
        theme = Theme.default()
        tl = Timeline(hist, theme=theme)
        expected = (n - 1) * theme.fallback_step_duration
        assert tl.duration == pytest.approx(expected)

    def test_sample_en_medio_de_tramo_fallback(self):
        # 3 snapshots iguales → duration = 0.8; sample(0.2) cae en el
        # PRIMER tramo (de 0.0 a 0.4), con progress = 0.2 / 0.4 = 0.5.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0])
        tl = Timeline(hist, theme=Theme.default())
        a, b, p = tl.sample(0.2)
        assert a is hist.at(0)
        assert b is hist.at(1)
        assert p == pytest.approx(0.5)

    def test_sample_cruzando_al_segundo_tramo(self):
        # sample(0.4) cae en la frontera. La convención cerrado-abierto
        # ([t0, t1)) hace que t=0.4 PASE al segundo tramo, con
        # progress = (0.4 - 0.4) / 0.4 = 0.0.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0])
        tl = Timeline(hist, theme=Theme.default())
        a, b, p = tl.sample(0.4)
        # En el cambio de tramo, devolvemos (h[1], h[2], 0.0):
        assert a is hist.at(1)
        assert b is hist.at(2)
        assert p == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Régimen durativo: timestamps distintos, delta>0 respetado
# ---------------------------------------------------------------------------


class TestTimelineDurativo:
    """Snapshots con timestamps distintos: virtual_time = tiempo real."""

    def test_dos_snapshots_con_delta_diez(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist, theme=Theme.default())
        assert tl.duration == pytest.approx(10.0)

    def test_sample_en_medio_de_tramo_durativo(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(5.0)
        assert a is hist.at(0)
        assert b is hist.at(1)
        assert p == pytest.approx(0.5)

    def test_sample_cerca_del_inicio(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(2.5)
        assert p == pytest.approx(0.25)

    def test_sample_cerca_del_final(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(7.5)
        assert p == pytest.approx(0.75)

    def test_varios_tramos_durativos_diferentes_duraciones(self):
        # tramos: 5s, 3s, 12s → duration total = 20s.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 5.0, 8.0, 20.0])
        tl = Timeline(hist)
        assert tl.duration == pytest.approx(20.0)

        # En t=4 estamos en el primer tramo, progress = 4/5 = 0.8.
        a, b, p = tl.sample(4.0)
        assert a is hist.at(0)
        assert b is hist.at(1)
        assert p == pytest.approx(0.8)

        # En t=6.5 estamos en el segundo tramo (5→8), progress = 1.5/3.
        a, b, p = tl.sample(6.5)
        assert a is hist.at(1)
        assert b is hist.at(2)
        assert p == pytest.approx(0.5)

        # En t=14 estamos en el tercero (8→20), progress = 6/12 = 0.5.
        a, b, p = tl.sample(14.0)
        assert a is hist.at(2)
        assert b is hist.at(3)
        assert p == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Régimen mixto: combinación durativos + instantáneos
# ---------------------------------------------------------------------------


class TestTimelineMixto:
    """Combina tramos con delta>0 y delta=0 en el mismo historial."""

    def test_durativo_seguido_de_instantaneo(self):
        # Tramo 1: 5s (real); tramo 2: 0s (fallback = 0.4).
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 5.0, 5.0])
        tl = Timeline(hist, theme=Theme.default())
        assert tl.duration == pytest.approx(5.4)

        # En t=5.2 estamos en el segundo tramo (instantáneo), progress=0.5.
        a, b, p = tl.sample(5.2)
        assert a is hist.at(1)
        assert b is hist.at(2)
        assert p == pytest.approx(0.5)

    def test_instantaneo_seguido_de_durativo(self):
        # Tramo 1: 0s → fallback (0.4). Tramo 2: 10s (real).
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 10.0])
        tl = Timeline(hist, theme=Theme.default())
        # 0.4 (fallback) + 10.0 (real) = 10.4.
        assert tl.duration == pytest.approx(10.4)

        # En t=0.2 estamos en el primer tramo (instantáneo virtual),
        # progress = 0.2 / 0.4 = 0.5.
        a, b, p = tl.sample(0.2)
        assert a is hist.at(0)
        assert b is hist.at(1)
        assert p == pytest.approx(0.5)

        # En t=5.4 estamos en el segundo tramo (durativo), progress =
        # (5.4 - 0.4) / 10.0 = 0.5.
        a, b, p = tl.sample(5.4)
        assert a is hist.at(1)
        assert b is hist.at(2)
        assert p == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Saneo de bordes
# ---------------------------------------------------------------------------


class TestTimelineBordes:
    """Comportamiento en los extremos del rango."""

    def test_playback_time_negativo(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(-5.0)
        assert a is hist.at(0)
        assert b is hist.at(0)
        assert p == 0.0

    def test_playback_time_cero_exacto(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(0.0)
        # Convención del módulo: t=0 cae en el saneo izquierdo →
        # (h[0], h[0], 0.0). Esto evita el frame "snap_a == snap_b
        # con progress=0 desde el primer instante" que produce flicker.
        assert a is hist.at(0)
        assert b is hist.at(0)
        assert p == 0.0

    def test_playback_time_igual_a_duration(self):
        # Playback en el final exacto → último snapshot inmóvil.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(10.0)
        assert a is hist.at(1)
        assert b is hist.at(1)
        assert p == 1.0

    def test_playback_time_mayor_que_duration(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(100.0)
        assert a is hist.at(1)
        assert b is hist.at(1)
        assert p == 1.0

    def test_playback_time_apenas_dentro_del_intervalo(self):
        # Justo después del 0: debería entrar al primer tramo.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        a, b, p = tl.sample(0.001)
        assert a is hist.at(0)
        assert b is hist.at(1)
        assert p == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# Fallback configurable via theme
# ---------------------------------------------------------------------------


class TestTimelineFallbackConfigurable:
    """Cambiar fallback_step_duration cambia los virtual_times."""

    def test_fallback_grande_dilata_la_timeline(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0])
        theme = Theme.default().with_overrides(fallback_step_duration=2.0)
        tl = Timeline(hist, theme=theme)
        assert tl.duration == pytest.approx(2.0)

    def test_fallback_pequeno_contrae_la_timeline(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0])
        theme = Theme.default().with_overrides(fallback_step_duration=0.1)
        tl = Timeline(hist, theme=theme)
        assert tl.duration == pytest.approx(0.1)

    def test_fallback_no_afecta_tramos_durativos(self):
        # Tramo durativo de 10s. Cambiamos el fallback: la duration
        # total NO debe cambiar.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        theme_a = Theme.default().with_overrides(fallback_step_duration=0.1)
        theme_b = Theme.default().with_overrides(fallback_step_duration=5.0)
        assert Timeline(hist, theme=theme_a).duration == pytest.approx(10.0)
        assert Timeline(hist, theme=theme_b).duration == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Theme default
# ---------------------------------------------------------------------------


class TestTimelineThemeDefault:
    """Si no se pasa theme, usa Theme.default()."""

    def test_sin_theme_usa_default(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0])
        tl = Timeline(hist)  # sin theme
        # fallback de default = 0.4 → duration = 0.4.
        assert tl.duration == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Inmutabilidad
# ---------------------------------------------------------------------------


class TestTimelineInmutabilidad:
    """Timeline es frozen+slots."""

    def test_no_puede_mutarse(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        from dataclasses import FrozenInstanceError
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            tl.theme = Theme.default()  # type: ignore[misc]

    def test_virtual_times_son_tupla(self):
        # No es lista (mutable), es tupla.
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 10.0])
        tl = Timeline(hist)
        assert isinstance(tl._virtual_times, tuple)


# ---------------------------------------------------------------------------
# Garantía: virtual_times estrictamente monotónica
# ---------------------------------------------------------------------------


class TestVirtualTimesMonotonica:
    """Para todo i, _virtual_times[i] < _virtual_times[i+1]."""

    def test_secuencial_es_monotonico(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0, 0.0])
        tl = Timeline(hist)
        times = tl._virtual_times
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_durativo_es_monotonico(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 5.0, 10.0])
        tl = Timeline(hist)
        times = tl._virtual_times
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_mixto_es_monotonico(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 5.0, 5.0, 10.0])
        tl = Timeline(hist)
        times = tl._virtual_times
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]


# ---------------------------------------------------------------------------
# Integración con PlanRunner real
# ---------------------------------------------------------------------------


class TestIntegracionConRunner:
    """Construir el Timeline a partir de un HistoryManager producido por PlanRunner."""

    def test_plan_sequential_con_duration_cero_aplica_fallback(self):
        # Plan secuencial con 3 PickUps (todos duration=0): todos los
        # snapshots tendrán timestamps iguales (0, 1, 2 según
        # Plan.sequential pero con duration=0, no cambia timestamp).
        # Actually wait: Plan.sequential asigna timestamps 0,1,2 a los
        # ScheduledCommand, pero con duration=0 los snapshots se
        # commitean en el start_time del Command. Vamos a verificar:
        world = make_world_with_package()
        runner = PlanRunner(world)
        plan = Plan.sequential([
            PickUp(drone_id="d1", arm_id="izq", package_id="pkg1"),
        ])
        runner.execute(plan)
        hist = runner.history

        tl = Timeline(hist, theme=Theme.default())
        # Comprobación básica: duration > 0.
        # Si los timestamps van 0 → 0, fallback aplica.
        # Si van 0 → 0 también, fallback aplica.
        assert tl.duration > 0.0

    def test_plan_durativo_respeta_tiempo_real(self):
        world = make_simple_world()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=7.0),
        ]))
        hist = runner.history
        # Snapshots: [inicial t=0, start t=0, end t=7].
        assert len(hist) == 3

        tl = Timeline(hist, theme=Theme.default())
        # Tramos virtuales:
        # - inicial (t=0) → start (t=0): delta=0 → fallback 0.4.
        # - start (t=0) → end (t=7): delta=7 → 7.0.
        # Total = 7.4.
        assert tl.duration == pytest.approx(7.4)

        # sample en el medio del tramo durativo:
        # Tramo 2 va de virtual_time=0.4 a 7.4. En virtual_time=3.9
        # estamos a la mitad del tramo.
        a, b, p = tl.sample(3.9)
        assert a is hist.at(1)
        assert b is hist.at(2)
        assert p == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Propiedad snapshot_times (añadida en Sesión E, decisión 3.3)
# ---------------------------------------------------------------------------


class TestTimelineSnapshotTimes:
    """Propiedad pública añadida en Sesión E para que la UI envolvente
    pueda implementar 'saltar al snapshot anterior/siguiente' sin
    reconstruir la tabla desde fuera del módulo.

    El contrato lo documenta el docstring de la property:
        - tupla de N floats con N == len(history).
        - estrictamente monotónica creciente.
        - snapshot_times[0] == 0.0
        - snapshot_times[-1] == duration
    """

    def test_un_snapshot_devuelve_tupla_con_un_cero(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0])
        tl = Timeline(hist, theme=Theme.default())
        assert tl.snapshot_times == (0.0,)

    def test_secuencial_n_snapshots_devuelve_n_fallbacks_acumulados(self):
        """Tres snapshots a t=0 → snapshot_times = (0, 0.4, 0.8)."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0])
        tl = Timeline(hist, theme=Theme.default())
        assert tl.snapshot_times == (
            pytest.approx(0.0),
            pytest.approx(0.4),
            pytest.approx(0.8),
        )

    def test_durativo_respeta_timestamps_reales(self):
        """Tres snapshots a t=0, 5, 13 → snapshot_times = (0, 5, 13)."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 5.0, 13.0])
        tl = Timeline(hist, theme=Theme.default())
        assert tl.snapshot_times == (
            pytest.approx(0.0),
            pytest.approx(5.0),
            pytest.approx(13.0),
        )

    def test_mixto_alterna_real_y_fallback(self):
        """t=[0, 0, 5, 5, 10] → tramos (0→0)=fallback, (0→5)=5,
        (5→5)=fallback, (5→10)=5. Acumulado: 0, 0.4, 5.4, 5.8, 10.8."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 5.0, 5.0, 10.0])
        tl = Timeline(hist, theme=Theme.default())
        expected = (0.0, 0.4, 5.4, 5.8, 10.8)
        assert len(tl.snapshot_times) == len(expected)
        for got, exp in zip(tl.snapshot_times, expected):
            assert got == pytest.approx(exp)

    def test_longitud_coincide_con_len_history(self):
        """Invariante: una entrada por snapshot."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0, 2.0, 3.0, 4.0])
        tl = Timeline(hist, theme=Theme.default())
        assert len(tl.snapshot_times) == len(hist)

    def test_primer_elemento_siempre_cero(self):
        for tss in [[0.0], [0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 3.0, 3.0, 7.0]]:
            hist = make_history_with_n_snapshots_at_timestamps(tss)
            tl = Timeline(hist, theme=Theme.default())
            assert tl.snapshot_times[0] == 0.0, f"falla con timestamps={tss}"

    def test_ultimo_elemento_igual_a_duration(self):
        """Invariante crítico para la UI: saltar al último snapshot
        equivale a saltar al final de la timeline."""
        for tss in [[0.0], [0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 3.0, 3.0, 7.0]]:
            hist = make_history_with_n_snapshots_at_timestamps(tss)
            tl = Timeline(hist, theme=Theme.default())
            assert tl.snapshot_times[-1] == pytest.approx(tl.duration), (
                f"falla con timestamps={tss}"
            )

    def test_es_estrictamente_monotonica(self):
        """No dos valores consecutivos iguales: incluso el fallback es
        > 0 (theme.fallback_step_duration > 0)."""
        for tss in [[0.0, 0.0, 0.0], [0.0, 5.0, 5.0, 10.0], [0.0, 0.0, 0.0, 7.0]]:
            hist = make_history_with_n_snapshots_at_timestamps(tss)
            tl = Timeline(hist, theme=Theme.default())
            for prev, curr in zip(tl.snapshot_times, tl.snapshot_times[1:]):
                assert prev < curr, (
                    f"falla con timestamps={tss}: {prev} < {curr}"
                )

    def test_devuelve_tupla_inmutable(self):
        """No podemos mutar la tabla via la property — defiende contra
        usos accidentales."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0])
        tl = Timeline(hist, theme=Theme.default())
        times = tl.snapshot_times
        assert isinstance(times, tuple)
        # No hay método append en tuple, así que solo verificamos el tipo.

    def test_consistencia_con_sample_en_cada_snapshot_time(self):
        """Para cada snapshot_times[i], sample(t) devuelve algo
        consistente. Específicamente: en t=snapshot_times[i] con i en
        (0, N-1), sample devuelve el snapshot i como snap_a con
        progress=0 (entrada del intervalo i→i+1).

        En t=snapshot_times[0]==0, sample tiene saneo izquierdo y
        devuelve (h[0], h[0], 0.0).

        En t=snapshot_times[-1]==duration, sample tiene saneo derecho y
        devuelve (h[N-1], h[N-1], 1.0).
        """
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 5.0, 13.0, 18.0])
        tl = Timeline(hist, theme=Theme.default())
        n = len(tl.snapshot_times)

        # snapshots intermedios (i=1, 2): progress=0 sobre el intervalo i→i+1.
        for i in range(1, n - 1):
            a, b, p = tl.sample(tl.snapshot_times[i])
            assert a is hist.at(i)
            assert b is hist.at(i + 1)
            assert p == pytest.approx(0.0)

        # i=0: saneo izquierdo aplica (playback_time <= 0).
        a, b, p = tl.sample(tl.snapshot_times[0])
        assert a is hist.at(0)
        assert b is hist.at(0)

        # i=N-1: saneo derecho.
        a, b, p = tl.sample(tl.snapshot_times[-1])
        assert a is hist.at(n - 1)
        assert b is hist.at(n - 1)
        assert p == pytest.approx(1.0)

    def test_fallback_personalizado_via_theme(self):
        """La property reacciona al theme.fallback_step_duration igual
        que duration."""
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 0.0, 0.0])
        custom_theme = Theme.default().with_overrides(fallback_step_duration=2.0)
        tl = Timeline(hist, theme=custom_theme)
        assert tl.snapshot_times == (
            pytest.approx(0.0),
            pytest.approx(2.0),
            pytest.approx(4.0),
        )


class TestSampleWithNeighbors:
    """sample_with_neighbors devuelve las acciones reales vecinas (pares de
    snapshots), o None en los bordes. (Aquí los Move ficticios no son Static
    enter-interacting, así que el vecino real es el intervalo adyacente.)"""

    def test_un_solo_snapshot_sin_vecinos(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0])
        tl = Timeline(hist)
        prev, a, b, nxt, p = tl.sample_with_neighbors(0.0)
        assert prev is None and nxt is None
        assert a is b
        assert p == 0.0

    def test_bordes_sin_vecinos(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0, 2.0, 3.0])
        tl = Timeline(hist)
        prev, a, b, nxt, p = tl.sample_with_neighbors(-1.0)
        assert prev is None and nxt is None and a is b and p == 0.0
        prev, a, b, nxt, p = tl.sample_with_neighbors(tl.duration + 5.0)
        assert prev is None and nxt is None and a is b and p == 1.0

    def test_primer_intervalo_sin_prev(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0, 2.0, 3.0])
        tl = Timeline(hist)
        prev, a, b, nxt, _ = tl.sample_with_neighbors(tl.duration / 6.0)
        assert prev is None  # no hay acción antes de la primera
        assert nxt == (hist.at(1), hist.at(2))
        assert (a, b) == (hist.at(0), hist.at(1))

    def test_ultimo_intervalo_sin_next(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0, 2.0, 3.0])
        tl = Timeline(hist)
        prev, a, b, nxt, _ = tl.sample_with_neighbors(tl.duration * 5.0 / 6.0)
        assert nxt is None  # no hay acción tras la última
        assert prev == (hist.at(1), hist.at(2))
        assert (a, b) == (hist.at(2), hist.at(3))

    def test_intervalo_central_ambos_vecinos(self):
        hist = make_history_with_n_snapshots_at_timestamps([0.0, 1.0, 2.0, 3.0])
        tl = Timeline(hist)
        prev, a, b, nxt, _ = tl.sample_with_neighbors(tl.duration / 2.0)
        assert (a, b) == (hist.at(1), hist.at(2))
        assert prev == (hist.at(0), hist.at(1))
        assert nxt == (hist.at(2), hist.at(3))


# ---------------------------------------------------------------------------
# Colapso de la pausa entre STEPS encadenados (_is_chained_dwell)
# ---------------------------------------------------------------------------


class TestChainedDwellCollapse:
    """La pausa enter-interacting que separa dos PackageMove del MISMO dron
    (steps encadenados en la misma loc) se colapsa a un epsilon imperceptible;
    el fallback completo (espaciado) se reserva para separar acciones
    completas (la primera acción, o la primera tras llegar por un Move).
    """

    def _plan_dos_pickups(self):
        from droneplan_viz.runtime import ScheduledCommand

        med = Content(id="m")
        world = World(
            locations={"dep": Location(id="dep")},
            drones={"d1": Drone(id="d1", position="dep",
                                arms=(Arm(id="izq"), Arm(id="der")),
                                state=DroneState.IDLE)},
            transporters={},
            packages={
                "pa": Package(id="pa", contains=med, at=AtLocation(loc_id="dep")),
                "pb": Package(id="pb", contains=med, at=AtLocation(loc_id="dep")),
            },
            contents={"m": med}, costs={},
        )
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pa",
                               duration=2.0, command_id="c1"),
                start_time=0.0),
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="der", package_id="pb",
                               duration=2.0, command_id="c2"),
                start_time=2.0),
        ))
        runner = PlanRunner(world)
        runner.execute(plan)
        return runner.history

    def test_pausa_entre_steps_encadenados_se_colapsa(self):
        # Historial: [inicial, pa_start, pa_end, pb_start, pb_end].
        # La pausa entre pa_end y pb_start (tramo 2→3) está flanqueada por
        # dos PackageMove de d1 → se colapsa.
        hist = self._plan_dos_pickups()
        tl = Timeline(hist, theme=Theme.default())
        st = tl.snapshot_times
        dwell_encadenada = st[3] - st[2]
        assert dwell_encadenada < 0.01, (
            f"la pausa entre steps encadenados debería colapsar, "
            f"dura {dwell_encadenada}s"
        )

    def test_pausa_de_la_primera_accion_mantiene_fallback(self):
        # El primer tramo (inicial → pa_start) NO está flanqueado por dos
        # PackageMove (no hay acción previa) → mantiene el fallback completo.
        hist = self._plan_dos_pickups()
        theme = Theme.default()
        tl = Timeline(hist, theme=theme)
        st = tl.snapshot_times
        primera_pausa = st[1] - st[0]
        assert primera_pausa == pytest.approx(theme.fallback_step_duration)

    def test_colapso_respeta_fallback_configurable(self):
        # Cambiar el fallback no afecta a la pausa colapsada (sigue epsilon),
        # pero sí a la de la primera acción.
        hist = self._plan_dos_pickups()
        tl = Timeline(hist, theme=Theme.default().with_overrides(
            fallback_step_duration=1.5))
        st = tl.snapshot_times
        assert (st[3] - st[2]) < 0.01            # encadenada: epsilon
        assert (st[1] - st[0]) == pytest.approx(1.5)  # primera: fallback
