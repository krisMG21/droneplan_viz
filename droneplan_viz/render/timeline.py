"""
Timeline: mapeo de tiempo de reproducción a (snap_a, snap_b, progress).

Este módulo encapsula la lógica que cierra el ciclo render: dada una
ejecución completa (HistoryManager) y un tiempo de reproducción, decide
qué par de snapshots consecutivos enmarcan el frame actual y con qué
progreso interpolar entre ellos. La UI envolvente (sesión posterior)
mantiene su propio `playback_time` (que avanza con clock.tick · velocidad)
y llama a `Timeline.sample(playback_time)` cada frame.

Decisiones de diseño:

1. EL TIMELINE PERTENECE A render/ Y NO A runtime/. Justificación
   (decisión elegida):
   - sample() decide cómo mapear tiempo a un par de snapshots con
     progress, lo cual implica conocer las reglas de interpolación
     visual (qué hacer en bordes, qué hacer con duration=0).
   - El runtime ya cerró su responsabilidad al producir el
     HistoryManager. Acoplar Timeline al runtime sería darle al
     runtime conocimiento de animaciones, lo cual rompería la
     frontera del proyecto.

2. EJE DE TIEMPO VIRTUAL HÍBRIDO. Decisión clave consensuada con el
   usuario:
   - Para snapshots consecutivos con timestamps distintos (PDDL parte 3,
     durativo): la diferencia virtual es el tiempo simulado real.
     Un Move de 10s dura 10s reales a velocidad 1×.
   - Para snapshots consecutivos con MISMO timestamp (PDDL parte 1-2,
     duration=0): la diferencia virtual es theme.fallback_step_duration
     (0.6s por defecto). Esto evita saltos instantáneos invisibles en
     planes secuenciales.

   El Timeline construye una tabla de virtual_times monotónicamente
   creciente y mapea playback_time a esa tabla. La UI solo ve UN eje
   lineal de tiempo, sin tener que saber del fallback.

3. INMUTABLE TRAS CONSTRUCCIÓN. El Timeline guarda referencia al
   HistoryManager y a la tabla de virtual_times (precalculada en
   __init__). Si la UI muta el HistoryManager (commitea nuevos
   snapshots), el Timeline NO se actualiza automáticamente; la UI
   debe reconstruirlo. Decisión documentada: la animación es de un
   plan completo ya ejecutado, no de uno en construcción.

4. PROGRESS SIEMPRE EN [0, 1] DENTRO DEL PAR. sample() no devuelve el
   progress curvado por easing; eso lo aplica painter.render_frame()
   internamente. Aquí devolvemos el progress LINEAL, que el painter
   pasa por ease_in_out_cubic.

5. ACEPTA HistoryManager CON UN SOLO SNAPSHOT. Timeline.duration = 0.0
   en ese caso. sample(t) devuelve (h[0], h[0], 0.0) para cualquier t.
   El painter ya trata el caso snap_a is snap_b como TransitionStatic.

6. ACEPTA theme=None. Si no se pasa Theme, usa Theme.default(). El
   único parámetro del theme que se consulta es fallback_step_duration.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Tuple

from droneplan_viz.history import HistoryManager, WorldSnapshot
from droneplan_viz.render.theme import Theme


# Duración virtual ínfima (segundos) que se asigna a la pausa
# "enter-interacting" cuando separa dos STEPS encadenados del mismo dron en
# la misma loc. En la práctica elimina la pausa (el dron fluye de un step al
# siguiente) manteniendo la tabla de virtual_times estrictamente creciente
# (lo que necesitan sample() y la navegación por snapshots). El espaciado
# visible (fallback_step_duration) se reserva para separar acciones completas.
_CHAINED_DWELL_EPSILON: float = 1e-3


@dataclass(frozen=True, slots=True)
class Timeline:
    """Mapeo de tiempo de reproducción a (snap_a, snap_b, progress).

    Atributos:
        history: HistoryManager del que se leen los snapshots. NO se
            modifica.
        theme: Theme cuyo fallback_step_duration se usa para tramos
            con duration=0. Por defecto Theme.default().

    Atributo precalculado (privado):
        _virtual_times: tupla de N floats donde N == len(history). El
            índice i contiene el virtual_time del snapshot i. Cumple:
                _virtual_times[0] == 0.0
                _virtual_times[i] < _virtual_times[i+1] estrictamente
                _virtual_times[N-1] == self.duration
    """

    history: HistoryManager
    theme: Theme = field(default_factory=Theme.default)
    _virtual_times: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precalcula la tabla de virtual_times.

        Recorre los snapshots por pares consecutivos y acumula el
        delta_t virtual: el real si es > 0, o el fallback si es 0.

        EXCEPCIÓN: la "pausa" entre STEPS encadenados (el snapshot Static
        "enter-interacting" que separa dos acciones de paquete consecutivas
        del MISMO dron, necesariamente en la misma loc) se COLAPSA a una
        duración ínfima (_CHAINED_DWELL_EPSILON), para que el dron fluya
        directamente de un step al siguiente sin pausa. El fallback completo
        (espaciado visible) se mantiene SOLO donde separa acciones completas
        (p.ej. el primer step tras llegar a la loc por un Move).

        Garantía: la tabla es estrictamente monotónica creciente (el epsilon
        es > 0), así que sample() y la navegación por snapshots siguen
        funcionando sin ambigüedad.
        """
        from droneplan_viz.render.interpolation import (
            classify_transition,
            TransitionPackageMove,
            TransitionStatic,
        )
        from droneplan_viz.render.painter import (
            _is_static_only_entering_interacting,
        )

        n = len(self.history)
        times: list[float] = [0.0]

        def _package_move_drone(a, b):
            """drone_id si (a→b) es un PackageMove; None en otro caso."""
            try:
                t = classify_transition(a, b)
            except Exception:
                return None
            return t.drone_id if isinstance(t, TransitionPackageMove) else None

        def _is_chained_dwell(i: int) -> bool:
            """True si el tramo (snap_{i-1}→snap_i) es la pausa
            enter-interacting ENTRE dos PackageMoves del mismo dron (steps
            encadenados en la misma loc): hay que colapsarla."""
            if not (2 <= i <= n - 2):
                return False
            sa, sb = self.history.at(i - 1), self.history.at(i)
            try:
                tr = classify_transition(sa, sb)
            except Exception:
                return False
            if not (
                isinstance(tr, TransitionStatic)
                and _is_static_only_entering_interacting(sa, sb)
            ):
                return False
            d_prev = _package_move_drone(self.history.at(i - 2), self.history.at(i - 1))
            d_next = _package_move_drone(self.history.at(i), self.history.at(i + 1))
            return d_prev is not None and d_prev == d_next

        for i in range(1, n):
            prev_snap = self.history.at(i - 1)
            curr_snap = self.history.at(i)
            real_delta = curr_snap.timestamp - prev_snap.timestamp

            # Si el delta real es positivo, lo respetamos (modo durativo).
            # Si es 0 (modo secuencial parte 1-2): fallback, salvo que sea la
            # pausa entre steps encadenados, que colapsamos a un epsilon
            # imperceptible (el dron fluye sin pausa entre steps de un grupo).
            # Si es negativo (caso patológico): tratamos como 0 → fallback.
            if real_delta > 0.0:
                virtual_delta = real_delta
            elif _is_chained_dwell(i):
                virtual_delta = _CHAINED_DWELL_EPSILON
            else:
                virtual_delta = self.theme.fallback_step_duration

            times.append(times[-1] + virtual_delta)

        # frozen=True ⇒ usamos object.__setattr__ para asignar el campo
        # init=False (patrón canónico para dataclasses frozen con
        # campos derivados).
        object.__setattr__(self, "_virtual_times", tuple(times))

    @property
    def duration(self) -> float:
        """Duración total de la reproducción en tiempo virtual.

        Igual al último virtual_time. Para un HistoryManager con un solo
        snapshot, devuelve 0.0.
        """
        return self._virtual_times[-1]

    @property
    def snapshot_times(self) -> tuple[float, ...]:
        """Tiempo virtual de cada snapshot del historial.

        Tupla de N floats donde N == len(history). El elemento i es el
        playback_time en el que aparece exactamente snapshot i (a
        progress = 0). Cumple:

            snapshot_times[0] == 0.0
            snapshot_times[i] < snapshot_times[i+1]  (estricto)
            snapshot_times[-1] == self.duration

        Añadida (decisión 3.3 elegida): permite a la UI
        envolvente implementar "saltar al snapshot anterior/siguiente"
        sin tener que reconstruir esta tabla desde fuera del módulo.

        Esta es una vista de lectura sobre el atributo interno
        `_virtual_times`, que ya es una tupla inmutable; devolverla
        directamente es seguro (las tuplas no se pueden mutar). Si en
        el futuro se sustituye la implementación interna por una
        estructura mutable, esta property debería copiar.
        """
        return self._virtual_times

    def sample(
        self,
        playback_time: float,
    ) -> Tuple[WorldSnapshot, WorldSnapshot, float]:
        """Devuelve el par de snapshots que enmarca playback_time, y el progress.

        Args:
            playback_time: tiempo de reproducción en segundos virtuales
                (eje monotónico creciente, mismo que self.duration).

        Returns:
            (snap_a, snap_b, progress):
                - snap_a: snapshot izquierdo del intervalo.
                - snap_b: snapshot derecho. Para playback_time fuera de
                  rango o historial trivial, snap_a == snap_b.
                - progress: float en [0, 1]. Lineal: el painter aplica
                  el easing.

        Saneo de bordes:
            playback_time <= 0       → (h[0], h[0], 0.0)
            playback_time >= duration → (h[N-1], h[N-1], 1.0)
            history con 1 snapshot   → (h[0], h[0], 0.0) para todo t

        Implementación: búsqueda binaria (bisect) sobre _virtual_times,
        que es estrictamente creciente. O(log N), frente al O(N) de un
        barrido lineal; relevante cuando el historial tiene miles de
        snapshots y sample() se llama varias veces por frame.
        """
        n = len(self.history)

        # Caso degenerado: un solo snapshot.
        if n == 1:
            snap = self.history.at(0)
            return (snap, snap, 0.0)

        # Saneo izquierdo: antes del inicio → primer snapshot inmóvil.
        if playback_time <= 0.0:
            snap = self.history.at(0)
            return (snap, snap, 0.0)

        # Saneo derecho: a partir del último virtual_time, último
        # snapshot inmóvil. Usamos >= para que en el instante exacto
        # del final no caigamos al intervalo previo.
        if playback_time >= self._virtual_times[-1]:
            snap = self.history.at(n - 1)
            return (snap, snap, 1.0)

        # Caso general: i tal que _virtual_times[i] <= playback_time <
        # _virtual_times[i+1]. bisect_right da el primer índice cuyo valor
        # es > playback_time; restando 1 obtenemos i. Los extremos ya están
        # filtrados, así que i ∈ [0, n-2].
        i = bisect.bisect_right(self._virtual_times, playback_time) - 1
        t0 = self._virtual_times[i]
        t1 = self._virtual_times[i + 1]
        progress = (playback_time - t0) / (t1 - t0)
        return (
            self.history.at(i),
            self.history.at(i + 1),
            progress,
        )

    def sample_with_neighbors(
        self,
        playback_time: float,
    ) -> Tuple[
        Tuple[WorldSnapshot, WorldSnapshot] | None,
        WorldSnapshot,
        WorldSnapshot,
        Tuple[WorldSnapshot, WorldSnapshot] | None,
        float,
    ]:
        """Como `sample`, pero devuelve también la ACCIÓN real anterior y la
        siguiente: (prev_action, snap_a, snap_b, next_action, progress).

        prev_action / next_action son pares (snap_a, snap_b) de la transición
        real adyacente, SALTANDO los snapshots Static "enter-interacting" que
        el runtime intercala antes de cada acción durativa (PickUp/Deliver/
        Load/Unload emiten un snap_start en el que el dron entra a INTERACTING
        sin moverse). Así el render puede encadenar acciones consecutivas en
        la misma loc sin que el dron vuelva al centro entre ellas. None si no
        hay acción real a ese lado.

        En los extremos saneados y en el caso degenerado, snap_a == snap_b y
        ambas acciones vecinas son None.
        """
        from droneplan_viz.render.interpolation import classify_transition
        from droneplan_viz.render.painter import (
            _is_static_only_entering_interacting,
        )
        from droneplan_viz.render.interpolation import TransitionStatic

        n = len(self.history)
        if n == 1 or playback_time <= 0.0:
            snap = self.history.at(0)
            return (None, snap, snap, None, 0.0)
        if playback_time >= self._virtual_times[-1]:
            snap = self.history.at(n - 1)
            return (None, snap, snap, None, 1.0)

        def _is_skip(j: int) -> bool:
            """True si el intervalo j es un Static 'enter-interacting'."""
            sa, sb = self.history.at(j), self.history.at(j + 1)
            tr = classify_transition(sa, sb)
            return isinstance(tr, TransitionStatic) and (
                _is_static_only_entering_interacting(sa, sb)
            )

        def _real_action(start: int, step: int):
            """Primer intervalo (saltando enter-interacting) en la dirección
            step desde start, como par (snap_a, snap_b); None si no hay."""
            j = start
            while 0 <= j <= n - 2:
                if not _is_skip(j):
                    return (self.history.at(j), self.history.at(j + 1))
                j += step
            return None

        for i in range(n - 1):
            t0 = self._virtual_times[i]
            t1 = self._virtual_times[i + 1]
            if t0 <= playback_time < t1:
                progress = (playback_time - t0) / (t1 - t0)
                prev_action = _real_action(i - 1, -1)
                next_action = _real_action(i + 1, +1)
                return (
                    prev_action,
                    self.history.at(i),
                    self.history.at(i + 1),
                    next_action,
                    progress,
                )
        snap = self.history.at(n - 1)
        return (None, snap, snap, None, 1.0)
