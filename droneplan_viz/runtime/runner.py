"""
PlanRunner: orquestador del bucle de ejecución del plan.

Cubre:

- Ejecución de planes con cualquier duración (>= 0).
- Bucle de eventos con tiebreak END-antes-que-START en mismo timestamp.
- Snapshots DUALES para Commands con duration > 0:
    * Snapshot 'start' en scheduled.start_time: solo cambia
      drone.state := MOVING (Move/MoveWithTransporter) o INTERACTING
      (las otras cuatro acciones). Sin tocar efectos PDDL.
    * Snapshot 'end' en scheduled.end_time: aplica los efectos PDDL
      mediante apply() y restaura drone.state := IDLE.
- Snapshot ÚNICO para Commands con duration = 0 (modos secuenciales
  PDDL parte 1-2): el snapshot start no aporta información, así que
  se omite por optimización.
- Validación PDDL por evento (delegada al validate()).
- Concurrencia PDDL parte 3: una ResourceTable interna detecta
  intersecciones temporales entre Commands sobre los recursos
  exclusivos definidos en el PDF (drone, paquete, transportador,
  persona). Si dos Commands solapan en un recurso, el segundo falla
  con kind="concurrency".
- Manejo de fallos: drone implicado a ERROR (sumidero forward
  de el runtime), failed_commands++, snapshot del fallo, ejecución del
  resto del plan continúa.
- total_time (makespan): el runner es su única autoridad. El makespan
  depende del momento absoluto en que cada acción termina (end_time),
  información temporal que solo el runtime conoce; los handlers de
  el runtime dejan total_time intacto deliberadamente. El runner lo fija
  en su evento END como max(total_time_previo, end_time del Command).
- dry_run(plan): ejecuta sobre un runner espejo sin tocar el original.

Motor de eventos:

    Cada ScheduledCommand produce DOS eventos:
        - START en scheduled.start_time
        - END   en scheduled.end_time

    La cola se ordena por tupla (timestamp, orig_index, prioridad):
        - timestamp asc: orden cronológico.
        - orig_index asc: para Commands distintos en el mismo
          timestamp, el declarado primero gana el desempate. Esto
          también garantiza que en el caso back-to-back (cmd1 acaba
          en t=10 y cmd2 empieza en t=10) el END de cmd1 (orig_index
          menor) se procesa antes que el START de cmd2.
        - prioridad asc DENTRO del mismo Command: START (0) antes que
          END (1). Crítico para Commands con duration=0, donde START
          y END coinciden en el mismo timestamp.

    Para Commands con duration=0, el START y el END coinciden. La
    optimización es: omitir el snapshot start (un cambio de drone.state
    que se revertiría en el mismo instante no aporta nada al
    historial). Para duration>0, snapshot start y end separados.

Decisiones de diseño:

1. El runner transiciona drone.state durante la ejecución. Esto saca
   provecho de _check_drone_ready del Validator: cualquier
   intento de dar otro Command al mismo drone mientras está MOVING o
   INTERACTING falla automáticamente con la validación PDDL existente,
   sin que el runner tenga que codificar la regla. Reuso elegante.

2. La transición se hace SIN llamar a apply(). El handler
   NO toca drone.state (decisión establecida); aquí el
   runner lo hace mediante dataclasses.replace directo, como una
   responsabilidad adicional del bucle temporal.

3. Si un fallo (PDDL o concurrencia) ocurre durante el procesamiento
   del START de un Command, su END NUNCA se procesa: el runner
   descarta el evento END correspondiente. Sin entrada en la cola =
   sin transición de vuelta a IDLE = sin apply() = sin reserva en la
   tabla de recursos.

4. drone.state en el snapshot start es MOVING para Move y
   MoveWithTransporter; INTERACTING para PickUp, Deliver, Load, Unload.

5. La validación PDDL se hace SOLO en el evento START, no en el END.
   Si pasa en START, asumimos que el world relevante al Command no
   cambia entre start y end. Esta propiedad la garantiza la
   ResourceTable: ningún otro Command que comparta recursos con el
   nuestro puede solapar nuestro intervalo. Por tanto, los efectos
   PDDL que otros Commands aplican entre START y END NUNCA tocan
   recursos relevantes para nuestro Command.

6. Orden de chequeo en START: PRIMERO PDDL, LUEGO concurrencia.
   Razones:
     - Si el Command es PDDL-inválido (drone inexistente, paquete no
       co-localizado, etc.), no tiene sentido comprobar recursos.
     - Mensajes de error PDDL son más útiles para el usuario del TFG
       (apuntan a errores semánticos del plan); los de concurrencia
       solo apuntan a 'choca con otra acción'.
     - El Validator ya cubre la regla 1 del PDF parte 3 (un dron, una
       acción) gracias a la transición drone.state := MOVING/INTERACTING
       que hace este runner. La ResourceTable la cubre también, como
       segunda red de seguridad, lo cual no es duplicación: la
       ResourceTable es necesaria para las reglas 2, 3 y 4 (caja,
       transportador, persona), que NO se reflejan en drone.state.

7. La ResourceTable se reinicializa en cada execute(). Una llamada
   nueva a execute() empieza con tabla vacía, coherente con que el
   runner sea reentrant para múltiples ejecuciones (caso de uso:
   facade aplica un plan, luego otro plan adicional sobre el mismo
   World).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
    apply,
    validate,
)
from droneplan_viz.commands.base import Command
from droneplan_viz.domain import DroneState, MetricsTracker, World
from droneplan_viz.history import HistoryManager, TimelineCursor, WorldSnapshot
from droneplan_viz.runtime.plan import Plan, ScheduledCommand
from droneplan_viz.runtime.resources import ResourceTable, resources_of
from droneplan_viz.runtime.result import CommandFailure, FailureKind, RunResult


# ---------------------------------------------------------------------------
# Prioridades de eventos para el tiebreak
# ---------------------------------------------------------------------------
# En el mismo timestamp ENTRE COMMANDS DISTINTOS, END se procesa antes
# que START. Esto resuelve el caso back-to-back: cmd1 acaba en t=10,
# cmd2 empieza en t=10. Procesando END(cmd1) primero, el drone vuelve
# a IDLE antes de que el START de cmd2 valide PDDL.
#
# IMPORTANTE: para el MISMO Command (mismo orig_index, p.ej. cuando
# duration=0 y START y END coinciden en el mismo timestamp), el START
# debe procesarse ANTES que el END. De lo contrario, el END intentaría
# aplicar antes de que el START hubiera validado.
#
# La ordenación canónica usa la tupla (timestamp, orig_index, prio):
#   - timestamp asc: orden cronológico natural.
#   - orig_index asc: para Commands distintos en el mismo timestamp,
#     el declarado primero gana el desempate; esto da determinismo.
#   - prio asc dentro de un mismo Command: START (0) antes que END (1).
#
# Con este orden, dos Commands distintos en el mismo timestamp se
# procesan en bloque, primero el del orig_index menor (START+END juntos)
# y luego el del orig_index mayor. La regla "END antes que START entre
# Commands distintos" se cumple automáticamente porque el END del
# primer Command sale antes del START del segundo.
_EVENT_START: Literal[0] = 0
_EVENT_END: Literal[1] = 1


# ---------------------------------------------------------------------------
# Helpers privados de mapeo
# ---------------------------------------------------------------------------
def _busy_state_for(command: Command) -> DroneState:
    """Devuelve el DroneState que el drone toma durante la ejecución
    de este Command.

    Move y MoveWithTransporter -> MOVING (el drone está en tránsito).
    PickUp, Deliver, LoadIntoTransporter, UnloadFromTransporter ->
    INTERACTING (el drone está manipulando objetos en su localización).

    Esta función NO se invoca para Commands con duration=0: en modo
    atemporal no hay intervalo durante el cual marcar al drone como
    ocupado.

    Raises:
        TypeError: si command no es un Command conocido.
    """
    match command:
        case Move() | MoveWithTransporter():
            return DroneState.MOVING
        case PickUp() | Deliver() | LoadIntoTransporter() | \
                UnloadFromTransporter():
            return DroneState.INTERACTING
        case _:
            raise TypeError(
                f"_busy_state_for() recibió un objeto que no es un "
                f"Command conocido: {type(command).__name__}"
            )


def _with_drone_state(
    world: World, drone_id: str, new_state: DroneState
) -> World:
    """Devuelve un World derivado con drone.state actualizado.

    Si el drone_id no existe en el World, devuelve el World sin cambios
    (defensa en profundidad). Si el state ya coincide, devuelve el
    World sin cambios (no construye un World nuevo innecesariamente).
    """
    if drone_id not in world.drones:
        return world
    old_drone = world.drones[drone_id]
    if old_drone.state == new_state:
        return world
    new_drone = replace(old_drone, state=new_state)
    new_drones = dict(world.drones)
    new_drones[drone_id] = new_drone
    return replace(world, drones=new_drones)


# ---------------------------------------------------------------------------
# PlanRunner
# ---------------------------------------------------------------------------
class PlanRunner:
    """Ejecutor de planes con historial y métricas.

    Atributos públicos (de lectura):
        history: HistoryManager con la secuencia de WorldSnapshots
            producida. Empieza con un único snapshot inicial.
        cursor: TimelineCursor sobre el history.
    """

    __slots__ = ("_history", "_cursor")

    def __init__(
        self, world: World, *, metrics: MetricsTracker | None = None
    ) -> None:
        """Inicializa el runner con un World y métricas opcionales.

        Args:
            world: estado de partida. Por la garantía de inmutabilidad
                estructural, se almacena por referencia
                directa en el snapshot inicial.
            metrics: métricas iniciales. Default: MetricsTracker vacío.
        """
        initial_metrics = metrics if metrics is not None else MetricsTracker()
        initial_snapshot = WorldSnapshot(
            world=world,
            metrics=initial_metrics,
            produced_by=None,
            timestamp=0.0,
        )
        self._history = HistoryManager(initial=initial_snapshot)
        self._cursor = TimelineCursor(self._history)

    # -----------------------------------------------------------------
    # Propiedades de introspección
    # -----------------------------------------------------------------
    @property
    def history(self) -> HistoryManager:
        """HistoryManager interno. Expuesto para que el render
         lo consuma tras execute()."""
        return self._history

    @property
    def cursor(self) -> TimelineCursor:
        """TimelineCursor interno. Posicionado en head al finalizar
        execute(). El facade puede moverlo para navegación."""
        return self._cursor

    @property
    def current_world(self) -> World:
        """World en la posición actual del cursor."""
        return self._cursor.current.world

    @property
    def current_metrics(self) -> MetricsTracker:
        """MetricsTracker en la posición actual del cursor."""
        return self._cursor.current.metrics

    # -----------------------------------------------------------------
    # Ejecución
    # -----------------------------------------------------------------
    def execute(self, plan: Plan) -> RunResult:
        """Ejecuta un Plan completo sobre el estado actual del runner.

        Construye una cola de eventos (START y END por cada
        ScheduledCommand) ordenada por (timestamp, orig_index,
        prioridad). Mantiene una ResourceTable local que se llena
        al procesar cada START exitoso y se consulta para detectar
        violaciones de concurrencia.

            - START de un Command:
                * Valida PDDL contra el world actual.
                * Si PDDL pasa: comprueba concurrencia contra la
                  ResourceTable (recursos).
                * Si ambos pasan y duration > 0: reserva recursos,
                  snapshot 'start' con drone.state := MOVING/INTERACTING.
                * Si ambos pasan y duration == 0: reserva inocua
                  (intervalo vacío) + el END (mismo timestamp) hará
                  el snapshot.
                * Si PDDL falla: drone a ERROR, failed_commands++,
                  snapshot del fallo, descarta END, kind='pddl'.
                * Si concurrencia falla: drone a ERROR,
                  failed_commands++, snapshot del fallo, descarta END,
                  kind='concurrency'.

            - END de un Command exitoso:
                * Aplica los efectos PDDL via apply().
                * Restaura drone.state := IDLE (solo si duration > 0).
                * Fija total_time (makespan) con el end_time del Command.
                * Snapshot 'end'.

        La ResourceTable es local a este execute(); cada llamada parte
        de tabla vacía. Esto permite ejecuciones repetidas sobre el
        mismo runner sin que las reservas de un plan anterior
        interfieran.

        Args:
            plan: Plan estructuralmente válido a ejecutar.

        Returns:
            RunResult con el estado final, fallos detectados, makespan
            y longitud del historial.
        """
        # Construir cola de eventos. Cada entrada:
        #   (timestamp, orig_index, prioridad, kind, scheduled)
        # ordenada por tupla natural:
        #   - timestamp asc: orden cronológico.
        #   - orig_index asc: para Commands distintos en el mismo
        #     timestamp, el declarado primero gana el desempate.
        #   - prioridad asc DENTRO del mismo Command: START (0) antes
        #     que END (1). Crítico para Commands con duration=0.
        events: list[tuple[float, int, int, str, ScheduledCommand]] = []
        for i, sched in enumerate(plan.scheduled):
            events.append(
                (sched.start_time, i, _EVENT_START, "start", sched)
            )
            events.append(
                (sched.end_time, i, _EVENT_END, "end", sched)
            )
        events.sort()

        # Tabla de recursos local a esta ejecución. Vacía al inicio.
        resources = ResourceTable()

        # Conjunto de command_ids cuyo END debe descartarse porque su
        # START falló (PDDL o concurrencia).
        cancelled_ends: set[str] = set()

        failures: list[CommandFailure] = []
        max_end_time_succeeded: float = 0.0

        for _ts, _idx, _prio, kind, scheduled in events:
            cmd_id = scheduled.command.command_id
            if kind == "start":
                failure = self._process_start(scheduled, resources)
                if failure is not None:
                    failures.append(failure)
                    cancelled_ends.add(cmd_id)
            else:  # kind == "end"
                if cmd_id in cancelled_ends:
                    # Su START falló; no procesamos su END.
                    continue
                self._process_end(scheduled)
                if scheduled.end_time > max_end_time_succeeded:
                    max_end_time_succeeded = scheduled.end_time

        return RunResult(
            plan=plan,
            final_world=self.current_world,
            final_metrics=self.current_metrics,
            failures=tuple(failures),
            history_length=len(self._history),
            makespan=max_end_time_succeeded,
        )

    def dry_run(self, plan: Plan) -> RunResult:
        """Ejecuta el plan sobre un runner espejo y devuelve el resultado.

        El runner original NO se modifica. Implementación: crea un
        runner espejo con el world y metrics actuales y ejecuta ahí.
        Coste: equivalente a una ejecución completa.

        Args:
            plan: el Plan a simular.

        Returns:
            RunResult equivalente al que produciría execute() sobre el
            estado actual del runner.
        """
        mirror = PlanRunner(
            world=self.current_world,
            metrics=self.current_metrics,
        )
        return mirror.execute(plan)

    # -----------------------------------------------------------------
    # Helpers privados: eventos
    # -----------------------------------------------------------------
    def _process_start(
        self,
        scheduled: ScheduledCommand,
        resources: ResourceTable,
    ) -> CommandFailure | None:
        """Procesa el evento START de un ScheduledCommand.

        Orden de chequeos:
            1. Validación PDDL contra el world actual.
               Si falla -> CommandFailure(kind="pddl").
            2. Comprobación de concurrencia contra la ResourceTable.
               Si falla -> CommandFailure(kind="concurrency").
            3. Si ambos pasan: reservar recursos, emitir snapshot start
               (si duration > 0).

        El orden es deliberado:
            - PDDL primero porque los errores PDDL apuntan a problemas
              semánticos del plan (más útiles didácticamente).
            - Concurrencia segundo porque solo tiene sentido reservar
              recursos para Commands que el dominio admite.

        La reserva se hace SIEMPRE (independiente de duration), pero
        para duration=0 el intervalo es vacío y la ResourceTable la
        descarta silenciosamente. Esto evita ramas adicionales.

        Args:
            scheduled: el ScheduledCommand a procesar.
            resources: la tabla de recursos de la ejecución actual.

        Returns:
            None si el START se procesó correctamente;
            CommandFailure con kind="pddl" o "concurrency" si falló.
        """
        current_snapshot = self._cursor.current
        current_world = current_snapshot.world
        cmd = scheduled.command

        # 1. Validación PDDL.
        validation = validate(current_world, cmd)
        if not validation:
            return self._handle_failure(
                scheduled, validation.reason or "", kind="pddl"
            )

        # 2. Comprobación de concurrencia.
        required_resources = resources_of(cmd)
        collision = resources.check(
            required_resources,
            scheduled.start_time,
            scheduled.end_time,
        )
        if collision is not None:
            return self._handle_failure(
                scheduled, collision.describe(), kind="concurrency"
            )

        # 3. Reservar recursos. Para duration=0 esta llamada es inocua
        # (intervalo vacío -> la tabla la ignora).
        resources.reserve(
            required_resources,
            scheduled.start_time,
            scheduled.end_time,
            owner=cmd.command_id,
        )

        # 4. Snapshot start si la acción tiene duración.
        if cmd.duration > 0:
            busy = _busy_state_for(cmd)
            new_world = _with_drone_state(
                current_world, cmd.drone_id, busy
            )
            start_snapshot = WorldSnapshot(
                world=new_world,
                metrics=current_snapshot.metrics,
                produced_by=cmd,
                timestamp=scheduled.start_time,
            )
            self._cursor.commit_new(start_snapshot)

        return None

    def _process_end(self, scheduled: ScheduledCommand) -> None:
        """Procesa el evento END de un ScheduledCommand exitoso.

        Aplica los efectos PDDL via apply(), restaura drone.state a
        IDLE (si la acción tenía duración > 0), fija total_time con el
        end_time real del Command y commitea snapshot 'end'.
        """
        current_snapshot = self._cursor.current
        current_world = current_snapshot.world
        current_metrics = current_snapshot.metrics
        cmd = scheduled.command

        # prev_total_time es el makespan acumulado hasta ahora. Lo
        # tomamos antes del apply() porque el handler deja total_time
        # intacto (los handlers no conocen el reloj); el runner es la
        # única autoridad de total_time y lo fija más abajo.
        prev_total_time = current_metrics.total_time

        # Aplicar efectos PDDL.
        new_world, new_metrics = apply(current_world, current_metrics, cmd)

        # Restaurar drone.state a IDLE solo si lo habíamos cambiado.
        if cmd.duration > 0:
            new_world = _with_drone_state(
                new_world, cmd.drone_id, DroneState.IDLE
            )

        # Fijar total_time (makespan) = max(prev, end_time real). Este
        # es el único punto del proyecto donde se establece el makespan:
        # el runtime es su autoridad porque es quien conoce los tiempos
        # absolutos del plan.
        final_metrics = replace(
            new_metrics,
            total_time=max(prev_total_time, scheduled.end_time),
        )

        end_snapshot = WorldSnapshot(
            world=new_world,
            metrics=final_metrics,
            produced_by=cmd,
            timestamp=scheduled.end_time,
        )
        self._cursor.commit_new(end_snapshot)

    def _handle_failure(
        self,
        scheduled: ScheduledCommand,
        reason: str,
        *,
        kind: FailureKind,
    ) -> CommandFailure:
        """Gestiona un fallo detectado en el START (PDDL o concurrencia).

        - Marca al drone implicado como ERROR (si existe). En el caso
          de fallo PDDL 'drone inexistente', el drone_id no está en el
          world; _with_drone_state lo maneja defensivamente sin crash.
        - Incrementa failed_commands en metrics.
        - Commitea un snapshot del fallo con produced_by=cmd y
          timestamp=scheduled.start_time.

        Args:
            scheduled: el ScheduledCommand que falló.
            reason: descripción legible del fallo.
            kind: categoría del fallo ('pddl' o 'concurrency').

        Returns:
            CommandFailure listo para incluir en el RunResult.failures.
        """
        current_snapshot = self._cursor.current
        current_world = current_snapshot.world
        current_metrics = current_snapshot.metrics
        cmd = scheduled.command

        new_world = _with_drone_state(
            current_world, cmd.drone_id, DroneState.ERROR
        )
        new_metrics = current_metrics.record_failed_command()

        failure_snapshot = WorldSnapshot(
            world=new_world,
            metrics=new_metrics,
            produced_by=cmd,
            timestamp=scheduled.start_time,
        )
        self._cursor.commit_new(failure_snapshot)

        return CommandFailure(
            scheduled=scheduled, reason=reason, kind=kind
        )
