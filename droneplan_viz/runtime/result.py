"""
Tipos de salida del runtime: CommandFailure y RunResult.

Dataclasses puramente pasivas (frozen + slots) que describen el
resultado de ejecutar un Plan. Aisladas en su propio módulo para que
los consumidores (render el runtime, facade el runtime) las importen sin
arrastrar la maquinaria del runner ni de la cola de eventos.

- CommandFailure: descripción inmutable del fallo de un Command
  concreto durante la ejecución. Distingue tres tipos:
    * "pddl" — el Validator rechazó el Command
                       (precondiciones del dominio: drone inexistente,
                       paquete no co-localizado, brazo ocupado, etc.).
    * "concurrency" — la tabla de recursos del runtime detectó una
                       intersección temporal con otra acción en vuelo
                       (regla del PDF Parte 3).
    * "structural"  — esta categoría se reserva para usos futuros
                       (errores estructurales detectados durante la
                       ejecución, no durante la construcción del Plan).
                       La validación estructural del Plan se hace en
                       Plan.__post_init__ y produce PlanValidationError,
                       que NO usa esta clase. Mantenemos "structural"
                       en el Literal por completitud y para no atar el
                       runtime a una categorización rígida.

- RunResult: dataclass agregada con todo lo que el caller necesita
  saber tras un execute() o un dry_run(). NO incluye el HistoryManager;
  el HistoryManager es propiedad del PlanRunner y se accede vía
  runner.history. Esto evita que dos consumidores compartan la misma
  referencia al manager (y por accidente acabaran muteándolo desde
  callers separados, aunque la mutación esté limitada a commit).

Decisiones de diseño:

1. Inmutables (frozen + slots). Coherente con el resto del proyecto:
   los resultados son fotos del pasado, no estado vivo. Si el caller
   quiere derivar otro objeto, hace replace.

2. RunResult.makespan se ALMACENA, no se deriva. Justificación: una vez
   terminada la ejecución, el makespan es un dato fijo. Calcularlo al
   vuelo recorriendo failures+scheduled+history para encontrar el
   end_time máximo entre Commands exitosos sería trabajo repetido y
   no aporta nada. El runner lo calcula una vez y lo guarda.

3. RunResult.history_length: longitud del HistoryManager al finalizar.
   También dato fijo; lo almacenamos para que el caller pueda
   asegurarse de que el manager no se ha mutado por terceros entre la
   ejecución y la lectura (caso patológico, pero comprobable).

4. RunResult.failures es una tupla, no una lista. Inmutable estructural.

5. CommandFailure referencia el ScheduledCommand original, no solo el
   command_id. Esto permite al consumidor saber EN QUÉ INSTANTE estaba
   programado el Command que falló, no solo qué Command era. Útil para
   mensajes de error legibles y para localización temporal en el
   render.

6. La propiedad RunResult.succeeded es trivial (failures vacía). Se
   ofrece como atajo idiomático para callers que solo quieren un
   booleano.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from droneplan_viz.domain import MetricsTracker, World
from droneplan_viz.runtime.plan import Plan, ScheduledCommand


# Tipos posibles de fallo. Mantenemos el Literal cerrado para que el
# type checker avise si alguien introduce una categoría nueva sin
# documentarla aquí.
FailureKind = Literal["pddl", "concurrency", "structural"]


@dataclass(frozen=True, slots=True)
class CommandFailure:
    """Descripción inmutable del fallo de un Command durante la ejecución.

    Atributos:
        scheduled: el ScheduledCommand que falló. Contiene el Command
            original y su start_time programado. Por las garantías de
            inmutabilidad del proyecto, referenciarlo es seguro.
        reason: mensaje legible de la causa del fallo. Proveniente del
            ValidationResult.reason del Validator si kind=pddl,
            o construido por el runtime para los otros kinds.
        kind: categoría del fallo. Ver el docstring del módulo para la
            semántica de cada valor.
    """

    scheduled: ScheduledCommand
    reason: str
    kind: FailureKind


@dataclass(frozen=True, slots=True)
class RunResult:
    """Resultado agregado de ejecutar un Plan.

    Atributos:
        plan: el Plan que se ejecutó. Almacenado para facilitar análisis
            posterior sin necesidad de mantener referencias externas.
        final_world: estado del World tras la ejecución completa. Por
            las garantías de inmutabilidad estructural del dominio,
            referenciarlo es seguro y barato.
        final_metrics: MetricsTracker tras la ejecución. Incluye
            total_cost, total_time (makespan), action_count,
            failed_commands.
        failures: tupla de CommandFailure, una por Command que falló
            durante la ejecución. Vacía si todo el plan se ejecutó
            con éxito.
        history_length: número de WorldSnapshots en el HistoryManager
            del runner al terminar. Incluye el snapshot inicial.
        makespan: end_time máximo entre los Commands que se ejecutaron
            con éxito. 0.0 si ninguno tuvo éxito o si el plan estaba
            vacío. Coincide con final_metrics.total_time en ejecuciones
            sin errores; cuando hay errores la métrica puede diverger
            porque solo cuenta los Commands aplicados.

    Propiedades derivadas:
        succeeded: True si failures está vacía.
    """

    plan: Plan
    final_world: World
    final_metrics: MetricsTracker
    failures: tuple[CommandFailure, ...] = field(default=())
    history_length: int = 1
    makespan: float = 0.0

    @property
    def succeeded(self) -> bool:
        """True si no hubo ningún fallo durante la ejecución.

        Es un atajo conveniente; equivale a `len(failures) == 0`.
        """
        return len(self.failures) == 0
