"""
Modelado del plan de entrada al runtime.

Un plan es una colección inmutable de Commands programados en el tiempo:
cada uno tiene un instante de inicio absoluto desde t=0 y una duración
heredada del propio Command. Este módulo NO ejecuta nada; solo modela.

Componentes:

- ScheduledCommand: par (Command, start_time) inmutable. Encapsula la
  decisión "este Command se ejecuta a partir de este instante". El
  end_time se deriva, no se almacena.

- Plan: contenedor inmutable de ScheduledCommands. Aplica las
  validaciones estructurales al construirse (start_time no negativos,
  duration no negativas, command_id únicos). NO aplica reglas de
  concurrencia ni de PDDL; esas se evalúan en runtime contra el World
  vivo y la tabla de recursos.

- Plan.sequential(commands): factoría que produce un Plan con timestamps
  enteros consecutivos a partir de una secuencia de Commands. Útil para
  las partes 1-2 del PDDL (planificación atemporal): los timestamps son
  arbitrarios mientras sean estrictamente crecientes, y "0, 1, 2, ..." es
  la representación natural.

- PlanValidationError: excepción específica para violaciones estructurales
  del plan, detectadas en el momento de su construcción.

Decisiones de diseño documentadas:

1. ScheduledCommand es el contenedor temporal del plan: empareja un
   Command con su instante de inicio absoluto. Referencia el Command
   directamente (no un id opaco), de modo que el bucle de ejecución y
   los tests trabajan con el objeto Command sin niveles de indirección
   ni tablas auxiliares id -> Command. El end_time se deriva de
   start_time + cmd.duration; no se almacena.

2. Timestamps absolutos desde t=0, no relativos al Command anterior.
   La concurrencia se detecta comparando intervalos [start, end);
   absolutos es la representación natural para esa comparación.
   Relativos forzarían re-derivar absolutos en cada chequeo.

3. start_time es float, no int. Las duraciones de vuelo dependen de
   fly-cost (parte 2 y 3), que puede ser cualquier número no negativo
   declarado en world.costs. No restringimos a enteros.

4. duration NO vive en ScheduledCommand: vive en cmd.duration. Esto
   refleja que la duración es una propiedad del Command (decisión 2.4
   de el runtime), no de su programación. Dos ScheduledCommand con el
   mismo Command tendrían la misma duración por definición.

5. Validación estructural eager en __post_init__ del Plan. Si un Plan
   se construye, es estructuralmente sano. Esto contrasta con la
   filosofía "el tipo no valida" del dominio (que sí permite valores
   incoherentes en las dataclasses), porque el Plan es la frontera de
   entrada del runtime: aquí es donde tiene sentido rechazar pronto.

6. command_id únicos dentro del plan. Dos Commands con el mismo
   command_id no se pueden distinguir en el historial: el snapshot
   produced_by sería ambiguo. Esto fuerza una propiedad básica que el
   patrón new_command_id() (uuid4) garantiza por defecto; pero si el
   facade inyecta ids legibles ("plan_step_3"), debe garantizar
   unicidad. El plan lo verifica.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from droneplan_viz.commands.base import Command


class PlanValidationError(ValueError):
    """Excepción lanzada cuando un Plan se construye con datos
    estructuralmente inválidos.

    Subclase de ValueError porque expresa "argumentos inválidos al
    constructor", semántica estándar de la stdlib. Tener una clase
    propia permite a los tests capturar exactamente este caso sin
    riesgo de tragar ValueError de otro origen.

    No tiene atributos adicionales; el mensaje contiene la razón.
    """


@dataclass(frozen=True, slots=True)
class ScheduledCommand:
    """Un Command programado para empezar en un instante temporal concreto.

    Atributos:
        command: el Command a ejecutar. Cualquiera de las seis dataclasses
            del paquete commands. Su duración se consulta vía
            command.duration.
        start_time: instante absoluto desde t=0 en el que el Command
            empieza. Debe ser >= 0.0. El Plan verifica esta propiedad
            al construirse.

    Propiedades derivadas:
        end_time: start_time + command.duration. Intervalo cerrado-abierto:
            la acción está activa en [start_time, end_time) y completada
            en [end_time, infinito). Esta convención es la que usa el
            runtime para detectar solapamientos temporales.
    """

    command: Command
    start_time: float

    @property
    def end_time(self) -> float:
        """Instante en que la acción termina.

        Derivado de start_time + command.duration; no se almacena.
        """
        return self.start_time + self.command.duration


@dataclass(frozen=True, slots=True)
class Plan:
    """Colección inmutable de ScheduledCommands.

    Aplica validaciones estructurales en __post_init__:
        - start_time >= 0 para todos.
        - command.duration >= 0 para todos.
        - command_id únicos.

    NO valida reglas PDDL (precondiciones de cada acción) ni reglas de
    concurrencia (intersección temporal de recursos). Esas se evalúan
    en runtime contra el World del momento y la tabla de recursos.

    El orden de los ScheduledCommand dentro de la tupla NO determina
    el orden de ejecución: el runtime ordena por start_time. Sin
    embargo, conservamos el orden de entrada para facilitar logs y
    trazabilidad en mensajes de error.

    Atributos:
        scheduled: tupla de ScheduledCommand. Inmutable.
    """

    scheduled: tuple[ScheduledCommand, ...] = field(default=())

    def __post_init__(self) -> None:
        """Valida estructuralmente el plan.

        Raises:
            PlanValidationError: si alguna de las invariantes
                estructurales se viola. El mensaje describe la razón
                concreta del primer fallo encontrado.
        """
        seen_ids: set[str] = set()
        for i, sched in enumerate(self.scheduled):
            if sched.start_time < 0:
                raise PlanValidationError(
                    f"ScheduledCommand en posición {i}: start_time "
                    f"negativo ({sched.start_time}); debe ser >= 0"
                )
            if sched.command.duration < 0:
                raise PlanValidationError(
                    f"ScheduledCommand en posición {i}: "
                    f"command.duration negativa ({sched.command.duration}); "
                    f"debe ser >= 0"
                )
            cid = sched.command.command_id
            if cid in seen_ids:
                raise PlanValidationError(
                    f"ScheduledCommand en posición {i}: command_id "
                    f"duplicado '{cid}'; los ids deben ser únicos en "
                    f"el plan"
                )
            seen_ids.add(cid)

    def __len__(self) -> int:
        """Número de ScheduledCommand en el plan."""
        return len(self.scheduled)

    def __iter__(self):
        """Iteración sobre los ScheduledCommand en orden de declaración.

        NO en orden de start_time. El runtime es responsable de ordenar
        por timestamp si su algoritmo lo requiere.
        """
        return iter(self.scheduled)

    @classmethod
    def sequential(cls, commands: Iterable[Command]) -> "Plan":
        """Construye un Plan secuencial atemporal.

        Asigna timestamps enteros consecutivos (0.0, 1.0, 2.0, ...) a
        los Commands en el orden recibido. Los Commands suelen tener
        duration=0.0 (partes 1-2 del PDDL), en cuyo caso los timestamps
        son arbitrarios y solo importa el orden relativo. Si algún
        Command tiene duration > 0, este constructor produce un plan
        donde las acciones se ejecutan una tras otra sin solapamiento
        si y solo si todas las duraciones son <= 1.0, lo cual NO es
        responsabilidad de esta factoría comprobar.

        Para planes con duraciones reales (parte 3), usar el
        constructor directo con ScheduledCommands de timestamps
        absolutos calculados por el facade.

        Args:
            commands: iterable de Commands en el orden deseado.

        Returns:
            Plan con ScheduledCommands programados en t=0, 1, 2, ...

        Raises:
            PlanValidationError: si los Commands recibidos violan las
                invariantes estructurales (p.ej. command_id duplicados).
        """
        scheduled = tuple(
            ScheduledCommand(command=cmd, start_time=float(i))
            for i, cmd in enumerate(commands)
        )
        return cls(scheduled=scheduled)
