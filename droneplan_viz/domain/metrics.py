"""MetricsTracker: contadores derivados de la simulación.

Refleja la información agregada del plan que el panel de la UI mostrará
y que la memoria del TFG puede analizar: coste acumulado, makespan,
cuántas acciones se han ejecutado, cuántos comandos rechazó el Validator.

Mapping con las métricas del PDDL real:

    total_cost       espejo de (:metric minimize (total-cost)) en parte 2.
                     Se incrementa con cada volar según fly-cost.
    total_time       espejo de (:metric minimize (total-time)) en parte 3.
                     Makespan: end_time máximo entre las acciones
                     ejecutadas.
    action_count     conteo trivial de acciones del plan. Útil ya en
                     parte 1, donde no hay métrica explícita.
    failed_commands  no es métrica PDDL; es información didáctica
                     ("tu plan generó 3 errores").

Decisiones de modelado:

    - Inmutable como el resto del dominio. Los métodos record_X
      devuelven una nueva instancia con el campo actualizado, en lugar
      de mutar. Esto encaja con el WorldSnapshot del Memento: cada
      snapshot puede agrupar (World, MetricsTracker) y la navegación
      hacia atrás restaura ambos íntegros.

    - Vive aparte de World, no como campo. Razón: las métricas son
      información derivada de la traza de acciones, no estado físico
      del mundo. Separar "qué hay" (World) de "cuánto cuesta lo que ha
      pasado" (MetricsTracker) mantiene cada pieza enfocada.

    - drones_in_error NO se almacena aquí. Es derivable del World
      actual (cuenta de drones con state == ERROR). Almacenarlo aquí
      invitaría a desincronización. El panel lo calculará al vuelo.

    - total_time se acumula tomando el MÁXIMO, no la suma. Refleja la
      semántica de makespan: acciones concurrentes terminan en
      distintos momentos, lo que importa es la última. Para planes
      secuenciales puros, makespan == sum_of_durations.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class MetricsTracker:
    """Contadores agregados del plan ejecutado hasta el momento.

    Atributos:
        total_cost: coste acumulado de las acciones de movimiento.
            Espejo de (:metric minimize (total-cost)) del PDDL parte 2.
        total_time: makespan en tiempo simulado. End_time máximo entre
            todas las acciones registradas. Espejo de
            (:metric minimize (total-time)) del PDDL parte 3.
        action_count: número de acciones del plan ejecutadas con éxito.
        failed_commands: número de comandos rechazados por el Validator.
    """

    total_cost: float = field(default=0.0)
    total_time: float = field(default=0.0)
    action_count: int = field(default=0)
    failed_commands: int = field(default=0)

    def record_move(self, cost: float) -> "MetricsTracker":
        """Registra el coste de un movimiento ejecutado.

        Incrementa total_cost y action_count. No toca total_time:
        eso es responsabilidad de record_action_completed(), que
        recibe el end_time y se llama también para esta acción.
        Mantenemos ambos métodos separados porque el coste solo
        aplica a movimientos, mientras que el end_time aplica a
        cualquier acción.

        Devuelve una nueva instancia; no muta self.
        """
        return replace(
            self,
            total_cost=self.total_cost + cost,
            action_count=self.action_count + 1,
        )

    def record_action_completed(self, end_time: float) -> "MetricsTracker":
        """Registra la finalización de una acción no-movimiento.

        Actualiza total_time tomando el máximo con el end_time recibido
        (semántica de makespan). Incrementa action_count.

        Para movimientos, los handlers llamarán a record_move(cost)
        para el coste y a record_action_completed(end_time) para el
        tiempo; cada método actualiza lo que le corresponde sin
        solapamiento. action_count se incrementa en ambos: a la hora
        de contar acciones, recolectarlas dos veces sería un bug, así
        que el dispatch tendrá cuidado de llamar a uno u otro según
        el tipo de acción.

        En el runtime, donde estos métodos se invocarán realmente, el
        Dispatcher decidirá el patrón exacto. Aquí ofrecemos los
        ladrillos.
        """
        return replace(
            self,
            total_time=max(self.total_time, end_time),
            action_count=self.action_count + 1,
        )

    def record_move_completed(
        self, cost: float, end_time: float
    ) -> "MetricsTracker":
        """Atajo para movimientos: registra a la vez coste y end_time
        en un solo cambio, incrementando action_count UNA vez.

        Equivalente a aplicar record_move + record_action_completed
        pero sin duplicar el incremento de action_count. Hacer dos
        llamadas separadas contaría la acción dos veces.

        Devuelve una nueva instancia.
        """
        return replace(
            self,
            total_cost=self.total_cost + cost,
            total_time=max(self.total_time, end_time),
            action_count=self.action_count + 1,
        )

    def record_failed_command(self) -> "MetricsTracker":
        """Registra un comando rechazado por el Validator.

        Solo incrementa failed_commands; no afecta a las métricas
        derivadas del PDDL (cost, time, action_count). Refleja la
        decisión didáctica: el alumno puede ver cuántos errores
        cometió al traducir el plan, separado del rendimiento del
        plan en sí.
        """
        return replace(self, failed_commands=self.failed_commands + 1)
