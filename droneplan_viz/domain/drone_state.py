"""DroneState: estados de la máquina de estados finita del drone.

La FSM tiene cuatro estados:

    IDLE         disponible para aceptar el siguiente comando
    MOVING       en tránsito entre dos localizaciones
    INTERACTING  ejecutando una acción atómica en su localización
                 actual (recoger, entregar, poner_en, sacar_de)
    ERROR        algún comando produjo una violación de precondiciones
                 y el drone queda paralizado

Topología de transiciones forward (durante el avance del tiempo simulado):

      ┌─────────────────────┐
      │                     │
      ▼                     │
    IDLE ──► MOVING ────────┤
      │       │             │
      │       │             │
      ▼       ▼             │
    INTERACTING ◄───────────┘
      │
      │  (al volver a IDLE tras terminar)
      ▼
    IDLE

    Cualquiera de los anteriores ──► ERROR  (al violar precondiciones)
    ERROR ──► (ningún estado): sumidero

ERROR es sumidero local a la rama temporal forward. Una vez dentro,
ningún comando posterior saca al drone de ERROR. La "salida" solo se
produce cuando el usuario navega hacia atrás en el historial, lo que
restaura un WorldSnapshot anterior íntegro; no es una transición de la
FSM, es un reemplazo de la foto del mundo. Detalles del razonamiento y
sus consecuencias en la documentación de diseño.

Los métodos del enum exponen únicamente preguntas estáticas que no
dependen de un comando concreto. La lógica de "qué transiciones son
válidas dado este comando" vive en el Validator (decisión 2 del diseño:
una sola fuente de verdad para reglas físicas).
"""

from enum import Enum, auto


class DroneState(Enum):
    """Estado discreto de un drone en la simulación."""

    IDLE = auto()
    MOVING = auto()
    INTERACTING = auto()
    ERROR = auto()

    def is_busy(self) -> bool:
        """¿El drone está actualmente ocupado en una acción?

        Útil para el Validator (rechazar comandos concurrentes sobre el
        mismo drone, manteniendo la secuencialidad intra-agente de la
        decisión 5) y para el renderer (animaciones de movimiento o
        interacción).

        ERROR no se considera "ocupado": el drone no está haciendo nada,
        está paralizado. Se distingue con is_terminal_forward().
        """
        return self in (DroneState.MOVING, DroneState.INTERACTING)

    def is_terminal_forward(self) -> bool:
        """¿Este estado es sumidero en la línea temporal forward?

        Devuelve True únicamente para ERROR. Es la codificación de la
        propiedad de sumidero: el Validator usa este método para rechazar
        cualquier comando dirigido a un drone en estado terminal, sin
        tener que conocer todos los valores del enum. Si en el futuro se
        añadiera un nuevo estado terminal (poco probable, pero posible),
        bastaría con ampliar este método y el resto del código heredaría
        el comportamiento correcto.
        """
        return self is DroneState.ERROR
