"""
TimelineCursor: dueño de la posición de lectura sobre un HistoryManager.

Separado intencionadamente del HistoryManager:
- HistoryManager es dueño del ALMACÉN (los snapshots).
- TimelineCursor es dueño de la POSICIÓN ACTUAL de lectura.

Esta separación da flexibilidad:
- Tests del manager sin necesidad de cursor.
- Posibilidad futura de tener varios cursores apuntando al mismo manager
  (p.ej. vista comparativa entre dos puntos del plan), sin reorganizar la
  API.
- Las dos clases tienen responsabilidades únicas y testables aisladamente.

API navegacional:
- back / forward: mueven una posición.
- goto(index): salta a una posición arbitraria.
- jump_to_head / jump_to_start: atajos al final y al inicio.
- at_head / at_start: introspección de posición.
- current: snapshot en la posición actual.
- index: índice actual.

API de extensión del historial:
- commit_new(snapshot): operación compuesta que encapsula la semántica
  Ctrl+Z. Si el cursor está en head, simplemente añade el snapshot y
  avanza. Si está en medio, descarta lo que sigue al cursor, añade el
  snapshot nuevo y avanza al recién creado head.

  Esta operación es el punto de entrada normal del runtime cuando aplica
  un Command durante una sesión interactiva. El runtime NO debe llamar
  directamente a HistoryManager.commit con truncate_from; la
  responsabilidad de calcular ese truncate_from a partir del cursor
  pertenece a TimelineCursor.commit_new.
"""
from __future__ import annotations

from droneplan_viz.history.manager import HistoryManager
from droneplan_viz.history.snapshot import WorldSnapshot


class TimelineCursor:
    """Posición de lectura sobre un HistoryManager.

    Atributos privados:
        _history: referencia al manager que contiene los snapshots.
        _index: posición actual, invariante 0 <= _index < len(_history).

    Empieza apuntando al head del manager en el momento de la creación.
    """

    __slots__ = ("_history", "_index")

    def __init__(self, history: HistoryManager) -> None:
        """Crea un cursor sobre el historial dado, posicionado en su head."""
        self._history = history
        self._index = history.head_index

    # -----------------------------------------------------------------
    # Introspección
    # -----------------------------------------------------------------
    @property
    def index(self) -> int:
        """Índice de la posición actual del cursor."""
        return self._index

    @property
    def current(self) -> WorldSnapshot:
        """Snapshot en la posición actual."""
        return self._history.at(self._index)

    @property
    def at_head(self) -> bool:
        """True si el cursor está en el último snapshot del historial."""
        return self._index == self._history.head_index

    @property
    def at_start(self) -> bool:
        """True si el cursor está en el snapshot inicial."""
        return self._index == 0

    # -----------------------------------------------------------------
    # Navegación
    # -----------------------------------------------------------------
    def back(self) -> WorldSnapshot | None:
        """Mueve el cursor una posición atrás y devuelve el nuevo snapshot.

        Returns:
            El snapshot en la nueva posición, o None si ya estaba en el
            inicio (en cuyo caso no se mueve).
        """
        if self.at_start:
            return None
        self._index -= 1
        return self.current

    def forward(self) -> WorldSnapshot | None:
        """Mueve el cursor una posición adelante y devuelve el nuevo snapshot.

        Returns:
            El snapshot en la nueva posición, o None si ya estaba en el
            head (en cuyo caso no se mueve).
        """
        if self.at_head:
            return None
        self._index += 1
        return self.current

    def goto(self, index: int) -> WorldSnapshot:
        """Salta a una posición arbitraria del historial.

        Args:
            index: índice no negativo en el rango [0, len(history)).

        Returns:
            Snapshot en la nueva posición.

        Raises:
            IndexError: si el índice está fuera del rango válido.
        """
        # Delegamos la validación de rango al manager.at
        snap = self._history.at(index)
        self._index = index
        return snap

    def jump_to_head(self) -> WorldSnapshot:
        """Salta al último snapshot del historial."""
        self._index = self._history.head_index
        return self.current

    def jump_to_start(self) -> WorldSnapshot:
        """Salta al snapshot inicial del historial."""
        self._index = 0
        return self.current

    # -----------------------------------------------------------------
    # Extensión del historial
    # -----------------------------------------------------------------
    def commit_new(self, snapshot: WorldSnapshot) -> int:
        """Añade un snapshot al historial respetando la semántica Ctrl+Z.

        Si el cursor está en head, simplemente añade y avanza.
        Si el cursor está en medio del historial, descarta todos los
        snapshots posteriores al cursor antes de añadir el nuevo, y
        deja al cursor en la nueva posición del head.

        Args:
            snapshot: snapshot a añadir.

        Returns:
            Índice del snapshot recién añadido (nuevo head).
        """
        if self.at_head:
            # No hace falta truncar; commit normal.
            new_index = self._history.commit(snapshot)
        else:
            # Truncamos desde la posición SIGUIENTE al cursor. El cursor
            # se queda conservado en el historial; lo que se descarta es
            # todo lo que venía después.
            truncate_from = self._index + 1
            new_index = self._history.commit(
                snapshot, truncate_from=truncate_from
            )
        self._index = new_index
        return new_index
