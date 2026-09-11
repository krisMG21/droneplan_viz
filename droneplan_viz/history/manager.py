"""
HistoryManager: almacén lineal de WorldSnapshots con soporte de truncado.

Responsabilidad única: dueño de la lista de snapshots. NO mantiene posición
de lectura (eso es del TimelineCursor) y NO aplica Commands (eso es del
runtime). Es deliberadamente pasivo.

Decisiones de diseño:

- Lista lineal con truncado. No hay ramificación tipo árbol/git.
  Documentado en la propuesta: aplicar un snapshot nuevo con
  cursor en medio descarta el futuro (tipo Ctrl+Z de cualquier editor).
  El truncado se pide explícitamente al commit; el manager no asume nada.

- El manager NO se acopla al TimelineCursor. Recibe órdenes (commit con
  o sin truncate_from); no consulta posiciones de lectura. Esto permite:
  - Tests del manager sin necesidad de un cursor.
  - Compartir un mismo manager entre varios cursores (no usado hoy,
    pero abre la puerta a vistas múltiples del mismo historial sin
    forzar un cambio futuro de API).

- IndexError canónico al acceder fuera de rango. No inventamos excepciones
  custom: list ya tiene su semántica clara y los usuarios la conocen.

- Inicialización con un snapshot inicial obligatorio. No existe un
  HistoryManager vacío: el snapshot 0 representa el estado de partida.
  Esta es la diferencia entre 'historial' (siempre tiene un origen) y
  'cola' (puede estar vacía).
"""
from __future__ import annotations

from droneplan_viz.history.snapshot import WorldSnapshot


class HistoryManager:
    """Almacén lineal de WorldSnapshots con soporte de truncado.

    Atributos privados:
        _snapshots: lista cronológica. Índice 0 = snapshot inicial;
            últimos índices = más recientes.

    No es frozen porque la lista misma es mutable por necesidad (commit
    añade entradas). Lo que es inmutable son los snapshots individuales.
    """

    __slots__ = ("_snapshots",)

    def __init__(self, initial: WorldSnapshot) -> None:
        """Crea el manager con un snapshot inicial.

        Args:
            initial: snapshot que representa el estado de partida del
                plan. Por convención lleva produced_by=None y
                timestamp=0.0, aunque no se valida.
        """
        self._snapshots: list[WorldSnapshot] = [initial]

    def __len__(self) -> int:
        """Número total de snapshots almacenados."""
        return len(self._snapshots)

    @property
    def head_index(self) -> int:
        """Índice del último snapshot (el más reciente).

        Siempre >= 0 porque el inicial nunca se borra.
        """
        return len(self._snapshots) - 1

    def at(self, index: int) -> WorldSnapshot:
        """Devuelve el snapshot en la posición dada.

        Args:
            index: índice no negativo en el rango [0, len(self)).

        Raises:
            IndexError: si el índice está fuera del rango válido.
        """
        if index < 0 or index >= len(self._snapshots):
            raise IndexError(
                f"índice {index} fuera de rango [0, {len(self._snapshots)})"
            )
        return self._snapshots[index]

    def commit(
        self,
        snapshot: WorldSnapshot,
        *,
        truncate_from: int | None = None,
    ) -> int:
        """Añade un snapshot al historial, opcionalmente truncando antes.

        Si truncate_from se proporciona, descarta todos los snapshots
        desde esa posición INCLUSIVE antes de añadir el nuevo. Esto
        implementa la semántica Ctrl+Z: cuando el runtime aplica un
        Command estando posicionado en medio del historial, todo lo
        que venía después se pierde.

        Args:
            snapshot: snapshot a añadir.
            truncate_from: índice desde el que descartar (inclusive).
                Si None, simplemente añade al final. Si 0, dejaría el
                historial vacío de cara al append (rompiendo el
                invariante de tener al menos un snapshot), por lo que
                truncate_from=0 está prohibido.

        Returns:
            Índice del snapshot recién añadido.

        Raises:
            ValueError: si truncate_from == 0 (borraría el snapshot
                inicial) o si truncate_from > len(self).
            IndexError: si truncate_from < 0.
        """
        if truncate_from is not None:
            if truncate_from < 0:
                raise IndexError(
                    f"truncate_from no puede ser negativo: {truncate_from}"
                )
            if truncate_from == 0:
                raise ValueError(
                    "truncate_from=0 borraría el snapshot inicial; prohibido"
                )
            if truncate_from > len(self._snapshots):
                raise ValueError(
                    f"truncate_from={truncate_from} fuera de rango "
                    f"(longitud actual {len(self._snapshots)})"
                )
            del self._snapshots[truncate_from:]

        self._snapshots.append(snapshot)
        return len(self._snapshots) - 1
