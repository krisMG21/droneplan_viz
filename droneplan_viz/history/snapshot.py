"""
WorldSnapshot: representación capturada de un punto en la línea temporal
del plan. Agrupa el estado del mundo, las métricas acumuladas, el Command
que produjo este snapshot (None para el snapshot inicial) y un timestamp.

Decisión arquitectónica central documentada aquí:

  El WorldSnapshot mantiene una REFERENCIA DIRECTA al World, sin deepcopy.

  Esto se apoya en la garantía de inmutabilidad estructural profunda
  establecida:
  - World es frozen + slots; sus dicts internos son MappingProxyType.
  - Todas las entidades referenciadas (Drone, Package, Person, Transporter,
    Location, Content) son frozen + slots con colecciones tuple internas.
  - MetricsTracker es frozen + slots con campos numéricos primitivos.
  - El tipo suma PackageLocation (AtLocation, HeldByArm, InTransporter)
    son los tres frozen + slots.

  Bajo estas garantías, el World capturado no puede ser modificado por
  ninguna operación legal del lenguaje. Cualquier "transición" produce
  un World nuevo (vía dataclasses.replace), dejando intactos los snapshots
  previos que apuntan al World anterior.

  Esta decisión sacrifica defensa en profundidad contra hipotéticas
  regresiones futuras (alguien introduce un campo mutable en una entidad
  del dominio) a cambio de eficiencia O(1) en captura y memoria por
  snapshot. El test-canario test_inmutabilidad_garantiza_snapshot_estable
  vigila proactivamente la propiedad de la que dependemos. Si ese test
  se rompe en una sesión futura, la solución es localizada: meter
  copy.deepcopy(world) en WorldSnapshot.capture() y todo el resto del
  sistema sigue funcionando.

  Motivación adicional: la herramienta sirve para visualizar replays de
  planes con interpolación visual entre keyframes (un snapshot por cada
  evento de interés: drone llega, sale, recoge, entrega). Un plan largo
  genera centenas o miles de snapshots. Deepcopy sería O(N) en memoria
  y tiempo por snapshot; referencia es O(1). El ahorro es relevante.
"""
from __future__ import annotations

from dataclasses import dataclass

from droneplan_viz.commands.base import Command
from droneplan_viz.domain import MetricsTracker, World


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Foto inmutable del estado en un punto de la línea temporal.

    Atributos:
        world: estado del mundo capturado. Por la garantía de inmutabilidad
            estructural, se almacena por referencia directa, no
            por copia. Ver el docstring del módulo para la justificación.
        metrics: métricas acumuladas hasta este punto.
        produced_by: Command que produjo esta transición. None solo en el
            snapshot inicial (estado de partida del plan, antes de aplicar
            cualquier acción).
        timestamp: tiempo simulado en el que se produjo este snapshot.
            0.0 para el snapshot inicial. El runtime lo provee al hacer
            commit; aquí es solo dato.
    """
    world: World
    metrics: MetricsTracker
    produced_by: Command | None = None
    timestamp: float = 0.0
