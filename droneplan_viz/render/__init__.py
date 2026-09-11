"""Paquete render: motor de dibujo puro sobre pygame.Surface.

el runtime del proyecto droneplan_viz. API pública:

- Theme: configuración visual centralizada.
- render_snapshot(surface, snapshot, *, theme=None): dibujo estático.
- render_frame(surface, snap_a, snap_b, progress, *, theme=None):
  dibujo interpolado entre dos snapshots consecutivos.
- locate_entity(world, entity_id, surface_size, *, theme=None): posición
  en píxeles (con zoom=1, pan=(0,0)) donde se dibujaría una entidad. La UI
  la usa para click-to-focus.
- ZOOM_MIN, ZOOM_MAX: cotas del zoom de cámara que la UI debe respetar.
- Timeline(history, theme=None): utilidad para que la UI mapee un
  tiempo de reproducción al par (snap_a, snap_b, progress) que
  render_frame consume.

Cámara (zoom + pan): render_snapshot y render_frame aceptan los kwargs
opcionales `zoom: float = 1.0` y `pan: tuple[int, int] = (0, 0)`. El zoom
cambia el ESPACIADO del grafo (las locations se separan/acercan), NO el
tamaño de los sprites; el pan desplaza la imagen en píxeles. Los defaults
reproducen el render clásico píxel-idéntico.

Filosofía del paquete (heredada de Sesiones A/B/C, ver memoria_sesion_d.md):

- Cero acoplamiento con el runtime: el render consume WorldSnapshots ya
  producidos, NO referencia PlanRunner ni invoca métodos de runtime.
- Cero loop, cero eventos: NO se crea pygame.display, NO se llama a
  pygame.event.get. El render recibe una Surface y dibuja en ella.
- Inmutabilidad del dominio: el render NUNCA muta entidades del World.
  Solo lee. Las únicas mutaciones son sobre la pygame.Surface recibida.
- Tema centralizado: paleta, dimensiones y tipografía viven en Theme.
  Cero colores ni tamaños hardcodeados fuera de theme.py.
"""

from droneplan_viz.render.painter import (
    ZOOM_MAX,
    ZOOM_MIN,
    locate_entity,
    locate_entity_interpolated,
    render_frame,
    locate_entity_at,
    render_world_at,
    render_snapshot,
    supersample_factor,
)
from droneplan_viz.render.theme import Theme
from droneplan_viz.render.timeline import Timeline

__all__ = [
    "Theme",
    "Timeline",
    "ZOOM_MAX",
    "ZOOM_MIN",
    "locate_entity",
    "locate_entity_interpolated",
    "render_frame",
    "locate_entity_at",
    "render_world_at",
    "render_snapshot",
    "supersample_factor",
]
