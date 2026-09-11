"""Cálculo puro de regiones de la ventana (rects para world y paneles HUD).

Función central: compute_regions(window_size). Recibe el tamaño actual
de la ventana y devuelve un dataclass Regions con cuatro pygame.Rect
disjuntos:

    +----------------------------------------+--------+
    |             hud_top                    |        |
    +----------------------------------------+ hud_   |
    |                                        | right  |
    |             world                      |        |
    |                                        |        |
    +----------------------------------------+        |
    |             hud_bottom                 |        |
    +----------------------------------------+--------+

Notas:

- hud_right ocupa toda la altura de la ventana (incluido el "hueco" a
  derecha de top y bottom). Esto deja el panel lateral como una columna
  continua, visualmente más limpio que un rectángulo recortado.
- world es subsurface central; lo consume render_frame.
- Lanzar ValueError si la ventana es demasiado pequeña (world < mínimo)
  es decisión consciente: layout.py mantiene su pureza (sin caps ni
  fallbacks) y la responsabilidad de imponer un mínimo recae en app.py
  (que lo hará en su handler de pygame.VIDEORESIZE).

Constantes con valores por defecto basados en la decisión §2.3 de la
propuesta. Se exponen como argumentos opcionales para
permitir tests con valores arbitrarios y futuras configuraciones via
Theme si alguna vez se centralizan en el render (no es el caso ahora).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pygame


#: Alto del panel HUD superior en píxeles (decisión §2.3).
DEFAULT_TOP_H: Final[int] = 56
#: Alto del panel HUD inferior en píxeles (timeline + velocidad + "Now:").
DEFAULT_BOTTOM_H: Final[int] = 100
#: Ancho del panel HUD lateral derecho (métricas + fallos).
DEFAULT_RIGHT_W: Final[int] = 380
#: Tamaño mínimo del world por debajo del cual no se puede operar
#: (sprites se superponen, layout circular ya no es legible). Coincide
#: con un escenario docente de 4-6 locations renderizado con holgura.
DEFAULT_MIN_WORLD: Final[tuple[int, int]] = (400, 300)


@dataclass(frozen=True, slots=True)
class Regions:
    """Tres rects HUD + un rect world, todos como pygame.Rect frozen.

    Los pygame.Rect son mutables por naturaleza, pero el dataclass frozen
    los almacena por valor. Como referencia funcional (consultar
    .x/.y/.width/.height) es lo que necesitamos; quien quisiera mutarlos
    tendría que sustituir el Regions entero.
    """
    world: pygame.Rect
    hud_top: pygame.Rect
    hud_bottom: pygame.Rect
    hud_right: pygame.Rect


def compute_regions(
    window_size: tuple[int, int],
    *,
    top_h: int = DEFAULT_TOP_H,
    bottom_h: int = DEFAULT_BOTTOM_H,
    right_w: int = DEFAULT_RIGHT_W,
    min_world: tuple[int, int] = DEFAULT_MIN_WORLD,
) -> Regions:
    """Calcula las cuatro regiones de la ventana.

    Args:
        window_size: (width, height) en píxeles. Debe ser positivo.
        top_h: alto del HUD superior.
        bottom_h: alto del HUD inferior.
        right_w: ancho del HUD lateral.
        min_world: (min_w, min_h) por debajo del cual ValueError.

    Returns:
        Regions con los cuatro rects no solapados:
          - hud_top    : (0, 0, W - right_w, top_h)
          - world      : (0, top_h, W - right_w, H - top_h - bottom_h)
          - hud_bottom : (0, H - bottom_h, W - right_w, bottom_h)
          - hud_right  : (W - right_w, 0, right_w, H)

    Raises:
        ValueError: si los rects HUD no caben en la ventana (left_w o
                    inner_h negativos) o si el world resultante es más
                    pequeño que min_world.
    """
    w, h = window_size
    if w <= 0 or h <= 0:
        raise ValueError(f"window_size debe ser positivo, recibido {window_size!r}")

    left_w = w - right_w
    inner_h = h - top_h - bottom_h
    min_world_w, min_world_h = min_world

    if left_w < min_world_w:
        raise ValueError(
            f"world resultante demasiado estrecho ({left_w} < {min_world_w}); "
            f"ventana ancho={w}, right_w={right_w}"
        )
    if inner_h < min_world_h:
        raise ValueError(
            f"world resultante demasiado bajo ({inner_h} < {min_world_h}); "
            f"ventana alto={h}, top_h={top_h}, bottom_h={bottom_h}"
        )

    return Regions(
        world=pygame.Rect(0, top_h, left_w, inner_h),
        hud_top=pygame.Rect(0, 0, left_w, top_h),
        hud_bottom=pygame.Rect(0, h - bottom_h, left_w, bottom_h),
        hud_right=pygame.Rect(left_w, 0, right_w, h),
    )


# ---------------------------------------------------------------------------
# HudLayout: constantes visuales del HUD centralizadas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HudLayout:
    """Constantes visuales del HUD (márgenes, tamaños de widgets, gaps).

    Centralizadas en un único dataclass frozen para que cualquier
    sesión futura (sprites, theming) pueda producir variantes con
    `dataclasses.replace(layout, btn_h=44)` sin tocar el código del HUD.

    Filosofía idéntica a `Theme` del paquete render: todas las
    decisiones visuales en un solo objeto, propagadas por argumento.

    Campos agrupados por región:

      HUD top — fila de 5 botones de transporte + título:
        - top_btn_w, top_btn_h: tamaño base de cada botón.
        - top_btn_gap: separación horizontal entre botones.
        - top_pad_x, top_pad_y: padding interior del panel superior.
        - top_play_extra_w: ancho extra del botón Play/Pause (más
          ancho porque alterna entre "Play" y "Pause").
        - top_title_offset_x: gap antes del label de título.

      HUD bottom — progress bar + tiempo + velocidades + "Now":
        - bottom_pad_x, bottom_pad_y: padding interior.
        - bottom_row1_h, bottom_row2_h: altos de las dos filas.
        - bottom_row_gap: separación vertical entre filas.
        - bottom_time_label_w: ancho del label "MM:SS".
        - bottom_speed_btn_w: ancho de cada botón de velocidad.
        - bottom_speed_btn_gap: gap entre botones de velocidad.

      HUD right — panel de métricas + panel de fallos:
        - right_pad: padding del panel lateral entero.
        - right_metrics_h: alto del subpanel de métricas.
        - right_subpanel_gap: separación entre subpanel de métricas
          y subpanel de fallos.
        - right_label_h: alto de cada label dentro de los subpaneles.
        - right_label_indent: indent horizontal de los labels dentro
          de UIPanel (margen para no pegarse al borde).
        - right_label_spacing: gap vertical entre labels.

      Overlay propio:
        - mark_color: color de la marca vertical sobre la barra.
        - mark_thickness, mark_selected_thickness: grosores.
        - halo_color: color RGBA del halo del fallo seleccionado.
        - halo_padding: cuánto sobresale el halo de la barra (px).

    Defaults consensuados durante el runtime con el screenshot de control.
    Pasar al constructor del Hud con `Hud(..., hud_layout=HudLayout())`
    permite a un alumno experimentar con valores nuevos sin tocar
    `hud.py`.
    """
    # HUD top
    top_btn_w: int = 64
    top_btn_h: int = 36
    top_btn_gap: int = 6
    top_pad_x: int = 8
    top_play_extra_w: int = 0      # antes 24 (texto "Play"/"Pause" más ancho);
                                    # con sprites de transporte ya no hace falta
                                    # ancho extra. Se mantiene el campo por si
                                    # se quisiera revertir.
    top_title_offset_x: int = 12

    # HUD bottom
    bottom_pad_x: int = 12
    bottom_pad_y_top: int = 12
    bottom_row1_h: int = 24
    bottom_row2_h: int = 24
    bottom_row_gap: int = 10
    bottom_time_label_w: int = 110
    bottom_speed_btn_w: int = 50
    bottom_speed_btn_gap: int = 4
    bottom_label_widget_gap: int = 8

    # HUD right
    right_pad: int = 12
    right_metrics_h: int = 200
    right_subpanel_gap: int = 12
    right_label_h: int = 22
    right_label_indent: int = 8
    right_label_spacing: int = 24
    right_title_top_offset: int = 4
    right_first_label_top_offset: int = 28

    # Tab container del panel lateral (introducido al rediseñar el panel
    # para soportar la pestaña de Inventario dinámico).
    # right_tab_title_h reserva espacio para la fila de títulos de las
    # tabs ("Inventario", "Métricas+Fallos") que pygame_gui pinta arriba
    # del UITabContainer; medido empíricamente con Monogram 24px.
    right_tab_title_h: int = 32
    right_tab_inner_pad: int = 6

    # Inventario rediseñado: cajas con icono + nombre + stats por cada
    # entidad, agrupadas bajo cabeceras de sección con flecha de plegado.
    # Todo dentro de un UIScrollingContainer.
    #
    # Dimensiones medidas para que el conjunto (cabecera 18px + 3 líneas
    # de texto 14px en cada caja + margen interno) queda cómodo y entra
    # 1-2 cajas por categoría sin scroll en alturas típicas, pero
    # scrollable cuando excede.
    inv_header_h: int = 26             # alto de la cabecera (font 20 + holgura)
    inv_header_font_size: int = 20     # fuente de cabecera, vs 16 de stats y 18 de nombre
    inv_chevron_size: int = 16         # ancho/alto del sprite de flecha
    inv_box_h: int = 52                # alto de caja: bloque texto 28 + márgenes 12+12 = icono 24 + márgenes 14+14
    inv_box_gap: int = 6               # separación entre cajas verticalmente
    inv_section_gap: int = 8           # separación entre fin de sección y siguiente cabecera
    inv_icon_size: int = 28            # ancho/alto del recuadro de icono de entidad
    inv_box_inner_pad_x: int = 12      # padding interno horizontal (borde ←→ icono)
    inv_box_inner_pad_y: int = 4       # padding interno vertical en la caja
    inv_box_text_left_pad: int = 12    # espacio entre icono y bloque de texto (igual al inner_pad_x)
    inv_box_side_margin: int = 4       # margen lateral DENTRO del scroll (caja vs scrollbar)

    # Overlays propios
    mark_color: tuple[int, int, int] = (255, 60, 60)
    mark_thickness: int = 1
    mark_selected_thickness: int = 3
    halo_color: tuple[int, int, int, int] = (255, 220, 80, 140)
    halo_padding_x: int = 4
    halo_padding_y: int = 4
