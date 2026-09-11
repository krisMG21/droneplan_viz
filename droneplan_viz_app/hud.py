"""HUD: widgets de pygame_gui, sincronización con AppState y dibujo.

Encapsula toda la complejidad de pygame_gui. El controller NO importa
pygame_gui; recibe HudIntent abstraídas que el HUD produce a partir de
los eventos de los widgets.

Estructura visual (layout consensuado §2.3):

    +----------------------------------------+--------+
    | hud_top  [⏮][⏪][Play/Pause][⏩][⏭] título      |
    +----------------------------------------+ panel  |
    |                                        | de     |
    |             world                      | métricas|
    |          (subsurface)                  | y de   |
    +----------------------------------------+ fallos |
    | hud_bottom                             |        |
    |  [████░░░░] 09.9/19.8s  [0.25][0.5][1×][2×][4×]  |
    |  Now: PickUp(d1, pkg_med1) — 60%       |        |
    +----------------------------------------+--------+

Funcionamiento:

  __init__   construye los widgets una vez. Recibe regions, theme,
             HudLayout (con default) y los datos derivados del
             RunResult (HudPlanInfo). Cada widget creado se registra
             en self._owned_widgets.

  relayout   destruye los widgets vigentes (.kill()) y los reconstruye
             en las nuevas regions. Usado por la app tras VIDEORESIZE.

  process_event  recibe un pygame.event.Event. Si es un evento de
             pygame_gui que reconoce, devuelve HudIntent; si no, None.

  sync_from_state  lee AppState y refresca los TEXTOS de los widgets.

  update     delega a UIManager.update(dt).

  draw       dibuja widgets via UIManager.draw_ui, luego pinta los
             overlays propios (marcas rojas con halo amarillo).

Sobre las constantes visuales: TODAS viven en HudLayout (dataclass
frozen en layout.py). Pasar un HudLayout personalizado al constructor
es la forma soportada de cambiar tamaños, gaps o paddings sin tocar
este módulo. Patrón análogo al `Theme` del paquete render.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import resources
from typing import Sequence

import pygame
import pygame_gui
from pygame_gui.core import UIElement
from pygame_gui.elements import (
    UIButton,
    UIImage,
    UILabel,
    UIPanel,
    UIProgressBar,
    UIScrollingContainer,
    UITabContainer,
)

from droneplan_viz.history import WorldSnapshot
from droneplan_viz.render import Theme
from droneplan_viz.render.interpolation import (
    TransitionStatic,
    classify_transition,
)
from droneplan_viz.runtime import CommandFailure
from droneplan_viz_app.app_state import ALLOWED_SPEEDS, AppState
from droneplan_viz_app.controller import HudIntent
from droneplan_viz_app.theme_loader import monogram_font_path
from droneplan_viz_app.inventory import build_inventory
from droneplan_viz_app.layout import HudLayout, Regions


# ---------------------------------------------------------------------------
# Subclase silenciosa de UIProgressBar
# ---------------------------------------------------------------------------


class _SilentProgressBar(UIProgressBar):
    """UIProgressBar sin texto interno.

    Por defecto pygame_gui pinta '0.0/100.0' en el centro de la barra;
    sobrescribir status_text() lo silencia. Documentado como punto de
    extensión por pygame_gui (ver docstring de UIProgressBar.status_text).
    """
    def status_text(self) -> str:  # type: ignore[override]
        return ""


# ---------------------------------------------------------------------------
# HudPlanInfo: datos derivados de RunResult que el HUD necesita
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HudPlanInfo:
    """Datos del plan derivados de RunResult, en la forma que el HUD usa.

    Tener este "plano informativo" desacopla al HUD de RunResult y
    HistoryManager: el HUD solo conoce floats, strings y tuplas
    inmutables. Cualquier sesión futura que cambie la API del runtime
    no requiere modificar `hud.py`, solo el constructor de HudPlanInfo.

    Atributos:
        duration: timeline.duration. Usado para mapear progress.
        snapshot_times: timeline.snapshot_times. Reservado para overlays
            futuros (tick marks por snapshot sobre la barra). No usado
            actualmente.
        failure_starts: start_time de cada CommandFailure, en orden.
            Usado para las marcas rojas sobre la progress bar.
        failure_labels: una línea de texto por fallo, lista para
            mostrar en el panel de fallos.
    """
    duration: float
    snapshot_times: tuple[float, ...]
    failure_starts: tuple[float, ...]
    failure_labels: tuple[str, ...]
    #: Inicio (tiempo VIRTUAL de timeline) de cada acción ejecutada, en orden
    #: cronológico. Usado para saltar al comienzo de la acción al clicarla.
    action_starts: tuple[float, ...] = ()
    #: Una línea por acción con sus estadísticas, para el panel de Acciones.
    action_labels: tuple[str, ...] = ()
    #: Dos líneas por acción (encabezado, estadísticas) para tarjetas de dos
    #: renglones en el panel de Acciones (más legibles que una sola línea).
    action_lines: tuple[tuple[str, str], ...] = ()


def build_hud_plan_info(
    duration: float,
    snapshot_times: tuple[float, ...],
    failures: Sequence[CommandFailure],
    plan=None,
    history=None,
) -> HudPlanInfo:
    """Construye un HudPlanInfo desde los datos brutos del runner.

    Si se proveen `plan` y `history`, se construye además la lista de
    ACCIONES ejecutadas: por cada ScheduledCommand que aparece en el
    historial se toma su tiempo VIRTUAL de inicio (el del primer snapshot
    que produjo) para poder saltar exactamente a su comienzo, y una etiqueta
    con sus estadísticas. Se ordenan cronológicamente.
    """
    starts = tuple(f.scheduled.start_time for f in failures)
    labels = tuple(_format_failure_label(f) for f in failures)

    action_starts: tuple[float, ...] = ()
    action_labels: tuple[str, ...] = ()
    action_lines: tuple[tuple[str, str], ...] = ()
    if plan is not None and history is not None:
        failed_cids = {f.scheduled.command.command_id for f in failures}
        # command_id → índice del PRIMER snapshot que lo produjo (su inicio).
        first_idx: dict[str, int] = {}
        for i in range(len(history)):
            pb = history.at(i).produced_by
            cid = getattr(pb, "command_id", None)
            if cid is not None and cid not in first_idx:
                first_idx[cid] = i
        rows: list[tuple[float, str, str]] = []
        for sc in plan.scheduled:
            cid = sc.command.command_id
            if cid not in first_idx:
                continue  # acción no ejecutada (plan truncado): no se lista.
            idx = first_idx[cid]
            vstart = snapshot_times[idx] if idx < len(snapshot_times) else 0.0
            l1, l2 = _format_action_label(sc, vstart, failed=cid in failed_cids)
            rows.append((vstart, l1, l2))
        rows.sort(key=lambda r: r[0])
        action_starts = tuple(r[0] for r in rows)
        action_lines = tuple((r[1], r[2]) for r in rows)
        # Etiqueta plana (una línea) por compatibilidad y para conteos.
        action_labels = tuple(f"{r[1]}  {r[2]}" for r in rows)

    return HudPlanInfo(
        duration=duration,
        snapshot_times=snapshot_times,
        failure_starts=starts,
        failure_labels=labels,
        action_starts=action_starts,
        action_labels=action_labels,
        action_lines=action_lines,
    )


#: Campo "objetivo" más representativo por tipo de Command, para la etiqueta.
_ACTION_TARGET_FIELDS = (
    "package_id", "transporter_id", "destination_id", "person_id", "arm_id",
)


def _format_action_label(sc, vstart: float, *, failed: bool) -> tuple[str, str]:
    """Dos renglones legibles para una tarjeta de Acción.

    Línea 1 (encabezado): '<drone> · <Tipo>(<objetivos>)'.
    Línea 2 (stats):       't=<vstart>s · Δ<dur>s [· FALLÓ]'.
    """
    cmd = sc.command
    cls = cmd.__class__.__name__
    drone = getattr(cmd, "drone_id", "?")
    dur = getattr(cmd, "duration", 0.0)
    targets = [
        str(getattr(cmd, f)) for f in _ACTION_TARGET_FIELDS
        if getattr(cmd, f, None) is not None
    ]
    tgt = ", ".join(targets)
    if len(tgt) > 34:
        tgt = tgt[:31] + "..."
    line1 = f"{drone} · {cls}({tgt})"
    line2 = f"t={vstart:.1f}s · Δ{dur:.1f}s"
    if failed:
        line2 += " · FALLÓ"
    return line1, line2


def _format_failure_label(f: CommandFailure) -> str:
    """Una línea legible para el panel de fallos.

    Formato: 't=<X>s · <kind> · <ClassName>(<arg1>,...)'

    No trunca: la tarjeta recorta visualmente con '…' lo que no quepa y el
    tooltip muestra el texto completo al pasar el ratón (igual que las
    tarjetas de acción).
    """
    cmd = f.scheduled.command
    cls = cmd.__class__.__name__
    args: list[str] = []
    for fld in fields(cmd):
        if fld.name in {"command_id", "duration"}:
            continue
        args.append(str(getattr(cmd, fld.name)))
    args_str = ",".join(args)
    return f"t={f.scheduled.start_time:.1f}s · {f.kind} · {cls}({args_str})"


# ---------------------------------------------------------------------------
# Hud principal
# ---------------------------------------------------------------------------


class Hud:
    """Encapsula UIManager + widgets + dibujo.

    El constructor crea los widgets una vez en las regions dadas y los
    registra en self._owned_widgets para poder limpiarlos en relayout.

    Args:
        manager: UIManager compartido con la app.
        regions: Regions (rects de hud_top/bottom/right + world).
        plan_info: HudPlanInfo derivado del RunResult.
        theme: Theme del render (para acceder al fallback duration y
            posibles colores compartidos en futuro).
        hud_layout: HudLayout con las constantes visuales. Default:
            HudLayout() con valores consensuados. Pasar uno propio
            permite variantes sin tocar este módulo.
    """

    def __init__(
        self,
        manager: pygame_gui.UIManager,
        regions: Regions,
        plan_info: HudPlanInfo,
        theme: Theme,
        *,
        hud_layout: HudLayout | None = None,
    ) -> None:
        self._manager = manager
        self._regions = regions
        self._plan = plan_info
        self._theme = theme
        self._hud_layout = hud_layout if hud_layout is not None else HudLayout()

        # Lista de widgets de los que somos dueños. Acumula durante
        # _build_* y se vacía en relayout via kill().
        self._owned_widgets: list[UIElement] = []

        # Atributos que apuntan a widgets concretos. Asignados en
        # _build_*. Declarados aquí (None) para que el type checker no
        # se queje y para hacer explícita la lista.
        self._btn_home: UIButton | None = None
        self._btn_step_back: UIButton | None = None
        self._btn_play_pause: UIButton | None = None
        self._btn_step_forward: UIButton | None = None
        self._btn_end: UIButton | None = None
        self._lbl_title: UILabel | None = None
        self._progress_bar: _SilentProgressBar | None = None
        self._lbl_time: UILabel | None = None
        self._btn_speeds: tuple[UIButton, ...] = ()
        self._lbl_now: UILabel | None = None
        # Tab container que ocupa toda la región derecha. Contiene dos
        # tabs: "Inventario" (dinámico, navegable, work-in-progress; ahora
        # solo placeholder) y "Métricas+Fallos" (lo que antes eran dos
        # subpaneles independientes, refundido aquí).
        self._tab_container: UITabContainer | None = None
        self._tab_idx_inventario: int = -1
        self._tab_idx_metrics: int = -1
        # Widgets dentro de la tab Métricas+Fallos.
        self._panel_metrics: UIPanel | None = None
        self._lbl_metrics_title: UILabel | None = None
        self._lbl_total_cost: UILabel | None = None
        self._lbl_total_time: UILabel | None = None
        self._lbl_action_count: UILabel | None = None
        self._lbl_failed_count: UILabel | None = None
        self._panel_failures: UIPanel | None = None
        self._lbl_failures_title: UILabel | None = None
        self._fail_scroll: UIScrollingContainer | None = None
        self._fail_cards: list = []  # [(UIImage, fail_idx, truncated, full)]
        # Widgets de la sección de Acciones (entre Métricas y Fallos).
        # Lista scrollable de tarjetas de dos líneas (mismo patrón que el
        # inventario): un UIScrollingContainer + un UIImage por acción, con
        # hit-test para el click.
        self._panel_actions: UIPanel | None = None
        self._lbl_actions_title: UILabel | None = None
        self._act_scroll: UIScrollingContainer | None = None
        self._act_cards: list = []  # [(UIImage, action_index)] para hit-test.
        # Placeholder de la tab Inventario (se sustituirá por el inventario
        # dinámico en una fase posterior). Lo guardamos para limpiarlo en
        # relayout().
        # Inventario rediseñado: UIScrollingContainer que aloja
        # cabeceras de sección + cajas (UIPanels) por entidad.
        # _inv_scroll: el scrolling container (creado en _build_right_panel).
        # _inventory_items: tupla de InventoryItem que respalda el contenido
        #   actualmente renderizado (para resolver hit-test de click a entity_id).
        # _inv_headers: dict {category: (header_label, chevron_button)} para
        #   detectar click sobre cabecera y poder rebuild.
        # _inv_boxes: lista de (UIPanel, entity_id) en orden para hit-test.
        self._inv_scroll: UIScrollingContainer | None = None
        self._inventory_items: tuple = ()  # tupla de InventoryItem
        self._inv_headers: dict = {}
        self._inv_boxes: list = []
        # Estado para la actualización INCREMENTAL del inventario. La
        # estructura del inventario (qué entidades, en qué orden y categoría)
        # es invariante durante la reproducción: solo cambian los textos al
        # moverse/cambiar de estado las entidades. Por eso no reconstruimos
        # los widgets cada frame ni en cada transición:
        #   _inv_last_snap / _inv_last_collapsed: gatean el trabajo O(N) — solo
        #     se rehace cuando cambia el snapshot activo o el plegado (no por
        #     frame intra-segmento, donde el snapshot es el mismo objeto).
        #   _inv_structure_key: identifica el conjunto/orden/plegado/ancho. Si
        #     no cambia, actualizamos in situ solo las cajas cuyo contenido
        #     cambió (set_image), sin destruir/recrear widgets.
        #   _inv_card_by_id / _inv_content_by_id: UIImage y firma de contenido
        #     por entity_id, para el diff incremental.
        self._inv_last_snap = None
        self._inv_last_collapsed: set = set()
        self._inv_structure_key: tuple | None = None
        self._inv_card_by_id: dict = {}
        self._inv_content_by_id: dict = {}

        self._build_all()

    # -----------------------------------------------------------------
    # Construcción y reconstrucción
    # -----------------------------------------------------------------
    def _build_all(self) -> None:
        """Construye los tres panels usando self._regions y
        self._hud_layout. Llamado en __init__ y desde relayout()."""
        self._build_top_panel(self._regions.hud_top)
        self._build_bottom_panel(self._regions.hud_bottom)
        self._build_right_panel(self._regions.hud_right, self._plan.failure_labels)

    def _register(self, widget: UIElement) -> UIElement:
        """Registra un widget como propiedad del Hud. Devuelve el mismo
        widget por conveniencia (cadena fluida).

        Usado dentro de _build_* para no escribir 2 líneas por widget.
        """
        self._owned_widgets.append(widget)
        return widget

    def relayout(self, new_regions: Regions) -> None:
        """Destruye los widgets actuales y los reconstruye en las
        nuevas regions.

        Invocado por la app tras pygame.VIDEORESIZE. Mantener la
        firma simple (solo `new_regions`) significa que un cambio de
        HudLayout o de plan_info NO desencadena relayout; eso es
        intencional (esos cambios no ocurren en runtime ahora).

        Estado interno que se pierde y debe restaurarse:
          - Scroll de las tarjetas de fallo (poco crítico).
          - Selección actual de fallo: la app, en su siguiente
            sync_from_state, vuelve a alinear el estado visual con
            state.selected_failure si fuera necesario.
            En la versión actual, el HUD lee selected_failure solo
            para el overlay rojo+halo, no para la selección visual
            interna del widget; la selección no se restaura
            automáticamente. Es aceptable: el usuario que redimensiona
            mientras tenía un fallo seleccionado pierde solo el
            highlight de pygame_gui sobre la fila, no la lógica
            (la marca roja + halo siguen pintándose).
        """
        # Purga los widgets actuales. kill() los desregistra del manager.
        for w in self._owned_widgets:
            try:
                w.kill()
            except Exception:
                # Si un widget ya estaba muerto (caso raro), continuar.
                pass
        self._owned_widgets.clear()

        # Reanular referencias para no dejar punteros a widgets muertos
        # (defensa contra usos accidentales entre relayout y el siguiente
        # sync_from_state).
        self._btn_home = None
        self._btn_step_back = None
        self._btn_play_pause = None
        self._btn_step_forward = None
        self._btn_end = None
        self._lbl_title = None
        self._progress_bar = None
        self._lbl_time = None
        self._btn_speeds = ()
        self._lbl_now = None
        self._tab_container = None
        self._tab_idx_inventario = -1
        self._tab_idx_metrics = -1
        self._panel_metrics = None
        self._lbl_metrics_title = None
        self._lbl_total_cost = None
        self._lbl_total_time = None
        self._lbl_action_count = None
        self._lbl_failed_count = None
        self._panel_failures = None
        self._lbl_failures_title = None
        self._fail_scroll = None
        self._fail_cards = []
        self._panel_actions = None
        self._lbl_actions_title = None
        self._act_scroll = None
        self._act_cards = []
        self._inv_scroll = None
        self._inventory_items = ()
        self._inv_headers = {}
        self._inv_boxes = []
        # Forzar reconstrucción completa en la próxima sync tras el relayout:
        # los widgets se han destruido y las regions/anchos cambian.
        self._inv_last_snap = None
        self._inv_last_collapsed = set()
        self._inv_structure_key = None
        self._inv_card_by_id = {}
        self._inv_content_by_id = {}

        # Invalidar la caché del logo de marca: si la barra superior
        # cambia de alto al redimensionar, debe reescalarse. El próximo
        # _draw_brand_logo lo recargará con las nuevas dimensiones.
        if hasattr(self, "_brand_logo_scaled"):
            del self._brand_logo_scaled

        # Reconstruir con las nuevas regions.
        self._regions = new_regions
        self._build_all()

    # -----------------------------------------------------------------
    # Construcción de paneles
    # -----------------------------------------------------------------
    def _build_top_panel(self, rect: pygame.Rect) -> None:
        """5 botones de transporte + label de título.

        Los botones se centran verticalmente en el panel via pad_y
        derivado de (rect.height - btn_h) / 2. El título ocupa el
        espacio sobrante a la derecha.

        Si los sprites de los botones están presentes en
        assets/hud_icons/, el texto de cada botón se deja vacío (el
        sprite contiene su propio pictograma). Si no, el texto sirve
        como fallback legible.
        """
        L = self._hud_layout
        pad_y = (rect.height - L.top_btn_h) // 2
        x = rect.x + L.top_pad_x
        y = rect.y + pad_y

        # Detectar si los 12 PNGs de botones de transporte están en su
        # sitio. Si lo están, los sprites se aplicarán vía theme y el
        # texto sobra (estaría visible encima del pictograma del sprite).
        sprites_loaded = self._transport_sprites_available()
        def t(legacy_text: str) -> str:
            return "" if sprites_loaded else legacy_text

        self._btn_home = self._register(UIButton(
            relative_rect=pygame.Rect(x, y, L.top_btn_w, L.top_btn_h),
            text=t("|<<"),
            manager=self._manager,
            tool_tip_text="Inicio (Home)",
            object_id="@transport_home",
        ))
        x += L.top_btn_w + L.top_btn_gap
        self._btn_step_back = self._register(UIButton(
            relative_rect=pygame.Rect(x, y, L.top_btn_w, L.top_btn_h),
            text=t("<-"),
            manager=self._manager,
            tool_tip_text="Snapshot anterior (Flecha izquierda)",
            object_id="@transport_step_back",
        ))
        x += L.top_btn_w + L.top_btn_gap
        # Play/Pause: el ancho extra (top_play_extra_w) era para el
        # texto "Play"/"Pause"; al pasar a sprite cuadrado coherente con
        # los demás botones se puso top_play_extra_w=0 por consistencia
        # visual. El sprite mostrado depende de state.paused y se
        # gestiona en sync_from_state vía change_object_id.
        play_w = L.top_btn_w + L.top_play_extra_w
        self._btn_play_pause = self._register(UIButton(
            relative_rect=pygame.Rect(x, y, play_w, L.top_btn_h),
            text=t("Play"),
            manager=self._manager,
            tool_tip_text="Play / Pause (barra espaciadora)",
            object_id="@transport_play",
        ))
        # Atributo auxiliar para que sync_from_state detecte cambios de
        # object_id sin tener que comparar el estado interno de pygame_gui
        # (que no expone esta info públicamente).
        self._btn_play_pause._inv_current_oid = "@transport_play"  # type: ignore[attr-defined]
        x += play_w + L.top_btn_gap
        self._btn_step_forward = self._register(UIButton(
            relative_rect=pygame.Rect(x, y, L.top_btn_w, L.top_btn_h),
            text=t("->"),
            manager=self._manager,
            tool_tip_text="Snapshot siguiente (Flecha derecha)",
            object_id="@transport_step_forward",
        ))
        x += L.top_btn_w + L.top_btn_gap
        self._btn_end = self._register(UIButton(
            relative_rect=pygame.Rect(x, y, L.top_btn_w, L.top_btn_h),
            text=t(">>|"),
            manager=self._manager,
            tool_tip_text="Final (End)",
            object_id="@transport_end",
        ))
        x += L.top_btn_w + L.top_btn_gap

        # Título a la derecha. Solo se crea si queda hueco razonable
        # (paneles muy estrechos se quedan sin título, no se solapa). Se
        # descuenta el ancho reservado por el logo del HUD (si existe),
        # de modo que el título quede centrado entre los botones y el
        # logo, y no entre los botones y el borde de la barra.
        title_x = x + L.top_title_offset_x
        logo_w = self._hud_logo_reserved_width()
        title_w = rect.x + rect.width - title_x - L.top_pad_x - logo_w
        if title_w > 80:
            self._lbl_title = self._register(UILabel(
                relative_rect=pygame.Rect(title_x, y, title_w, L.top_btn_h),
                text="droneplan_viz · visualizador interactivo",
                manager=self._manager,
            ))

    def _build_bottom_panel(self, rect: pygame.Rect) -> None:
        """Progress bar + label tiempo + 5 botones velocidad + label 'Now:'.

        Layout en dos filas:
          fila 1: [progress] [time] [0.25×][0.5×][1×][2×][4×]
          fila 2: Now: ...

        El label de tiempo y los botones de velocidad están alineados
        al borde DERECHO del panel; la progress bar absorbe el ancho
        restante. Esto se calcula reverso: primero ancho de
        time+speeds, luego progress_w.
        """
        L = self._hud_layout
        # Posiciones absolutas de partida.
        x_left = rect.x + L.bottom_pad_x
        y_row1 = rect.y + L.bottom_pad_y_top

        # Ancho del bloque time + speeds, a la derecha.
        speeds_total_w = (
            len(ALLOWED_SPEEDS) * L.bottom_speed_btn_w
            + (len(ALLOWED_SPEEDS) - 1) * L.bottom_speed_btn_gap
        )
        right_block_w = (
            L.bottom_time_label_w
            + L.bottom_label_widget_gap
            + speeds_total_w
        )

        # Posición x donde empieza el bloque derecho (alineado a la
        # derecha del panel).
        x_right_block = rect.x + rect.width - L.bottom_pad_x - right_block_w
        # Progress bar va de x_left a x_right_block - gap.
        progress_x = x_left
        progress_w = max(
            x_right_block - L.bottom_label_widget_gap - progress_x, 100
        )

        self._progress_bar = self._register(_SilentProgressBar(
            relative_rect=pygame.Rect(progress_x, y_row1, progress_w, L.bottom_row1_h),
            manager=self._manager,
        ))
        self._progress_bar.set_current_progress(0.0)

        # Label de tiempo, alineado verticalmente con la barra.
        time_x = x_right_block
        self._lbl_time = self._register(UILabel(
            relative_rect=pygame.Rect(time_x, y_row1, L.bottom_time_label_w, L.bottom_row1_h),
            text=self._format_time_label(0.0, self._plan.duration),
            manager=self._manager,
        ))

        # Botones de velocidad, secuencialmente a la derecha del label.
        speed_x = time_x + L.bottom_time_label_w + L.bottom_label_widget_gap
        speed_buttons: list[UIButton] = []
        for speed in ALLOWED_SPEEDS:
            btn = self._register(UIButton(
                relative_rect=pygame.Rect(
                    speed_x, y_row1, L.bottom_speed_btn_w, L.bottom_row1_h
                ),
                text=self._format_speed_label(speed, active=False),
                manager=self._manager,
                tool_tip_text=f"Velocidad {speed}×",
                object_id="@speed_button",
            ))
            speed_buttons.append(btn)
            speed_x += L.bottom_speed_btn_w + L.bottom_speed_btn_gap
        self._btn_speeds = tuple(speed_buttons)

        # Fila 2: label 'Now:'.
        y_row2 = y_row1 + L.bottom_row1_h + L.bottom_row_gap
        now_w = rect.width - 2 * L.bottom_pad_x
        self._lbl_now = self._register(UILabel(
            relative_rect=pygame.Rect(x_left, y_row2, now_w, L.bottom_row2_h),
            text="Now: —",
            manager=self._manager,
        ))

    def _build_right_panel(
        self, rect: pygame.Rect, failure_labels: tuple[str, ...]
    ) -> None:
        """Panel lateral con DOS tabs vía UITabContainer.

        Tab "Inventario": placeholder por ahora (se rellenará con la lista
        dinámica de entidades en una fase posterior).

        Tab "Métricas+Fallos": refunde lo que antes eran dos subpaneles
        independientes (métricas arriba, lista de fallos abajo) dentro de
        la misma tab. La estructura interna se mantiene tal cual: los
        widgets son los mismos, solo cambia el contenedor padre (ahora el
        UIPanel que devuelve UITabContainer.get_tab_container(idx)).

        La tab activa al arrancar es "Inventario" (decisión: la novedad
        del rediseño debe ser visible al abrir; Métricas+Fallos es
        información agregada a un click de distancia).
        """
        L = self._hud_layout

        # 1. UITabContainer ocupa toda la región derecha.
        self._tab_container = self._register(UITabContainer(
            relative_rect=pygame.Rect(
                rect.x + L.right_pad,
                rect.y + L.right_pad,
                rect.width - 2 * L.right_pad,
                rect.height - 2 * L.right_pad,
            ),
            manager=self._manager,
        ))
        self._tab_idx_inventario = self._tab_container.add_tab(" Inventario ")
        self._tab_idx_metrics = self._tab_container.add_tab(" Métricas+Fallos ")

        # 2. Contenido de la tab "Inventario": UIScrollingContainer que
        # aloja las cabeceras de sección y las cajas por entidad.
        # _build_right_panel solo crea el contenedor; el contenido
        # (cabeceras + cajas) lo monta _rebuild_inventory_ui en cada
        # sync_from_state cuando los items o el plegado cambien.
        inv_panel = self._tab_container.get_tab_container(self._tab_idx_inventario)
        inv_w = inv_panel.relative_rect.width
        inv_h = inv_panel.relative_rect.height
        scroll_rect = pygame.Rect(
            L.right_tab_inner_pad,
            L.right_tab_inner_pad,
            inv_w - 2 * L.right_tab_inner_pad,
            inv_h - 2 * L.right_tab_inner_pad,
        )
        self._inv_scroll = self._register(UIScrollingContainer(
            relative_rect=scroll_rect,
            manager=self._manager,
            container=inv_panel,
            allow_scroll_x=False,
            allow_scroll_y=True,
        ))

        # 3. Contenido de la tab "Métricas+Fallos".
        met_panel = self._tab_container.get_tab_container(self._tab_idx_metrics)
        # Activamos temporalmente esta tab mientras construimos su contenido:
        # pygame_gui llama hide() al añadir un elemento a un contenedor oculto,
        # y un UIScrollingContainer recién creado falla en hide() (su
        # vert_scroll_bar aún no existe). El switch final (a Inventario) deja
        # el estado correcto al terminar el build.
        self._tab_container.switch_current_container(self._tab_idx_metrics)
        met_w = met_panel.relative_rect.width
        met_h = met_panel.relative_rect.height
        # Subpanel de métricas (reutilizamos el patrón anterior pero
        # ahora anidado dentro del container de la tab).
        metrics_rect = pygame.Rect(
            L.right_tab_inner_pad,
            L.right_tab_inner_pad,
            met_w - 2 * L.right_tab_inner_pad,
            L.right_metrics_h - 54,
        )
        self._panel_metrics = self._register(UIPanel(
            relative_rect=metrics_rect,
            manager=self._manager,
            container=met_panel,
        ))
        inner_w = metrics_rect.width - 2 * L.right_label_indent
        self._lbl_metrics_title = self._register(UILabel(
            relative_rect=pygame.Rect(
                L.right_label_indent, L.right_title_top_offset, inner_w, L.right_label_h
            ),
            text="Métricas",
            manager=self._manager,
            container=self._panel_metrics,
        ))
        metric_specs = [
            ("_lbl_total_cost", "total_cost: —"),
            ("_lbl_total_time", "total_time: —"),
            ("_lbl_action_count", "actions: —"),
            ("_lbl_failed_count", "failures: —"),
        ]
        for i, (attr, default_text) in enumerate(metric_specs):
            y = L.right_first_label_top_offset + i * L.right_label_spacing
            lbl = self._register(UILabel(
                relative_rect=pygame.Rect(
                    L.right_label_indent, y, inner_w, L.right_label_h
                ),
                text=default_text,
                manager=self._manager,
                container=self._panel_metrics,
            ))
            setattr(self, attr, lbl)

        # Espacio bajo métricas, repartido entre ACCIONES y FALLOS (con un
        # gap entre cada subpanel). Acciones recibe algo más de alto porque
        # suele listar más elementos; ambas listas tienen scroll propio.
        sect_top = L.right_tab_inner_pad + L.right_metrics_h + L.right_subpanel_gap
        remaining = (met_h - sect_top - L.right_tab_inner_pad
                     - L.right_subpanel_gap)
        actions_h = max(L.right_label_h * 3, int(remaining * 0.55))
        failures_h = remaining - actions_h

        # --- Subpanel de ACCIONES ---
        action_lines = self._plan.action_lines
        actions_rect = pygame.Rect(
            L.right_tab_inner_pad, sect_top - 54,
            met_w - 2 * L.right_tab_inner_pad, actions_h + 54,
        )
        self._panel_actions = self._register(UIPanel(
            relative_rect=actions_rect,
            manager=self._manager,
            container=met_panel,
        ))
        inner_w_a = actions_rect.width - 2 * L.right_label_indent
        self._lbl_actions_title = self._register(UILabel(
            relative_rect=pygame.Rect(
                L.right_label_indent, L.right_title_top_offset,
                inner_w_a, L.right_label_h,
            ),
            text=f"Acciones ({len(action_lines)})",
            manager=self._manager,
            container=self._panel_actions,
        ))
        if action_lines:
            # Scroll container AJUSTADO a los bordes del panel (solo un pad
            # mínimo), bajo el título: aprovecha el ancho/alto disponibles y
            # queda centrado. Las tarjetas (dos líneas) se montan dentro.
            pad = 3
            sc_y = L.right_first_label_top_offset
            sc_rect = pygame.Rect(
                pad, sc_y,
                actions_rect.width - 2 * pad,
                actions_rect.height - sc_y - pad,
            )
            self._act_scroll = self._register(UIScrollingContainer(
                relative_rect=sc_rect,
                manager=self._manager,
                container=self._panel_actions,
                allow_scroll_x=False,
                allow_scroll_y=True,
            ))
            self._build_action_cards(action_lines)
        else:
            self._register(UILabel(
                relative_rect=pygame.Rect(
                    L.right_label_indent, L.right_first_label_top_offset,
                    inner_w_a, L.right_label_h,
                ),
                text="Sin acciones en este plan.",
                manager=self._manager,
                container=self._panel_actions,
            ))

        # Subpanel de fallos (debajo del de acciones dentro de la misma tab).
        failures_top = sect_top + actions_h + L.right_subpanel_gap
        failures_rect = pygame.Rect(
            L.right_tab_inner_pad,
            failures_top,
            met_w - 2 * L.right_tab_inner_pad,
            failures_h,
        )
        self._panel_failures = self._register(UIPanel(
            relative_rect=failures_rect,
            manager=self._manager,
            container=met_panel,
        ))
        inner_w_f = failures_rect.width - 2 * L.right_label_indent
        self._lbl_failures_title = self._register(UILabel(
            relative_rect=pygame.Rect(
                L.right_label_indent, L.right_title_top_offset,
                inner_w_f, L.right_label_h,
            ),
            text=f"Fallos ({len(failure_labels)})",
            manager=self._manager,
            container=self._panel_failures,
        ))
        if failure_labels:
            list_y = L.right_first_label_top_offset
            list_h = failures_rect.height - list_y - L.right_label_indent
            self._fail_scroll = self._register(UIScrollingContainer(
                relative_rect=pygame.Rect(
                    L.right_label_indent, list_y, inner_w_f, list_h
                ),
                manager=self._manager,
                container=self._panel_failures,
                allow_scroll_x=False,
                allow_scroll_y=True,
            ))
            self._build_failure_cards(failure_labels)
        else:
            self._register(UILabel(
                relative_rect=pygame.Rect(
                    L.right_label_indent, L.right_first_label_top_offset,
                    inner_w_f, L.right_label_h,
                ),
                text="Sin fallos en este plan.",
                manager=self._manager,
                container=self._panel_failures,
            ))

        # 4. Tab activa al arrancar: Inventario.
        self._tab_container.switch_current_container(self._tab_idx_inventario)

    # -----------------------------------------------------------------
    # Control del panel lateral (pestañas)
    # -----------------------------------------------------------------
    def show_metrics_failures_tab(self) -> None:
        """Activa la pestaña Métricas+Fallos del panel lateral derecho.

        Útil para revelar las métricas y la lista de fallos sin que el
        usuario tenga que pulsar la pestaña (p.ej. al generar capturas de
        un escenario con fallos, o como reacción a seleccionar un fallo).
        """
        if self._tab_container is not None and self._tab_idx_metrics >= 0:
            self._tab_container.switch_current_container(self._tab_idx_metrics)

    def show_inventory_tab(self) -> None:
        """Activa la pestaña Inventario del panel lateral derecho."""
        if self._tab_container is not None and self._tab_idx_inventario >= 0:
            self._tab_container.switch_current_container(self._tab_idx_inventario)

    # -----------------------------------------------------------------
    # Inventario rediseñado: rebuild + helpers
    # -----------------------------------------------------------------
    def _rebuild_inventory_ui(self, items: tuple, collapsed_categories) -> None:
        """Reconstruye todo el contenido visible del UIScrollingContainer
        del inventario a partir de los items y el set de categorías
        plegadas.

        Por simplicidad y robustez, el patrón es destructivo: matamos
        todos los widgets internos y los recreamos desde cero. La
        alternativa (diffing parcial) es mucho más compleja y no se
        justifica salvo que sea cuello de botella (no lo es: rebuild
        solo ocurre cuando los datos del world cambian o el plegado
        muta, no por frame).

        Estructura interna del scroll:
            ┌─────────────────────────────────────┐
            │ ▼ DRONES                             │  ← cabecera 24px alta
            │ ┌─────────────────────────────────┐ │
            │ │ [icon] d1                        │ │  ← caja 64px alta
            │ │        IDLE en deposito          │ │
            │ │        brazos 2                  │ │
            │ └─────────────────────────────────┘ │
            │ ┌─────────────────────────────────┐ │
            │ │ [icon] d2 ...                    │ │
            │ └─────────────────────────────────┘ │
            │                                      │  ← gap entre secciones
            │ ▶ LOCATIONS                          │  ← plegada: sin cajas
            │ ▼ PERSONAS                           │
            │ ...                                  │
            └─────────────────────────────────────┘
        """
        L = self._hud_layout
        if self._inv_scroll is None:
            return

        # Destruir widgets previos del scroll y SACARLOS de
        # _owned_widgets para que relayout no intente killarlos otra vez
        # (un widget killed dos veces lanza excepción) y para evitar el
        # leak de referencias entre rebuilds.
        widgets_a_purgar = set()
        for header_widget, chevron_widget in self._inv_headers.values():
            widgets_a_purgar.add(header_widget)
            widgets_a_purgar.add(chevron_widget)
        for box_img, _ in self._inv_boxes:
            widgets_a_purgar.add(box_img)

        for w in widgets_a_purgar:
            try:
                w.kill()
            except Exception:
                pass
        # Purga de _owned_widgets.
        if widgets_a_purgar:
            self._owned_widgets = [
                w for w in self._owned_widgets if w not in widgets_a_purgar
            ]
        self._inv_headers = {}
        self._inv_boxes = []

        # Acceso al contenedor interno scrollable.
        scroll_inner = self._inv_scroll.get_container()
        # Ancho disponible dentro del scroll (descontando scrollbar + márgenes).
        avail_w = self._inv_avail_width()

        # Recorremos los items y vamos colocando widgets verticalmente.
        y = 0
        current_category_collapsed = False
        for item in items:
            if item.kind == "header":
                current_category_collapsed = item.category in collapsed_categories
                # Cabecera CLICABLE entera: UIButton con texto alineado
                # a la izquierda + UIImage del chevron encima a la derecha.
                # El UIImage no captura clicks (es solo render), así que
                # un click dentro del rect del chevron también dispara
                # el UIButton subyacente. Resultado: toda la cabecera
                # (texto y flecha) es un solo objetivo de click, lo que
                # cumple con la decisión G del rediseño.
                header_rect = pygame.Rect(
                    L.inv_box_side_margin, y,
                    avail_w, L.inv_header_h,
                )
                header_btn = self._register(UIButton(
                    relative_rect=header_rect,
                    text=item.text,
                    manager=self._manager,
                    container=scroll_inner,
                    object_id="@inventory_header",
                ))
                # Flecha a la derecha: sprite del chevron según estado.
                chevron_size = L.inv_chevron_size
                # Margen de 4 px desde el borde derecho de la cabecera.
                chevron_x = (
                    L.inv_box_side_margin + avail_w - chevron_size - 4
                )
                chevron_y = y + (L.inv_header_h - chevron_size) // 2
                chevron_rect = pygame.Rect(
                    chevron_x, chevron_y, chevron_size, chevron_size,
                )
                chevron_surf = self._make_chevron_surface(
                    collapsed=current_category_collapsed
                )
                chevron_img = self._register(UIImage(
                    relative_rect=chevron_rect,
                    image_surface=chevron_surf,
                    manager=self._manager,
                    container=scroll_inner,
                ))
                self._inv_headers[item.category] = (header_btn, chevron_img)
                y += L.inv_header_h + 2
            else:
                if current_category_collapsed:
                    continue
                # Caja para la entidad: pre-rendereada como Surface y
                # mostrada con un único UIImage. Esto reduce drásticamente
                # el número de widgets: en lugar de UIPanel + UIImage +
                # 3 UILabels por caja (5 widgets), un único UIImage.
                # Para una tab con 12 entidades, pasamos de ~60 widgets
                # a 12. La interactividad (hover, click) se gestionará
                # vía hit-test en el paso 4.
                box_surface = self._render_inventory_box_surface(
                    item, avail_w,
                )
                box_rect = pygame.Rect(
                    L.inv_box_side_margin, y,
                    avail_w, L.inv_box_h,
                )
                box_img = self._register(UIImage(
                    relative_rect=box_rect,
                    image_surface=box_surface,
                    manager=self._manager,
                    container=scroll_inner,
                ))
                self._inv_boxes.append((box_img, item.entity_id))
                y += L.inv_box_h + L.inv_box_gap
            # Tras una sección plegada o desplegada, dejamos algo de aire
            # antes de la siguiente cabecera. La forma simple: añadir un
            # gap extra al detectar el cambio a otra categoría sería más
            # robusto, pero en la práctica el gap entre la última caja y
            # la cabecera siguiente queda razonable con inv_box_gap.

        # Ajustar dimensiones del área scrollable al alto total que ocupamos.
        total_h = max(y, self._inv_scroll.relative_rect.height)
        self._inv_scroll.set_scrollable_area_dimensions(
            (self._inv_scroll.relative_rect.width - self._SCROLLBAR_RESERVE,
             total_h)
        )

    #: Espacio que pygame_gui reserva para la scrollbar vertical (px).
    _SCROLLBAR_RESERVE = 24

    def _inv_avail_width(self) -> int:
        """Ancho disponible para las cajas dentro del scroll del inventario.

        Descuenta la reserva de scrollbar y los márgenes laterales, igual que
        la reconstrucción, para que la ruta incremental rasterice cajas del
        mismo ancho.
        """
        L = self._hud_layout
        scroll_w = self._inv_scroll.relative_rect.width
        return scroll_w - self._SCROLLBAR_RESERVE - 2 * L.inv_box_side_margin

    def _render_inventory_box_surface(self, item, width: int) -> pygame.Surface:
        """Renderiza una caja del inventario completa (fondo + borde +
        icono + nombre + 2 líneas de stats) como Surface única, lista
        para mostrar con un UIImage.

        Centralizar el render aquí (en lugar de componer múltiples
        widgets pygame_gui) reduce el número total de widgets a ~22 en
        lugar de ~75, lo que evita problemas de memoria al ejecutar
        suites de tests largas con muchas reconstrucciones del HUD.

        Args:
            item: InventoryItem entity (no header) con display_name y stats.
            width: ancho disponible en píxeles para la caja.

        Returns:
            Surface de tamaño (width, inv_box_h) con la caja renderizada.
        """
        L = self._hud_layout
        h = L.inv_box_h
        surface = pygame.Surface((width, h), pygame.SRCALPHA)

        # Fondo + borde.
        bg_color = (37, 45, 56)         # #252D38
        border_color = (60, 71, 84)     # #3C4754
        pygame.draw.rect(surface, bg_color, (0, 0, width, h), border_radius=2)
        pygame.draw.rect(surface, border_color, (0, 0, width, h), width=1, border_radius=2)

        # Icono a la izquierda, centrado vertical.
        icon = self._make_inventory_icon_surface(item.category)
        icon_y = (h - L.inv_icon_size) // 2
        surface.blit(icon, (L.inv_box_inner_pad_x, icon_y))

        # Texto: nombre arriba, 2 líneas de stats debajo.
        text_x = L.inv_box_inner_pad_x + L.inv_icon_size + L.inv_box_text_left_pad
        text_color = (230, 235, 240)
        stat_color = (180, 190, 200)

        # Las fuentes se cargan desde el TTF empaquetado. Las cacheamos
        # como atributos del Hud para no abrir el TTF en cada rebuild.
        # Tamaños: nombre 20px, stats 16px. La cabecera de sección
        # usa 20px (configurada vía theme.json en @inventory_header).
        if not hasattr(self, "_inv_font_name"):
            font_path = (
                resources.files("droneplan_viz_app")
                / "assets" / "fonts" / "monogram.ttf"
            )
            from pathlib import Path
            ttf = str(Path(str(font_path)))
            self._inv_font_name = pygame.font.Font(ttf, 20)
            self._inv_font_stat = pygame.font.Font(ttf, 16)

        # Bloque de texto centrado verticalmente: el bloque mide 30 px
        # (nombre 15 + gap 2 + stats 13) y la caja mide 52 px, así que
        # el bloque empieza a y=(52-30)/2 = 11 px desde arriba.
        # El espaciado vertical del bloque (11 px) queda muy próximo al
        # espaciado vertical del icono ((52-24)/2 = 14 px); la diferencia
        # de 3 px es imperceptible visualmente.
        text_block_h = 30
        text_block_y = (L.inv_box_h - text_block_h) // 2

        # Nombre (línea 1).
        name_surf = self._inv_font_name.render(
            item.display_name or "", True, text_color,
        )
        surface.blit(name_surf, (text_x, text_block_y))

        # Stats (línea 2): 2 px de gap respecto al nombre (apenas
        # perceptible pero evita que los glifos parezcan amontonados).
        if item.stats:
            stats_text = " · ".join(item.stats)
            stat_surf = self._inv_font_stat.render(stats_text, True, stat_color)
            surface.blit(stat_surf, (text_x, text_block_y + 17))

        return surface

    # -----------------------------------------------------------------
    # Tarjetas del panel de Acciones (una línea, columnas alineadas)
    # -----------------------------------------------------------------
    def _build_action_cards(self, action_lines: tuple) -> None:
        """Monta una tarjeta (UIImage de UNA línea con columnas) por acción
        dentro del UIScrollingContainer, y ajusta el área scrollable a su
        alto total.

        Cada tarjeta es una Surface única (como el inventario). Llena el
        ancho del contenedor; las columnas 't=...' y 'Δ...' arrancan en un
        x fijo (mismas en todas las filas → alineación vertical). Guarda su
        índice de acción para el hit-test del click.
        """
        if self._act_scroll is None:
            return
        self._act_cards = []
        inner = self._act_scroll.get_container()
        scroll_w = self._act_scroll.relative_rect.width
        container_h = self._act_scroll.relative_rect.height
        card_h = 30
        gap = 4
        # La scrollbar de pygame_gui se solapa con el contenido cuando
        # aparece; reservamos su ancho SOLO si el contenido desborda el alto
        # visible, así las tarjetas llegan al borde derecho cuando no hay
        # scroll (y no quedan tapadas por la barra cuando sí lo hay).
        SCROLLBAR_W = 20
        n = len(action_lines)
        total_needed = 2 + n * (card_h + gap)
        will_scroll = total_needed > container_h
        reserve = SCROLLBAR_W if will_scroll else 0
        avail_w = max(80, scroll_w - reserve)
        y = 2
        for idx, lines in enumerate(action_lines):
            l1, l2 = lines
            t_str, dur_str = self._split_action_stats(l2)
            failed = "FALLÓ" in l2
            surf = self._render_action_card_surface(
                l1, t_str, dur_str, failed, avail_w, card_h)
            img = self._register(UIImage(
                relative_rect=pygame.Rect(0, y, avail_w, card_h),
                image_surface=surf,
                manager=self._manager,
                container=inner,
            ))
            # Guardar si la descripción se recortó (para el tooltip en
            # hover) junto con el texto completo.
            truncated = self._action_desc_truncated(l1, avail_w)
            self._act_cards.append((img, idx, truncated, l1))
            y += card_h + gap
        total_h = max(y, container_h)
        self._act_scroll.set_scrollable_area_dimensions((avail_w, total_h))

    def _ensure_card_fonts(self) -> None:
        """Crea (una vez) las fuentes Monogram usadas por las tarjetas de
        acción y de fallo. _act_font_l1 (20) es la común para el texto
        principal; _act_font_l2 (16) para las stats de las acciones.
        """
        if not hasattr(self, "_act_font_l1"):
            from pathlib import Path
            font_path = (
                resources.files("droneplan_viz_app")
                / "assets" / "fonts" / "monogram.ttf"
            )
            ttf = str(Path(str(font_path)))
            self._act_font_l1 = pygame.font.Font(ttf, 20)
            self._act_font_l2 = pygame.font.Font(ttf, 16)

    def _build_failure_cards(self, failure_labels: tuple) -> None:
        """Monta una tarjeta (UIImage de una línea) por fallo dentro del
        UIScrollingContainer de fallos, con la misma fuente y estilo que
        las tarjetas de acción. Guarda el índice del fallo para el
        hit-test de selección y si el texto se recortó (para el tooltip).
        """
        if self._fail_scroll is None:
            return
        self._fail_cards = []
        inner = self._fail_scroll.get_container()
        scroll_w = self._fail_scroll.relative_rect.width
        container_h = self._fail_scroll.relative_rect.height
        card_h = 30
        gap = 4
        SCROLLBAR_W = 20
        n = len(failure_labels)
        total_needed = 2 + n * (card_h + gap)
        will_scroll = total_needed > container_h
        reserve = SCROLLBAR_W if will_scroll else 0
        avail_w = max(80, scroll_w - reserve)
        y = 2
        for idx, label in enumerate(failure_labels):
            surf = self._render_failure_card_surface(label, avail_w, card_h)
            img = self._register(UIImage(
                relative_rect=pygame.Rect(0, y, avail_w, card_h),
                image_surface=surf,
                manager=self._manager,
                container=inner,
            ))
            truncated = self._failure_text_truncated(label, avail_w)
            self._fail_cards.append((img, idx, truncated, label))
            y += card_h + gap
        total_h = max(y, container_h)
        self._fail_scroll.set_scrollable_area_dimensions((avail_w, total_h))

    def _failure_text_truncated(self, text: str, width: int) -> bool:
        """True si el texto del fallo no cabe en una tarjeta de ese ancho
        (misma métrica que _render_failure_card_surface).
        """
        self._ensure_card_fonts()
        pad_x = 8
        text_max = max(20, width - pad_x * 2)
        return self._act_font_l1.size(text)[0] > text_max

    def _render_failure_card_surface(
        self, text: str, width: int, height: int
    ) -> pygame.Surface:
        """Renderiza una tarjeta de fallo de una línea: texto recortado con
        '…' si no cabe, borde rojo, misma fuente (_act_font_l1) y color de
        fondo que las tarjetas de acción.
        """
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_color = (37, 45, 56)          # #252D38
        border_color = (150, 70, 70)     # rojo (todos los fallos)
        pygame.draw.rect(surface, bg_color, (0, 0, width, height), border_radius=2)
        pygame.draw.rect(
            surface, border_color, (0, 0, width, height), width=1, border_radius=2
        )
        self._ensure_card_fonts()
        pad_x = 8
        text_max = max(20, width - pad_x * 2)
        txt = self._act_font_l1.render(
            self._fit_text(self._act_font_l1, text, text_max),
            True, (224, 150, 150),
        )
        surface.blit(txt, (pad_x, (height - txt.get_height()) // 2))
        return surface

    def _action_desc_truncated(self, desc: str, width: int) -> bool:
        """True si `desc` no cabe en la columna de descripción de una
        tarjeta de ancho `width` (misma métrica que _render_action_card_
        surface), es decir, si se mostrará recortada con '…'.
        """
        if not hasattr(self, "_act_font_l1"):
            # Aún no se creó la fuente; se creará en el primer render. Para
            # no duplicar la lógica, se asume no truncado hasta entonces.
            return False
        DUR_SLOT, T_SLOT, col_gap, pad_x = 52, 58, 6, 8
        desc_max = max(20, (width - DUR_SLOT - T_SLOT) - col_gap - pad_x)
        return self._act_font_l1.size(desc)[0] > desc_max

    @staticmethod
    def _split_action_stats(line2: str) -> tuple[str, str]:
        """Separa 't=Xs · ΔYs [· FALLÓ]' en (t_str, dur_str)."""
        parts = [p.strip() for p in line2.split("·")]
        t_str = parts[0] if len(parts) >= 1 else ""
        dur_str = parts[1] if len(parts) >= 2 else ""
        return t_str, dur_str

    def _fit_text(self, font: "pygame.font.Font", text: str, max_w: int) -> str:
        """Recorta `text` con '…' para que no exceda `max_w` px con `font`."""
        if font.size(text)[0] <= max_w:
            return text
        ell = "…"
        s = text
        while s and font.size(s + ell)[0] > max_w:
            s = s[:-1]
        return (s + ell) if s else ell

    def _render_action_card_surface(
        self, desc: str, t_str: str, dur_str: str, failed: bool,
        width: int, height: int,
    ) -> pygame.Surface:
        """Renderiza una tarjeta de acción de UNA línea con tres bloques:
        descripción (izquierda) y 't=...' / 'Δ...' anclados a la derecha en
        columnas de INICIO FIJO (mismas x en todas las filas → se alinean
        verticalmente). Cada bloque se recorta con '…' si excede su columna.
        Borde y stats en rojo si la acción falló.
        """
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_color = (37, 45, 56)          # #252D38
        border_color = (150, 70, 70) if failed else (60, 71, 84)
        pygame.draw.rect(surface, bg_color, (0, 0, width, height), border_radius=2)
        pygame.draw.rect(
            surface, border_color, (0, 0, width, height), width=1, border_radius=2
        )

        # Fuentes (cacheadas como atributos del Hud).
        if not hasattr(self, "_act_font_l1"):
            from pathlib import Path
            font_path = (
                resources.files("droneplan_viz_app")
                / "assets" / "fonts" / "monogram.ttf"
            )
            ttf = str(Path(str(font_path)))
            self._act_font_l1 = pygame.font.Font(ttf, 20)   # descripción
            self._act_font_l2 = pygame.font.Font(ttf, 16)   # stats (t, Δ)

        name_color = (230, 235, 240)
        stat_color = (224, 130, 130) if failed else (180, 190, 200)

        # Columnas de inicio FIJO ancladas al borde derecho. DUR_SLOT y
        # T_SLOT son el ancho (constante) de cada columna; al ser fijos, el
        # primer carácter de 'Δ...' y de 't=...' cae siempre en la misma x
        # (mismo espaciado al borde derecho) en todas las tarjetas.
        pad_x = 8
        DUR_SLOT = 52   # ancho de la columna 'Δ...' (px hasta el borde der.)
        T_SLOT = 58     # ancho de la columna 't=...'
        col_gap = 6
        dur_x = width - DUR_SLOT
        t_x = dur_x - T_SLOT
        desc_max = max(20, t_x - col_gap - pad_x)

        def blit_vcenter(text_surf, x):
            surface.blit(text_surf, (x, (height - text_surf.get_height()) // 2))

        sd = self._act_font_l1.render(
            self._fit_text(self._act_font_l1, desc, desc_max), True, name_color)
        blit_vcenter(sd, pad_x)
        st = self._act_font_l2.render(
            self._fit_text(self._act_font_l2, t_str, T_SLOT - col_gap),
            True, stat_color)
        blit_vcenter(st, t_x)
        sdur = self._act_font_l2.render(
            self._fit_text(self._act_font_l2, dur_str, DUR_SLOT - col_gap),
            True, stat_color)
        blit_vcenter(sdur, dur_x)
        return surface

    def _transport_sprites_available(self) -> bool:
        """¿Están los 12 PNGs de botones de transporte en la carpeta
        de assets? Si lo están, el theme_loader los habrá aplicado al
        theme JSON y el texto de los UIButton debe vaciarse (el sprite
        contiene el pictograma). Si falta alguno, los botones muestran
        el texto fallback.

        El resultado se cachea para evitar 12 stat() en cada relayout.
        """
        if hasattr(self, "_sprites_avail_cache"):
            return self._sprites_avail_cache
        try:
            from pathlib import Path
            anchor = resources.files("droneplan_viz_app")
            icons_dir = Path(str(anchor / "assets" / "hud_icons"))
            actions = (
                "home", "step_back", "play", "pause",
                "step_forward", "end",
            )
            ok = all(
                (icons_dir / f"btn_{a}_{state}.png").is_file()
                for a in actions for state in ("normal", "pressed")
            )
        except Exception:
            ok = False
        self._sprites_avail_cache = ok
        return ok

    def _make_chevron_surface(self, *, collapsed: bool) -> pygame.Surface:
        """Devuelve el sprite del chevron de cabecera.

        - collapsed=True  → inv_chevron_right.png (apunta a la derecha,
                            indica "sección plegada, click para desplegar").
        - collapsed=False → inv_chevron_down.png (apunta hacia abajo,
                            indica "sección desplegada, click para plegar").

        Los dos surfaces se cachean en self._chevron_cache para evitar
        leer el PNG en cada rebuild.

        Si los sprites no se encuentran, devuelve un placeholder cuadrado
        gris claro con un símbolo ASCII (">" o "v") para que el HUD
        funcione sin assets.
        """
        if not hasattr(self, "_chevron_cache"):
            self._chevron_cache = {}
        key = "right" if collapsed else "down"
        if key in self._chevron_cache:
            return self._chevron_cache[key]

        fname = f"inv_chevron_{key}.png"
        try:
            from pathlib import Path
            ref = (
                resources.files("droneplan_viz_app")
                / "assets" / "hud_icons" / fname
            )
            p = Path(str(ref))
            if p.is_file():
                surf = pygame.image.load(str(p)).convert_alpha()
                self._chevron_cache[key] = surf
                return surf
        except Exception:
            pass

        # Fallback: placeholder ASCII.
        L = self._hud_layout
        size = L.inv_chevron_size
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        font = pygame.font.SysFont(None, size - 2)
        glyph = ">" if collapsed else "v"
        label = font.render(glyph, True, (200, 210, 220))
        rect = label.get_rect(center=(size // 2, size // 2))
        surf.blit(label, rect)
        self._chevron_cache[key] = surf
        return surf

    def _make_inventory_icon_surface(self, category: str) -> pygame.Surface:
        """Devuelve el icono 24×24 para una categoría del inventario.

        Comportamiento:
        1. Intenta cargar el PNG correspondiente de
           droneplan_viz_app/assets/hud_icons/inv_<categoria>.png
           y devolver una composición: cuadrado de color de la categoría
           con esquinas redondeadas + el pictograma blanco encima.
        2. Si el PNG no se encuentra, cae al placeholder antiguo
           (cuadrado de color con letra inicial).

        Los iconos cargados se cachean por categoría en self._inv_icon_cache
        para no volver a leer el PNG ni recomponer en cada rebuild.

        Args:
            category: nombre de la categoría (DRONES, LOCATIONS, ...).

        Returns:
            Surface 24×24 con canal alfa.
        """
        L = self._hud_layout
        size = L.inv_icon_size

        # Cache.
        if not hasattr(self, "_inv_icon_cache"):
            self._inv_icon_cache = {}
        if category in self._inv_icon_cache:
            return self._inv_icon_cache[category]

        # Mapeo categoría → nombre de fichero. Notar inv_dron.png (sin 'e')
        # coincide con el nombre tal cual lo provee el usuario.
        filenames = {
            "DRONES":       "inv_dron.png",
            "LOCATIONS":    "inv_location.png",
            "PERSONAS":     "inv_person.png",
            "PAQUETES":     "inv_package.png",
            "TRANSPORTERS": "inv_transporter.png",
        }
        # Paleta de fondo recoloreada para integrarse en la paleta del
        # HUD (azul-grisáceo + amarillo). Coherente con los colores de
        # acento del world: azul para drones (mismo acento de paneles),
        # amarillo para locations (igual al borde de location en el
        # render), verde-azulado para personas, ámbar para paquetes
        # (similar a cajas marrones), morado para transporters.
        bg_colors = {
            "DRONES":       (122, 184, 255),  # #7AB8FF azul claro
            "LOCATIONS":    (200, 70, 70),   # #FFC850 amarillo
            "PERSONAS":     (122, 191, 170),  # #7ABFAA verde-azulado
            "PAQUETES":     (200, 152, 90),   # #C8985A ámbar
            "TRANSPORTERS": (176, 128, 224),  # #B080E0 morado
        }

        fname = filenames.get(category)
        png_path = None
        if fname:
            try:
                ref = (
                    resources.files("droneplan_viz_app")
                    / "assets" / "hud_icons" / fname
                )
                from pathlib import Path
                p = Path(str(ref))
                if p.is_file():
                    png_path = str(p)
            except Exception:
                png_path = None

        if png_path is not None:
            # Modo PNG real: cuadrado de color + pictograma blanco encima.
            # El recuadro de fondo ocupa todo `size` (inv_icon_size); el
            # pictograma se mantiene a su tamaño nativo y se CENTRA, en
            # lugar de escalarlo: escalar pixel-art a tamaños no múltiplos
            # (24→28) produce filas de píxeles irregulares. Centrar deja un
            # margen uniforme de color alrededor del pictograma.
            surface = pygame.Surface((size, size), pygame.SRCALPHA)
            bg = bg_colors.get(category, (180, 180, 180))
            pygame.draw.rect(surface, bg, (0, 0, size, size), border_radius=3)
            pictogram = pygame.image.load(png_path).convert_alpha()
            pw, ph = pictogram.get_size()
            # Si el pictograma es MAYOR que el recuadro, lo escalamos
            # hacia abajo (caso atípico); si es menor o igual, lo
            # centramos sin tocar.
            if pw > size or ph > size:
                pictogram = pygame.transform.scale(pictogram, (size, size))
                pw, ph = size, size
            off = ((size - pw) // 2, (size - ph) // 2)
            surface.blit(pictogram, off)
            self._inv_icon_cache[category] = surface
            return surface

        # Fallback: placeholder con letra inicial.
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        bg = bg_colors.get(category, (180, 180, 180))
        pygame.draw.rect(surface, bg, (0, 0, size, size), border_radius=3)
        font = pygame.font.SysFont(None, size - 4)
        initial = category[0] if category else "?"
        label = font.render(initial, True, (30, 30, 30))
        rect = label.get_rect(center=(size // 2, size // 2))
        surface.blit(label, rect)
        self._inv_icon_cache[category] = surface
        return surface

    # -----------------------------------------------------------------
    # Procesamiento de eventos
    # -----------------------------------------------------------------
    def process_event(self, event: pygame.event.Event) -> HudIntent | None:
        """Procesa un evento. Devuelve HudIntent o None."""
        self._manager.process_events(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            return self._intent_from_button_press(event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            act = self._intent_from_action_click(event)
            if act is not None:
                return act
            fail = self._intent_from_failure_click(event)
            if fail is not None:
                return fail
            return self._intent_from_inventory_click(event)
        return None

    def _active_tab_index(self) -> int:
        """Índice de la pestaña actualmente visible del panel derecho.

        Los hit-tests manuales (cajas de inventario, tarjetas de acciones)
        comparan rects en coordenadas de pantalla, pero pygame_gui mantiene
        esos rects con su posición lógica AUNQUE la pestaña esté oculta. Sin
        este filtro, un click en la zona del panel dispara los hit-tests de
        AMBAS pestañas a la vez (la oculta y la visible). Filtramos por la
        pestaña activa para que solo responda la que se está viendo.
        """
        if self._tab_container is None:
            return -1
        idx = getattr(self._tab_container, "current_container_index", None)
        return idx if idx is not None else -1

    def _intent_from_action_click(
        self, event: pygame.event.Event
    ) -> HudIntent | None:
        """Hit-test sobre las tarjetas del panel de Acciones.

        Cada tarjeta es un UIImage (no emite click propio), así que
        comprobamos manualmente si el click cayó dentro de su rect, y solo
        si cae dentro del viewport visible del scroll (una tarjeta puede
        estar parcialmente recortada por el scroll).

        Solo actúa si la pestaña Métricas+Fallos (que contiene las
        tarjetas) es la visible; en otra pestaña sus rects siguen vivos
        pero ocultos y no deben capturar clicks.

        Returns:
            HudIntent action_select con el índice de la acción, o None.
        """
        if self._active_tab_index() != self._tab_idx_metrics:
            return None
        if self._act_scroll is None or not self._act_cards:
            return None
        pos = event.pos
        if not self._act_scroll.rect.collidepoint(pos):
            return None
        for card_img, action_idx, _trunc, _full in self._act_cards:
            if card_img.rect.collidepoint(pos):
                return HudIntent(kind="action_select", payload=action_idx)
        return None

    def _intent_from_inventory_click(
        self, event: pygame.event.Event
    ) -> HudIntent | None:
        """Hit-test sobre las cajas del inventario.

        Cada caja es un UIImage que no emite eventos de click por sí
        mismo, así que comprobamos manualmente si el click cayó dentro
        del rect (en coordenadas de pantalla) de alguna caja visible.

        Solo procesamos clicks dentro del área visible del scroll
        container: una caja puede estar parcialmente fuera del viewport
        (scrolled), y pygame_gui mantiene el rect del widget en su
        posición lógica aunque esté recortado. Comprobamos contra el
        rect del scroll para no enfocar entidades cuya caja está fuera
        de vista pero cuyo rect lógico coincide con el click.

        Solo actúa si la pestaña Inventario es la visible; en otra
        pestaña las cajas siguen vivas pero ocultas y no deben capturar
        clicks.

        Returns:
            HudIntent inventory_focus con el entity_id si el click cae
            sobre una caja visible; None en otro caso.
        """
        if self._active_tab_index() != self._tab_idx_inventario:
            return None
        if self._inv_scroll is None or not self._inv_boxes:
            return None
        pos = event.pos  # (x, y) en coordenadas de pantalla.

        # El click debe caer dentro del viewport del scroll container.
        # Si no, ignoramos (puede ser un click en la scrollbar, en otra
        # tab, o fuera del panel).
        scroll_rect = self._inv_scroll.rect
        if not scroll_rect.collidepoint(pos):
            return None

        for box_img, entity_id in self._inv_boxes:
            if entity_id is None:
                continue
            # box_img.rect está en coordenadas de pantalla y refleja la
            # posición tras el scroll (pygame_gui lo actualiza).
            if box_img.rect.collidepoint(pos):
                return HudIntent(
                    kind="inventory_focus", payload=entity_id,
                )
        return None

    def _intent_from_button_press(self, event: pygame.event.Event) -> HudIntent | None:
        ui_el = event.ui_element  # type: ignore[attr-defined]
        if ui_el is self._btn_play_pause:
            return HudIntent(kind="play_pause")
        if ui_el is self._btn_home:
            return HudIntent(kind="home")
        if ui_el is self._btn_end:
            return HudIntent(kind="end")
        if ui_el is self._btn_step_back:
            return HudIntent(kind="step_back")
        if ui_el is self._btn_step_forward:
            return HudIntent(kind="step_forward")
        for btn, speed in zip(self._btn_speeds, ALLOWED_SPEEDS):
            if ui_el is btn:
                return HudIntent(kind="speed", payload=speed)
        # Cabeceras del inventario: cada (header_btn, chevron_img). El
        # chevron es UIImage (no clicable), así que solo el header_btn
        # dispara el UI_BUTTON_PRESSED. La cobertura visual del click
        # incluye el área del chevron porque el UIImage está dentro del
        # rect del botón sin interceptar el evento.
        for category, (header_btn, _chevron_img) in self._inv_headers.items():
            if ui_el is header_btn:
                return HudIntent(
                    kind="inventory_toggle_section", payload=category,
                )
        return None

    def _intent_from_failure_click(
        self, event: pygame.event.Event
    ) -> HudIntent | None:
        """Hit-test sobre las tarjetas del panel de Fallos.

        Cada tarjeta es un UIImage (no emite click propio), así que
        comprobamos manualmente si el click cayó dentro de su rect. Igual
        que las tarjetas de acción: solo actúa en la pestaña Métricas+
        Fallos y si el click cae dentro del viewport del scroll de fallos.

        Returns:
            HudIntent failure_select con el índice del fallo, o None.
        """
        if self._active_tab_index() != self._tab_idx_metrics:
            return None
        if self._fail_scroll is None or not self._fail_cards:
            return None
        pos = event.pos
        if not self._fail_scroll.rect.collidepoint(pos):
            return None
        for card_img, fail_idx, _trunc, _full in self._fail_cards:
            if card_img.rect.collidepoint(pos):
                return HudIntent(kind="failure_select", payload=fail_idx)
        return None

    # -----------------------------------------------------------------
    # Sincronización con AppState
    # -----------------------------------------------------------------
    def sync_from_state(
        self,
        state: AppState,
        snap_a: WorldSnapshot,
        snap_b: WorldSnapshot,
        progress: float,
    ) -> None:
        """Refresca textos de widgets según AppState y el frame actual.

        Tolera widgets a None (en el hueco entre relayout y la siguiente
        construcción, en teoría inexistente porque relayout reconstruye
        sincrónicamente, pero defensivo)."""
        if self._btn_play_pause is not None:
            sprites_loaded = self._transport_sprites_available()
            # El texto solo se actualiza si NO hay sprites (modo legacy).
            # Con sprites, el botón se construyó con text="" y el
            # pictograma viene del sprite del theme.
            if not sprites_loaded:
                new_text = "Play" if state.paused else "Pause"
                if self._btn_play_pause.text != new_text:
                    self._btn_play_pause.set_text(new_text)
            # object_id alterna entre @transport_play (mostrar Play =
            # estamos pausados) y @transport_pause (mostrar Pause =
            # reproduciendo). pygame_gui aplica el sprite de la sección
            # del theme correspondiente al object_id.
            target_oid = "@transport_play" if state.paused else "@transport_pause"
            current_oid = getattr(
                self._btn_play_pause, "_inv_current_oid", "@transport_play"
            )
            if current_oid != target_oid:
                self._btn_play_pause.change_object_id(target_oid)
                self._btn_play_pause._inv_current_oid = target_oid  # type: ignore[attr-defined]

        for btn, speed in zip(self._btn_speeds, ALLOWED_SPEEDS):
            active = speed == state.playback_speed
            target = self._format_speed_label(speed, active=active)
            if btn.text != target:
                btn.set_text(target)

        if self._progress_bar is not None:
            if self._plan.duration > 0:
                frac = min(state.playback_time / self._plan.duration, 1.0)
            else:
                frac = 0.0
            self._progress_bar.set_current_progress(frac)

        if self._lbl_time is not None:
            new_time = self._format_time_label(state.playback_time, self._plan.duration)
            if self._lbl_time.text != new_time:
                self._lbl_time.set_text(new_time)

        if self._lbl_now is not None:
            new_now = self._format_now_label(snap_a, snap_b, progress)
            if self._lbl_now.text != new_now:
                self._lbl_now.set_text(new_now)

        m = snap_a.metrics
        for lbl, text in (
            (self._lbl_total_cost, f"total_cost: {m.total_cost:.1f}"),
            (self._lbl_total_time, f"total_time: {m.total_time:.1f}s"),
            (self._lbl_action_count, f"actions: {m.action_count}"),
            (self._lbl_failed_count, f"failures: {m.failed_commands}"),
        ):
            if lbl is not None and lbl.text != text:
                lbl.set_text(text)

        # Inventario: su ESTRUCTURA (qué entidades, en qué orden y categoría)
        # es invariante durante la reproducción; solo cambian los textos al
        # moverse o cambiar de estado las entidades. Por eso evitamos el
        # trabajo O(N) por frame y la destrucción/recreación de widgets en las
        # transiciones:
        #   - El snapshot activo (snap_a) es el MISMO objeto en todos los
        #     frames de un segmento; solo cambia al cruzar un límite. Si ni el
        #     snapshot ni el plegado han cambiado, no hay nada que sincronizar
        #     (antes se reconstruían los datos y la firma O(N) cada frame).
        #   - Cuando cambian, si la estructura es la misma (caso típico de una
        #     transición: solo cambia el texto de algún dron) actualizamos in
        #     situ solo las cajas cuyo contenido cambió. Solo reconstruimos los
        #     widgets si cambia el conjunto de entidades, el plegado o el ancho.
        if self._inv_scroll is not None:
            collapsed = state.inventory_collapsed
            if (snap_a is not self._inv_last_snap
                    or collapsed != self._inv_last_collapsed):
                self._inv_last_snap = snap_a
                self._inv_last_collapsed = set(collapsed)
                self._sync_inventory(snap_a.world, collapsed)

    def _sync_inventory(self, world, collapsed) -> None:
        """Sincroniza el inventario con `world` de forma incremental.

        Reconstruye los widgets solo si cambia la estructura (conjunto/orden de
        entidades, plegado o ancho); en otro caso re-rasteriza y reasigna
        (set_image) únicamente las cajas cuyo contenido cambió, sin destruir ni
        recrear widgets.
        """
        new_items = build_inventory(world)
        avail_w = self._inv_avail_width()
        structure_key = (
            tuple((it.kind, it.entity_id, it.category) for it in new_items),
            frozenset(collapsed),
            avail_w,
        )
        if structure_key != self._inv_structure_key:
            # Cambio estructural → reconstrucción completa (destructiva).
            self._inventory_items = new_items
            self._inv_structure_key = structure_key
            self._rebuild_inventory_ui(new_items, collapsed)
            # Indexar las cajas recién creadas por entity_id y registrar su
            # contenido actual, para los diffs incrementales posteriores.
            self._inv_card_by_id = {
                eid: img for img, eid in self._inv_boxes if eid is not None
            }
            self._inv_content_by_id = {
                it.entity_id: (it.display_name, it.stats)
                for it in new_items
                if it.kind == "entity" and it.category not in collapsed
            }
            return
        # Misma estructura → actualizar in situ solo las cajas cambiadas.
        self._inventory_items = new_items
        for it in new_items:
            if it.kind != "entity" or it.category in collapsed:
                continue
            sig = (it.display_name, it.stats)
            if self._inv_content_by_id.get(it.entity_id) != sig:
                img = self._inv_card_by_id.get(it.entity_id)
                if img is not None:
                    img.set_image(
                        self._render_inventory_box_surface(it, avail_w)
                    )
                self._inv_content_by_id[it.entity_id] = sig

    # -----------------------------------------------------------------
    # update / draw
    # -----------------------------------------------------------------
    def update(self, dt: float) -> None:
        self._manager.update(dt)

    def draw(self, surface: pygame.Surface, state: AppState) -> None:
        """Dibuja widgets via UIManager y luego pinta los overlays propios."""
        self._manager.draw_ui(surface)
        self._draw_brand_logo(surface)
        self._draw_zoom_indicator(surface, state)
        self._draw_failure_marks(surface, state)
        self._draw_action_tooltip(surface)
        self._draw_help(surface, state)

    #: Controles que lista el overlay de ayuda, agrupados por categoría.
    _HELP_SECTIONS = (
        ("Reproducción", (
            ("Espacio", "Reproducir / pausar"),
            ("← / →", "Acción anterior / siguiente"),
            (", / .", "Fotograma a fotograma\n(mantener para avance continuo)"),
            ("Inicio / Fin", "Ir al principio / final del plan"),
        )),
        ("Cámara", (
            ("Rueda del ratón", "Acercar / alejar"),
            ("+ / −", "Acercar / alejar"),
            ("0", "Restablecer la cámara"),
            ("Arrastrar (botón central)", "Desplazar la vista"),
            ("L", "Mostrar / ocultar las etiquetas"),
        )),
        ("Capturas", (
            ("F", "Guardar captura de la ventana"),
            ("Mayús + F", "Guardar captura solo del visor"),
        )),
        ("General", (
            ("Alt + Enter / F11", "Pantalla completa"),
            ("Esc", "Salir de la pantalla completa"),
            ("H", "Mostrar / ocultar esta ayuda"),
        )),
    )

    def _draw_help(self, surface: pygame.Surface, state: AppState) -> None:
        """Pista fija 'H · ayuda' y, si state.show_help, el panel completo
        de controles sobre el área del visor.
        """
        world = self._regions.world
        font_path = monogram_font_path()

        def font(size: int) -> pygame.font.Font:
            return (pygame.font.Font(font_path, size) if font_path
                    else pygame.font.SysFont(None, size))

        # Pista permanente en la esquina inferior derecha del visor. Se
        # oculta mientras el panel está abierto (sería redundante).
        if not state.show_help:
            hint_font = font(18)
            hint = hint_font.render("H · ayuda", True, (150, 160, 172))
            hw, hh = hint.get_size()
            pad, margin = 5, 8
            bx = world.right - hw - pad * 2 - margin
            by = world.bottom - hh - pad * 2 - margin
            box = pygame.Surface((hw + pad * 2, hh + pad * 2), pygame.SRCALPHA)
            pygame.draw.rect(box, (28, 32, 38, 180),
                             box.get_rect(), border_radius=4)
            surface.blit(box, (bx, by))
            surface.blit(hint, (bx + pad, by + pad))
            return

        # Panel completo: velo semitransparente sobre el visor + tarjeta
        # centrada con los controles agrupados.
        veil = pygame.Surface(world.size, pygame.SRCALPHA)
        veil.fill((16, 19, 24, 200))
        surface.blit(veil, world.topleft)

        title_font = font(30)
        head_font = font(22)
        key_font = font(20)
        desc_font = font(20)
        key_color = (235, 238, 242)
        desc_color = (188, 196, 206)
        head_color = (240, 170, 70)

        # Medir el ancho de la columna de teclas para alinear descripciones.
        key_col = 0
        for _sec, rows in self._HELP_SECTIONS:
            for k, _d in rows:
                key_col = max(key_col, key_font.size(k)[0])
        key_col += 18  # separación teclas → descripción

        line_h = 26
        sec_gap = 14
        pad = 24
        title_surf = title_font.render("Controles", True, key_color)

        # Alto total del contenido. Una descripción puede ocupar varias
        # líneas (separadas por '\n'), que cuentan como filas extra.
        content_h = title_surf.get_height() + 16
        for _sec, rows in self._HELP_SECTIONS:
            content_h += head_font.get_height() + 4
            for _k, d in rows:
                content_h += line_h * d.count("\n")
            content_h += line_h * len(rows)
            content_h += sec_gap
        content_w = 520

        px = world.centerx - content_w // 2
        py = world.centery - content_h // 2 - pad

        panel = pygame.Rect(px - pad, py - pad,
                            content_w + pad * 2, content_h + pad * 2)
        pygame.draw.rect(surface, (30, 35, 42), panel, border_radius=8)
        pygame.draw.rect(surface, (70, 82, 96), panel, width=1, border_radius=8)

        y = py
        surface.blit(title_surf, (px, y))
        y += title_surf.get_height() + 16

        for sec, rows in self._HELP_SECTIONS:
            head = head_font.render(sec, True, head_color)
            surface.blit(head, (px, y))
            y += head_font.get_height() + 4
            for k, d in rows:
                surface.blit(key_font.render(k, True, key_color), (px + 6, y))
                # La descripción puede tener varias líneas ('\n'); la tecla
                # se alinea con la primera.
                for i, dline in enumerate(d.split("\n")):
                    surface.blit(
                        desc_font.render(dline, True, desc_color),
                        (px + 6 + key_col, y + i * line_h),
                    )
                y += line_h * (d.count("\n") + 1)
            y += sec_gap

        foot = desc_font.render("H o Esc para cerrar", True, (140, 150, 162))
        surface.blit(foot, (px, py + content_h - foot.get_height()))

    def _draw_action_tooltip(self, surface: pygame.Surface) -> None:
        """Muestra el texto completo bajo el cursor cuando la tarjeta (de
        acción o de fallo) lo tiene recortado con '…'.

        Solo actúa en la pestaña de métricas/fallos. El recuadro se
        superpone a la propia fila, con la misma fuente y color de fondo
        que las tarjetas, para que se perciba como continuación de la
        lista sin tapar la fila de encima.
        """
        if self._active_tab_index() != self._tab_idx_metrics:
            return
        if not hasattr(self, "_act_font_l1"):
            return
        pos = pygame.mouse.get_pos()
        # Tooltip de acciones (texto claro) o de fallos (texto rojo claro).
        if self._act_scroll is not None and self._act_cards \
                and self._act_scroll.rect.collidepoint(pos):
            self._draw_card_tooltip(
                surface, self._act_cards, pos, (230, 235, 240))
        elif self._fail_scroll is not None and self._fail_cards \
                and self._fail_scroll.rect.collidepoint(pos):
            self._draw_card_tooltip(
                surface, self._fail_cards, pos, (224, 150, 150))

    def _draw_card_tooltip(self, surface, cards, pos, text_color) -> None:
        """Dibuja el tooltip de texto completo sobre la tarjeta bajo el
        cursor, si su texto estaba recortado. Común a acciones y fallos.
        """
        for card_img, _idx, truncated, full in cards:
            if not truncated or not card_img.rect.collidepoint(pos):
                continue

            bg_color = (37, 45, 56)          # #252D38, igual que la tarjeta
            border_color = (90, 104, 120)
            pad_x = 8

            txt = self._act_font_l1.render(full, True, text_color)
            tw, th = txt.get_size()
            bw = tw + pad_x * 2

            # El recuadro se superpone a la PROPIA fila (mismo borde
            # superior y misma altura que la tarjeta), extendiéndose solo a
            # lo ancho para mostrar el texto completo. Así no tapa la fila
            # de encima. Se ancla al borde izquierdo de la tarjeta y se
            # recorta contra el borde derecho de la ventana.
            win_w, _win_h = surface.get_size()
            rect = card_img.rect
            bh = rect.height
            x = rect.left
            y = rect.top
            if x + bw > win_w:
                x = max(0, win_w - bw)

            pygame.draw.rect(surface, bg_color, (x, y, bw, bh), border_radius=2)
            pygame.draw.rect(
                surface, border_color, (x, y, bw, bh), width=1, border_radius=2
            )
            surface.blit(txt, (x + pad_x, y + (bh - th) // 2))
            break  # a lo sumo una tarjeta bajo el cursor

    def _draw_zoom_indicator(
        self, surface: pygame.Surface, state: AppState
    ) -> None:
        """Pinta el nivel de zoom actual en una esquina del visor.

        El zoom es la escala absoluta del sprite respecto a su resolucion
        nativa (1.0 = pixel nativo 1:1). Se muestra como porcentaje
        redondeado (100% = escala nativa), mas legible que un multiplicador
        con decimales. Se ancla a la esquina inferior izquierda del area de
        mundo, con un recuadro de fondo para legibilidad, en el mismo
        estilo que las etiquetas de la escena.
        """
        world = self._regions.world
        text = f"{round(state.zoom * 100)}%"

        font_path = monogram_font_path()
        font = (
            pygame.font.Font(font_path, 20)
            if font_path
            else pygame.font.SysFont(None, 20)
        )
        label = font.render(text, True, (235, 238, 242))
        tw, th = label.get_size()

        pad_x, pad_y, margin = 6, 3, 8
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        x = world.left + margin
        y = world.bottom - bh - margin

        box = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(
            box, (28, 32, 38, 210), pygame.Rect(0, 0, bw, bh),
            border_radius=max(3, bh // 3),
        )
        surface.blit(box, (x, y))
        surface.blit(label, (x + pad_x, y + pad_y))

    def _ensure_brand_logo(self) -> None:
        """Carga y escala el logo del HUD una vez (cacheo perezoso).

        Deja self._brand_logo_scaled a la Surface escalada o a None si no
        hay asset. Se usa tanto al dibujar el logo como al reservar su
        espacio para centrar el título.
        """
        if not hasattr(self, "_brand_logo_scaled"):
            from droneplan_viz_app.branding import load_hud_logo
            logo = load_hud_logo()
            if logo is None:
                self._brand_logo_scaled = None
            else:
                self._brand_logo_scaled = pygame.transform.scale(
                    logo, (logo.get_width() * 2, logo.get_height() * 2)
                )

    def _hud_logo_reserved_width(self) -> int:
        """Ancho que el logo ocupa en el borde derecho de la barra (con
        márgenes), 0 si no hay logo. Sirve para que el título quede
        centrado entre los botones y el logo, no entre los botones y el
        borde de la barra.
        """
        self._ensure_brand_logo()
        if self._brand_logo_scaled is None:
            return 0
        # Ancho del logo + margen derecho (14 ≈ el hueco 12+2 del blit) +
        # una holgura izquierda para que el título no roce el logo.
        return self._brand_logo_scaled.get_width() + 14 + 10

    def _draw_brand_logo(self, surface: pygame.Surface) -> None:
        """Pinta el logo de marca en la barra superior, si el asset existe.

        Se cachea (Surface ya escalada) en el primer draw para no recargar
        ni reescalar cada frame. El logo se ancla al borde DERECHO de la
        barra, junto al título, para no solaparse con los botones de
        transporte que ocupan la izquierda.

        Pixel-art: escalado x2 con nearest-neighbor (scale) y sin
        suavizado, coherente con SpriteManager. El logo base es de 28 px
        de alto, de modo que x2 (56 px) encaja en la barra superior.
        Degrada con elegancia: si no hay logo, no pinta nada y el título
        de texto de la barra se mantiene.
        """
        # Cacheo perezoso: la primera vez intentamos cargar y escalar; a
        # partir de ahí reutilizamos (o sabemos que no hay logo).
        self._ensure_brand_logo()

        if self._brand_logo_scaled is not None:
            top = self._regions.hud_top
            lw = self._brand_logo_scaled.get_width()
            lh = self._brand_logo_scaled.get_height()
            # Anclado a la derecha de la barra superior, centrado en su
            # alto, con un ajuste fino de +2 px a la derecha y +4 px hacia
            # abajo (el logo puede sobresalir de la barra por su parte
            # inferior; es intencional).
            x = top.right - lw - 12 + 2
            y = top.y + (top.height - lh) // 2 + 4
            surface.blit(self._brand_logo_scaled, (x, y))

    def _draw_failure_marks(self, surface: pygame.Surface, state: AppState) -> None:
        """Marcas verticales rojas con halo amarillo sobre la barra de
        progreso, una por fallo."""
        if (
            not self._plan.failure_starts
            or self._plan.duration <= 0
            or self._progress_bar is None
        ):
            return
        L = self._hud_layout
        bar_rect = self._progress_bar.rect
        for i, start_time in enumerate(self._plan.failure_starts):
            frac = min(start_time / self._plan.duration, 1.0)
            x = bar_rect.x + int(frac * bar_rect.width)
            is_selected = state.selected_failure == i
            # Halo PRIMERO (debajo) → se ve como aureola alrededor de
            # la línea roja, no encima.
            if is_selected:
                halo_w = 2 * L.halo_padding_x
                halo_h = bar_rect.height + 2 * L.halo_padding_y
                halo_rect = pygame.Rect(
                    x - L.halo_padding_x, bar_rect.y - L.halo_padding_y,
                    halo_w, halo_h,
                )
                halo_surface = pygame.Surface(halo_rect.size, pygame.SRCALPHA)
                halo_surface.fill(L.halo_color)
                surface.blit(halo_surface, halo_rect.topleft)
            thickness = L.mark_selected_thickness if is_selected else L.mark_thickness
            pygame.draw.line(
                surface,
                L.mark_color,
                (x, bar_rect.y - 2),
                (x, bar_rect.y + bar_rect.height + 2),
                thickness,
            )

    # -----------------------------------------------------------------
    # Helpers de formateo
    # -----------------------------------------------------------------
    @staticmethod
    def _format_time_label(playback_time: float, duration: float) -> str:
        """Formato 'X.Xs / Y.Ys'. Incluye la duración total para que el
        usuario sepa cuánto queda. Si duration es 0 (caso degenerado),
        solo muestra el tiempo actual."""
        if duration <= 0:
            return f"{playback_time:.1f}s"
        return f"{playback_time:.1f}s / {duration:.1f}s"

    @staticmethod
    def _format_speed_label(speed: float, *, active: bool) -> str:
        """Marca el activo con corchetes. Cuando una sesión futura
        introduzca theming visual, esta función desaparece a favor de
        cambiar el object_id del botón."""
        if speed == int(speed):
            base = f"{int(speed)}×"
        else:
            # Decimales sin el 0 a la izquierda: 0.25 → '.25', 0.5 → '.5'.
            base = f"{speed}×".lstrip("0")
        return f"[{base}]" if active else base

    @staticmethod
    def _format_now_label(
        snap_a: WorldSnapshot,
        snap_b: WorldSnapshot,
        progress: float,
    ) -> str:
        """Etiqueta "Now: ..." del HUD.

        Usa `classify_transition` como fuente de verdad para
        decidir si hay un Command animándose: misma decisión que toma el
        render para pintar el frame. Mantener una única fuente de verdad
        evita la discrepancia que existía en E.5 cuando el label se
        encendía durante tramos administrativos de 0.6s (snap_start del
        primer Command, end_anterior→start_siguiente entre Commands
        durativos) que el render trata como TransitionStatic.

        Política:
          - Transición estática (incluido el caso edge snap_a is snap_b,
            que classify_transition ya devuelve como TransitionStatic
            por su caso 2)  →  "Now: —"
          - Cualquier otra transición animable → "Now: {ClassName}(args) — {pct}%"

        El snap_b.produced_by tiene que existir cuando hay animación;
        si por alguna razón fuera None (defensivo, no debería pasar tras
        una transición no-estática), devolvemos también el dash.
        """
        transition = classify_transition(snap_a, snap_b)
        if isinstance(transition, TransitionStatic):
            return "Now: —"
        cmd = snap_b.produced_by
        if cmd is None:
            return "Now: —"
        args: list[str] = []
        for fld in fields(cmd):
            if fld.name in {"command_id", "duration"}:
                continue
            args.append(str(getattr(cmd, fld.name)))
        args_str = ",".join(args)
        if len(args_str) > 40:
            args_str = args_str[:37] + "..."
        pct = int(progress * 100)
        return f"Now: {cmd.__class__.__name__}({args_str}) — {pct}%"