"""Loop principal de la aplicación: clase DroneplanVizApp.

Compone todos los módulos (scenarios, app_state, layout, controller,
hud) sobre una ventana pygame real. Es el único módulo del paquete que
junta pygame.display, pygame_gui y el render.

Diseño:

  - El constructor hace todo el setup (init, ventana, ejecución del
    plan, Timeline, Hud). Tras él, run() entra en el loop hasta QUIT.
  - El loop sigue exactamente el patrón consensuado:
        dt = clock.tick(60) / 1000
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.VIDEORESIZE: _on_resize(event)
            intent = hud.process_event(event)
            if intent:
                if intent.kind == "failure_select":
                    handle_hud_failure_select(...)
                else:
                    handle_hud_intent(...)
            else:
                handle_pygame_event(state, event, timeline)
        controller.tick(state, dt, timeline)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        hud.update(dt)
        window.fill(theme.background)
        render_frame(world_surface, snap_a, snap_b, progress, theme)
        hud.draw(window, state)
        pygame.display.flip()

  - run_one_frame() está expuesto como método separado para que los
    tests inyecten eventos sintéticos y verifiquen el flujo sin
    arrancar un loop infinito.

  - El método save_screenshot() es público para que la app pueda
    generar capturas para la memoria (lo invoca un argumento de CLI
    en main.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final
from collections.abc import Mapping

import pygame
import pygame_gui

from droneplan_viz.render import (
    Theme,
    Timeline,
    render_world_at,
    supersample_factor,
)
from droneplan_viz.runtime import PlanRunner
from droneplan_viz_app.app_state import AppState, fitting_zoom
from droneplan_viz_app.controller import (
    on_frame_back,
    on_frame_forward,
    focus_on_entity_at,
    handle_hud_action_select,
    handle_hud_failure_select,
    handle_hud_intent,
    handle_hud_inventory_focus,
    handle_hud_inventory_toggle_section,
    handle_pygame_event,
    tick,
)
from droneplan_viz_app.hud import Hud, HudIntent, build_hud_plan_info
from droneplan_viz_app.layout import compute_regions
from droneplan_viz_app.scenarios import SCENARIOS
from droneplan_viz_app.theme_loader import monogram_font_path, prepare_ui_theme

if TYPE_CHECKING:
    # Solo para la anotación de `scenario`; no se importan en runtime para no
    # acoplar la app a estos tipos más allá de lo que ya usa.
    from droneplan_viz.domain import World
    from droneplan_viz.runtime import Plan


#: Tamaño mínimo de ventana con el que el layout sigue siendo operable.
#: compute_regions() lanza ValueError por debajo del mínimo de world; la
#: responsabilidad de clampear es de la app (ver layout.py), y este es el
#: único sitio donde vive ese número.
MIN_WINDOW_SIZE: Final[tuple[int, int]] = (1024, 640)


def _clamp_to_min(size: tuple[int, int]) -> tuple[int, int]:
    """Eleva `size` al mínimo operable del layout (sin reducirlo nunca)."""
    return (max(size[0], MIN_WINDOW_SIZE[0]), max(size[1], MIN_WINDOW_SIZE[1]))


def _desktop_size() -> tuple[int, int]:
    """Resolución del escritorio, para la pantalla completa.

    Se usa get_desktop_sizes() y NO pygame.display.Info(): tras un
    set_mode(), Info().current_w/h devuelven el tamaño de la VENTANA, no el
    de la pantalla, que era justo el valor incorrecto que rompía el cálculo.
    Si SDL no puede informar (entornos sin pantalla), se degrada al mínimo.
    """
    try:
        sizes = pygame.display.get_desktop_sizes()
    except pygame.error:  # pragma: no cover - depende del backend SDL
        sizes = []
    return sizes[0] if sizes else MIN_WINDOW_SIZE


class DroneplanVizApp:
    """Aplicación interactiva pygame + pygame_gui sobre droneplan_viz.

    Construye una ventana redimensionable, ejecuta un escenario al
    arranque, y entra en un loop que reproduce la animación con
    controles HUD.

    Args:
        scenario_name: clave del registro SCENARIOS (default "demo").
        scenario: par (World, Plan) construido al vuelo (p. ej. por la
            fachada DronePlanViz). Si se da, TOMA PRECEDENCIA sobre
            scenario_name y se salta la resolución vía SCENARIOS. Permite a
            la fachada abrir la app sobre un escenario propio sin pasar por
            el registro de demos. None (default) => se usa scenario_name.
        window_size: tamaño inicial de la ventana.
        fps_cap: tope de frames por segundo. 60 por defecto.
    """

    def __init__(
        self,
        *,
        scenario_name: str = "demo",
        scenario: "tuple[World, Plan] | None" = None,
        window_size: tuple[int, int] = (1280, 800),
        fps_cap: int = 60,
        content_colors: "Mapping[str, tuple[int, int, int]] | None" = None,
    ) -> None:
        # El registro solo se consulta (y valida) cuando NO se inyecta un
        # escenario al vuelo. Así la fachada puede pasar su propio
        # (world, plan) y el camino por scenario_name (CLI, tests de E)
        # sigue idéntico.
        if scenario is None and scenario_name not in SCENARIOS:
            raise ValueError(
                f"escenario '{scenario_name}' desconocido; "
                f"disponibles: {sorted(SCENARIOS.keys())}"
            )

        self._fps_cap = fps_cap
        self._running = False

        # Init pygame + window.
        pygame.init()
        self._window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("droneplan_viz · visualizador interactivo")
        self._clock = pygame.time.Clock()

        # Estado de pantalla completa. _windowed_size guarda el tamaño de
        # la ventana para restaurarlo al salir del modo completo.
        self._fullscreen = False
        self._windowed_size = window_size

        # Pantalla de carga: pinta un frame ANTES del trabajo pesado
        # (ejecución del plan, Timeline, SpriteManager, HUD), que es lo
        # que mantenía la ventana en negro unos segundos al arrancar.
        self._draw_loading_screen(window_size)

        # Ejecución del escenario (una sola vez al arranque). El escenario
        # inyectado tiene precedencia; si no, se resuelve por nombre.
        world, plan = scenario if scenario is not None else SCENARIOS[scenario_name]()
        self._runner = PlanRunner(world)
        self._result = self._runner.execute(plan)

        # Estado y datos derivados.
        # Inyectamos la fuente Monogram (la misma del inventario) en las
        # etiquetas del render para que los nombres de los objetos se lean
        # claros y coherentes con la estética pixel-art. Si el asset no
        # estuviera disponible, monogram_font_path() devuelve None y el
        # render cae a SysFont sin romper.
        self._theme = Theme.default().with_overrides(
            font_path=monogram_font_path(),
            content_colors=content_colors,
        )
        # SpriteManager: activa los sprites .png de
        # droneplan_viz/render/assets/. Si falta alguno, ese
        # elemento cae a primitiva. Se crea UNA vez y se reutiliza.
        from droneplan_viz.render.sprite_manager import SpriteManager
        self._sprite_manager = SpriteManager(self._theme)
        self._timeline = Timeline(self._runner.history, theme=self._theme)
        self._state = AppState(window_size=window_size)
        self._regions = compute_regions(window_size)
        # Zoom de ARRANQUE: el nivel de la rejilla (escala absoluta) al que
        # toda la escena entra en pantalla. Depende de la densidad del grafo
        # (vía el supersampling) y del tamaño de la región de escena, pero la
        # rejilla de niveles es fija. Se guarda en home_zoom (destino del reset).
        self._state.home_zoom = self._fit_zoom_to_scene()
        self._state.zoom = self._state.home_zoom
        self._failures_start_times = tuple(
            f.scheduled.start_time for f in self._result.failures
        )

        # HUD.
        # El theme JSON contiene la declaración de Monogram como fuente
        # por defecto. `prepare_ui_theme()` resuelve la ruta del TTF
        # empaquetado vía importlib.resources y produce un theme válido
        # que pygame_gui consume al construir el manager.
        theme_path = prepare_ui_theme()
        self._ui_manager = pygame_gui.UIManager(window_size, theme_path=theme_path)
        plan_info = build_hud_plan_info(
            self._timeline.duration,
            self._timeline.snapshot_times,
            self._result.failures,
            self._result.plan,
            self._runner.history,
        )
        #: Tiempos virtuales de inicio de cada acción (para el salto al
        #: clicar una acción en el panel de Acciones).
        self._action_start_times = plan_info.action_starts
        self._hud = Hud(self._ui_manager, self._regions, plan_info, self._theme)

        # Garantiza que la pantalla de carga se haya visto al menos 1 s,
        # aunque todo el setup anterior haya sido muy rápido.
        self._hold_loading_screen(min_ms=1000)

    # -----------------------------------------------------------------
    # Loop público
    # -----------------------------------------------------------------
    def run(self) -> int:
        """Ejecuta el loop principal hasta QUIT. Devuelve 0."""
        self._running = True
        try:
            while self._running:
                self.run_one_frame()
        finally:
            pygame.quit()
        return 0

    def run_one_frame(self, *, dt_override: float | None = None) -> None:
        """Procesa un único frame.

        Expuesto separadamente para los tests, que inyectan eventos
        sintéticos vía pygame.event.post() y luego invocan este método
        un número fijo de veces.

        Args:
            dt_override: si se proporciona, se usa este dt en vez de
                medirlo con el clock. Útil para tests que ejecutan
                muchos frames sin pausa real. En producción es None y
                self._clock.tick(fps_cap) gobierna la cadencia.
        """
        if dt_override is None:
            dt = self._clock.tick(self._fps_cap) / 1000.0
        else:
            dt = dt_override

        # 1. Procesar eventos.
        for event in pygame.event.get():
            self._handle_event(event)

        # Nota: si un QUIT ha puesto self._running a False, run() saldrá
        # del while en la próxima iteración. Aquí seguimos pintando un
        # frame final coherente (no abortamos a mitad).

        # 1b. Avance fotograma a fotograma sostenido. Las teclas ',' y '.'
        # se consultan por estado (no por evento discreto) para que, al
        # mantenerlas pulsadas, la reproducción avance de forma continua,
        # como el desplazamiento cuadro a cuadro de un reproductor de
        # vídeo. Al sondearse cada frame, la primera pulsación ya responde
        # de inmediato y no se duplica con ningún KEYDOWN. Si ambas están
        # pulsadas se cancelan (no se avanza).
        pressed = pygame.key.get_pressed()
        back = pressed[pygame.K_COMMA]
        forward = pressed[pygame.K_PERIOD]
        if back and not forward:
            on_frame_back(self._state, timeline=self._timeline)
        elif forward and not back:
            on_frame_forward(self._state, timeline=self._timeline)

        # 2. Tick (avance automático si no está pausado).
        tick(self._state, dt, timeline=self._timeline)

        # 3. Estado lógico del frame para el HUD (snap activo + progreso). El
        # render por entidad (render_world_at) ya NO necesita las acciones
        # vecinas: cada drone se interpola sobre el span completo de su acción.
        snap_a, snap_b, progress = self._timeline.sample(
            self._state.playback_time
        )

        # 3b. Seguimiento de cámara: si hay una entidad seguida, recentramos
        # la cámara sobre su posición INTERPOLADA este frame. Así la cámara
        # acompaña a un drone en vuelo en vez de quedarse en su nodo de
        # salida. Si la entidad dejó de ser localizable (id obsoleto, o un
        # paquete que pasó a estar sostenido), cancelamos el seguimiento.
        if self._state.followed_entity is not None:
            try:
                focus_on_entity_at(
                    self._state, self._timeline, self._state.playback_time,
                    self._state.followed_entity,
                    self._regions.world.size,
                    theme=self._theme,
                )
            except (KeyError, ValueError):
                self._state.followed_entity = None

        # 4. Sincronizar HUD.
        self._hud.sync_from_state(self._state, snap_a, snap_b, progress)
        self._hud.update(dt)

        # 5. Dibujar.
        self._window.fill(self._theme.background)
        world_surface = self._window.subsurface(self._regions.world)
        # Cámara: zoom (ESCALA ABSOLUTA del sprite, 1.0 = nativo 1:1) y pan del
        # AppState, reenviados al render uniforme por entidad. El render compone
        # el buffer nativo a esa escala (independiente de la densidad de la
        # escena), así que un nivel de zoom dibuja el sprite del mismo tamaño en
        # cualquier plan. render_world_at interpola CADA entidad sobre el span
        # completo de su acción: coreografía encadenada y movimientos concurrentes
        # correctos, de forma uniforme. El zoom de arranque (home_zoom) hace
        # entrar toda la escena en pantalla.
        render_world_at(
            world_surface, self._timeline, self._state.playback_time,
            theme=self._theme, sprite_manager=self._sprite_manager,
            zoom=self._state.zoom, pan=self._state.pan,
            anim_time=pygame.time.get_ticks() / 1000.0,
            labels_hidden=self._state.labels_hidden,
        )
        self._hud.draw(self._window, self._state)
        pygame.display.flip()

    # -----------------------------------------------------------------
    # Despachador interno de un solo evento
    # -----------------------------------------------------------------
    def _handle_event(self, event: pygame.event.Event) -> None:
        """Despacha un evento individual.

        Orden de prioridad:
          1. QUIT cierra la app.
          2. VIDEORESIZE recalcula regions y avisa al UIManager. NO se
             reenvía a handle_pygame_event para evitar que asigne
             state.window_size sin pasar por el clamp.
          3. Eventos HUD (UI_BUTTON_PRESSED, UI_SELECTION_LIST_*) → si el
             HUD reconoce el widget, intent → controller.
          4. Eventos genéricos (teclado, etc.) → handle_pygame_event.
        """
        if event.type == pygame.QUIT:
            self._running = False
            return

        if event.type == pygame.VIDEORESIZE:
            # En pantalla completa el tamaño lo fija el modo de vídeo, no el
            # usuario: los avisos de redimensionado que algunos gestores de
            # ventanas emiten al entrar en ella se ignoran. Si no, el
            # set_mode(..., RESIZABLE) de _on_resize devolvería la app a modo
            # ventana justo después de haber activado la pantalla completa.
            if not self._fullscreen:
                self._on_resize((event.w, event.h))
            # NO reenviamos: _on_resize ya actualiza state.window_size
            # con el tamaño clampeado.
            return

        intent = self._hud.process_event(event)
        if intent is not None:
            self._handle_intent(intent)
            return

        # Tecla F: guardar una captura. Se atiende aquí, en la app, porque
        # es una acción sobre la ventana (guardar un PNG), no sobre el
        # estado de reproducción. Se captura el frame ya compuesto sin
        # volver a renderizar (evita recursión con save_screenshot, que
        # llama a run_one_frame).
        #   - F:        ventana completa.
        #   - Shift+F:  solo el visor (el área del escenario), sin barras
        #               ni panel lateral.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            only_world = bool(event.mod & pygame.KMOD_SHIFT)
            self._save_screenshot_auto(only_world=only_world)
            return

        # Pantalla completa: Alt+Enter (atajo clásico en juegos) o F11
        # (común en navegadores y editores). Alterna ventana/completo.
        if event.type == pygame.KEYDOWN and (
            event.key == pygame.K_F11
            or (event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                and event.mod & pygame.KMOD_ALT)
        ):
            self._toggle_fullscreen()
            return

        # Tecla H: alterna el overlay de ayuda. Escape lo cierra si está
        # abierto. Es estado de presentación de la app, no de reproducción.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
            self._state.show_help = not self._state.show_help
            return
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                and self._state.show_help):
            self._state.show_help = False
            return

        # Tecla L: alterna la visibilidad manual de las etiquetas de las
        # entidades (independiente del ocultado automático por zoom).
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            self._state.labels_hidden = not self._state.labels_hidden
            return
        # Escape también sale de pantalla completa (comportamiento
        # esperado). Solo si la ayuda no estaba abierta (ese caso ya se
        # atendió arriba).
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                and self._fullscreen):
            self._toggle_fullscreen()
            return

        handle_pygame_event(
            self._state, event,
            timeline=self._timeline,
            world_region=self._regions.world,
        )

    def _handle_intent(self, intent: HudIntent) -> None:
        """Despacha un HudIntent al controller.

        Algunos intents requieren contexto extra que el despachador
        genérico (handle_hud_intent) no maneja:
        - failure_select: necesita los start_times (precalculados).
        - inventory_focus: necesita el world snapshot actual y el tamaño
          de la subsurface donde el render dibuja el grafo.
        - inventory_toggle_section: solo necesita mutar AppState.
        """
        if intent.kind == "failure_select":
            handle_hud_failure_select(
                self._state,
                intent,
                self._failures_start_times,
                timeline=self._timeline,
            )
        elif intent.kind == "action_select":
            handle_hud_action_select(
                self._state,
                intent,
                self._action_start_times,
                timeline=self._timeline,
            )
        elif intent.kind == "inventory_focus":
            # Snapshot actual = snap_a del frame visible.
            snap_a, _snap_b, _progress = self._timeline.sample(
                self._state.playback_time
            )
            handle_hud_inventory_focus(
                self._state,
                intent,
                snap_a.world,
                self._regions.world.size,
                theme=self._theme,
            )
        elif intent.kind == "inventory_toggle_section":
            handle_hud_inventory_toggle_section(self._state, intent)
        else:
            handle_hud_intent(self._state, intent, timeline=self._timeline)

    def _fit_zoom_to_scene(self) -> float:
        """Nivel de zoom (escala absoluta) al que toda la escena entra en la
        región de escena actual: fitting_zoom del factor de supersampling.

        Las locations son estáticas durante el plan, así que cualquier snapshot
        da el mismo factor; usamos el inicial.
        """
        base, _b, _p = self._timeline.sample(0.0)
        ss = supersample_factor(
            base.world, self._regions.world.size, self._theme
        )
        return fitting_zoom(ss)

    def _toggle_fullscreen(self) -> None:
        """Alterna entre ventana y pantalla completa (Alt+Enter / F11).

        Al entrar se hace set_mode(resolución_del_escritorio,
        FULLSCREEN | SCALED):

          - El tamaño se pide EXPLÍCITO, vía get_desktop_sizes(). Pasar
            (0, 0) no vale con SCALED: pygame-ce lo rechaza con "Cannot set
            0 sized SCALED display mode", y como el error se capturaba abajo
            el atajo se quedaba en nada.
          - SCALED hace la pantalla completa "de escritorio", SIN cambiar el
            modo de vídeo del monitor. El FULLSCREEN clásico sí cambia la
            resolución física, que es lo que provocaba parpadeos, frames en
            negro y saltos de ventana. Además, si la pantalla resultara ser
            más pequeña que el mínimo del layout, SCALED escala la superficie
            lógica en vez de dejar el HUD fuera de cuadro.

        Al volver se restaura el último tamaño de ventana. En ambos casos se
        recompone el layout con el tamaño efectivo (_relayout).
        """
        if not self._fullscreen:
            # Guardar el tamaño actual de ventana para restaurarlo luego.
            self._windowed_size = self._window.get_size()
            try:
                self._window = pygame.display.set_mode(
                    _clamp_to_min(_desktop_size()),
                    pygame.FULLSCREEN | pygame.SCALED,
                )
            except pygame.error:
                # Entorno sin pantalla utilizable: se deja todo como estaba
                # en lugar de recomponer un layout que no ha cambiado.
                return
            self._fullscreen = True
        else:
            self._window = pygame.display.set_mode(
                _clamp_to_min(self._windowed_size), pygame.RESIZABLE
            )
            self._fullscreen = False

        # El cambio de modo puede dejar en cola un aviso de redimensionado
        # con el tamaño ANTERIOR; procesarlo desharía la transición.
        pygame.event.clear(pygame.VIDEORESIZE)

        # Tamaño efectivo de dibujo tras el cambio (get_size da el real).
        self._relayout(self._window.get_size())

    def _relayout(self, size: tuple[int, int]) -> None:
        """Recompone regions, UIManager, HUD y zoom de ajuste para `size`.

        Camino único de los dos cambios de tamaño de la ventana (resize del
        usuario y pantalla completa), para que ambos dejen exactamente el
        mismo estado. `size` debe ser ya el tamaño real de la superficie de
        dibujo: aquí no se clampea nada (si no, las regiones podrían caer
        fuera de la superficie y subsurface() reventaría al pintar).
        """
        self._regions = compute_regions(size)
        self._ui_manager.set_window_resolution(size)
        self._hud.relayout(self._regions)
        self._state.window_size = size
        # La región de escena cambió → recalcular el zoom de ajuste para el
        # reset. El zoom ACTUAL (escala absoluta) se preserva: un sprite a 1.0
        # sigue siendo nativo aunque cambie la ventana.
        self._state.home_zoom = self._fit_zoom_to_scene()

    def _on_resize(self, new_size: tuple[int, int]) -> None:
        """Recalcula regions, actualiza UIManager y recompone el HUD.

        El layout impone un mínimo (MIN_WINDOW_SIZE) que clampeamos aquí.
        Por debajo de ese tamaño, compute_regions lanzaría ValueError; al
        clampear, el usuario ve la app intacta aunque su ventana sea
        más pequeña (los bordes simplemente quedan recortados por el
        gestor de ventanas, no por nuestro código).

        Tres operaciones tras el resize:

          1. pygame.display.set_mode con el tamaño clampeado.
          2. UIManager.set_window_resolution para que pygame_gui sepa
             del nuevo viewport (necesario para tooltips, layout
             interno, etc.).
          3. hud.relayout para que TODOS los widgets se reposicionen y
             redimensionen según las nuevas regions. Esto es lo que
             diferencia a el runtime de una app "que se ve mal al
             redimensionar": el HUD se reconstruye limpiamente.
        """
        clamped = _clamp_to_min(new_size)
        self._window = pygame.display.set_mode(clamped, pygame.RESIZABLE)
        self._relayout(clamped)

    # -----------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------
    def _draw_loading_screen(self, window_size: tuple[int, int]) -> None:
        """Pinta un frame de carga (fondo + logo + texto) y lo muestra.

        Se invoca al principio del constructor, antes del setup pesado,
        para que el usuario vea de inmediato una pantalla de marca en
        lugar de una ventana en negro. Es un único frame estático: no
        anima ni bloquea; el flip() lo hace visible y el constructor
        continúa con la carga real acto seguido.

        Registra el instante de inicio (self._loading_started_at) para que
        el constructor pueda garantizar un tiempo mínimo de exhibición
        (ver _hold_loading_screen), de modo que la pantalla se vea aunque
        el resto del arranque sea muy rápido.

        Degrada con elegancia: si no hay logo, muestra solo el texto; si
        algo fallara, no debe impedir el arranque (por eso el try/except).

        El logo es pixel-art: se escala x2 con nearest-neighbor (scale) y
        sin suavizado, para preservar los bordes duros (coherente con
        SpriteManager, que evita smoothscale por lo mismo).
        """
        # Instante de inicio, para el tiempo mínimo de exhibición.
        self._loading_started_at = pygame.time.get_ticks()

        # Color de fondo constante (el mismo azul-gris oscuro del tema de
        # render). No usamos self._theme porque aún no está construido en
        # este punto del arranque.
        BG = (28, 32, 38)
        FG = (235, 238, 242)

        try:
            from droneplan_viz_app.branding import load_logo

            w, h = window_size
            self._window.fill(BG)

            # Logo centrado, escalado x2 con nearest-neighbor (scale), sin
            # suavizado, para conservar la estética pixel-art.
            text_y = h // 2
            logo = load_logo()
            if logo is not None:
                lw, lh = logo.get_size()
                scaled = pygame.transform.scale(logo, (lw * 2, lh * 2))
                rect = scaled.get_rect(center=(w // 2, h // 2 - 30))
                self._window.blit(scaled, rect)
                text_y = rect.bottom + 30

            # Texto "Cargando…" con Monogram si está disponible.
            font_path = monogram_font_path()
            font = (
                pygame.font.Font(font_path, 28)
                if font_path
                else pygame.font.SysFont(None, 28)
            )
            label = font.render("Cargando…", True, FG)
            self._window.blit(label, label.get_rect(center=(w // 2, text_y)))

            pygame.display.flip()
        except Exception:
            # La pantalla de carga es puramente cosmética: cualquier fallo
            # aquí no debe abortar el arranque de la aplicación.
            self._window.fill(BG)
            pygame.display.flip()

    def _hold_loading_screen(self, min_ms: int = 1000) -> None:
        """Garantiza que la pantalla de carga se vea al menos min_ms.

        Se llama al final del constructor, tras el setup pesado. Si la
        carga real terminó antes del mínimo, espera el tiempo restante;
        si tardó más, no espera nada. Así la pantalla siempre es visible
        un instante, incluso cuando el entorno carga muy rápido.

        En modo headless (controlador de vídeo 'dummy', usado por tests y
        por los generadores de capturas) no hay ventana visible, así que
        la espera no aporta nada y se omite para no penalizar esos usos.
        """
        import os

        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return
        started = getattr(self, "_loading_started_at", None)
        if started is None:
            return
        elapsed = pygame.time.get_ticks() - started
        remaining = min_ms - elapsed
        if remaining > 0:
            pygame.time.wait(remaining)

    def save_screenshot(self, path: str) -> None:
        """Guarda la ventana actual como PNG.

        Útil para generar las capturas que acompañan la memoria. Llama
        a run_one_frame() con dt=0 (sin avance temporal) para asegurar
        un frame fresco SIN modificar el estado.
        """
        self.run_one_frame(dt_override=0.0)
        pygame.image.save(self._window, path)

    def _save_screenshot_auto(self, *, only_world: bool = False) -> None:
        """Guarda una captura (tecla F) con nombre autonumerado.

        Pensado para generar el material gráfico de la memoria: captura
        exactamente lo que se ve, siempre al mismo tamaño y encuadre, sin
        depender de recortes manuales de pantalla. A diferencia de
        save_screenshot, NO vuelve a renderizar (esto se invoca durante el
        procesado de eventos, ya dentro de run_one_frame): guarda el frame
        ya compuesto en pantalla.

        only_world: si es True, guarda solo el área del visor (el recuadro
            del escenario), sin la barra superior, el panel lateral ni la
            barra de progreso. Si es False, guarda la ventana completa.

        El archivo se escribe en la carpeta 'screenshots/' (creada si no
        existe) del directorio de trabajo actual, con el patrón
        'droneplan_captura[_visor]_AAAAMMDD_HHMMSS_mmm.png', de modo que
        pulsaciones sucesivas no se sobrescriben.
        """
        import os
        from datetime import datetime

        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
        tag = "_visor" if only_world else ""
        try:
            os.makedirs("screenshots", exist_ok=True)
            path = os.path.join("screenshots", f"droneplan_captura{tag}_{stamp}.png")
            if only_world:
                # Recortar exactamente la región del visor (misma que usa
                # el render para el world).
                region = self._window.subsurface(self._regions.world).copy()
                pygame.image.save(region, path)
            else:
                pygame.image.save(self._window, path)
        except Exception:
            # Guardar una captura nunca debe tirar la aplicación.
            pass

    @property
    def state(self) -> AppState:
        """Estado actual (acceso de solo lectura para tests)."""
        return self._state

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def fullscreen(self) -> bool:
        """True si la ventana está en pantalla completa (solo lectura)."""
        return self._fullscreen

    @property
    def hud(self) -> Hud:
        """HUD de la app (acceso de solo lectura para tests y demos, p.ej.
        para revelar la pestaña de métricas/fallos)."""
        return self._hud