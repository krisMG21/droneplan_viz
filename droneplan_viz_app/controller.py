"""Controller: traduce eventos a mutaciones de AppState.

Tres tipos de handlers:

  1. tick(state, dt, *, timeline)
     Avance automático de playback_time cada frame. Auto-pausa al llegar
     a Timeline.duration.

  2. handle_pygame_event(state, event, *, timeline)
     Eventos crudos de pygame: teclado (SPACE play/pause; LEFT/RIGHT
     step; HOME/END), VIDEORESIZE. NO procesa clicks sobre widgets — eso
     es responsabilidad del HUD, que abstrae a HudIntent.

  3. Handlers de HudIntent
     Una función por intent: on_play_pause, on_step_back, on_step_forward,
     on_home, on_end, on_speed_button, on_failure_clicked. El HUD se
     encarga de mapear sus widgets de pygame_gui a estas intenciones, así
     el controller NO importa pygame_gui (frontera arquitectónica
     consensuada en la propuesta).

Diseño:

- Los handlers mutan state IN-SITU. AppState es mutable por diseño.
  Devolver copias introduciría un patrón pseudo-funcional sin ganancia.
- Los handlers son PUROS desde el punto de vista de testing: dado un
  state inicial y un evento, el state' es determinista. No tocan disco,
  ni red, ni globales.
- selected_failure se RESETEA a None en cualquier navegación que no sea
  click sobre un fallo. Es decisión consciente: la selección "se
  desactiva" al moverse, así no queda la marca pegada cuando el usuario
  navega.
- handle_pygame_event no muta state.playback_time en VIDEORESIZE; solo
  actualiza window_size. El recálculo de regions es responsabilidad del
  loop principal de la app.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pygame

from droneplan_viz.render import Timeline
from droneplan_viz_app.app_state import (
    ALLOWED_SPEEDS,
    AppState,
    DEFAULT_PAN,
    FRAME_STEP,
    ZOOM_MAX,
    ZOOM_MIN,
    ZOOM_STEP,
)

if TYPE_CHECKING:
    # Solo para anotaciones; con `from __future__ import annotations` no
    # se evalúan en tiempo de ejecución, así que evitamos importar el
    # dominio aquí en runtime (manteniendo controller liviano).
    from droneplan_viz.domain import World
    from droneplan_viz.render import Theme


#: ZOOM_STEP (factor por tick de rueda / atajos +/-) se importa de app_state,
#: que es la fuente única de la rejilla de zoom. Desde el zoom de carga (1.0):
#: 1 tick de zoom-out satura a ZOOM_MIN (0.8) y 5 de zoom-in a ZOOM_MAX
#: (3.0517578125); más ticks saturan en los extremos. Se re-exporta aquí para que
#: `from controller import ZOOM_STEP` siga funcionando.


#: Coeficiente de easing exponencial del seguimiento de cámara: fracción de la
#: distancia al pan-objetivo que se recorre cada frame (pan += (objetivo-pan)·k).
#: ~0.25 da una constante de tiempo de ~3-4 frames (≈60 ms a 60 fps): la cámara
#: acompaña al objetivo con un ligero "peso" que suaviza el escalonado sub-píxel
#: y los saltos de borde de acción, sin lag perceptible. Asume la cadencia de
#: ~60 fps de la app; un valor mayor = más pegada al objetivo (más brusca), menor
#: = más suave pero con más retardo.
_FOLLOW_SMOOTHING: float = 0.25


#: Estado MÓDULO-LOCAL para el arrastre con botón central (pan).
#:
#: Mantener el punto previo del ratón fuera de AppState es deliberado:
#: es información puramente transitoria de UI (válida solo entre
#: MOUSEBUTTONDOWN y MOUSEBUTTONUP del botón central). Si perdiéramos
#: este valor (p.ej. reinicio de la app), el arrastre simplemente se
#: aborta — no hay pérdida de datos del usuario. Meterlo en AppState
#: contaminaría el estado lógico con un detalle puramente visual.
#:
#: None ⇔ no hay arrastre en curso. Tupla (x, y) ⇔ último punto del
#: ratón registrado durante el arrastre activo.
_pan_drag_last: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# HudIntent: abstracción para desacoplar controller de pygame_gui
# ---------------------------------------------------------------------------

#: Tipos posibles de intento del HUD. Literal cerrado para que el type
#: checker avise si se introduce una categoría nueva sin documentarla.
HudIntentKind = Literal[
    "play_pause",
    "step_back",
    "step_forward",
    "home",
    "end",
    "speed",
    "failure_select",
    "action_select",
    "inventory_focus",
    "inventory_toggle_section",
]


@dataclass(frozen=True, slots=True)
class HudIntent:
    """Intento abstraído del HUD a procesar por el controller.

    Atributos:
        kind: categoría del intent (botón pulsado, item seleccionado).
        payload: datos asociados. Convención por kind:
            - "play_pause", "step_back", "step_forward",
              "home", "end"  : payload ignorado (None).
            - "speed"        : payload = float (uno de ALLOWED_SPEEDS).
            - "failure_select": payload = int (índice en failures).
            - "action_select": payload = int (índice en la lista de acciones).
            - "inventory_focus": payload = str (entity_id a enfocar).
            - "inventory_toggle_section": payload = str (categoría a
              plegar/desplegar, e.g. "DRONES").
    """
    kind: HudIntentKind
    payload: object = None


# ---------------------------------------------------------------------------
# Tick automático
# ---------------------------------------------------------------------------


def tick(state: AppState, dt: float, *, timeline: Timeline) -> None:
    """Avanza playback_time si no está pausado.

    Política al alcanzar duration: auto-pausa, NO loop (decisión §2.4.a
    de la propuesta confirmada). El playback_time se clampea a duration
    para evitar drift positivo.

    Args:
        state: AppState mutable. Modificado in-situ.
        dt: incremento de tiempo en segundos (típicamente del clock).
        timeline: Timeline en curso, necesario para conocer la duración
            total y aplicar el clamp.
    """
    if state.paused:
        return
    state.playback_time += dt * state.playback_speed
    if state.playback_time >= timeline.duration:
        state.playback_time = timeline.duration
        state.paused = True


# ---------------------------------------------------------------------------
# Eventos pygame crudos
# ---------------------------------------------------------------------------


def handle_pygame_event(
    state: AppState,
    event: pygame.event.Event,
    *,
    timeline: Timeline,
    world_region: "pygame.Rect | None" = None,
) -> None:
    """Despacha un pygame.event.Event al handler correspondiente.

    Eventos manejados:
      - KEYDOWN SPACE         → toggle paused.
      - KEYDOWN LEFT          → step_back (saltar al snapshot anterior).
      - KEYDOWN RIGHT         → step_forward (al siguiente).
      - KEYDOWN HOME          → playback_time = 0, pausa.
      - KEYDOWN END           → playback_time = duration, pausa.
      - KEYDOWN +/= ó KP_PLUS → zoom in (factor ZOOM_STEP).
      - KEYDOWN -   ó KP_MINUS→ zoom out (factor 1/ZOOM_STEP).
      - KEYDOWN 0   ó KP_0    → reset de cámara a defaults.
      - MOUSEWHEEL            → zoom in/out según event.y (signed), SOLO
                                si el cursor está sobre la región del
                                world (ver world_region).
      - MOUSEBUTTONDOWN btn=2 → inicia arrastre de pan.
      - MOUSEMOTION (con drag activo) → suma delta a state.pan.
      - MOUSEBUTTONUP btn=2   → termina arrastre.
      - VIDEORESIZE           → actualiza state.window_size.

    Args:
        world_region: rect (en coordenadas de ventana) de la zona donde
            se dibuja el world. Si se pasa, la rueda del ratón solo
            produce zoom cuando el cursor está DENTRO de esa región;
            sobre el panel lateral o el HUD el zoom se ignora (y el
            scroll del inventario lo gestiona pygame_gui por su cuenta).
            Si es None (default), la rueda siempre hace zoom
            (retrocompatibilidad).

    Cualquier otro evento se ignora silenciosamente. El loop de la app
    se encarga de QUIT por su cuenta (no es responsabilidad del state).
    """
    global _pan_drag_last

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
            on_play_pause(state)
        elif event.key == pygame.K_LEFT:
            on_step_back(state, timeline=timeline)
        elif event.key == pygame.K_RIGHT:
            on_step_forward(state, timeline=timeline)
        elif event.key == pygame.K_HOME:
            on_home(state)
        elif event.key == pygame.K_END:
            on_end(state, timeline=timeline)
        # Atajos de cámara.
        # K_PLUS suele requerir Shift en teclados con +/= en la misma
        # tecla; aceptamos también K_EQUALS (la tecla física sin Shift)
        # y K_KP_PLUS del numpad para que el usuario no tenga que pelear
        # con el layout.
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            on_zoom(state, ZOOM_STEP)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            on_zoom(state, 1.0 / ZOOM_STEP)
        elif event.key in (pygame.K_0, pygame.K_KP_0):
            on_camera_reset(state)
    elif event.type == pygame.MOUSEWHEEL:
        # event.y > 0 = rueda hacia arriba = zoom in. event.y < 0 = zoom out.
        # Si en algún driver llega 0, no hacemos nada (no rompe).
        #
        # IMPORTANTE: el zoom solo aplica si el cursor está sobre la
        # región del world. Sobre el panel lateral (inventario/métricas)
        # la rueda debe hacer scroll, no zoom — ese scroll lo gestiona
        # pygame_gui en el UIScrollingContainer y aquí simplemente NO
        # hacemos zoom. El evento MOUSEWHEEL de pygame no incluye la
        # posición del cursor, así que la consultamos con get_pos().
        if world_region is not None:
            mx, my = pygame.mouse.get_pos()
            if not world_region.collidepoint(mx, my):
                return  # Fuera del world: no hacemos zoom.
        if event.y > 0:
            on_zoom(state, ZOOM_STEP)
        elif event.y < 0:
            on_zoom(state, 1.0 / ZOOM_STEP)
    elif event.type == pygame.MOUSEBUTTONDOWN:
        # Botón 2 = botón central (la rueda pulsada). Elegido en plan de
        # cámara para no chocar con click izquierdo (selección futura).
        if event.button == 2:
            _pan_drag_last = (int(event.pos[0]), int(event.pos[1]))
    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 2:
            _pan_drag_last = None
    elif event.type == pygame.MOUSEMOTION:
        # Solo respondemos si hay un arrastre activo. event.rel ya da el
        # delta, pero confiamos en nuestra propia cuenta para no depender
        # de la limpieza de rel entre frames perdidos.
        if _pan_drag_last is not None:
            x, y = int(event.pos[0]), int(event.pos[1])
            dx = x - _pan_drag_last[0]
            dy = y - _pan_drag_last[1]
            on_pan(state, dx, dy)
            _pan_drag_last = (x, y)
    elif event.type == pygame.VIDEORESIZE:
        # event.size = (w, h) en pygame-ce / pygame.
        state.window_size = (int(event.w), int(event.h))


# ---------------------------------------------------------------------------
# Intenciones de HUD (también accesibles desde teclado)
# ---------------------------------------------------------------------------


def handle_hud_intent(
    state: AppState,
    intent: HudIntent,
    *,
    timeline: Timeline,
) -> None:
    """Despacha un HudIntent simple (sin contexto extra) al handler.

    Args:
        state: AppState mutable.
        intent: HudIntent producido por el HUD.
        timeline: Timeline en curso (necesario para algunos handlers).

    Maneja: play_pause, step_back, step_forward, home, end, speed.

    NO maneja "failure_select" ni "inventory_focus" porque requieren
    piezas extra de contexto que no encajan en este despachador
    genérico (los start_times de los fallos para failure_select; el
    world snapshot y el tamaño de surface para inventory_focus). Para
    esos la app llama a sus handlers dedicados directamente.
    """
    if intent.kind == "play_pause":
        on_play_pause(state)
    elif intent.kind == "step_back":
        on_step_back(state, timeline=timeline)
    elif intent.kind == "step_forward":
        on_step_forward(state, timeline=timeline)
    elif intent.kind == "home":
        on_home(state)
    elif intent.kind == "end":
        on_end(state, timeline=timeline)
    elif intent.kind == "speed":
        assert isinstance(intent.payload, (int, float)), (
            f"speed intent requiere payload numérico, recibido {intent.payload!r}"
        )
        on_speed_button(state, float(intent.payload))
    elif intent.kind == "failure_select":
        raise ValueError(
            "intent 'failure_select' no se procesa via handle_hud_intent; "
            "usa handle_hud_failure_select"
        )
    elif intent.kind == "inventory_focus":
        raise ValueError(
            "intent 'inventory_focus' no se procesa via handle_hud_intent; "
            "usa handle_hud_inventory_focus"
        )
    elif intent.kind == "inventory_toggle_section":
        raise ValueError(
            "intent 'inventory_toggle_section' no se procesa via "
            "handle_hud_intent; usa handle_hud_inventory_toggle_section"
        )
    else:
        # Por el Literal cerrado, esto es inalcanzable; pero blindamos.
        raise ValueError(f"HudIntent desconocido: {intent.kind!r}")


# ---------------------------------------------------------------------------
# Handlers individuales (públicos para tests granulares)
# ---------------------------------------------------------------------------


def on_play_pause(state: AppState) -> None:
    """Alterna paused. Si reanuda desde el final, rebobina a 0.

    El caso "reanudar desde el final": si el usuario llegó al final
    (state.paused = True por auto-pausa, playback_time == duration) y
    pulsa Play, el comportamiento natural es volver al inicio y empezar
    de nuevo, no quedarse pegado al final. Esto es lo que espera
    cualquiera acostumbrado a un reproductor multimedia.

    Decisión consciente: NO comparamos con timeline.duration aquí (el
    handler no recibe timeline). En su lugar, la regla práctica es: si
    al despausar veríamos un único frame estático antes de auto-pausar
    de nuevo, ese fue el final. La condición exacta la aplicamos en
    tick() — auto-pausa cuando playback_time >= duration. Aquí, al
    despausar, no hacemos nada especial: el siguiente tick disparará
    auto-pause en el mismo frame. Para evitar ese caso flickerante,
    delegamos al usuario que use HOME explícitamente. Mantiene este
    handler sin dependencias.

    Esta decisión simplifica los tests del controller (no necesitan
    Timeline para test_play_pause). Si se demuestra molesta para el
    usuario en el pulido visual de E.5, lo revisamos.
    """
    state.paused = not state.paused
    state.selected_failure = None  # navegación: borra selección.


def on_step_back(state: AppState, *, timeline: Timeline) -> None:
    """Salta al snapshot inmediatamente anterior al playback_time actual.

    Usa Timeline.snapshot_times (propiedad pública añadida en E.3.0).
    Si playback_time está exactamente sobre un snapshot, salta al
    anterior. Si está antes del primero, no hace nada. Pausa la
    reproducción.

    Implementación: bisect.bisect_left sobre snapshot_times. El índice
    devuelto es la posición donde insertaríamos playback_time en orden;
    el snapshot inmediatamente anterior está en (idx - 1). Si idx == 0,
    no hay snapshot anterior.
    """
    state.paused = True
    state.selected_failure = None
    times = timeline.snapshot_times
    idx = bisect.bisect_left(times, state.playback_time)
    if idx == 0:
        # Ya estamos en o antes del primer snapshot; no hay anterior.
        return
    state.playback_time = times[idx - 1]


def on_step_forward(state: AppState, *, timeline: Timeline) -> None:
    """Salta al snapshot inmediatamente posterior al playback_time actual.

    Pausa la reproducción. Si ya estamos en o después del último
    snapshot, no hace nada (la reproducción acabó).

    Implementación: bisect.bisect_right encuentra la posición tras
    cualquier ocurrencia de playback_time. El índice ahí ES el siguiente
    snapshot (si existe).
    """
    state.paused = True
    state.selected_failure = None
    times = timeline.snapshot_times
    idx = bisect.bisect_right(times, state.playback_time)
    if idx >= len(times):
        # Ya estamos en el final o más allá; no hay siguiente.
        return
    state.playback_time = times[idx]


def on_frame_back(state: AppState, *, timeline: Timeline) -> None:
    """Retrocede un fotograma (FRAME_STEP segundos). Pausa.

    A diferencia de on_step_back, que salta al snapshot anterior del
    plan, este avance es de grano fino (1/60 s por defecto) y permite
    inspeccionar la interpolación entre dos snapshots, como el avance
    fotograma a fotograma de un reproductor de vídeo. El resultado se
    acota a 0.0 para no retroceder antes del inicio.
    """
    state.paused = True
    state.selected_failure = None
    state.playback_time = max(0.0, state.playback_time - FRAME_STEP)


def on_frame_forward(state: AppState, *, timeline: Timeline) -> None:
    """Avanza un fotograma (FRAME_STEP segundos). Pausa.

    Contraparte de on_frame_back. El resultado se acota a la duración
    del plan para no rebasar el final.
    """
    state.paused = True
    state.selected_failure = None
    state.playback_time = min(timeline.duration, state.playback_time + FRAME_STEP)


def on_home(state: AppState) -> None:
    """Rebobina al inicio. Pausa."""
    state.playback_time = 0.0
    state.paused = True
    state.selected_failure = None


def on_end(state: AppState, *, timeline: Timeline) -> None:
    """Salta al final. Pausa."""
    state.playback_time = timeline.duration
    state.paused = True
    state.selected_failure = None


def on_speed_button(state: AppState, multiplier: float) -> None:
    """Cambia la velocidad de reproducción.

    Valida que multiplier esté en ALLOWED_SPEEDS (los botones del HUD
    solo emiten valores permitidos, pero blindamos contra futuras
    invocaciones programáticas).

    NO cambia paused: el usuario decide independientemente cuándo pausar.
    NO resetea selected_failure: cambiar la velocidad mientras se mira
    un fallo es una operación tangencial, no una navegación.
    """
    if multiplier not in ALLOWED_SPEEDS:
        raise ValueError(
            f"velocidad {multiplier} no permitida; debe estar en {ALLOWED_SPEEDS}"
        )
    state.playback_speed = multiplier


def on_failure_clicked(
    state: AppState,
    index: int,
    start_time: float,
    *,
    timeline: Timeline,
    failures_count: int,
) -> None:
    """Salta al instante de un fallo y lo marca como seleccionado.

    El index es relativo a RunResult.failures; sirve solo para que
    selected_failure se quede apuntando a ese fallo (el HUD lo usará
    para resaltarlo).

    El start_time es la posición a la que saltar (típicamente
    failures[index].scheduled.start_time). Lo pasa el caller (la app)
    para evitar que el controller dependa de CommandFailure o
    RunResult.

    Si start_time excede timeline.duration (caso teórico: un Command
    programado más allá del último éxito), lo clampeamos defensivamente.

    Args:
        state: AppState mutable.
        index: índice del fallo en RunResult.failures, [0, failures_count).
        start_time: tiempo de reproducción al que saltar.
        timeline: Timeline en curso (para conocer duration).
        failures_count: total de fallos (para validar index).

    Raises:
        ValueError: si index está fuera de rango o start_time < 0.
    """
    if not (0 <= index < failures_count):
        raise ValueError(
            f"failure index {index} fuera de rango [0, {failures_count})"
        )
    if start_time < 0.0:
        raise ValueError(f"start_time debe ser >= 0, recibido {start_time}")

    state.playback_time = min(start_time, timeline.duration)
    state.paused = True
    state.selected_failure = index


def handle_hud_failure_select(
    state: AppState,
    intent: HudIntent,
    failures_start_times: tuple[float, ...],
    *,
    timeline: Timeline,
) -> None:
    """Helper para el caso failure_select del handle_hud_intent.

    El payload del intent es el índice. La tupla failures_start_times
    contiene los start_time de cada fallo (la app la construye una vez
    al arranque y la pasa por referencia).

    Separado de handle_hud_intent para mantener este último simple: el
    intent failure_select tiene un payload de un solo entero, pero
    necesita una pieza adicional de contexto (los start_times). En vez
    de inflar la firma de handle_hud_intent con un argumento más, este
    helper específico lo procesa.

    La app llamará a este helper directamente cuando reciba un intent
    failure_select desde el HUD.
    """
    assert intent.kind == "failure_select"
    assert isinstance(intent.payload, int)
    idx = intent.payload
    on_failure_clicked(
        state,
        idx,
        failures_start_times[idx],
        timeline=timeline,
        failures_count=len(failures_start_times),
    )


def on_action_clicked(
    state: AppState,
    index: int,
    start_time: float,
    *,
    timeline: Timeline,
    actions_count: int,
) -> None:
    """Salta al COMIENZO de una acción SIN cambiar el estado de reproducción.

    A diferencia de on_failure_clicked (que pausa y marca el fallo
    seleccionado), aquí solo se mueve `playback_time`: si la reproducción
    estaba en play, sigue en play desde el inicio de la acción; si estaba en
    pausa, sigue en pausa ahí. Igual que el seguimiento de entidades, la
    selección de una acción no debe alterar play/pausa.

    start_time es el tiempo VIRTUAL de inicio de la acción (el del primer
    snapshot que produjo), de modo que el salto cae exactamente en su
    comienzo dentro de la animación.

    Raises:
        ValueError: si index está fuera de rango o start_time < 0.
    """
    if not (0 <= index < actions_count):
        raise ValueError(
            f"action index {index} fuera de rango [0, {actions_count})"
        )
    if start_time < 0.0:
        raise ValueError(f"start_time debe ser >= 0, recibido {start_time}")
    state.playback_time = min(start_time, timeline.duration)
    # NO se toca state.paused ni state.selected_failure: el salto a una
    # acción es ortogonal al estado de reproducción.


def handle_hud_action_select(
    state: AppState,
    intent: HudIntent,
    action_start_times: tuple[float, ...],
    *,
    timeline: Timeline,
) -> None:
    """Helper para el caso action_select de handle_hud_intent.

    El payload del intent es el índice en la lista de acciones. La tupla
    action_start_times contiene el tiempo VIRTUAL de inicio de cada acción
    (la app la construye una vez al arranque y la pasa por referencia).
    """
    assert intent.kind == "action_select"
    assert isinstance(intent.payload, int)
    idx = intent.payload
    if not (0 <= idx < len(action_start_times)):
        return
    on_action_clicked(
        state,
        idx,
        action_start_times[idx],
        timeline=timeline,
        actions_count=len(action_start_times),
    )


def handle_hud_inventory_focus(
    state: AppState,
    intent: HudIntent,
    world,
    surface_size: tuple[int, int],
    *,
    theme=None,
) -> None:
    """Procesa un click sobre un item entity del inventario.

    Resuelve la cámara para que la entidad seleccionada quede centrada
    via focus_on_entity. El zoom se mantiene al valor actual del state
    (el usuario eligió un zoom, no lo cambiamos en el focus); solo
    movemos el pan.

    Args:
        state: AppState a mutar (pan).
        intent: kind="inventory_focus", payload=entity_id (str).
        world: WorldSnapshot actual (snap_a del frame visible).
        surface_size: tamaño en píxeles de la subsurface del world.
        theme: Theme con el que se renderiza.

    Cualquier entidad del world (drone, transporter, person, location o
    package en CUALQUIER placement) activa el seguimiento. El encuadre
    inicial estático se intenta best-effort: si la entidad no es localizable
    estáticamente (paquete HeldByArm/InTransporter, que dependen del
    portador), no pasa nada — el seguimiento por frame la situará con su
    posición interpolada en cuanto se dibuje el siguiente frame.

    Si el entity_id NO existe en el world (condición de carrera con un
    inventario obsoleto), no activamos nada.
    """
    assert intent.kind == "inventory_focus"
    assert isinstance(intent.payload, str)
    entity_id = intent.payload
    if not _entity_exists(world, entity_id):
        return
    # Encuadre inicial estático (best-effort). Para paquetes sostenidos o
    # dentro de un transporter, locate_entity lanza ValueError: lo ignoramos
    # porque el seguimiento por frame (focus_on_entity_frame) recentra sobre
    # la posición interpolada correcta en el acto.
    try:
        focus_on_entity(state, world, entity_id, surface_size, theme=theme)
    except (KeyError, ValueError):
        pass
    # A partir de ahora la cámara SIGUE a esta entidad a lo largo de TODAS
    # sus acciones (vuelo, coreografía intra-loc, transporte, etc.).
    state.followed_entity = entity_id


def _entity_exists(world, entity_id: str) -> bool:
    """¿Existe entity_id como alguna entidad del world?"""
    return (
        entity_id in world.drones
        or entity_id in world.transporters
        or entity_id in world.persons
        or entity_id in world.locations
        or entity_id in world.packages
    )


def handle_hud_inventory_toggle_section(
    state: AppState,
    intent: HudIntent,
) -> None:
    """Procesa un click sobre una cabecera de sección del inventario.

    Si la categoría estaba plegada, la despliega (la quita del set);
    si estaba desplegada, la pliega (la añade). El HUD detecta el
    cambio en la próxima sync_from_state vía el signature del rebuild.

    Args:
        state: AppState a mutar (inventory_collapsed).
        intent: kind="inventory_toggle_section", payload=str con la
            categoría (e.g. "DRONES").
    """
    assert intent.kind == "inventory_toggle_section"
    assert isinstance(intent.payload, str)
    category = intent.payload
    if category in state.inventory_collapsed:
        state.inventory_collapsed.discard(category)
    else:
        state.inventory_collapsed.add(category)


# ---------------------------------------------------------------------------
# Handlers de cámara (pan + zoom)
# ---------------------------------------------------------------------------
#
# Tres handlers puros sobre AppState, sin pygame, testables aisladamente.
# Los eventos crudos de pygame (rueda, click central, teclas +/-/0) se
# despachan a estos handlers en handle_pygame_event más abajo.
#
# Política de saturación: aunque el render satura internamente fuera de
# [ZOOM_MIN, ZOOM_MAX] (defensivo según nota de D2), nosotros clampeamos
# AQUÍ para mantener AppState válida en todo momento: el invariante
# `ZOOM_MIN <= state.zoom <= ZOOM_MAX` del state.validate() debe
# cumplirse SIEMPRE, no solo en el frame que se renderiza.


def on_zoom(state: AppState, factor: float) -> None:
    """Multiplica state.zoom por factor, saturando al rango permitido.

    factor > 1.0 ⇒ zoom in (acerca el grafo).
    factor < 1.0 ⇒ zoom out (aleja).

    Saturar (en vez de lanzar) es coherente con la decisión de D2 para
    el render: alcanzar el extremo del rango no es un error, es el tope
    natural; el usuario simplemente nota que el control no avanza más.
    """
    new_zoom = state.zoom * factor
    state.zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))


def on_pan(state: AppState, dx: int, dy: int) -> None:
    """Desplaza state.pan en (dx, dy) píxeles.

    Sin clamp: el usuario puede arrastrar muy lejos y sacar el world del
    marco visible — eso es intencional, se recupera con reset (tecla 0).

    El pan manual CANCELA el seguimiento de cámara: si el usuario arrastra,
    toma el control y la cámara deja de perseguir a la entidad seguida (de
    lo contrario el recentrado por frame anularía el arrastre).
    """
    state.followed_entity = None
    state.follow_pan = None
    px, py = state.pan
    state.pan = (px + int(dx), py + int(dy))


def on_camera_reset(state: AppState) -> None:
    """Restaura zoom y pan a la vista de arranque de la escena.

    Atajo: tecla 0. Útil cuando el usuario se ha perdido tras arrastrar
    o hacer mucho zoom; vuelve siempre a "todo visible". También cancela
    el seguimiento de cámara (vuelta a cámara libre). El zoom vuelve a
    `home_zoom` (el nivel de ajuste de ESTA escena, fijado al cargar), no
    a un 1.0 fijo que en escenas densas estaría muy acercado.
    """
    state.followed_entity = None
    state.follow_pan = None
    state.zoom = state.home_zoom
    state.pan = DEFAULT_PAN


def focus_on_entity(
    state: AppState,
    world: World,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
    zoom: float | None = None,
) -> None:
    """Centra la cámara sobre la entidad en la surface dada.

    Calcula y aplica el `pan` necesario para que la entidad quede en el
    centro de la subsurface del world, respetando el zoom indicado (o el
    zoom actual del state si no se pasa uno).

    Args:
        state: AppState a mutar. Se actualizan state.zoom y state.pan.
        world: WorldSnapshot del que extraer la posición de la entidad.
        entity_id: id de drone, location, transporter, person o package.
            Si es un package InTransporter/HeldByArm, locate_entity lanza
            ValueError — esta función propaga el error: el llamante debe
            resolver el sujeto (el drone/transporter que lo sostiene) y
            enfocar sobre él. Id inexistente propaga KeyError.
        surface_size: (sw, sh) en píxeles de la subsurface del world.
            Importante: NO el tamaño de la ventana completa, sino el
            rect donde el render dibuja el grafo.
        theme: theme con el que se está renderizando. Debe ser el mismo
            que se pasa a render_frame; el padding y los offsets de las
            anclas afectan a la posición resultante.
        zoom: nivel de zoom destino. Si None, se mantiene state.zoom.
            Si se pasa, se aplica antes de calcular el pan.

    La fórmula es la documentada por D2 en nota_sesion_e2_camara.txt:
    el render aplica zoom expandiendo desde el centro de la surface y
    luego suma el pan. Por tanto, para que la entidad acabe centrada:

        pan_x = sw//2 - (sw/2 + (lx - sw/2) * z)
        pan_y = sh//2 - (sh/2 + (ly - sh/2) * z)

    No animación: el cambio es instantáneo. Animar la transición de
    cámara (lerp del par actual al destino) es responsabilidad de la
    app si lo quiere; este helper solo calcula y aplica el destino.

    Tras ejecutar, state.validate() sigue pasando (zoom saturado al
    rango permitido, pan dentro de los enteros).
    """
    # Lazy import: estos tipos solo son necesarios cuando el helper se
    # usa, y manteniéndolos lazy mantenemos `controller.py` sin un
    # acoplamiento estructural con el render que no es estrictamente
    # necesario.
    from droneplan_viz.render import locate_entity, supersample_factor

    # Resolución del zoom: kwarg explícito o el actual del state.
    if zoom is not None:
        # Saturar igual que on_zoom para mantener el invariante de
        # AppState. Render satura también internamente pero queremos
        # state.validate() consistente.
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        state.zoom = zoom

    lx, ly = locate_entity(world, entity_id, surface_size, theme=theme)
    ss = supersample_factor(world, surface_size, theme)
    _apply_focus_pan(state, lx, ly, surface_size, ss)


def _apply_focus_pan(
    state: AppState,
    lx: float,
    ly: float,
    surface_size: tuple[int, int],
    ss: float,
    *,
    ease: bool = False,
) -> None:
    """Fija state.pan para centrar el punto lógico (lx, ly) en la surface.

    Fórmula de D2 (verificada en demo_camera.py): el render aplica el zoom
    expandiendo desde el centro de la surface y luego suma el pan, así que
    para que (lx, ly) acabe en el centro:

        pan = centro - (centro + (l - centro) · z_int)

    `state.zoom` es la escala ABSOLUTA del sprite (relativa al nativo); el zoom
    INTERNO que aplica el render sobre el buffer (de tamaño viewport×ss) es
    z_int = zoom · ss. (lx, ly) viene de locate_* en coords de vista completa
    (÷ss), así que el pan se calcula con z_int. Para ss=1 (escena sin
    supersampling) z_int = zoom y la fórmula coincide con la histórica.

    ease=False (por defecto): salta directamente al pan-objetivo (comportamiento
    instantáneo del focus one-shot). ease=True (seguimiento continuo): relaja el
    acumulador float state.follow_pan hacia el objetivo con easing exponencial y
    redondea UNA vez a state.pan, para que la cámara no vaya "a tirones" a
    velocidades sub-píxel. El objetivo se calcula en coma flotante y solo se
    redondea al final (una única cuantización, con round(), no int()).
    """
    z = state.zoom * ss
    sw, sh = surface_size
    target_x = sw / 2 - (sw / 2 + (lx - sw / 2) * z)
    target_y = sh / 2 - (sh / 2 + (ly - sh / 2) * z)
    if ease and state.follow_pan is not None:
        fx, fy = state.follow_pan
        fx += (target_x - fx) * _FOLLOW_SMOOTHING
        fy += (target_y - fy) * _FOLLOW_SMOOTHING
    else:
        # Primer frame de seguimiento (acumulador sin inicializar) o focus
        # one-shot: salta al objetivo sin suavizar (no hay lag de arranque).
        fx, fy = target_x, target_y
    state.follow_pan = (fx, fy)
    state.pan = (round(fx), round(fy))


def focus_on_entity_frame(
    state: AppState,
    snap_a,
    snap_b,
    progress: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
    zoom: float | None = None,
    prev_action=None,
    next_action=None,
) -> None:
    """Centra la cámara sobre la posición INTERPOLADA de la entidad.

    Variante de focus_on_entity para frames en transición: usa
    locate_entity_interpolated(snap_a, snap_b, progress, ...) en vez de la
    posición estática del snapshot, de modo que la cámara acompaña a un
    drone en vuelo (no apunta a su nodo de salida). Es la función que la
    app llama cada frame mientras state.followed_entity no sea None.

    snap_prev/snap_next-equivalente: prev_action/next_action son los pares
    (snap_a, snap_b) de la acción real anterior/siguiente (sin los Static
    enter-interacting), para que el seguimiento espeje el encadenado de la
    coreografía sin desfase respecto al dibujo.

    Mismo contrato de errores que focus_on_entity: KeyError (id inexistente)
    y ValueError (package no-AtLocation) se propagan; el llamante decide
    si cancelar el seguimiento.
    """
    from droneplan_viz.render import (
        locate_entity_interpolated,
        supersample_factor,
    )

    if zoom is not None:
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        state.zoom = zoom
    lx, ly = locate_entity_interpolated(
        snap_a, snap_b, progress, entity_id, surface_size, theme=theme,
        prev_action=prev_action, next_action=next_action,
    )
    ss = supersample_factor(snap_a.world, surface_size, theme)
    _apply_focus_pan(state, lx, ly, surface_size, ss, ease=True)


def focus_on_entity_at(
    state: AppState,
    timeline,
    t: float,
    entity_id: str,
    surface_size: tuple[int, int],
    *,
    theme: Theme | None = None,
    zoom: float | None = None,
) -> None:
    """Centra la cámara sobre la entidad seguida bajo el render por entidad.

    Variante de focus_on_entity_frame para render_world_at: localiza la
    posición de la entidad en el tiempo virtual `t` con locate_entity_at
    (mismos segmentos por drone que el dibujo), de modo que el seguimiento no
    se desfasa respecto a lo que se ve. Mismo contrato de errores (KeyError /
    ValueError se propagan; el llamante decide si cancelar el seguimiento).
    """
    from droneplan_viz.render import locate_entity_at, supersample_factor

    if zoom is not None:
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        state.zoom = zoom
    lx, ly = locate_entity_at(timeline, t, entity_id, surface_size, theme=theme)
    base, _b, _p = timeline.sample(t)
    ss = supersample_factor(base.world, surface_size, theme)
    _apply_focus_pan(state, lx, ly, surface_size, ss, ease=True)
