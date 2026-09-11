"""Estado mutable acotado de la aplicación interactiva.

Esta es la ÚNICA fuente de mutación de la UI. Las funciones del paquete
controller.py reciben AppState y mutan sus campos in situ (no devuelven
copias: AppState es mutable por diseño). El módulo hud.py LEE AppState
para decidir qué dibujar, nunca lo muta.

Mantener esa frontera es el equivalente, para la UI, de la inmutabilidad
estricta del dominio: hay un único lugar donde el estado cambia, lo cual
hace la depuración y los tests tratables.

Campos del estado:

- playback_time : float ≥ 0. Posición en el eje VIRTUAL de Timeline,
                  no en el tiempo real de la simulación. Lo consume
                  Timeline.sample() en cada frame.
- playback_speed: float ∈ ALLOWED_SPEEDS. Multiplicador del avance
                  automático cuando no está pausado.
- paused        : bool. Si True, tick() no avanza playback_time.
- window_size   : (int, int). Tamaño actual de la ventana. Se actualiza
                  en pygame.VIDEORESIZE.
- selected_failure: int | None. Índice del fallo seleccionado en el
                  panel lateral (si el usuario hizo click en uno),
                  para resaltarlo en la marca de la timeline. None
                  significa ninguno seleccionado.
- zoom          : float ∈ [ZOOM_MIN, ZOOM_MAX]. Factor de espaciado del
                  grafo en el render (NO escala los sprites, solo separa
                  o acerca las localizaciones). Lo consume render_frame
                  vía el kwarg zoom. 1.0 = vista normal.
- pan           : (int, int). Desplazamiento en píxeles de la imagen
                  renderizada, aplicado DESPUÉS del zoom. (0, 0) = vista
                  centrada. Lo consume render_frame vía el kwarg pan.
- inventory_collapsed: set[str]. Nombres de categorías del inventario
                  (DRONES, LOCATIONS, PERSONAS, PAQUETES, TRANSPORTERS)
                  que el usuario ha plegado. Una categoría plegada solo
                  muestra su cabecera; sus cajas se ocultan. Persiste a
                  través de redimensionados (vive en AppState, no en el
                  HUD local). Default: vacío = todas desplegadas.

Invariantes (validados por validate()):

- playback_time >= 0
- playback_speed ∈ ALLOWED_SPEEDS
- window_size[0] >= 1 and window_size[1] >= 1
- selected_failure is None or selected_failure >= 0
- ZOOM_MIN <= zoom <= ZOOM_MAX (los handlers del controller deben
  clampear; render satura defensivamente si recibe fuera de rango).
- pan es tupla de dos enteros.
- inventory_collapsed: set de strings (las categorías concretas no se
  validan estrictamente; cualquier string es legal pero solo las que
  coinciden con cabeceras reales del inventario tienen efecto visible).

Cota superior de playback_time: NO la valida AppState porque la duración
total la conoce Timeline, no AppState. El controller la clampea cuando
recibe un tick que se pasaría.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


#: Multiplicadores de velocidad permitidos para los botones de la UI.
#: Lista discreta consensuada (decisión 3.4 confirmada en Paso E.1):
#: cinco valores fijos en lugar de slider continuo.
#: Orden = orden en que se pintarán los botones de izquierda a derecha.
ALLOWED_SPEEDS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Velocidad por defecto al arrancar la aplicación: 1.0× (tiempo real).
DEFAULT_SPEED: Final[float] = 1.0

#: Tamaño por defecto de la ventana. Coincide con la propuesta de
#: arquitectura aceptada en Paso E.1 §2.3.
DEFAULT_WINDOW_SIZE: Final[tuple[int, int]] = (1280, 800)

#: Estado inicial de cámara cuando no hay plan cargado. El render con
#: zoom=DEFAULT_ZOOM (escala nativa 1.0) y pan=DEFAULT_PAN es PÍXEL-IDÉNTICO al
#: render sin esos kwargs para escenas que no necesitan supersampling (ss=1),
#: donde 1.0 = sprite nativo = vista completa. Al cargar un plan, la app fija el
#: zoom de ARRANQUE con `fitting_zoom(ss)` (ver abajo): el nivel de la rejilla
#: que hace entrar todo en pantalla.
DEFAULT_ZOOM: Final[float] = 1.0
DEFAULT_PAN: Final[tuple[int, int]] = (0, 0)

#: Paso multiplicativo entre niveles de zoom (= tick de rueda/atajos del
#: controller). 1.25 da una progresión cómoda.
ZOOM_STEP: Final[float] = 1.25

#: Salto temporal (segundos de tiempo virtual) del avance fotograma a
#: fotograma con las teclas ',' y '.'. Se fija en 1/60 para que coincida
#: con la cadencia de reproducción por defecto (fps_cap=60): un paso
#: equivale a un fotograma reproducido a velocidad 1x. A diferencia de
#: LEFT/RIGHT, que saltan entre snapshots del plan, este avance es de
#: grano fino y permite inspeccionar la interpolación entre snapshots.
FRAME_STEP: Final[float] = 1.0 / 60.0

#: REJILLA FIJA de zoom, definida como ESCALA ABSOLUTA del sprite respecto a su
#: resolución nativa (1.0 = píxel nativo 1:1), NO relativa a la vista de ajuste.
#: Es IDÉNTICA en toda ejecución y escena: un mismo nivel dibuja el sprite al
#: mismo tamaño en pantalla con 9 o con 100 locations (el supersampling solo
#: cambia la NITIDEZ del buffer, no la escala —ver painter._SUPERSAMPLE_MAX—).
#:
#: Niveles (1.25^k, k = -5..+4):
#:   0.3277  ← zoom-out máximo (= 1/SS_MAX: ajuste de las escenas más densas)
#:   0.4096, 0.512, 0.64, 0.8
#:   1.0     ← sprite a resolución NATIVA 1:1
#:   1.25, 1.5625, 1.9531
#:   2.4414  ← zoom-in máximo (~2.4× nativo, inspección de cerca)
#:
#: Al CARGAR, la app elige como zoom de arranque `fitting_zoom(ss)` = el mayor
#: nivel de la rejilla al que TODO entra en pantalla (el ajuste cae en 1/ss).
#: Para escenas densas eso es un nivel bajo de la lista (queda casi toda la
#: rejilla para hacer zoom-in); para escenas dispersas, hacia el nativo. La
#: LISTA no varía: solo varía qué nivel se toma por defecto.
#:
#: Este es el rango INTERACTIVO; es un subconjunto del clamp del render
#: (painter.ZOOM_MIN/MAX), también en escala absoluta.
ZOOM_MIN: Final[float] = ZOOM_STEP ** -5      # 0.32768  (zoom-out máximo)
ZOOM_MAX: Final[float] = ZOOM_STEP ** 4       # 2.44140625 (zoom-in máximo)

#: Lista explícita de niveles (ordenada), escala absoluta. Útil para UI y tests.
ZOOM_LEVELS: Final[tuple[float, ...]] = tuple(
    ZOOM_STEP ** k for k in range(-5, 5)
)


def fitting_zoom(ss: float) -> float:
    """Zoom de arranque para una escena con factor de supersampling `ss`.

    Devuelve el MAYOR nivel de la rejilla (escala absoluta) al que la escena
    entra entera en pantalla. La vista completa ocurre a escala 1/ss (a ese
    zoom el buffer nativo, de tamaño viewport×ss, se reduce justo al viewport),
    así que se toma el mayor nivel ≤ 1/ss. Si 1/ss cae por debajo del mínimo
    (escena densísima), se satura a ZOOM_MIN. Determinista.
    """
    target = 1.0 / ss if ss > 0 else 1.0
    best = ZOOM_MIN
    for level in ZOOM_LEVELS:
        if level <= target + 1e-9 and level > best:
            best = level
    return best


#: Re-exportamos los límites/niveles para que los handlers del controller
#: y los tests del paquete app accedan a ellos sin duplicar constantes.
__all__ = [
    "ALLOWED_SPEEDS",
    "FRAME_STEP",
    "DEFAULT_SPEED",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_ZOOM",
    "DEFAULT_PAN",
    "ZOOM_STEP",
    "ZOOM_MIN",
    "ZOOM_MAX",
    "ZOOM_LEVELS",
    "fitting_zoom",
    "AppState",
]


@dataclass(slots=True)
class AppState:
    """Estado mutable acotado de la UI interactiva.

    NO frozen: se muta in situ desde controller.py. El uso de slots
    evita typos accidentales (asignar a un campo que no existe lanza
    AttributeError) y reduce ligeramente el coste de memoria.

    Defaults sensatos para arrancar la app sin parámetros: pausado en
    t=0 con velocidad 1× y la ventana al tamaño consensuado. La app
    decide si arranca pausada (defendible — el usuario lee primero el
    HUD) o reproduciendo (más vivo). Por ahora arranca PAUSADA: que el
    usuario pulse Play deja la primera impresión más controlada y evita
    que el plan termine antes de que el usuario lea el panel de
    métricas.
    """
    playback_time: float = 0.0
    playback_speed: float = DEFAULT_SPEED
    paused: bool = True
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
    selected_failure: int | None = None
    zoom: float = DEFAULT_ZOOM
    pan: tuple[int, int] = DEFAULT_PAN
    # Overlay de ayuda visible (lista de controles). Se alterna con H y se
    # cierra con H o Escape. Es estado de presentación puro (no afecta al
    # modelo ni a la reproducción).
    show_help: bool = False
    # Ocultar manualmente las etiquetas de nombre de las entidades. Se
    # alterna con L. Cuando es True, las etiquetas no se dibujan a ningún
    # zoom; cuando es False, siguen el criterio automático por zoom (se
    # ocultan al alejar la cámara). Presentación pura.
    labels_hidden: bool = False
    # Zoom de ARRANQUE de la escena actual (escala absoluta): el nivel de la
    # rejilla al que todo entra en pantalla, fijado por la app al cargar con
    # fitting_zoom(ss). El reset de cámara vuelve aquí (no a un 1.0 fijo, que
    # en una escena densa estaría muy acercado). Default 1.0 = nativo (ss=1).
    home_zoom: float = DEFAULT_ZOOM
    # Set de categorías del inventario plegadas. default_factory es
    # obligatorio para colecciones mutables en dataclasses: usar
    # `set()` como default directo compartiría la misma instancia entre
    # todas las AppState (bug clásico de Python).
    inventory_collapsed: set[str] = field(default_factory=set)
    # Entidad que la cámara SIGUE de forma continua. Cuando no es None, la
    # app recentra la cámara cada frame sobre la posición INTERPOLADA de
    # esta entidad (vía focus_on_entity_frame), de modo que acompaña a un
    # drone en vuelo en vez de quedarse en su nodo de salida. Se activa al
    # hacer focus desde el inventario y se cancela con pan manual o reset
    # de cámara. None = cámara libre (comportamiento previo).
    followed_entity: str | None = None
    # Acumulador FLOAT interno del suavizado de cámara durante el seguimiento.
    # state.pan es (int, int) por contrato (lo blittea el render en píxeles
    # enteros), pero un seguimiento que salta directamente al pan-objetivo cada
    # frame se ve "a tirones" a velocidades sub-píxel. Este acumulador mantiene
    # la posición de cámara en coma flotante y se relaja hacia el objetivo
    # (easing exponencial), redondeándose a state.pan solo al final. None = sin
    # inicializar: el próximo frame de seguimiento salta al objetivo (sin lag de
    # arranque). Lo resetean a None el pan manual y el reset de cámara.
    follow_pan: tuple[float, float] | None = None

    def validate(self) -> None:
        """Comprueba los invariantes documentados arriba.

        Llamar a este método NO es obligatorio en el flujo normal del
        controller (las transiciones del controller mantienen los
        invariantes por construcción). Existe para:

          1. Los tests: pueden invocarlo tras una secuencia de
             mutaciones para detectar regresiones.
          2. Debugging: si algo va mal, una llamada manual aquí lanza
             un AssertionError descriptivo.

        Raises:
            AssertionError con mensaje explicativo si algún invariante
            se viola.
        """
        assert self.playback_time >= 0.0, (
            f"playback_time debe ser >= 0, es {self.playback_time}"
        )
        assert self.playback_speed in ALLOWED_SPEEDS, (
            f"playback_speed debe estar en {ALLOWED_SPEEDS}, "
            f"es {self.playback_speed}"
        )
        assert (
            isinstance(self.window_size, tuple)
            and len(self.window_size) == 2
            and self.window_size[0] >= 1
            and self.window_size[1] >= 1
        ), f"window_size inválido: {self.window_size!r}"
        assert (
            self.selected_failure is None or self.selected_failure >= 0
        ), f"selected_failure debe ser None o entero >= 0, es {self.selected_failure}"
        assert ZOOM_MIN <= self.zoom <= ZOOM_MAX, (
            f"zoom debe estar en [{ZOOM_MIN}, {ZOOM_MAX}], es {self.zoom}"
        )
        assert ZOOM_MIN <= self.home_zoom <= ZOOM_MAX, (
            f"home_zoom debe estar en [{ZOOM_MIN}, {ZOOM_MAX}], es {self.home_zoom}"
        )
        assert (
            isinstance(self.pan, tuple)
            and len(self.pan) == 2
            and isinstance(self.pan[0], int)
            and isinstance(self.pan[1], int)
        ), f"pan debe ser tupla (int, int), es {self.pan!r}"
        assert isinstance(self.inventory_collapsed, set), (
            f"inventory_collapsed debe ser set, es {type(self.inventory_collapsed).__name__}"
        )
        assert all(isinstance(c, str) for c in self.inventory_collapsed), (
            f"inventory_collapsed debe contener solo strings, "
            f"contiene {self.inventory_collapsed!r}"
        )
        assert self.followed_entity is None or isinstance(self.followed_entity, str), (
            f"followed_entity debe ser None o str, es {self.followed_entity!r}"
        )