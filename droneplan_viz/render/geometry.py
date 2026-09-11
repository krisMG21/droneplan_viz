"""
Tipos y utilidades geométricas puras, sin dependencia de pygame.

Este módulo es la fundación matemática del paquete render. Todo lo que
hay aquí es testeable sin entorno gráfico, lo que se traduce en tests
rápidos y fiables.

Contenido:

- Point: par (x, y) inmutable. Es lo que devuelve el módulo layout para
  cada entidad posicionable, y lo que sprites.py consume para decidir
  dónde dibujar.
- lerp, lerp_point: interpolación lineal saneada. La saturación a [0, 1]
  hace que el render NO tenga que comprobar progress en los bordes;
  El progreso ya llega saturado como float, pero además
  blindamos aquí por defensa en profundidad.
- ease_in_out_cubic: easing por defecto para movimiento de drones. Más
  natural que lineal para entradas/salidas de cámara, y barato.
- ViewBox: rectángulo lógico con utilidades de mapeo a Surface. Permite
  desacoplar coordenadas semánticas (Location.position_screen, layout
  circular) del tamaño real de la Surface que el caller entrega. La
  Surface decide el tamaño; ViewBox decide cómo encaja el world dentro.

Decisiones explícitas:

1. Point es frozen+slots, paralelo al resto del dominio.
   __add__ y __sub__ trabajan componente a componente; son útiles en
   layout para componer offsets (centro de loc + desplazamiento del
   drone n-ésimo dentro de la loc).

2. lerp(a, b, t) satura t a [0, 1] internamente. Aunque el progreso
   ya devuelve progress saturado, blindamos aquí: si el render se
   llamara directamente con progress=1.5 (caso patológico), no romperíamos.

3. ease_in_out_cubic implementa la fórmula estándar de Smootherstep
   adaptada: 4t³ para t<0.5, 1 - 4(1-t)³ para t≥0.5. Continuidad C1 en
   0.5 garantizada. f(0)=0, f(1)=1, f(0.5)=0.5, derivada nula en los
   extremos. Aspecto natural en pantalla.

4. ViewBox.inset reduce TODOS los bordes en la misma cantidad. Si en el
   futuro hace falta padding asimétrico, se añade. YAGNI por ahora.

5. ViewBox.fit_into(surface_size) calcula un sub-viewbox dentro de una
   surface real, preservando el aspect ratio del viewbox lógico. Esto
   es lo que permite que un layout circular de radio 1.0 (normalizado)
   se mapee a CUALQUIER Surface sin distorsión.
"""
from __future__ import annotations

import math

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """Punto 2D inmutable. Coords en float para tolerar interpolación.

    Las coordenadas pueden venir de:
    - Location.position_screen (ints) → se castean a float al crear el Point.
    - Layout circular automático → ya son floats.
    - lerp_point durante interpolación → floats.

    La conversión a int para pygame.draw se hace en sprites.py al
    invocar las primitivas. Aquí mantenemos float para precisión.
    """

    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "Point":
        """Escalado uniforme desde el origen."""
        return Point(self.x * factor, self.y * factor)

    def as_int_tuple(self) -> tuple[int, int]:
        """Conversión a (int, int) para pygame.draw, que exige ints."""
        return (int(round(self.x)), int(round(self.y)))


def lerp(a: float, b: float, t: float) -> float:
    """Interpolación lineal entre a y b parametrizada por t.

    Satura t a [0, 1] antes de interpolar. Esto significa:
        lerp(0, 10, -0.5) == 0    (t saturado a 0)
        lerp(0, 10, 0.0)  == 0
        lerp(0, 10, 0.5)  == 5
        lerp(0, 10, 1.0)  == 10
        lerp(0, 10, 1.5)  == 10   (t saturado a 1)

    Defensa en profundidad: el progreso ya llega saturado, pero
    aquí el saneo es local y permite que cualquier caller que pase un
    progress fuera de rango por error no rompa visualmente.
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return a + (b - a) * t


def lerp_point(a: Point, b: Point, t: float) -> Point:
    """Interpolación lineal componente a componente entre dos Point.

    Equivale a Point(lerp(a.x, b.x, t), lerp(a.y, b.y, t)). El saneo de
    t se hace una sola vez aquí para no duplicar la lógica.
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def ease_in_out_cubic(t: float) -> float:
    """Easing de entrada/salida, más lineal que un cúbico puro.

    Mezcla del cúbico estándar con la identidad lineal (BLEND=0.55 cúbico
    + 0.45 lineal). El cúbico puro arranca y frena muy despacio (f'(0)=
    f'(1)=0), lo que daba una sensación demasiado "elástica" en el Move.
    Mezclando con lineal se conserva la suavidad en los extremos pero la
    velocidad en el tramo medio es más constante (movimiento más lineal,
    sin llegar a serlo del todo): aceleraciones algo mayores, como pidió
    el usuario.

    Propiedades:
        f(0) = 0, f(0.5) = 0.5, f(1) = 1 (se mantienen).
        f'(0) y f'(1) ya no son 0 (arranque/frenada menos extremos).
        Continuidad C1 en t=0.5.

    Saneo: t fuera de [0, 1] se satura.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    # Cúbico estándar (smootherstep de entrada/salida).
    if t < 0.5:
        cubic = 4.0 * t * t * t
    else:
        u = 1.0 - t
        cubic = 1.0 - 4.0 * u * u * u
    # Mezcla con lineal: más velocidad constante en el medio.
    BLEND = 0.55  # peso del cúbico; (1-BLEND) es el peso lineal
    return BLEND * cubic + (1.0 - BLEND) * t


def tilt_envelope(t: float, steepness: float = 7.0) -> float:
    """Envoltura (0→1→0) para la inclinación lateral del dron en un Move.

    Sustituye al antiguo `sin(π·t)`. Aunque el seno es suave como función,
    su PENDIENTE en los extremos es ±π (no nula): la inclinación arrancaba
    y terminaba con un "tirón" perceptible respecto al estado de reposo
    (tilt constante 0). Esta envoltura, construida con tanh (la sigmoide
    que pediste), tiene pendiente ~0 en t=0 y t=1, de modo que el dron
    se inclina y se endereza de forma gradual: entra al balanceo, lo
    mantiene en el crucero y sale de él suavemente antes de llegar.

    Construcción: diferencia de dos tanh centradas en 1/4 y 3/4 del
    trayecto (una sube, otra baja), normalizada para que:

        f(0) = 0,  f(0.5) = 1,  f(1) = 0   (exactos, por simetría)

    `steepness` (k) controla lo marcada que es la rampa: valores altos →
    el balanceo sube antes y se sostiene más plano (más "meseta"); valores
    bajos → forma más redondeada, parecida al seno. 7.0 da una entrada/
    salida claramente más suave que el seno conservando un pico nítido.

    Saneo: t fuera de [0, 1] se satura (devuelve 0 en los bordes).
    """
    if t <= 0.0 or t >= 1.0:
        return 0.0
    k = steepness
    a, b = 0.25, 0.75  # centros de las dos sigmoides (simétricos en 0.5)

    def _raw(x: float) -> float:
        return math.tanh(k * (x - a)) - math.tanh(k * (x - b))

    raw0 = _raw(0.0)            # = _raw(1.0) por simetría
    peak = _raw(0.5)           # valor máximo (en el centro)
    denom = peak - raw0
    if denom <= 1e-12:         # defensivo: k degenerado → sin envoltura
        return 0.0
    return (_raw(t) - raw0) / denom


def lateral_tilt_deg(
    dx: float,
    dy: float,
    progress: float,
    max_deg: float,
    steepness: float,
    scale: float = 0.5,
) -> float:
    """Inclinación lateral del dron en grados, PROPORCIONAL a lo horizontal
    que sea el movimiento.

    Antes la magnitud era constante (sign · max_deg · scale · envelope): un
    movimiento casi vertical se inclinaba lo mismo que uno horizontal. Aquí
    la escalamos por la FRACCIÓN HORIZONTAL del desplazamiento:

        horiz = |dx| / hypot(dx, dy)   ∈ [0, 1]

    es decir, el coseno del ángulo del vector respecto a la horizontal de
    pantalla: 1 si el movimiento es puramente horizontal (inclinación
    máxima), 0 si es puramente vertical (sin inclinación), √½≈0.71 a 45°.
    Así un drone que sube/baja casi en vertical apenas se inclina y uno que
    cruza en horizontal se inclina del todo, como pediste.

    El signo lo da dx (convención: +dx → inclinación horaria). La forma
    temporal la da geometry.tilt_envelope (tanh) sobre `progress`.

    Args:
        dx, dy: componentes EN PANTALLA del desplazamiento (origen→destino).
            dy crece hacia abajo, pero solo usamos su magnitud para la
            normalización, así que el sentido vertical es indiferente.
        progress: avance del movimiento en [0, 1] (raw, no eased).
        max_deg: pico de inclinación de referencia (theme.drone_move_max_tilt_deg).
        steepness: k de la envoltura tanh (theme.drone_tilt_envelope_steepness).
        scale: factor adicional sobre el pico (0.5 por defecto → 6° con
            max_deg=12°, el valor que ya se venía usando para no romper
            la lectura del sprite pixel-art).

    Returns:
        Ángulo en grados (con signo). 0.0 si no hay desplazamiento o si el
        movimiento es puramente vertical.
    """
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return 0.0
    horiz_frac = abs(dx) / dist
    if horiz_frac <= 1e-9:
        return 0.0
    sign = 1.0 if dx > 0 else -1.0
    env = tilt_envelope(max(0.0, min(1.0, progress)), steepness)
    return sign * max_deg * scale * env * horiz_frac


@dataclass(frozen=True, slots=True)
class ViewBox:
    """Rectángulo lógico para mapear coordenadas del world a la Surface.

    Pensado para vivir como dato puro durante el render: el painter
    calcula UN viewbox a partir del Surface y del layout, y se lo pasa
    a sprites.py.

    Atributos:
        left, top: esquina superior izquierda del viewbox en coords
            del destino (típicamente píxeles de la Surface).
        width, height: dimensiones del viewbox en las mismas coords.

    No almacenamos la Surface ni su tamaño: el viewbox es geometría
    pura. La conexión con la Surface real la hace fit_into() como
    constructor.
    """

    left: float
    top: float
    width: float
    height: float

    @classmethod
    def fit_into(
        cls,
        surface_size: tuple[int, int],
        padding: int = 0,
    ) -> "ViewBox":
        """Construye un ViewBox que ocupa la Surface entera menos un padding.

        Args:
            surface_size: (width, height) de la Surface destino, en píxeles.
            padding: margen uniforme en píxeles. 0 = ocupar toda la Surface.

        Returns:
            ViewBox con esquina (padding, padding) y dimensiones reducidas
            en 2*padding. Si el padding excede la mitad de alguna dimensión,
            el viewbox resultante puede tener width/height <= 0; los
            consumidores son responsables de manejarlo (típicamente
            haciendo nada).
        """
        w, h = surface_size
        return cls(
            left=float(padding),
            top=float(padding),
            width=float(w - 2 * padding),
            height=float(h - 2 * padding),
        )

    @property
    def right(self) -> float:
        """Coord x del borde derecho. Equivale a left + width."""
        return self.left + self.width

    @property
    def bottom(self) -> float:
        """Coord y del borde inferior. Equivale a top + height."""
        return self.top + self.height

    @property
    def center(self) -> Point:
        """Centro geométrico del viewbox como Point."""
        return Point(self.left + self.width / 2.0, self.top + self.height / 2.0)

    def inset(self, amount: float) -> "ViewBox":
        """Devuelve un ViewBox reducido uniformemente en `amount` por cada lado.

        Útil para garantizar que entidades de tamaño finito (drones,
        locations) caben enteras dentro del viewbox sin recortarse.
        Inset uniforme: si en el futuro hace falta asimétrico, se añade
        otro método.
        """
        return ViewBox(
            left=self.left + amount,
            top=self.top + amount,
            width=self.width - 2 * amount,
            height=self.height - 2 * amount,
        )

    def contains(self, p: Point) -> bool:
        """¿El Point p cae dentro del viewbox (bordes incluidos)?

        Cerrado en los cuatro lados. Útil para asserts en tests del
        layout.
        """
        return (
            self.left <= p.x <= self.right
            and self.top <= p.y <= self.bottom
        )
