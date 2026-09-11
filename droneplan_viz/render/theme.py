"""
Theme: paleta, dimensiones y tipografía del render.

Este módulo centraliza TODAS las constantes visuales del paquete render.
Cero colores hardcodeados fuera de aquí. Cero tamaños hardcodeados fuera
de aquí (excepto sub-detalles puramente geométricos como "el grosor del
borde es theme.border_width", que también vive aquí).

Filosofía:

- Una sola dataclass frozen+slots con todos los parámetros visuales.
- Factoría Theme.default() con la paleta elegida.
- Método with_overrides(**kwargs) para producir variantes sin tocar el
  default (delegado a dataclasses.replace).
- Funciones puras color_for_*() que aplican las políticas (heurística
  de Location por id, lookup de Content por id, etc.).

Decisiones documentadas:

1. Las políticas de coloreado de Locations y Contents son booleanos o
   estrategias enumeradas, NO callbacks. Esto se hace por dos razones:
   - El Theme es frozen+slots; un callback sería difícil de serializar.
   - Para el alcance docente, las políticas predeterminadas bastan.
   Si en el futuro se requiere personalización fina, se cambia el tipo
   del campo a Callable y se aceptan callbacks. YAGNI por ahora.

2. La diferenciación de Locations es OPT-OUT (True por defecto). Razón
   discutida en la propuesta: el TFG querrá poder mostrarlo desactivado
   ("aquí está sin asunciones semánticas sobre los ids"), y dejar la
   puerta abierta a ids no españoles en sesiones futuras.

3. El color de Contents usa estrategia híbrida:
   - Diccionario fijo KNOWN_CONTENT_COLORS para los contenidos canónicos
     del PDF docente (medicina, comida, agua, con sus plurales).
   - Fallback determinista via zlib.adler32(id) mod len(palette) para
     cualquier content_id desconocido. Determinista entre procesos
     (a diferencia de hash() de Python, que tiene PYTHONHASHSEED random).

4. fallback_step_duration = 0.6s para PDDL parte 1-2 (duration=0).
   Decisión elegida: deja ver mejor cada paso sin
   romper la fluidez. Para parte 3 (duration > 0), se respeta el tiempo
   simulado real.

5. La fuente es SysFont(None, size) por defecto. None deja que pygame
   elija la fuente del sistema más razonable, evitando tener que
   distribuir un .ttf con el ZIP. Si en el futuro se quiere una fuente
   concreta (Inter, Roboto), se pone font_name="Inter" y pygame intenta
   resolverlo; si no la encuentra, cae a la del sistema.

6. NO hay caché de fuentes en Theme. SysFont se llama al vuelo cuando
   sprites.py necesita renderizar texto. pygame.font.SysFont es
   internamente eficiente y cachea. Si en testing observamos overhead
   real, se añade un cache local sin tocar la API.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

# Tipos cortos para legibilidad
RGB = tuple[int, int, int]
ContentColorStrategy = Literal["hybrid", "fallback_only", "id_only"]


# ---------------------------------------------------------------------------
# Paleta base — colores fijos del Theme.default()
# ---------------------------------------------------------------------------

# Fondo del world. Gris pizarra oscuro, suficientemente neutro para que
# los acentos de color (drones, paquetes) destaquen sin saturar.
_BACKGROUND: RGB = (28, 32, 38)

# Aristas del grafo (líneas entre Locations donde existe coste). Tono
# apagado para no competir con las entidades, pero visible sobre el fondo.
_EDGE: RGB = (70, 78, 92)

# Locations:
_LOC_FILL_NEUTRAL: RGB = (104, 106, 110)     # gris neutro (como location.png)
_LOC_BORDER_NEUTRAL: RGB = (58, 60, 64)
# Heurística semántica (sólo si differentiate_locations=True):
_LOC_FILL_HOUSE: RGB = (180, 145, 100)       # casas: tono tierra cálido
_LOC_FILL_HOSPITAL: RGB = (220, 220, 230)    # hospitales: blanco
_LOC_BORDER_HOSPITAL: RGB = (200, 80, 80)    # con borde rojo (cruz roja)
_LOC_FILL_DEPOT: RGB = (80, 140, 200)        # depósitos: azul

# Drones por estado FSM:
_DRONE_IDLE: RGB = (240, 144, 48)            # naranja (color del sprite drone.png)
_DRONE_MOVING: RGB = (240, 144, 48)          # naranja (el estado lo marca la cara, no el color)
_DRONE_INTERACTING: RGB = (240, 144, 48)     # naranja
_DRONE_ERROR: RGB = (214, 78, 66)            # rojo (estado ERROR distinto + X blanca)
_DRONE_BORDER: RGB = (20, 25, 30)            # contorno oscuro común
_ERROR_X: RGB = (255, 255, 255)              # X blanca sobre rojo

# Transportador:
_TRANSPORTER_FILL: RGB = (142, 166, 166)     # gris-verdoso (como carrier.png)
_TRANSPORTER_BORDER: RGB = (72, 84, 86)

# Persona y sus needs:
_PERSON_FILL: RGB = (86, 122, 190)           # azul (como person2.png; contrasta con etiqueta blanca)
_PERSON_BORDER: RGB = (38, 54, 92)
_NEEDS_INDICATOR: RGB = (220, 80, 70)        # anillo rojo si tiene needs

# Texto e IDs estructurales (los pinta el render como anotaciones discretas;
# el HUD completo es responsabilidad de la UI envolvente):
_TEXT: RGB = (225, 230, 240)
_TEXT_DIM: RGB = (140, 145, 155)             # versión apagada (debug, ids)

# Contenidos canónicos del PDDL docente (medicina, comida, agua). Se
# duplican plural y singular porque las descomposiciones SHOP2 del PDF
# usan "lleva-medicinas" / "lleva-comidas" en plural.
KNOWN_CONTENT_COLORS: dict[str, RGB] = {
    "medicina":  (240, 130, 130),  # rojo claro
    "medicinas": (240, 130, 130),
    "comida":    (170, 200, 100),  # verde lima
    "comidas":   (170, 200, 100),
    "agua":      (130, 180, 230),  # azul claro
    "aguas":     (130, 180, 230),
}

# Paleta de fallback para contenidos no canónicos. 8 colores
# suficientemente distinguibles, ordenados para que cualquier mezcla
# razonable produzca un buen contraste. La selección se hace via
# zlib.adler32(content_id) % 8.
FALLBACK_CONTENT_PALETTE: tuple[RGB, ...] = (
    (200, 160, 100),  # ocre
    (180, 130, 200),  # malva
    (100, 200, 180),  # turquesa
    (220, 180, 120),  # mostaza
    (160, 160, 220),  # lavanda
    (200, 200, 130),  # amarillo apagado
    (180, 100, 140),  # rosa oscuro
    (130, 180, 140),  # verde apagado
)


# ---------------------------------------------------------------------------
# Mapeo de sprites por defecto (clave lógica -> filename)
# ---------------------------------------------------------------------------
#
# Nombres de archivo que el SpriteManager busca en el directorio de assets.
# Declararlos aquí (no hardcodearlos en el manager ni en sprites.py)
# mantiene la regla "cero rutas/nombres fuera del theme".
#
# MODELO DE CAPAS (rediseño): el dron ya NO se dibuja como un sprite entero
# por estado FSM. Se COMPONE apilando capas independientes:
#
#   - cuerpo: "drone" (base, 58x58, brazos recogidos, pantalla sin cara) o
#     "drone_interacting" (58x64, brazos extendidos, cara ya incrustada).
#   - cara/expresión (solo sobre el cuerpo base): se elige por estado y, en
#     movimiento, por dirección cuantizada del vector de desplazamiento.
#   - objeto agarrado (excluyente): "box" (paquete suelto) o "carrier_N"
#     (carrier con su nivel de llenado, 6 niveles).
#
# Esto convierte la explosión multiplicativa de PNG (dirección x objeto x
# llenado x expresión = cientos) en una suma aditiva (decenas).
#
# El resto de entidades (package, transporter, person, location) siguen el
# modelo simple de sprite base, igual que antes.
_DEFAULT_SPRITE_FILES: Mapping[str, str] = {
    # --- Dron: capas componibles ---
    # Cuerpos
    "drone": "drone.png",                       # base, sin cara incrustada
    "drone_interacting": "drone_interacting.png",  # brazos extendidos, cara propia
    # Caras de movimiento (direccionales, 6 sectores)
    "face_N": "face_N.png",
    "face_S": "face_S.png",
    "face_NE": "face_NE.png",
    "face_NW": "face_NW.png",
    "face_SE": "face_SE.png",
    "face_SW": "face_SW.png",
    # Caras de estado (sin dirección)
    "face": "face.png",                # reposo a la espera de otra acción
    "face_idle": "face_idle.png",      # ejecución terminada (ojos cerrados)
    "face_error1": "face_error1.png",  # error (signo de exclamación)
    "face_error2": "face_error2.png",  # warning/intermitencia (reservada)
    "face_talking1": "face_talking1.png",  # reservadas para mensajes futuros
    "face_talking2": "face_talking2.png",
    "face_talking3": "face_talking3.png",
    # Objeto agarrado
    "box": "box.png",
    # Carrier por nivel de llenado: carrier.png=0%, carrier1..5=20..100%.
    "carrier_0": "carrier.png",
    "carrier_1": "carrier1.png",
    "carrier_2": "carrier2.png",
    "carrier_3": "carrier3.png",
    "carrier_4": "carrier4.png",
    "carrier_5": "carrier5.png",
    # --- Resto de entidades: sprite base simple ---
    "package": "package.png",
    "transporter": "transporter.png",
    "person": "person.png",
    "location": "location.png",
}


# ---------------------------------------------------------------------------
# Anclas de composición de sprites del dron
# ---------------------------------------------------------------------------
#
# Todos los offsets en píxeles a RESOLUCIÓN NATIVA (antes de escalar). El
# usuario que dibuja los assets ajusta estos números sin tocar lógica de
# composición. Verificados contra los PNG reales entregados.
#
# Sistema de coordenadas: el origen del lienzo compuesto es su esquina
# superior izquierda (0,0). El cuerpo del dron se pega arriba (su techo en
# y=0). Todo lo que crece (brazos extendidos del cuerpo interacting,
# carrier) crece HACIA ABAJO, por lo que el dron nunca se desplaza
# verticalmente al añadir capas inferiores.
def _si(v: int, s: float) -> int:
    """Escala un tamaño/offset ENTERO por s, redondeando. Preserva el signo y
    una magnitud mínima de 1 para offsets/anchos no nulos (no anular líneas ni
    desplazamientos al reducir)."""
    if v == 0:
        return 0
    r = int(round(v * s))
    if r == 0:
        r = 1 if v > 0 else -1
    return r


def _st(t: tuple[int, int], s: float) -> tuple[int, int]:
    """Escala una tupla (w, h) entera por s."""
    return (_si(t[0], s), _si(t[1], s))


@dataclass(frozen=True, slots=True)
class SpriteAnchors:
    """Dimensiones nativas y anclas para componer el sprite del dron.

    Atributos de dimensión (px nativos):
        body_size: cuerpo base (brazos recogidos).
        body_interacting_size: cuerpo interactuando (brazos extendidos).
        face_size: capa de cara/expresión.
        box_size: caja (paquete suelto sostenido).
        carrier_size: carrier (cualquier nivel de llenado).
        canvas_size: lienzo compuesto, dimensionado para el peor caso
            (cuerpo + carrier colgando). 58x80.

    Atributos de ancla (offset desde el techo del cuerpo, X centrada salvo
    indicación):
        face_top: Y del techo de la cara (31px bajo el techo del cuerpo).
        carrier_overlap: cuántos px del techo del carrier quedan DENTRO del
            cuerpo (por encima de su borde inferior). 8px → el carrier
            sobresale carrier_h - 8 = 22px, dando canvas alto 58+22=80.
        registration_dx, registration_dy: desplazamiento desde el punto de
            layout (que el resto del sistema entiende como "centro del
            cuerpo del dron") hasta la esquina superior izquierda del
            lienzo compuesto. Permite colocar el compuesto en pantalla
            anclando por su top-left mientras el layout razona en centros.
            El centro del cuerpo base está en (29, 29) desde el top-left
            del lienzo, así que el offset es (-29, -29).
    """
    body_size: tuple[int, int] = (58, 58)
    body_interacting_size: tuple[int, int] = (58, 64)
    face_size: tuple[int, int] = (16, 8)
    box_size: tuple[int, int] = (14, 14)
    carrier_size: tuple[int, int] = (38, 30)
    canvas_size: tuple[int, int] = (58, 80)

    face_top: int = 31
    carrier_overlap: int = 8

    registration_dx: int = -29
    registration_dy: int = -29

    def scaled(self, s: float) -> "SpriteAnchors":
        """Copia con todas las dimensiones y anclas escaladas por s.

        s=1.0 devuelve self (idéntico). Todos los campos escalan por el MISMO
        factor, así que el composite del dron mantiene sus proporciones y su
        alineación relativa (con redondeo de ±1px, asumible en la vista
        reducida)."""
        if s >= 1.0:
            return self
        return SpriteAnchors(
            body_size=_st(self.body_size, s),
            body_interacting_size=_st(self.body_interacting_size, s),
            face_size=_st(self.face_size, s),
            box_size=_st(self.box_size, s),
            carrier_size=_st(self.carrier_size, s),
            canvas_size=_st(self.canvas_size, s),
            face_top=_si(self.face_top, s),
            carrier_overlap=_si(self.carrier_overlap, s),
            registration_dx=_si(self.registration_dx, s),
            registration_dy=_si(self.registration_dy, s),
        )


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Theme:
    """Configuración visual completa del render.

    Centraliza paleta, dimensiones y tipografía. Toda función de dibujo
    recibe (o tiene acceso indirecto a) una instancia de Theme para
    consultar colores y tamaños. Cero constantes visuales fuera de aquí.

    Construcción típica:
        theme = Theme.default()
        # o variantes:
        theme = Theme.default().with_overrides(padding=40, drone_radius=22)
    """

    # ---- Paleta ----------------------------------------------------------
    background: RGB = _BACKGROUND
    edge: RGB = _EDGE

    loc_fill_neutral: RGB = _LOC_FILL_NEUTRAL
    loc_border_neutral: RGB = _LOC_BORDER_NEUTRAL
    loc_fill_house: RGB = _LOC_FILL_HOUSE
    loc_fill_hospital: RGB = _LOC_FILL_HOSPITAL
    loc_border_hospital: RGB = _LOC_BORDER_HOSPITAL
    loc_fill_depot: RGB = _LOC_FILL_DEPOT

    drone_idle: RGB = _DRONE_IDLE
    drone_moving: RGB = _DRONE_MOVING
    drone_interacting: RGB = _DRONE_INTERACTING
    drone_error: RGB = _DRONE_ERROR
    drone_border: RGB = _DRONE_BORDER
    error_x: RGB = _ERROR_X

    transporter_fill: RGB = _TRANSPORTER_FILL
    transporter_border: RGB = _TRANSPORTER_BORDER

    person_fill: RGB = _PERSON_FILL
    person_border: RGB = _PERSON_BORDER
    needs_indicator: RGB = _NEEDS_INDICATOR

    text: RGB = _TEXT
    text_dim: RGB = _TEXT_DIM

    # ---- Dimensiones (todas en píxeles, base para una Surface típica
    #      ~ 800x600; el layout las usa como referencia y NO se escalan
    #      automáticamente con la Surface. Si la UI redimensiona muy
    #      lejos de la base, se sobreescriben aquí con with_overrides) ---
    #
    # NOTA (rediseño de sprites por capas): el dron se compone a resolución
    # nativa 58x58 y se dibuja 1:1 (sin reescalado) para pixel art perfecto.
    # Por eso drone_radius=29 (diámetro 58), y el resto de dimensiones del
    # layout se recalibraron proporcionalmente (factor ~1.6 respecto a los
    # valores previos de drones de 36px) para mantener las proporciones
    # relativas del grafo. La cámara/zoom (encargo posterior) permitirá ver
    # el grafo entero aunque ahora ocupe más superficie.
    drone_radius: int = 29          # diámetro 58 = tamaño nativo del sprite
    package_size: int = 14          # los paquetes son cuadrados; lado entero
    # Grosor (px) del outline de color por contenido alrededor de la caja.
    # Recupera la distinción por tipo de contenido en el suelo, que se pierde
    # cuando box.png (monocromo) se usa para todas las cajas. 0 = sin outline.
    package_outline_width: int = 2
    transporter_width: int = 48
    transporter_height: int = 30
    # location_radius: media-ancho del LIENZO del sprite de location en
    # pantalla. El sprite es isométrico (110x88 nativo) y se renderiza con
    # escala 2x → 220x176 en pantalla. location_radius=110 es media-ancho.
    # Se usa como "tamaño de referencia" en clearance, viewbox y layout;
    # NO implica que la location sea un círculo de ese radio.
    location_radius: int = 110
    person_radius: int = 18         # NO escala con la location: la persona
                                    # mantiene su tamaño relativo al dron.
    arm_radius: int = 5             # marcador del brazo del drone
    border_width: int = 2
    edge_width: int = 2             # grosor de aristas del grafo
    error_x_width: int = 3          # grosor de la X sobre drone en ERROR

    # ---- Anclajes del sprite isométrico de location ----------------------
    # El sprite de location (110x88 nativo) es isométrico: el "suelo hábil"
    # donde aterrizan drones y se apilan cajas es un ROMBO interior, no
    # coincide con el centro geométrico del lienzo. Estos parámetros
    # definen la zona hábil dentro del sprite, en píxeles DE PANTALLA
    # (tras la escala 2x).
    #
    #   - sprite nativo: 110x88. En pantalla (x2): 220x176.
    #   - rombo nativo:  80x52, a 13px del techo y 23px del suelo,
    #                    centrado en X. Centro del rombo en (55, 39)
    #                    nativo → 5px arriba del centro del lienzo (44).
    #   - en pantalla:   rombo 160x104, centro 10px arriba del centro
    #                    del lienzo.
    location_sprite_native_size: tuple[int, int] = (110, 88)
    location_sprite_scale: int = 2  # factor entero (nearest-neighbor limpio)
    #: La elipse de FALLBACK (sin sprite) se dibuja al lienzo × este factor,
    #: para dejar separación entre localizaciones vecinas (evita el solape).
    #: Adimensional: scaled() no lo toca.
    location_fallback_scale: float = 0.66
    # Offset del CENTRO LÓGICO (donde aterrizan drones, donde se apilan
    # cajas) respecto al CENTRO DEL LIENZO del sprite, en px DE PANTALLA.
    # Negativo = el centro lógico está más arriba que el centro del lienzo.
    location_ground_offset_y: int = -10
    # Dimensiones del rombo hábil en píxeles DE PANTALLA. Define el área
    # útil para clearance interno (co-localización, pirámide de cajas).
    location_ground_size: tuple[int, int] = (160, 104)

    # ---- Layout ----------------------------------------------------------
    padding: int = 24               # margen del world dentro de la Surface
    # Factor de SEPARACIÓN del layout circular (fallback cuando las locations
    # no traen position_screen): semieje de la elipse como fracción de la
    # dimensión del viewbox. El layout circular lo CAPA a 0.50 (las locations
    # llegan como mucho al borde de la región segura que reserva
    # painter._scene_viewbox), de modo que las decoraciones (dron flotante,
    # sprite/etiqueta) NUNCA se recortan, sea cual sea el tamaño de la surface.
    # 0.50 = separación máxima que cabe con esas decoraciones (~×1.25 respecto
    # al 0.40 histórico). Valores >0.50 se saturan (no aumentan la separación
    # pero tampoco recortan); valores <0.50 acercan las locations.
    location_spread_factor: float = 0.50
    # "Hover": elevación PERMANENTE del dron sobre el centro del rombo
    # mientras está en una location. Idea: los drones "vuelan
    # estacionario" cuando están en una loc, dejando el rombo entero
    # libre para cajas, carriers y otros objetos. Aplicado siempre, sea
    # 1 dron solo o varios co-localizados. Negativo = arriba en pantalla.
    drone_hover_y_offset: int = -50
    # Inclinación máxima (en grados) del compuesto del drone durante un
    # Move/MoveWithTransporter. Convención drone/avión: sentido horario
    # (positivo) al viajar hacia la derecha, antihorario (negativo) hacia
    # la izquierda. La inclinación sigue una curva campana sobre el
    # progress: 0 al inicio, máximo en progress=0.5, 0 al llegar. Si el
    # movimiento es puramente vertical (dx≈0), no hay inclinación. Sin
    # inclinación en hover, PickUp, Deliver, Load, Unload (la coreografía
    # de esas acciones no rota el sprite).
    drone_move_max_tilt_deg: float = 12.0
    # Pendiente (k) de la envoltura tanh que modula la inclinación a lo
    # largo del Move (geometry.tilt_envelope). Reemplaza al antiguo
    # sin(π·t): con tanh la inclinación entra y sale con pendiente ~0 en
    # los extremos (sin tirón). k alto → rampa más marcada y "meseta" más
    # plana en el crucero; k bajo → forma más redondeada (similar al seno).
    drone_tilt_envelope_steepness: float = 7.0
    # Radio del anillo de drones co-localizados. Cuando hay 2+ drones
    # en la misma loc, se distribuyen alrededor de un centro elevado
    # (drone_hover_y_offset + drone_colocation_y_lift).
    drone_colocation_offset: int = 30
    # Lift adicional cuando hay co-localización (sobre el hover). Pequeño
    # ahora que el hover hace la elevación principal. Negativo = arriba.
    drone_colocation_y_lift: int = -10

    # ---- Sombras de objetos ----------------------------------------------
    # Sombra ovalada plana, oscura y semitransparente, bajo objetos SUELTOS
    # apoyados en el suelo (transporters y cajas) para separarlos del sprite
    # de location y darles solidez/apoyo visual. La pinta el painter SOLO
    # cuando el objeto está realmente en el suelo (no dentro de un
    # transporter, agarrado por un brazo, ni arrastrado por el aire): la
    # decisión de "está apoyado" la conoce el orquestador, no la primitiva.
    # Las locations NO llevan sombra (son el propio suelo). Se dibuja ANTES
    # del sprite del objeto (queda debajo) y es determinista: mismas
    # entradas → mismos píxeles (preserva la identidad de píxel con
    # zoom=1/pan=(0,0), solo cambia DE FORMA DETERMINISTA qué se dibuja).
    shadow_enabled: bool = True          # interruptor maestro
    shadow_color: RGB = (0, 0, 0)        # color base (se combina con alpha)
    shadow_alpha: int = 130              # opacidad 0..255 (translúcida)
    shadow_width_factor: float = 1.05    # ancho de la elipse / ancho del objeto
                                         # (>1 → la sombra asoma por los lados)
    shadow_height_ratio: float = 0.42    # alto elipse / ancho elipse (aplanamiento)
    # Ajuste vertical del centro de la sombra respecto a la BASE del objeto
    # (borde inferior del lienzo), en px de pantalla. 0 = centrada en la
    # base (la mitad inferior asoma bajo el objeto). Negativo = sube y se
    # mete bajo el objeto; positivo = baja y asoma más.
    shadow_offset_y: int = 1

    # ---- Tipografía ------------------------------------------------------
    font_name: str | None = None    # None = SysFont(None, ...)
    # Ruta a un .ttf concreto para las etiquetas del render (ids de
    # location, drone, transporter, contenido de paquete, persona). Si se
    # fija, tiene PRIORIDAD sobre font_name: las etiquetas se renderizan con
    # pygame.font.Font(font_path, size) en vez de SysFont. Pensado para que
    # la app inyecte la MISMA fuente que usa el inventario (Monogram), de
    # modo que las etiquetas del render se lean igual de claras y coherentes
    # con la estética pixel-art. None = comportamiento previo (SysFont).
    # La librería de render NO empaqueta el .ttf (vive en la app): es la
    # app quien resuelve la ruta y la inyecta vía with_overrides, evitando
    # que la capa render dependa de un asset del paquete de aplicación.
    # Si la ruta no existe o falla la carga, se cae a SysFont (sin romper).
    font_path: str | None = None
    font_size_label: int = 16       # IDs de location (16 = tamaño nativo
                                    # nítido de Monogram, su rejilla pixel)
    font_size_id: int = 16          # IDs de drone, transporter, etc.

    # ---- Políticas -------------------------------------------------------
    # Diferenciación de Locations por heurística de id (casa/hospital/depósito).
    # OPT-OUT: True por defecto, desactivable para entornos no españoles.
    differentiate_locations: bool = False

    # Estrategia de color para contenidos:
    # - "hybrid": KNOWN_CONTENT_COLORS para canónicos, fallback determinista para el resto.
    # - "fallback_only": ignora KNOWN_CONTENT_COLORS, todos vía zlib.adler32.
    # - "id_only": gris fijo (text_dim) para todos; distinción solo por texto.
    content_color_strategy: ContentColorStrategy = "hybrid"

    # Override explícito de color por content_id (case-insensitive). Si está
    # presente y contiene la clave, color_for_content lo devuelve ANTES de
    # aplicar content_color_strategy. Lo usa el escenario para inyectar una
    # paleta (p.ej. colores asignados al crear el problema) sin tocar la
    # lógica de estrategia. None = sin override. Es un Mapping (frozen-friendly,
    # se comparte por referencia; no se muta).
    content_colors: Mapping[str, RGB] | None = None

    # ---- Animación -------------------------------------------------------
    # Duración visual de una transición instantánea (PDDL parte 1-2,
    # duration=0) en la línea temporal virtual del Timeline. Es también la
    # pausa entre acciones consecutivas (el snap_start "enter-interacting").
    fallback_step_duration: float = 0.4

    # ---- Sprites ---------------------------------------------------------
    # Directorio de assets .png. None = usar los empaquetados en
    # droneplan_viz/render/assets/ (resueltos por el SpriteManager vía
    # importlib.resources). Un path explícito lo overridea (útil en tests
    # para apuntar a un tmpdir con PNGs sintéticos, o para themes que
    # carguen otra colección de sprites).
    sprite_dir: str | None = None

    # Mapeo entity_type / entity_type_variant -> nombre de archivo. Por
    # defecto el mapeo estándar; un theme alternativo puede sustituirlo
    # para usar otra nomenclatura o set de assets. Frozen-friendly: es un
    # Mapping y se comparte por referencia (los dicts literales de módulo
    # no se mutan).
    sprite_files: Mapping[str, str] = field(
        default_factory=lambda: _DEFAULT_SPRITE_FILES
    )

    # Anclas para componer el sprite del dron por capas. Frozen dataclass
    # con todos los offsets nativos; el usuario que dibuja los assets los
    # reajusta sin tocar la lógica de composición de sprites.py.
    sprite_anchors: SpriteAnchors = field(default_factory=SpriteAnchors)

    # ---- API público de construcción/derivación --------------------------

    @classmethod
    def default(cls) -> "Theme":
        """Theme con la paleta elegida.

        Es el punto de entrada habitual. La factoría existe para dar un
        nombre semántico al constructor sin argumentos y dejar sitio a
        variantes futuras (Theme.dark(), Theme.high_contrast(), etc.)
        sin romper compatibilidad.
        """
        return cls()

    def with_overrides(self, **kwargs) -> "Theme":
        """Devuelve un Theme nuevo con los campos indicados sobreescritos.

        Es un wrapper sobre dataclasses.replace con nombre semántico.
        Útil para tests y para que la UI futura ajuste un par de
        parámetros sin reconstruir el Theme entero:

            theme = Theme.default().with_overrides(
                padding=40,
                differentiate_locations=False,
            )
        """
        return replace(self, **kwargs)

    def scaled(self, factor: float) -> "Theme":
        """Copia del Theme con los TAMAÑOS y OFFSETS geométricos (px) escalados
        por `factor` ∈ (0, 1].

        Escala SOLO geometría de dibujo: radios/tamaños de entidades, sprite y
        rombo de location, offsets de vuelo/co-localización, grosores de línea,
        tamaños de fuente y las anclas del composite del dron. NO toca colores,
        fuentes, factores adimensionales (tilt en grados, pendiente, ratios de
        sombra), duraciones, el padding ni el spread del layout.

        `factor >= 1.0` devuelve self (idéntico), de modo que el camino por
        defecto (la inmensa mayoría de escenas) produce salida byte-idéntica.

        Lo usa el render cuando hay tantas locations que, ya repartidas por el
        viewbox, sus sprites a tamaño nativo se solaparían: dibuja todo a este
        factor (un "zoom out" de los TAMAÑOS; las posiciones no cambian).
        """
        if factor >= 1.0:
            return self
        return replace(
            self,
            drone_radius=_si(self.drone_radius, factor),
            package_size=_si(self.package_size, factor),
            package_outline_width=_si(self.package_outline_width, factor),
            transporter_width=_si(self.transporter_width, factor),
            transporter_height=_si(self.transporter_height, factor),
            location_radius=_si(self.location_radius, factor),
            person_radius=_si(self.person_radius, factor),
            arm_radius=_si(self.arm_radius, factor),
            border_width=_si(self.border_width, factor),
            edge_width=_si(self.edge_width, factor),
            error_x_width=_si(self.error_x_width, factor),
            location_sprite_native_size=_st(
                self.location_sprite_native_size, factor
            ),
            location_ground_offset_y=_si(self.location_ground_offset_y, factor),
            location_ground_size=_st(self.location_ground_size, factor),
            drone_hover_y_offset=_si(self.drone_hover_y_offset, factor),
            drone_colocation_offset=_si(self.drone_colocation_offset, factor),
            drone_colocation_y_lift=_si(self.drone_colocation_y_lift, factor),
            shadow_offset_y=_si(self.shadow_offset_y, factor),
            font_size_label=max(8, _si(self.font_size_label, factor)),
            font_size_id=max(8, _si(self.font_size_id, factor)),
            sprite_anchors=self.sprite_anchors.scaled(factor),
        )


# ---------------------------------------------------------------------------
# Funciones puras de aplicación de políticas
# ---------------------------------------------------------------------------


def color_for_location(loc_id: str, theme: Theme) -> tuple[RGB, RGB]:
    """Devuelve (fill, border) para una Location según la heurística del Theme.

    Si theme.differentiate_locations es False, devuelve siempre el par
    neutral. Si es True, examina prefijos del loc_id:

        casa*    | house*    | home*       → tono tierra cálido
        hospital | clinica*  | clinic*     → blanco con borde rojo
        deposito*| almacen*  | warehouse*  | depot* → azul depósito
        cualquier otro                     → neutral

    El matching es case-insensitive y por prefijo. La comparación se hace
    con str.startswith() para evitar matchear sub-cadenas accidentales
    (ej. "encasamiento" NO matchea "casa" porque no empieza por "casa").

    Devuelve siempre la tupla (fill, border) lista para pygame.draw.
    """
    if not theme.differentiate_locations:
        return (theme.loc_fill_neutral, theme.loc_border_neutral)

    norm = loc_id.lower()

    if norm.startswith(("casa", "house", "home")):
        return (theme.loc_fill_house, theme.loc_border_neutral)
    if norm.startswith(("hospital", "clinica", "clinic")):
        return (theme.loc_fill_hospital, theme.loc_border_hospital)
    if norm.startswith(("deposito", "almacen", "warehouse", "depot")):
        return (theme.loc_fill_depot, theme.loc_border_neutral)

    return (theme.loc_fill_neutral, theme.loc_border_neutral)


def color_for_content(content_id: str, theme: Theme) -> RGB:
    """Devuelve el color asociado a un Content.id según theme.content_color_strategy.

    Estrategias:

    - "hybrid" (default):
        1. Si content_id (case-insensitive) está en KNOWN_CONTENT_COLORS,
           devuelve ese color.
        2. Si no, fallback determinista vía zlib.adler32(content_id).

    - "fallback_only":
        Ignora KNOWN_CONTENT_COLORS; siempre fallback determinista.

    - "id_only":
        Devuelve theme.text_dim. La distinción visual queda en manos del
        texto del id pintado encima del paquete.

    El uso de zlib.adler32 en vez de hash() es deliberado: hash() de
    Python tiene PYTHONHASHSEED aleatorio entre procesos, lo que rompería
    los tests cross-run y la consistencia visual entre ejecuciones. adler32
    es determinista, estable entre versiones, y suficientemente uniforme
    para una paleta de 8 colores.
    """
    norm = content_id.lower()

    # Override explícito del escenario (paleta inyectada): tiene prioridad
    # sobre CUALQUIER estrategia (incluida id_only). Case-insensitive.
    if theme.content_colors is not None:
        override = theme.content_colors.get(norm)
        if override is not None:
            return override

    if theme.content_color_strategy == "id_only":
        return theme.text_dim

    if theme.content_color_strategy == "hybrid":
        known = KNOWN_CONTENT_COLORS.get(norm)
        if known is not None:
            return known

    # "fallback_only" llega aquí siempre; "hybrid" llega aquí si no es canónico.
    idx = zlib.adler32(norm.encode("utf-8")) % len(FALLBACK_CONTENT_PALETTE)
    return FALLBACK_CONTENT_PALETTE[idx]


# Mapeo DroneState → color, encapsulado para que sprites.py no tenga que
# hacer match sobre el enum. Se queda aquí porque la decisión "qué color
# tiene cada estado" es política visual, no semántica del dominio.
# Importamos DroneState lazy para evitar acoplamiento de orden de
# importación (theme.py NO debería depender del dominio en general, pero
# DroneState es un enum estable y es la fuente de verdad de los estados;
# si en el futuro queremos invertir la dependencia, mapeamos por nombre
# de estado en vez de por el enum).
def color_for_drone_state(state, theme: Theme) -> RGB:
    """Color asociado a un DroneState.

    Mapping:
        IDLE        → drone_idle (verde)
        MOVING      → drone_moving (azul)
        INTERACTING → drone_interacting (ámbar)
        ERROR       → drone_error (rojo)

    Recibe el state como Enum (DroneState); por evitar la importación
    circular potencial en el futuro, hacemos match por name del enum.
    Esto también permite testear sin importar DroneState (pasando un
    mock con .name="IDLE").
    """
    name = state.name if hasattr(state, "name") else str(state)
    if name == "IDLE":
        return theme.drone_idle
    if name == "MOVING":
        return theme.drone_moving
    if name == "INTERACTING":
        return theme.drone_interacting
    if name == "ERROR":
        return theme.drone_error
    # Fallback defensivo: estado desconocido → gris neutral del fondo.
    # En la práctica no debería ocurrir: DroneState tiene 4 valores y
    # están todos cubiertos arriba.
    return theme.text_dim
