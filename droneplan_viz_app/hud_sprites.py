"""Carga de sprites pixel-art de los botones de transporte del HUD.

Los botones de transporte (Home, StepBack, Play, Pause, StepForward,
End) son **sprites del botón completo**: el PNG incluye fondo, borde y
pictograma. pygame_gui los aplica vía el theme JSON con propiedades
`normal_image_path` y `selected_image_path` (estado pulsado/seleccionado).

El usuario provee dos sprites por botón:
    btn_<accion>_normal.png    - estado base.
    btn_<accion>_pressed.png   - estado pulsado.

El estado hovered se genera automáticamente a partir del normal aplicando
un overlay blanco translúcido al 15 %: produce una versión "iluminada"
del sprite sin requerir trabajo artístico adicional.

Comportamiento si los sprites no están presentes: los botones se quedan
con texto Monogram (comportamiento legacy). Esto permite que la app
funcione antes de tener los sprites preparados y los aplique
automáticamente cuando aparezcan en la carpeta esperada.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pygame


#: Subcarpeta donde el usuario coloca los PNG de iconos del HUD.
_HUD_ICONS_SUBDIR = ("assets", "hud_icons")


#: Acciones disponibles y sus filenames base (sin sufijo de estado).
#: La clave se usa también para asignar object_id en el theme.
_TRANSPORT_ACTIONS = (
    "home",
    "step_back",
    "play",
    "pause",
    "step_forward",
    "end",
)


@dataclass(frozen=True, slots=True)
class TransportSprites:
    """Conjunto de sprites generados para un botón.

    Attributes:
        normal_path:   ruta absoluta al PNG normal (el del repo).
        pressed_path:  ruta absoluta al PNG pulsado.
        hovered_path:  ruta absoluta a un PNG generado en tempfile que
                       contiene el normal con overlay blanco para
                       indicar hover (lo genera build_transport_sprites).
    """
    normal_path: str
    pressed_path: str
    hovered_path: str


def _icons_dir() -> Path:
    """Devuelve la carpeta donde deberían vivir los sprites del HUD,
    aunque no exista todavía (el caller comprueba si los archivos están)."""
    anchor = resources.files("droneplan_viz_app")
    for sub in _HUD_ICONS_SUBDIR:
        anchor = anchor / sub
    return Path(str(anchor))


def _generate_hover_variant(source_png: Path) -> Path:
    """Lee el PNG normal, le aplica un overlay blanco al 15 % de alpha
    para producir un "hover" visualmente más claro, y guarda el
    resultado en un tempfile.

    El overlay solo afecta a los píxeles NO transparentes del original
    (filtrado por máscara alfa), para que los rincones transparentes del
    sprite no se rellenen.

    Args:
        source_png: ruta al PNG base.

    Returns:
        Ruta al PNG hover generado.
    """
    import tempfile
    surface = pygame.image.load(str(source_png)).convert_alpha()

    # Generar la variante hover aclarando ligeramente los píxeles
    # VISIBLES del sprite, sin tocar el canal alfa (para no rellenar
    # las zonas transparentes) y sin saturar a blanco.
    #
    # El intento anterior usaba BLEND_RGBA_ADD con un overlay blanco:
    # eso (a) sumaba al canal alfa, volviendo opacas las zonas
    # transparentes — "se recubría de blanco entero" — y (b) empujaba
    # los píxeles ya claros a blanco puro. La técnica correcta:
    #   1. BLEND_RGB_ADD (no RGBA): suma solo a los canales de color,
    #      deja el alfa intacto.
    #   2. Un incremento pequeño y uniforme (no proporcional al brillo)
    #      con un overlay gris oscuro, de modo que aclara sutilmente
    #      sin reventar las zonas claras.
    #   3. Re-aplicar la máscara alfa del original, por si el blit
    #      hubiera tocado bordes antialias.
    hover = surface.copy()
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    # Incremento RGB pequeño (≈ +28/255 ≈ 11 %). Alfa 255 en el overlay
    # es irrelevante con BLEND_RGB_ADD (ese flag ignora el alfa del
    # overlay y no lo suma al destino).
    overlay.fill((28, 28, 28, 255))
    hover.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    # Restaurar el canal alfa original exactamente (BLEND_RGB_ADD no lo
    # toca, pero lo forzamos por seguridad ante distintas versiones de
    # SDL).
    alpha_mask = pygame.surfarray.array_alpha(surface)
    pygame.surfarray.pixels_alpha(hover)[:] = alpha_mask

    with tempfile.NamedTemporaryFile(
        suffix="_hover.png", delete=False
    ) as f:
        tmp_path = Path(f.name)
    pygame.image.save(hover, str(tmp_path))
    return tmp_path


def load_transport_sprites() -> dict[str, TransportSprites] | None:
    """Carga los 12 PNG de los botones de transporte y genera los 6
    PNG de hover.

    Returns:
        Diccionario {action_name: TransportSprites} con paths absolutos.
        None si CUALQUIERA de los 12 PNG faltan (todo o nada): es
        explícitamente mejor caer al modo "texto Monogram" en todos
        los botones que tener un HUD mixto inconsistente.

    Notes:
        - La función NO levanta error si la carpeta no existe: devuelve
          None y la app sigue funcionando con texto.
        - Los PNG de hover se escriben en /tmp y NO se limpian
          explícitamente. Son ~12 KB cada uno, irrelevantes.
        - Esta función debe llamarse DESPUÉS de pygame.init() porque
          load+save necesitan video inicializado.
    """
    icons_dir = _icons_dir()
    if not icons_dir.is_dir():
        return None

    result: dict[str, TransportSprites] = {}
    for action in _TRANSPORT_ACTIONS:
        normal = icons_dir / f"btn_{action}_normal.png"
        pressed = icons_dir / f"btn_{action}_pressed.png"
        if not (normal.is_file() and pressed.is_file()):
            # Falta al menos uno: abortar la carga completa para no
            # dejar el HUD en estado mixto.
            return None
        hover_tmp = _generate_hover_variant(normal)
        result[action] = TransportSprites(
            normal_path=str(normal),
            pressed_path=str(pressed),
            hovered_path=str(hover_tmp),
        )
    return result


def inject_sprites_into_theme(
    theme_dict: dict, sprites: dict[str, TransportSprites]
) -> None:
    """Modifica el theme dict in-place para añadir las imágenes de cada
    estado a los object_ids @transport_<action>.

    Hace MERGE en lugar de sobreescribir: si el theme template ya tiene
    una sección @transport_<action> con `misc` (border, shadow…), se
    preservan y solo añadimos la sección `images` con las rutas a los
    tres estados visuales del sprite.

    Args:
        theme_dict: dict con la estructura del theme JSON cargado.
        sprites: dict {action: TransportSprites} de load_transport_sprites.
    """
    for action, spr in sprites.items():
        key = f"@transport_{action}"
        existing = theme_dict.get(key, {})
        # Merge: las imágenes son nuevas; el resto (misc, colours...)
        # se preserva del template si existía.
        #
        # Cubrimos TODOS los estados de imagen que pygame_gui reconoce
        # (normal, hovered, selected, disabled). Mapear también
        # 'disabled' al sprite normal evita que un botón temporalmente
        # deshabilitado muestre el fondo desnudo. El estado 'selected'
        # es el que pygame_gui activa al pulsar.
        existing["images"] = {
            "normal_image":   {"path": spr.normal_path},
            "hovered_image":  {"path": spr.hovered_path},
            "selected_image": {"path": spr.pressed_path},
            "disabled_image": {"path": spr.normal_path},
        }
        # Colores de fondo/borde OSCUROS para todos los estados. Aunque
        # el sprite cubre el botón, en el instante del click pygame_gui
        # puede repintar el fondo del botón un frame antes de blittear
        # la imagen del nuevo estado: si ese fondo fuese el blanco por
        # defecto, se vería un flash. Forzándolo a un gris muy oscuro
        # (casi el fondo del propio sprite) el flash es imperceptible.
        # border/shadow a transparente para que no asome marco.
        dark = "#1E2832"
        existing.setdefault("colours", {})
        existing["colours"].update({
            "normal_bg": dark,
            "hovered_bg": dark,
            "selected_bg": dark,
            "active_bg": dark,
            "disabled_bg": dark,
            "normal_border": dark,
            "hovered_border": dark,
            "selected_border": dark,
            "active_border": dark,
            "disabled_border": dark,
        })
        theme_dict[key] = existing
