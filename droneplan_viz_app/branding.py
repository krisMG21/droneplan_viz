"""Carga de los recursos de marca (logo) de la aplicación.

Sigue el mismo patrón que theme_loader/hud_sprites: resuelve las rutas
vía importlib.resources y degrada con elegancia (devuelve None) si el
asset no está empaquetado, de modo que la app funcione igual sin logo.

El usuario coloca los PNG en:
    droneplan_viz_app/assets/branding/logo.png      - logo principal
    droneplan_viz_app/assets/branding/logo_hud.png  - variante HUD (opcional)

Los logos son pixel-art, coherentes con el resto de sprites del proyecto.
Por eso el ESCALADO se hace fuera de este módulo con nearest-neighbor
(pygame.transform.scale) y preferentemente por múltiplos enteros: aquí
solo se cargan a su resolución nativa.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import pygame

#: Subcarpeta donde viven los recursos de marca dentro del paquete.
_BRANDING_SUBDIR = ("assets", "branding")


def _branding_path(filename: str) -> Path:
    """Ruta (exista o no) de un asset de branding dentro del paquete.

    Usa importlib.resources para funcionar tanto en instalación editable
    (apunta al fichero del repo) como en wheel (fichero extraído en
    site-packages). El caller comprueba si el archivo existe.
    """
    anchor = resources.files("droneplan_viz_app")
    for sub in _BRANDING_SUBDIR:
        anchor = anchor / sub
    return Path(str(anchor)) / filename


def load_logo(filename: str = "logo.png") -> "pygame.Surface | None":
    """Carga un logo como Surface con alfa, o None si no está disponible.

    Tolerante a fallos: si el PNG no existe o no puede cargarse, devuelve
    None y el caller decide el fallback (texto, no pintar nada, etc.).

    Requiere que el display de pygame ya esté inicializado, porque usa
    convert_alpha() para optimizar los blits posteriores. La Surface se
    devuelve a su RESOLUCIÓN NATIVA (sin escalar): el escalado pixel-art
    es responsabilidad del caller.

    Args:
        filename: nombre del PNG dentro de assets/branding/.

    Returns:
        La Surface cargada, o None si el asset falta o no puede leerse.
    """
    path = _branding_path(filename)
    if not path.is_file():
        return None
    try:
        return pygame.image.load(str(path)).convert_alpha()
    except (pygame.error, FileNotFoundError, OSError):
        return None


def load_hud_logo() -> "pygame.Surface | None":
    """Logo para la barra del HUD.

    Carga la variante específica logo_hud.png (pixel-art pequeño, pensado
    para la barra superior). Devuelve None si no está presente, en cuyo
    caso el HUD conserva su título de texto. No cae al logo principal:
    logo.png es grande y no está diseñado para esta esquina, así que
    mostrarlo escalado ahí daría un resultado incorrecto.
    """
    return load_logo("logo_hud.png")