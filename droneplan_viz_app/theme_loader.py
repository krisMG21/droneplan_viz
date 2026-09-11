"""Carga del theme de pygame_gui con la fuente Monogram empaquetada.

El theme JSON de pygame_gui exige rutas a los TTF en `regular_path`. La
ruta es absoluta (o relativa al cwd, que es inestable) y depende de
dónde esté instalado el paquete (editable, wheel, venv, etc.). El JSON
estático del repositorio no puede saberla.

Solución: el JSON del repo contiene un placeholder `FONT_PATH_PLACEHOLDER`
que se sustituye en runtime por la ruta real del TTF, resuelta vía
`importlib.resources`. El resultado se vuelca a un fichero temporal
único por proceso y se pasa al `UIManager`.

Razón de un fichero temporal y no un dict en memoria: la API estable
de `pygame_gui.UIManager` acepta `theme_path: str` apuntando a un
fichero. Su `load_theme(dict)` también existe pero es más quisquilloso
con la estructura. El fichero temporal es la vía documentada.

El tempfile se limpia automáticamente al terminar el proceso (usamos
`tempfile.NamedTemporaryFile` sin delete y aceptamos que Python no
borre archivos al cerrar; es un coste irrisorio, KB no MB).
"""
from __future__ import annotations

import json
import tempfile
from importlib import resources
from pathlib import Path


#: Marcador en el JSON del repo que se sustituye en runtime.
_FONT_PATH_PLACEHOLDER: str = "FONT_PATH_PLACEHOLDER"


def _resolve_font_path() -> str:
    """Devuelve la ruta absoluta del TTF de Monogram empaquetado.

    Vía `importlib.resources` para funcionar tanto en instalación
    editable (apunta al fichero del repo) como en wheel (apunta al
    fichero extraído en site-packages).

    Returns:
        Ruta absoluta al .ttf como string. pygame_gui no acepta Path.

    Raises:
        FileNotFoundError si el TTF no está en el paquete instalado
        (suele significar que `package-data` del pyproject.toml está
        mal configurado o se rompió la inclusión del asset).
    """
    anchor = resources.files("droneplan_viz_app") / "assets" / "fonts" / "monogram.ttf"
    path = Path(str(anchor))
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encuentra Monogram en el paquete instalado: {path}. "
            "Verifica [tool.setuptools.package-data] del pyproject.toml."
        )
    return str(path)


def monogram_font_path() -> str | None:
    """Ruta del .ttf de Monogram para inyectar en el Theme de render.

    Envoltorio TOLERANTE A FALLOS sobre `_resolve_font_path`: devuelve la
    ruta (normalizada con forward slashes, igual que para pygame_gui) o
    None si el asset no está empaquetado. Pensado para que la app haga:

        theme = Theme.default().with_overrides(font_path=monogram_font_path())

    Si devuelve None, el render cae a SysFont sin romper. A diferencia de
    `_resolve_font_path` (que LANZA si falta el TTF, porque pygame_gui lo
    necesita sí o sí), aquí la fuente fina es una mejora opcional del
    render y su ausencia no debe abortar el arranque.
    """
    try:
        return _resolve_font_path().replace("\\", "/")
    except (FileNotFoundError, OSError, ValueError):
        return None


def _resolve_theme_template() -> str:
    """Lee el JSON-template del HUD desde el paquete instalado.

    Returns:
        Contenido del JSON como string, con el placeholder sin sustituir.
    """
    anchor = resources.files("droneplan_viz_app") / "assets" / "theme.json"
    return Path(str(anchor)).read_text(encoding="utf-8")


def prepare_ui_theme() -> str:
    """Prepara el theme.json de pygame_gui con la ruta real del TTF.

    Lee el JSON-template del paquete, sustituye el placeholder por la
    ruta resuelta del TTF, escribe el resultado en un fichero temporal
    y devuelve la ruta.

    Esta función se llama una vez al construir la app. La ruta
    resultante se pasa a `pygame_gui.UIManager(window_size, theme_path=...)`.

    Returns:
        Ruta del fichero temporal con el JSON listo para pygame_gui.

    Raises:
        FileNotFoundError si el TTF o el JSON no están empaquetados
        correctamente.
        json.JSONDecodeError si el template está corrupto.
    """
    font_path = _resolve_font_path()
    template = _resolve_theme_template()

    # En Windows _resolve_font_path devuelve algo como
    # 'C:\\Users\\nombre\\...\\monogram.ttf', y al insertarlo crudo en el
    # JSON los backslashes se interpretan como inicios de escape JSON
    # inválidos (\U, \c, \n…) y json.loads revienta. Normalizar a
    # forward slashes evita el problema: tanto JSON como pygame_gui como
    # las APIs de OS aceptan rutas con '/' en Windows sin pestañear.
    font_path = font_path.replace("\\", "/")

    # Verificación defensiva: si el JSON no contiene el placeholder es
    # señal de que algo está mal (ya estaría sustituido, o el theme se
    # ha editado a mano dejándolo inválido). Fallar pronto con mensaje
    # útil.
    if _FONT_PATH_PLACEHOLDER not in template:
        raise ValueError(
            f"El theme.json no contiene el placeholder "
            f"{_FONT_PATH_PLACEHOLDER!r}. ¿Se editó a mano? "
            "El placeholder es necesario para que el TTF de Monogram "
            "se resuelva al instalar el paquete."
        )

    rendered = template.replace(_FONT_PATH_PLACEHOLDER, font_path)

    # Validación: el resultado debe seguir siendo JSON parseable. Con la
    # normalización de backslashes anterior esto debería pasar siempre;
    # se mantiene la verificación como red de seguridad contra
    # futuros cambios en el template.
    theme_dict = json.loads(rendered)

    # Inyectar sprites de botones de transporte si están presentes.
    # load_transport_sprites devuelve None si la carpeta no existe o
    # falta algún PNG → la app sigue con texto Monogram en los botones
    # (comportamiento legacy graceful).
    try:
        from droneplan_viz_app.hud_sprites import (
            inject_sprites_into_theme,
            load_transport_sprites,
        )
        sprites = load_transport_sprites()
        if sprites is not None:
            # Las rutas pueden tener backslashes en Windows (igual que el
            # TTF). Normalizamos antes de inyectar al theme.
            from dataclasses import replace as dc_replace
            sprites_norm = {
                action: dc_replace(
                    spr,
                    normal_path=spr.normal_path.replace("\\", "/"),
                    pressed_path=spr.pressed_path.replace("\\", "/"),
                    hovered_path=spr.hovered_path.replace("\\", "/"),
                )
                for action, spr in sprites.items()
            }
            inject_sprites_into_theme(theme_dict, sprites_norm)
    except Exception:
        # Cualquier error cargando sprites NO debe romper el arranque:
        # caer en el modo texto legacy es siempre aceptable.
        pass

    # Volcado a tempfile. delete=False porque pygame_gui lee el fichero
    # cuando le da la gana (no necesariamente justo después); que el
    # sistema operativo lo limpie al cerrar el proceso es suficiente.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(theme_dict, f)
        return f.name
