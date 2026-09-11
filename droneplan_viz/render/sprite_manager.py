"""
SpriteManager: carga, escala y cachea sprites .png para el render.

Responsabilidad ÚNICA: dado un (entity_type, variant, size), devolver una
pygame.Surface escalada lista para blittear, o None si no hay asset
disponible. El painter consulta al manager y, si recibe None, cae a la
primitiva geométrica de sprites.py. El manager NO dibuja, NO conoce el
World, NO conoce entidades del dominio: solo mapea strings a Surfaces.

Decisiones de diseño:

1. AUTODETECCIÓN, NO FLAG. No hay un parámetro `use_sprites`. El manager
   intenta resolver cada asset bajo demanda; si el archivo existe se usa,
   si no, get() devuelve None y el painter usa primitivas. Consecuencia:
   - Sin assets en el repo (estado actual) → todo cae a primitivas → los
     tests headless siguen verdes sin empaquetar binarios.
   - El día que se añadan los .png a assets/, los sprites se activan
     solos, sin tocar código ni configuración.
   Esto cumple el requisito del usuario ("que se usen siempre que estén
   disponibles") sin obligar a nadie a recordar encender un flag.

2. CACHÉ POR (entity_type, variant, size). pygame.image.load + scale es
   caro; se hace UNA vez por combinación y se guarda. La caché es un dict
   mutable interno: el manager es infraestructura de la UI, no dominio,
   así que puede tener estado. No toca nada inmutable.

3. NEGATIVE CACHE. Si un asset no existe, lo recordamos (valor None en la
   caché) para no volver a tocar disco en cada frame buscando un archivo
   que sabemos que falta. Sin esto, una entidad sin sprite haría un
   os.path.exists/abrir-fallido 60 veces por segundo.

4. RUTAS Y NOMBRES EN EL THEME. El manager NO hardcodea nombres de
   archivo ni tamaños. Lee theme.sprite_files (entity_type/variant →
   filename) y theme.sprite_sizes / dimensiones del theme para el escalado.
   Mantiene la regla del proyecto "cero rutas/tamaños fuera del theme".

5. RESOLUCIÓN DE DIRECTORIO. Por defecto los assets viven empaquetados en
   droneplan_viz/render/assets/ y se localizan vía importlib.resources
   (robusto ante instalación como wheel, editable, o zip). El theme puede
   fijar un sprite_dir explícito (str path) para overridear, útil en
   tests (apuntar a un tmpdir con PNGs sintéticos) o para themes que
   carguen assets de otra carpeta.

6. convert_alpha BAJO DEMANDA Y DEFENSIVO. convert_alpha() requiere un
   display mode activo; en headless con SDL dummy puede no haberlo. Si
   convert_alpha falla, usamos la Surface tal cual (sin optimizar el
   formato de píxel). El render sigue funcionando, solo un pelín más
   lento en el blit — irrelevante para los tamaños del proyecto.

7. NEAREST-NEIGHBOR PARA PIXEL ART. Escalamos con pygame.transform.scale
   (nearest-neighbor), NO smoothscale (interpolación bilineal). Los
   assets del proyecto son pixel art a resolución pequeña que se escala
   al alza (factor entero p.ej. 2x para location 110x88 → 220x176).
   Con smoothscale, la interpolación bilineal introduce gradientes que
   emborronan el pixel art — destruye la estética. Nearest-neighbor
   mantiene los píxeles nítidos: con factor entero el resultado es
   perfecto, y con factor no-entero hay alguna fila/columna duplicada
   irregularmente pero los píxeles siguen siendo píxeles.
   La composición de capas del drone ya usaba nearest (ver
   get_drone_composite); este cambio unifica el criterio para todos
   los assets (location, package, transporter, person).
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pygame

from droneplan_viz.render.theme import Theme

#: Patrón de archivo de sprite de persona: personN.png (set base). La variante
#: personN_box.png (persona con caja entregada) se resuelve en get_person; aquí
#: solo detectamos los sets disponibles por el archivo base.
_PERSON_SPRITE_RE = re.compile(r"^(person(\d+))\.png$", re.IGNORECASE)


class SpriteManager:
    """Cargador y caché de sprites con fallback transparente a None.

    Uso típico desde el painter:

        spr = sprite_manager.get("drone", variant="idle", size=(36, 36))
        if spr is not None:
            surface.blit(spr, _top_left_centered(pos, spr))
        else:
            draw_drone(...)   # primitiva

    El manager se construye una vez (lo hace la app o el painter) y se
    reutiliza durante toda la sesión: la caché vive mientras viva el
    manager.
    """

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        # Caché: (entity_type, variant, size) -> Surface | None.
        # None significa "ya buscamos y no existe" (negative cache).
        self._cache: dict[tuple[str, str | None, tuple[int, int]], pygame.Surface | None] = {}
        # Caché de PNG a resolución nativa por clave lógica (capas del dron).
        self._native_cache: dict[str, pygame.Surface | None] = {}
        # Caché de compuestos del dron por (tag, body, face, obj, dest_size).
        self._composite_cache: dict[tuple, pygame.Surface | None] = {}
        # Directorio de assets resuelto perezosamente la primera vez.
        self._asset_dir: Path | None = None
        self._asset_dir_resolved = False
        # Sets de sprite de persona (personN) detectados en assets; None hasta
        # el primer escaneo. Caché de sprites de persona por (base, box, size).
        self._person_bases: tuple[str, ...] | None = None
        self._person_cache: dict[tuple[str, bool, tuple[int, int]], pygame.Surface | None] = {}

    # -- API pública -------------------------------------------------------

    def get(
        self,
        entity_type: str,
        *,
        size: tuple[int, int],
        variant: str | None = None,
    ) -> pygame.Surface | None:
        """Devuelve el sprite escalado para (entity_type, variant, size).

        Args:
            entity_type: "drone", "package", "transporter", "person",
                "location". Clave para resolver el nombre de archivo en
                theme.sprite_files.
            size: (ancho, alto) en píxeles al que escalar el sprite.
            variant: subtipo opcional. Para drones, el estado FSM en
                minúsculas ("idle", "interacting", "moving", "error").
                Para paquetes, el content_id ("medicina"). None = sin
                variante (usa la clave base entity_type).

        Returns:
            pygame.Surface escalada y cacheada, o None si no hay asset
            para esa combinación (el caller debe caer a primitiva).

        Resolución de nombre de archivo (candidatos en orden):
            1. Si variant no es None y la clave f"{entity_type}_{variant}"
               está en theme.sprite_files, su archivo es el primer
               candidato.
            2. La clave entity_type a secas (sprite base) es el segundo
               candidato.
            Se prueba cada candidato EN ORDEN y se usa el primero que
            exista físicamente en disco. Así, si una variante está
            declarada en el theme pero su .png no se ha entregado todavía,
            se cae al sprite base en vez de a primitiva. Si ningún
            candidato existe, devuelve None y el painter usa primitiva.
        """
        key = (entity_type, variant, size)
        if key in self._cache:
            return self._cache[key]

        surface: pygame.Surface | None = None
        for filename in self._candidate_filenames(entity_type, variant):
            surface = self._load_and_scale(filename, size)
            if surface is not None:
                break
        # surface es None si ningún candidato existe en disco (negative
        # cache); el painter caerá a primitiva.
        self._cache[key] = surface
        return surface

    def person_sprite_bases(self) -> tuple[str, ...]:
        """Escanea el dir de assets y devuelve los 'sets' de sprite de persona
        disponibles, en orden determinista.

        Un set es un nombre base ``personN`` para el que existe ``personN.png``
        (la variante ``personN_box.png`` es opcional y se resuelve en
        get_person). Se ordena por el número N (person1, person2, person10…),
        no alfabéticamente, para que la asignación a personas sea estable e
        intuitiva. Cacheado: el escaneo de disco se hace una sola vez.

        Devuelve () si no hay dir de assets o ningún personN.png. En ese caso
        el painter cae al sprite genérico 'person' o a la primitiva.
        """
        if self._person_bases is not None:
            return self._person_bases
        bases: list[str] = []
        asset_dir = self._get_asset_dir()
        if asset_dir is not None:
            seen: dict[str, int] = {}
            for p in asset_dir.iterdir():
                m = _PERSON_SPRITE_RE.match(p.name)
                if m is not None:
                    seen[m.group(1)] = int(m.group(2))  # base -> N
            bases = sorted(seen, key=lambda b: seen[b])
        self._person_bases = tuple(bases)
        return self._person_bases

    def get_person(
        self, base: str, *, box: bool, size: tuple[int, int]
    ) -> pygame.Surface | None:
        """Sprite de una persona por nombre base directo (sin pasar por
        theme.sprite_files).

        base: nombre del set, p.ej. "person1" (de person_sprite_bases()).
        box: True para la variante "con caja" (persona que ya recibió su
            entrega). Carga ``{base}_box.png``; si esa variante no existe,
            cae a ``{base}.png`` (una persona sin sprite _box no parpadea:
            simplemente no cambia al recibir). Devuelve None si ni siquiera
            ``{base}.png`` existe (el painter cae a primitiva).
        """
        candidates = [f"{base}_box.png", f"{base}.png"] if box else [f"{base}.png"]
        key = (base, box, size)
        if key in self._person_cache:
            return self._person_cache[key]
        surface: pygame.Surface | None = None
        for filename in candidates:
            surface = self._load_and_fit(filename, size)
            if surface is not None:
                break
        self._person_cache[key] = surface
        return surface

    def clear_cache(self) -> None:
        """Vacía las cachés. Útil si el theme cambia en caliente (la app
        podría permitir cambiar de skin). No se usa en el flujo normal."""
        self._cache.clear()
        self._native_cache.clear()
        self._composite_cache.clear()
        self._person_cache.clear()
        self._person_bases = None

    # -- Internos ----------------------------------------------------------

    def _candidate_filenames(
        self, entity_type: str, variant: str | None
    ) -> list[str]:
        """Lista ordenada de archivos candidatos para (entity_type, variant).

        Orden de preferencia: archivo de la variante específica (si la
        clave está declarada en el theme), luego archivo base. Sin
        duplicados y sin None. Puede devolver lista vacía si ni la
        variante ni la base están declaradas.
        """
        files = self._theme.sprite_files
        candidates: list[str] = []
        if variant is not None:
            specific = f"{entity_type}_{variant}"
            if specific in files:
                candidates.append(files[specific])
        if entity_type in files:
            base = files[entity_type]
            if base not in candidates:
                candidates.append(base)
        return candidates

    def _load_and_fit(
        self, filename: str, box: tuple[int, int]
    ) -> pygame.Surface | None:
        """Carga un .png y lo escala PRESERVANDO el aspect ratio para caber en
        `box` (ancho, alto), sin deformar.

        A diferencia de _load_and_scale (que estira al tamaño exacto), aquí se
        calcula el factor que hace caber el sprite nativo dentro de la caja
        manteniendo su proporción. Necesario para sprites no cuadrados como las
        personas (p.ej. 20x35): forzarlas a un cuadrado las achataría. Sigue
        usando nearest-neighbor (pixel art). Devuelve None si el archivo no
        existe.
        """
        asset_dir = self._get_asset_dir()
        if asset_dir is None:
            return None
        path = asset_dir / filename
        if not path.is_file():
            return None
        try:
            raw = pygame.image.load(str(path))
        except (pygame.error, FileNotFoundError):
            return None
        try:
            raw = raw.convert_alpha()
        except pygame.error:
            pass
        nw, nh = raw.get_size()
        bw, bh = box
        if nw <= 0 or nh <= 0:
            return raw
        scale = min(bw / nw, bh / nh)
        target = (max(1, round(nw * scale)), max(1, round(nh * scale)))
        if target != (nw, nh):
            raw = pygame.transform.scale(raw, target)
        return raw

    def _load_and_scale(
        self, filename: str, size: tuple[int, int]
    ) -> pygame.Surface | None:
        """Carga un .png del directorio de assets y lo escala a size.

        Devuelve None si el archivo no existe físicamente (asset
        declarado en el theme pero no entregado todavía), de modo que el
        painter caiga a primitiva sin romperse.
        """
        asset_dir = self._get_asset_dir()
        if asset_dir is None:
            return None
        path = asset_dir / filename
        if not path.is_file():
            return None
        try:
            raw = pygame.image.load(str(path))
        except (pygame.error, FileNotFoundError):
            # Archivo corrupto o ilegible: degradamos a primitiva.
            return None
        # convert_alpha optimiza el formato de píxel para blits rápidos,
        # pero requiere un display mode. En headless puede lanzar; si
        # falla, seguimos con la Surface sin convertir.
        try:
            raw = raw.convert_alpha()
        except pygame.error:
            pass
        if raw.get_size() != size:
            raw = pygame.transform.scale(raw, size)
        return raw

    # -- Carga nativa y composición de capas (dron) ------------------------

    def _load_native_by_key(self, key: str) -> pygame.Surface | None:
        """Carga el PNG asociado a una clave lógica del theme, SIN escalar.

        La clave (p.ej. "drone", "face_NE", "carrier_3") se resuelve a un
        nombre de archivo vía theme.sprite_files. Devuelve la Surface a
        resolución nativa o None si la clave no está declarada o el
        archivo no existe. Cachea por clave en _native_cache para no
        recargar de disco.
        """
        if key in self._native_cache:
            return self._native_cache[key]
        files = self._theme.sprite_files
        filename = files.get(key)
        surface: pygame.Surface | None = None
        if filename is not None:
            asset_dir = self._get_asset_dir()
            if asset_dir is not None:
                path = asset_dir / filename
                if path.is_file():
                    try:
                        raw = pygame.image.load(str(path))
                        try:
                            raw = raw.convert_alpha()
                        except pygame.error:
                            pass
                        surface = raw
                    except (pygame.error, FileNotFoundError):
                        surface = None
        self._native_cache[key] = surface
        return surface

    def get_drone_composite(
        self,
        *,
        body: str,
        face: str | None,
        obj: str | None,
        dest_size: tuple[int, int],
    ) -> pygame.Surface | None:
        """Compone el sprite del dron apilando capas y lo escala una vez.

        Modelo de capas (rediseño): en vez de un PNG entero por estado FSM,
        el dron se arma apilando capas independientes sobre un lienzo a
        resolución nativa, y el compuesto se reescala UNA sola vez al
        destino con nearest-neighbor (pixel art). El resultado se cachea
        por (body, face, obj, dest_size).

        Args:
            body: clave del cuerpo ("drone" o "drone_interacting").
            face: clave de la cara a superponer, o None. Solo se aplica
                sobre el cuerpo base; con "drone_interacting" el caller
                debe pasar None (la cara ya viene incrustada en el PNG).
            obj: clave del objeto agarrado ("box", "carrier_0".."carrier_5")
                o None.
            dest_size: tamaño final en píxeles. Para el caso 1:1 nativo es
                (2*drone_radius, ...), igual al cuerpo, y no hay reescalado.

        Returns:
            Surface compuesta y escalada, o None si NI SIQUIERA el cuerpo
            existe (en ese caso el painter cae a primitiva). Si el cuerpo
            existe pero falta la cara o el objeto, se compone con las capas
            disponibles (fallback en cascada por capa).

        Anclas: todas salen de theme.sprite_anchors; cero números mágicos.
        """
        cache_key = ("drone_composite", body, face, obj, dest_size)
        if cache_key in self._composite_cache:
            return self._composite_cache[cache_key]

        anchors = self._theme.sprite_anchors

        body_surf = self._load_native_by_key(body)
        if body_surf is None:
            # Sin cuerpo no hay composición posible → primitiva.
            self._composite_cache[cache_key] = None
            return None

        # Lienzo a resolución nativa, dimensionado para el peor caso
        # (cuerpo + carrier colgando). El cuerpo se pega arriba (techo en
        # y=0); todo lo demás cuelga hacia abajo.
        canvas_w, canvas_h = anchors.canvas_size
        canvas = pygame.Surface((canvas_w, canvas_h), pygame.SRCALPHA)

        body_w, body_h = body_surf.get_size()
        body_x = (canvas_w - body_w) // 2  # centrado en X
        canvas.blit(body_surf, (body_x, 0))

        # Capa de cara (solo si se pidió; el caller decide si aplica según
        # el cuerpo). Centrada en X, a face_top desde el techo.
        if face is not None:
            face_surf = self._load_native_by_key(face)
            if face_surf is None:
                # Fallback en cascada: si falta la cara direccional/estado,
                # probar la cara básica "face".
                face_surf = self._load_native_by_key("face")
            if face_surf is not None:
                fw, _fh = face_surf.get_size()
                face_x = (canvas_w - fw) // 2
                canvas.blit(face_surf, (face_x, anchors.face_top))

        # Capa de objeto agarrado (excluyente: caja(s) o carrier).
        if obj is not None:
            # "box2" no es un asset propio: reutiliza el sprite "box"
            # pintado DOS veces (una "por brazo"). El resto de claves
            # ("box", "carrier_N") se cargan tal cual.
            load_key = "box" if obj == "box2" else obj
            obj_surf = self._load_native_by_key(load_key)
            if obj_surf is not None:
                ow, oh = obj_surf.get_size()
                if obj == "box" or obj.startswith("box"):
                    # Caja(s): pegadas al borde inferior del cuerpo.
                    obj_y = body_h - oh
                    if obj == "box2":
                        # Dos cajas: una a cada lado del eje (una por brazo).
                        gap = max(1, ow // 5)
                        total = 2 * ow + gap
                        left_x = (canvas_w - total) // 2
                        canvas.blit(obj_surf, (left_x, obj_y))
                        canvas.blit(obj_surf, (left_x + ow + gap, obj_y))
                    else:
                        canvas.blit(obj_surf, ((canvas_w - ow) // 2, obj_y))
                else:
                    # Carrier: su techo queda carrier_overlap px DENTRO del
                    # cuerpo (por encima del borde inferior).
                    obj_y = body_h - anchors.carrier_overlap
                    canvas.blit(obj_surf, ((canvas_w - ow) // 2, obj_y))

        # Reescalado ÚNICO del compuesto, nearest-neighbor (pixel art).
        # El factor de escala se deriva del ANCHO del cuerpo (58 en base e
        # interacting), NO de la altura. Si usáramos la altura, el cuerpo
        # interacting (58x64) se aplastaría a la altura destino (58),
        # perdiendo los 6px de los brazos extendidos. Con factor por ancho
        # se preserva el aspecto: el interacting conserva sus 64px nativos
        # (los 6px extra crecen hacia abajo, techo fijo). El lienzo entero
        # se escala con ese mismo factor único.
        dest_body_w, _dest_body_h = dest_size
        scale = dest_body_w / body_w
        if scale != 1.0:
            scaled_canvas_size = (
                max(1, round(canvas_w * scale)),
                max(1, round(canvas_h * scale)),
            )
            composite = pygame.transform.scale(canvas, scaled_canvas_size)
        else:
            # 1:1 nativo: sin reescalado, pixel art intacto.
            composite = canvas

        self._composite_cache[cache_key] = composite
        return composite

    def _get_asset_dir(self) -> Path | None:
        """Resuelve (una vez) el directorio de assets.

        Prioridad:
            1. theme.sprite_dir si está fijado (override explícito).
            2. droneplan_viz/render/assets/ vía importlib.resources.
        Devuelve None si ninguno existe (estado actual del repo: sin
        carpeta assets → todo cae a primitivas).
        """
        if self._asset_dir_resolved:
            return self._asset_dir

        self._asset_dir_resolved = True

        # 1. Override explícito del theme.
        if self._theme.sprite_dir is not None:
            candidate = Path(self._theme.sprite_dir)
            self._asset_dir = candidate if candidate.is_dir() else None
            return self._asset_dir

        # 2. Assets empaquetados. importlib.resources es robusto ante
        #    instalación como wheel/editable.
        try:
            anchor = resources.files("droneplan_viz.render") / "assets"
            # anchor puede ser un MultiplexedPath o Traversable; lo
            # convertimos a Path real si es posible.
            candidate = Path(str(anchor))
            self._asset_dir = candidate if candidate.is_dir() else None
        except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError):
            self._asset_dir = None
        return self._asset_dir
