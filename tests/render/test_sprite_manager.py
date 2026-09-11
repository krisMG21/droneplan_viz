"""Tests para droneplan_viz.render.sprite_manager.SpriteManager.

Dos grandes grupos:

1. Contrato de FALLBACK (sin assets): el manager devuelve None para
   cualquier petición cuando no hay directorio de assets o el archivo
   pedido no existe. Esto es lo que mantiene los tests headless verdes
   sin empaquetar binarios: si no hay PNG, get() devuelve None y el
   painter cae a primitivas.

2. Camino CON sprite: generamos PNGs sintéticos en un tmpdir, apuntamos
   theme.sprite_dir a ese directorio, y verificamos que el manager los
   carga, los escala al tamaño pedido, y los cachea (no recarga de disco
   en la segunda llamada).

Todos headless (SDL dummy vía conftest del paquete).
"""
from __future__ import annotations

import pygame
import pytest

from droneplan_viz.render.sprite_manager import SpriteManager
from droneplan_viz.render.theme import Theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_png(path, size=(64, 64), color=(200, 50, 50, 255)) -> None:
    """Crea un PNG sintético relleno de un color (con alfa) en `path`."""
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill(color)
    pygame.image.save(surf, str(path))


# ---------------------------------------------------------------------------
# Grupo 1: contrato de fallback (sin assets)
# ---------------------------------------------------------------------------


class TestSpriteManagerFallback:
    """Sin assets, todo get() devuelve None: el painter usará primitivas."""

    def test_sin_assets_en_dir_get_devuelve_none(self, tmp_path):
        # El fallback a None debe darse cuando el directorio de assets no
        # contiene los PNG pedidos. Usamos un tmpdir vacío explícito en vez
        # de confiar en que el paquete no traiga assets/ (ya los trae: los
        # PNG reales del dron se entregaron). Lo que se verifica es el
        # CONTRATO de fallback, no el estado del repo.
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        assert sm.get("drone", size=(36, 36), variant="idle") is None
        assert sm.get("package", size=(14, 14)) is None
        assert sm.get("transporter", size=(48, 30)) is None

    def test_sprite_dir_inexistente_get_devuelve_none(self):
        theme = Theme.default().with_overrides(sprite_dir="/no/existe/este/path")
        sm = SpriteManager(theme)
        assert sm.get("drone", size=(36, 36)) is None

    def test_sprite_dir_vacio_get_devuelve_none(self, tmp_path):
        # Directorio existe pero está vacío: el archivo declarado en el
        # theme no está físicamente → None.
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        assert sm.get("drone", size=(36, 36), variant="idle") is None

    def test_entity_type_no_declarado_devuelve_none(self, tmp_path):
        # Un entity_type que no está en sprite_files: None aunque haya dir.
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        assert sm.get("entidad_inventada", size=(10, 10)) is None

    def test_negative_cache_no_recarga(self, tmp_path, monkeypatch):
        # Tras un get() que devuelve None, una segunda llamada NO debe
        # volver a tocar disco (negative cache). Lo verificamos espiando
        # _load_and_scale.
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)

        calls = {"n": 0}
        original = sm._load_and_scale

        def spy(filename, size):
            calls["n"] += 1
            return original(filename, size)

        monkeypatch.setattr(sm, "_load_and_scale", spy)

        sm.get("drone", size=(36, 36), variant="idle")
        sm.get("drone", size=(36, 36), variant="idle")
        # Solo la primera llamada llega a _load_and_scale; la segunda usa
        # la negative cache.
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Grupo 2: camino con sprite sintético
# ---------------------------------------------------------------------------


class TestSpriteManagerConSprite:
    """Con PNGs sintéticos en un tmpdir, el manager los carga y escala."""

    def test_carga_sprite_base(self, tmp_path):
        _write_png(tmp_path / "package.png", size=(64, 64))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        spr = sm.get("package", size=(14, 14))
        assert spr is not None
        assert spr.get_size() == (14, 14)  # escalado al tamaño pedido

    def test_carga_variante_de_contenido(self, tmp_path):
        # El dron ya NO usa variantes por estado (se compone por capas, ver
        # get_drone_composite). La cascada de variantes sigue viva para
        # otras entidades; la probamos con package + variant=content_id.
        _write_png(tmp_path / "package_medicina.png", size=(32, 32))
        theme = Theme.default().with_overrides(
            sprite_dir=str(tmp_path),
            sprite_files={
                **Theme.default().sprite_files,
                "package_medicina": "package_medicina.png",
            },
        )
        sm = SpriteManager(theme)
        spr = sm.get("package", size=(14, 14), variant="medicina")
        assert spr is not None
        assert spr.get_size() == (14, 14)

    def test_variante_inexistente_cae_a_base(self, tmp_path):
        # Hay package.png (base) pero NO package_agua.png. Pedir
        # variant="agua" debe caer a la clave base "package".
        _write_png(tmp_path / "package.png", size=(32, 32))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        spr = sm.get("package", size=(14, 14), variant="agua")
        assert spr is not None
        assert spr.get_size() == (14, 14)

    def test_cache_no_recarga_de_disco(self, tmp_path, monkeypatch):
        _write_png(tmp_path / "package.png", size=(32, 32))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)

        calls = {"n": 0}
        original = sm._load_and_scale

        def spy(filename, size):
            calls["n"] += 1
            return original(filename, size)

        monkeypatch.setattr(sm, "_load_and_scale", spy)

        a = sm.get("package", size=(14, 14))
        b = sm.get("package", size=(14, 14))
        # Misma Surface cacheada, una sola carga de disco.
        assert a is b
        assert calls["n"] == 1

    def test_distintos_tamanos_son_entradas_distintas(self, tmp_path):
        _write_png(tmp_path / "package.png", size=(64, 64))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        s14 = sm.get("package", size=(14, 14))
        s28 = sm.get("package", size=(28, 28))
        assert s14.get_size() == (14, 14)
        assert s28.get_size() == (28, 28)
        assert s14 is not s28

    def test_clear_cache_fuerza_recarga(self, tmp_path, monkeypatch):
        _write_png(tmp_path / "package.png", size=(32, 32))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)

        calls = {"n": 0}
        original = sm._load_and_scale

        def spy(filename, size):
            calls["n"] += 1
            return original(filename, size)

        monkeypatch.setattr(sm, "_load_and_scale", spy)

        sm.get("package", size=(14, 14))
        sm.clear_cache()
        sm.get("package", size=(14, 14))
        # Tras limpiar la caché, la segunda llamada recarga.
        assert calls["n"] == 2

    def test_color_del_sprite_se_preserva(self, tmp_path):
        # Verifica que el contenido del PNG llega intacto: el píxel
        # central del sprite escalado tiene el color con el que lo
        # generamos.
        _write_png(tmp_path / "package.png", size=(32, 32), color=(10, 200, 30, 255))
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        spr = sm.get("package", size=(16, 16))
        assert spr is not None
        cx, cy = 8, 8
        r, g, b, a = spr.get_at((cx, cy))
        assert (r, g, b) == (10, 200, 30)


# ---------------------------------------------------------------------------
# Grupo 3: composición de capas del dron (rediseño)
# ---------------------------------------------------------------------------


class TestDroneComposite:
    """get_drone_composite: apilado de capas, anclas, niveles, fallback, caché."""

    def _make_layers(self, tmp_path, *, body=True, face=True, box=True,
                     carriers=False):
        """Genera PNGs de capas a tamaño nativo con colores distinguibles.

        Cada capa un color plano distinto para poder verificar por píxel
        que aparece donde toca tras la composición.
        """
        a = Theme.default().sprite_anchors
        if body:
            s = pygame.Surface(a.body_size, pygame.SRCALPHA)
            s.fill((100, 100, 100, 255))  # cuerpo gris
            pygame.image.save(s, str(tmp_path / "drone.png"))
        if face:
            s = pygame.Surface(a.face_size, pygame.SRCALPHA)
            s.fill((0, 0, 255, 255))  # cara azul
            pygame.image.save(s, str(tmp_path / "face.png"))
        if box:
            s = pygame.Surface(a.box_size, pygame.SRCALPHA)
            s.fill((255, 0, 0, 255))  # caja roja
            pygame.image.save(s, str(tmp_path / "box.png"))
        if carriers:
            for i, name in enumerate(
                ["carrier.png", "carrier1.png", "carrier2.png",
                 "carrier3.png", "carrier4.png", "carrier5.png"]
            ):
                s = pygame.Surface(a.carrier_size, pygame.SRCALPHA)
                # tono verde creciente con el nivel para distinguirlos
                s.fill((0, 100 + i * 25, 0, 255))
                pygame.image.save(s, str(tmp_path / name))

    def test_compone_cuerpo_solo(self, tmp_path):
        self._make_layers(tmp_path, body=True, face=False, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        comp = sm.get_drone_composite(
            body="drone", face=None, obj=None, dest_size=(58, 58),
        )
        assert comp is not None
        # El lienzo es 58x80 (canvas_size); 1:1 sin reescalar.
        assert comp.get_size() == (58, 80)
        # El cuerpo (gris) está en la franja superior:
        r, g, b, _ = comp.get_at((29, 29))
        assert (r, g, b) == (100, 100, 100)

    def test_sin_cuerpo_devuelve_none(self, tmp_path):
        # Falta drone.png → no se puede componer → None (painter usa primitiva).
        self._make_layers(tmp_path, body=False, face=True, box=True)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        comp = sm.get_drone_composite(
            body="drone", face="face", obj="box", dest_size=(58, 58),
        )
        assert comp is None

    def test_cara_se_ancla_a_face_top(self, tmp_path):
        self._make_layers(tmp_path, body=True, face=True, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        a = theme.sprite_anchors
        comp = sm.get_drone_composite(
            body="drone", face="face", obj=None, dest_size=(58, 58),
        )
        assert comp is not None
        # La cara (azul) debe estar a face_top=31 desde arriba, centrada
        # en X. El centro de la cara cae en (29, 31 + face_h/2).
        face_cx = 29
        face_cy = a.face_top + a.face_size[1] // 2
        r, g, b, _ = comp.get_at((face_cx, face_cy))
        assert (r, g, b) == (0, 0, 255)

    def test_caja_pegada_al_borde_inferior_del_cuerpo(self, tmp_path):
        self._make_layers(tmp_path, body=True, face=False, box=True)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        a = theme.sprite_anchors
        comp = sm.get_drone_composite(
            body="drone", face=None, obj="box", dest_size=(58, 58),
        )
        assert comp is not None
        # La caja (roja) cabe entera dentro del cuerpo, pegada al borde
        # inferior (y = body_h - box_h .. body_h). Su centro:
        body_h = a.body_size[1]
        box_cy = body_h - a.box_size[1] // 2
        r, g, b, _ = comp.get_at((29, box_cy))
        assert (r, g, b) == (255, 0, 0)

    def test_carrier_cuelga_por_debajo_del_cuerpo(self, tmp_path):
        self._make_layers(tmp_path, body=True, face=False, box=False, carriers=True)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        a = theme.sprite_anchors
        comp = sm.get_drone_composite(
            body="drone", face=None, obj="carrier_3", dest_size=(58, 58),
        )
        assert comp is not None
        # El carrier nivel 3 (verde 0,175,0) cuelga: su techo a
        # body_h - carrier_overlap, y se extiende hacia abajo. Probamos
        # un píxel claramente dentro del carrier, por debajo del cuerpo.
        body_h = a.body_size[1]
        carrier_top = body_h - a.carrier_overlap
        probe_y = carrier_top + a.carrier_size[1] // 2  # centro vertical del carrier
        r, g, b, _ = comp.get_at((29, probe_y))
        assert (r, g, b) == (0, 175, 0)

    def test_niveles_de_carrier_distintos(self, tmp_path):
        self._make_layers(tmp_path, body=True, face=False, box=False, carriers=True)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        a = theme.sprite_anchors
        body_h = a.body_size[1]
        probe_y = body_h - a.carrier_overlap + a.carrier_size[1] // 2
        c0 = sm.get_drone_composite(body="drone", face=None, obj="carrier_0", dest_size=(58, 58))
        c5 = sm.get_drone_composite(body="drone", face=None, obj="carrier_5", dest_size=(58, 58))
        # Distintos niveles → distinto color en el carrier.
        assert c0.get_at((29, probe_y))[:3] != c5.get_at((29, probe_y))[:3]

    def test_fallback_cara_a_face_basica(self, tmp_path):
        # Pedimos face_NE (no existe) pero face.png sí: debe caer a face.
        self._make_layers(tmp_path, body=True, face=True, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        a = theme.sprite_anchors
        comp = sm.get_drone_composite(
            body="drone", face="face_NE", obj=None, dest_size=(58, 58),
        )
        assert comp is not None
        # La cara básica (azul) debe estar en su ancla:
        face_cy = a.face_top + a.face_size[1] // 2
        r, g, b, _ = comp.get_at((29, face_cy))
        assert (r, g, b) == (0, 0, 255)

    def test_compuesto_se_cachea(self, tmp_path, monkeypatch):
        self._make_layers(tmp_path, body=True, face=True, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        calls = {"n": 0}
        original = sm._load_native_by_key

        def spy(key):
            calls["n"] += 1
            return original(key)

        monkeypatch.setattr(sm, "_load_native_by_key", spy)
        a = sm.get_drone_composite(body="drone", face="face", obj=None, dest_size=(58, 58))
        n_after_first = calls["n"]
        b = sm.get_drone_composite(body="drone", face="face", obj=None, dest_size=(58, 58))
        # Segunda llamada: mismo objeto cacheado, sin nuevas cargas nativas.
        assert a is b
        assert calls["n"] == n_after_first

    def test_escalado_1_a_1_sin_reescalar(self, tmp_path):
        # dest_size == body_size nativo → el compuesto conserva el lienzo
        # nativo 58x80 sin reescalar (pixel art intacto).
        self._make_layers(tmp_path, body=True, face=False, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        comp = sm.get_drone_composite(body="drone", face=None, obj=None, dest_size=(58, 58))
        assert comp.get_size() == (58, 80)

    def test_escalado_a_destino_distinto(self, tmp_path):
        # dest_size != nativo → reescala el lienzo entero por el mismo factor.
        self._make_layers(tmp_path, body=True, face=False, box=False)
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)
        # destino 29x29 (mitad) → lienzo 58x80 escala a 29x40.
        comp = sm.get_drone_composite(body="drone", face=None, obj=None, dest_size=(29, 29))
        assert comp.get_size() == (29, 40)


# ---------------------------------------------------------------------------
# Sprites de persona: escaneo de sets (personN.png) y loader directo get_person
# (variante _box con fallback a la base).
# ---------------------------------------------------------------------------
def _center_color(surf):
    return surf.get_at((surf.get_width() // 2, surf.get_height() // 2))[:3]


class TestPersonSprites:
    def test_scan_detecta_sets_ordenados_por_n(self, tmp_path):
        _write_png(tmp_path / "person2.png")
        _write_png(tmp_path / "person10.png")
        _write_png(tmp_path / "person1.png")
        _write_png(tmp_path / "person1_box.png")  # variante no cuenta como set
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        # Orden por N (no alfabético: person10 va DESPUÉS de person2).
        assert sm.person_sprite_bases() == ("person1", "person2", "person10")

    def test_sin_person_sprites_devuelve_vacio(self, tmp_path):
        _write_png(tmp_path / "box.png")
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        assert sm.person_sprite_bases() == ()

    def test_scan_se_cachea(self, tmp_path):
        _write_png(tmp_path / "person1.png")
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        first = sm.person_sprite_bases()
        _write_png(tmp_path / "person2.png")  # aparece tras el primer escaneo
        assert sm.person_sprite_bases() == first  # cacheado, no re-escanea

    def test_get_person_base(self, tmp_path):
        _write_png(tmp_path / "person1.png", color=(10, 20, 30, 255))
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        spr = sm.get_person("person1", box=False, size=(36, 36))
        assert spr is not None and spr.get_size() == (36, 36)

    def test_get_person_box_usa_la_variante(self, tmp_path):
        _write_png(tmp_path / "person1.png", color=(200, 0, 0, 255))      # base roja
        _write_png(tmp_path / "person1_box.png", color=(0, 0, 200, 255))  # box azul
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        spr = sm.get_person("person1", box=True, size=(8, 8))
        assert _center_color(spr) == (0, 0, 200)  # la variante box

    def test_get_person_box_cae_a_base_si_no_hay_variante(self, tmp_path):
        _write_png(tmp_path / "person2.png", color=(0, 200, 0, 255))  # sin person2_box
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        spr = sm.get_person("person2", box=True, size=(8, 8))
        assert spr is not None and _center_color(spr) == (0, 200, 0)  # cae a la base

    def test_get_person_inexistente_devuelve_none(self, tmp_path):
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        assert sm.get_person("person9", box=False, size=(8, 8)) is None

    def test_get_person_preserva_aspect_ratio(self, tmp_path):
        # Persona no cuadrada (20x35, como los assets reales): al pedir un
        # tamaño cuadrado NO se deforma; se escala preservando proporción para
        # caber dentro de la caja (fit, no stretch).
        _write_png(tmp_path / "person1.png", size=(20, 35))
        sm = SpriteManager(Theme.default().with_overrides(sprite_dir=str(tmp_path)))
        spr = sm.get_person("person1", box=False, size=(36, 36))
        w, h = spr.get_size()
        assert (w, h) == (21, 36)   # 20:35 escalado a alto 36 → ~21 de ancho
        assert w < h                 # sigue siendo retrato, no achatado a cuadrado
