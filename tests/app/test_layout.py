"""Tests del módulo droneplan_viz_app.layout.

Verifican que compute_regions:
  1. Devuelve los cuatro rects esperados con la geometría documentada.
  2. Los rects no se solapan (invariante crítico para subsurface).
  3. Lanza ValueError en casos de input inválido.

pygame.Rect funciona sin SDL_VIDEODRIVER ni pygame.init(); por tanto
estos tests son lógica pura.
"""
from __future__ import annotations

import pygame
import pytest

from droneplan_viz_app.layout import (
    DEFAULT_BOTTOM_H,
    DEFAULT_MIN_WORLD,
    DEFAULT_RIGHT_W,
    DEFAULT_TOP_H,
    Regions,
    compute_regions,
)


# ---------------------------------------------------------------------------
# Tamaños por defecto
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_top_h(self):
        assert DEFAULT_TOP_H == 56

    def test_default_bottom_h(self):
        assert DEFAULT_BOTTOM_H == 100

    def test_default_right_w(self):
        assert DEFAULT_RIGHT_W == 380

    def test_default_min_world(self):
        assert DEFAULT_MIN_WORLD == (400, 300)


# ---------------------------------------------------------------------------
# Geometría para el tamaño consensuado 1280×800
# ---------------------------------------------------------------------------


class TestRegionsAt1280x800:
    @pytest.fixture
    def regions(self) -> Regions:
        return compute_regions((1280, 800))

    def test_world_dimensiones(self, regions):
        # Ancho: 1280 - 380 = 900. Alto: 800 - 56 - 100 = 644.
        assert regions.world.width == 900
        assert regions.world.height == 644

    def test_world_posicion(self, regions):
        assert regions.world.x == 0
        assert regions.world.y == 56  # debajo del hud_top.

    def test_hud_top_dimensiones(self, regions):
        assert regions.hud_top == pygame.Rect(0, 0, 900, 56)

    def test_hud_bottom_dimensiones(self, regions):
        # Comienza a y = 800 - 100 = 700.
        assert regions.hud_bottom == pygame.Rect(0, 700, 900, 100)

    def test_hud_right_dimensiones(self, regions):
        # Toda la altura, ancho 380, desde x = 1280 - 380 = 900.
        assert regions.hud_right == pygame.Rect(900, 0, 380, 800)


# ---------------------------------------------------------------------------
# Invariantes geométricos
# ---------------------------------------------------------------------------


class TestRegionsInvariantes:
    def _rects(self, regions: Regions) -> list[pygame.Rect]:
        return [regions.world, regions.hud_top, regions.hud_bottom, regions.hud_right]

    @pytest.mark.parametrize(
        "size",
        [(1280, 800), (1024, 640), (1920, 1080), (800, 500)],
    )
    def test_ningun_par_de_rects_se_solapa(self, size):
        """Crítico para usar subsurface sobre la ventana sin que el HUD
        pinte encima del world (o viceversa). El test cubre varios
        tamaños razonables."""
        # 800x500: ancho left = 800-300 = 500 (>= min 400), alto inner =
        # 500-56-100 = 344 (>= min 300). Cabe justo: caso límite.
        regions = compute_regions(size)
        rects = self._rects(regions)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                # pygame.Rect.colliderect devuelve True si tienen
                # intersección de área positiva.
                assert not rects[i].colliderect(rects[j]), (
                    f"rect {i} {rects[i]} colisiona con rect {j} {rects[j]}"
                )

    def test_rects_cubren_la_ventana_entera(self):
        """La unión de los 4 rects debe cubrir la ventana entera, sin
        huecos. Lo verificamos por suma de áreas == área total."""
        size = (1280, 800)
        regions = compute_regions(size)
        total_area = size[0] * size[1]
        union_area = sum(r.width * r.height for r in self._rects(regions))
        assert union_area == total_area

    def test_world_area_positiva(self):
        regions = compute_regions((1280, 800))
        assert regions.world.width > 0
        assert regions.world.height > 0


# ---------------------------------------------------------------------------
# Casos de error
# ---------------------------------------------------------------------------


class TestComputeRegionsErrors:
    def test_window_size_no_positivo_falla(self):
        with pytest.raises(ValueError, match="positivo"):
            compute_regions((0, 800))
        with pytest.raises(ValueError, match="positivo"):
            compute_regions((1280, 0))
        with pytest.raises(ValueError, match="positivo"):
            compute_regions((-10, -10))

    def test_ancho_insuficiente_falla(self):
        """Si la ventana es más estrecha que right_w + min_world_w, el
        world resultante quedaría por debajo del mínimo. Default:
        300 + 400 = 700. Una ventana de 600 de ancho debe fallar."""
        with pytest.raises(ValueError, match="estrecho"):
            compute_regions((600, 800))

    def test_alto_insuficiente_falla(self):
        """Default: top + bottom + min_world_h = 56 + 100 + 300 = 456.
        Una ventana de 400 de alto debe fallar."""
        with pytest.raises(ValueError, match="bajo"):
            compute_regions((1280, 400))

    def test_mensaje_de_error_incluye_dimensiones_concretas(self):
        """Útil para debugging en producción: el mensaje debe permitir
        diagnosticar el problema sin ejecutar bajo pdb."""
        with pytest.raises(ValueError) as exc_info:
            compute_regions((500, 800))  # left_w = 500 - 380 = 120, min = 400.
        msg = str(exc_info.value)
        assert "120" in msg  # left_w resultante.
        assert "400" in msg  # mínimo esperado.


# ---------------------------------------------------------------------------
# Argumentos opcionales
# ---------------------------------------------------------------------------


class TestCustomDimensions:
    def test_top_h_personalizado(self):
        regions = compute_regions((1280, 800), top_h=80)
        assert regions.hud_top.height == 80
        # world se ajusta: 800 - 80 - 100 = 620.
        assert regions.world.height == 620

    def test_right_w_personalizado(self):
        regions = compute_regions((1280, 800), right_w=400)
        assert regions.hud_right.width == 400
        assert regions.world.width == 880  # 1280 - 400.

    def test_min_world_personalizado_permite_ventana_mas_pequena(self):
        """Si bajamos el mínimo, una ventana antes inválida ahora vale."""
        # Con default (400, 300), (600, 500) falla.
        with pytest.raises(ValueError):
            compute_regions((600, 500))
        # Con mínimo (200, 200), debe pasar.
        regions = compute_regions((600, 500), min_world=(200, 200))
        assert regions.world.width >= 200
        assert regions.world.height >= 200


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------


class TestRegionsType:
    def test_devuelve_dataclass_regions(self):
        regions = compute_regions((1280, 800))
        assert isinstance(regions, Regions)

    def test_regions_es_frozen(self):
        regions = compute_regions((1280, 800))
        # Intentar reasignar un campo debe lanzar (frozen + slots).
        with pytest.raises(Exception):  # FrozenInstanceError o AttributeError
            regions.world = pygame.Rect(0, 0, 1, 1)  # type: ignore[misc]

    def test_campos_son_pygame_rect(self):
        regions = compute_regions((1280, 800))
        assert isinstance(regions.world, pygame.Rect)
        assert isinstance(regions.hud_top, pygame.Rect)
        assert isinstance(regions.hud_bottom, pygame.Rect)
        assert isinstance(regions.hud_right, pygame.Rect)


# ---------------------------------------------------------------------------
# HudLayout
# ---------------------------------------------------------------------------


from droneplan_viz_app.layout import HudLayout


class TestHudLayout:
    def test_defaults_construyen(self):
        L = HudLayout()
        assert L.top_btn_h > 0
        assert L.bottom_row1_h > 0
        assert L.right_label_h > 0

    def test_es_frozen(self):
        L = HudLayout()
        with pytest.raises(Exception):
            L.top_btn_h = 999  # type: ignore[misc]

    def test_acepta_overrides_via_replace(self):
        """Patrón documentado: dataclasses.replace produce variantes
        sin tocar el módulo."""
        import dataclasses
        L1 = HudLayout()
        L2 = dataclasses.replace(L1, top_btn_h=44, bottom_row1_h=30)
        assert L2.top_btn_h == 44
        assert L2.bottom_row1_h == 30
        # Los demás campos se mantienen.
        assert L2.top_btn_w == L1.top_btn_w

    def test_colores_de_overlay_definidos(self):
        L = HudLayout()
        assert len(L.mark_color) == 3
        assert len(L.halo_color) == 4
        # Halo con alpha < 255 (translúcido).
        assert L.halo_color[3] < 255
