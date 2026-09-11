"""Tests del módulo theme_loader (carga de Monogram en el theme de pygame_gui).

No requieren pygame ni pygame_gui: solo verifican que el JSON generado
resuelve correctamente la ruta del TTF empaquetado y queda parseable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneplan_viz_app.theme_loader import (
    _FONT_PATH_PLACEHOLDER,
    monogram_font_path,
    prepare_ui_theme,
)


class TestPrepareUiTheme:
    def test_genera_un_fichero_existente(self):
        path = prepare_ui_theme()
        assert Path(path).is_file()

    def test_resultado_es_json_parseable(self):
        path = prepare_ui_theme()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Sanity: estructura esperada presente.
        assert "button" in data
        assert "label" in data
        assert "selection_list" in data

    def test_sustituye_el_placeholder_por_una_ruta_real(self):
        path = prepare_ui_theme()
        with open(path, encoding="utf-8") as f:
            contenido = f.read()
        # El placeholder NO debe sobrevivir.
        assert _FONT_PATH_PLACEHOLDER not in contenido
        # Y la ruta debe apuntar a un .ttf que existe.
        data = json.loads(contenido)
        ttf_path = data["button"]["font"]["regular_path"]
        assert Path(ttf_path).is_file(), (
            f"La ruta del TTF generada no existe: {ttf_path}"
        )
        assert ttf_path.endswith(".ttf")

    def test_todas_las_categorias_apuntan_al_mismo_ttf(self):
        """Coherencia: button, label y selection_list deben usar el
        mismo fichero TTF, aunque a distintos tamaños."""
        path = prepare_ui_theme()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rutas = {
            data["button"]["font"]["regular_path"],
            data["label"]["font"]["regular_path"],
            data["selection_list"]["font"]["regular_path"],
        }
        assert len(rutas) == 1, (
            f"Categorías apuntan a TTFs distintos: {rutas}"
        )

    def test_tamanos_son_los_consensuados(self):
        """Tamaños del HUD según decisiones del plan:
        - botones y labels: 24px (texto principal).
        - selection_list: 14px (texto denso del panel de fallos,
          cabe holgadamente sin truncamiento aun con strings largos).
        Si cambias estos valores, este test te recuerda actualizar la
        documentación / mockups."""
        path = prepare_ui_theme()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["button"]["font"]["size"] == "24"
        assert data["label"]["font"]["size"] == "24"
        assert data["selection_list"]["font"]["size"] == "14"

    def test_invocaciones_repetidas_generan_ficheros_independientes(self):
        """Cada llamada produce su propio tempfile (no se cachea), para
        no fastidiar a tests que mockean el contenido o lo manipulan."""
        path1 = prepare_ui_theme()
        path2 = prepare_ui_theme()
        assert path1 != path2

    def test_path_windows_con_backslashes_no_rompe_json(self, monkeypatch, tmp_path):
        """Regresión: en Windows _resolve_font_path devuelve rutas como
        'C:\\Users\\... \\monogram.ttf'. Si se insertan crudas en el JSON,
        los backslashes se interpretan como escape sequences JSON
        inválidas (\\U, \\c, etc.) y json.loads revienta. La función
        normaliza la ruta a forward slashes antes de insertarla.
        """
        # Construimos un .ttf falso en una ruta con apariencia Windows
        # (en realidad usamos tmp_path en POSIX, pero monkeypatcheamos la
        # función resolver para que devuelva un string con backslashes
        # como si fuera Windows).
        fake_ttf = tmp_path / "Users" / "c.marquez" / "monogram.ttf"
        fake_ttf.parent.mkdir(parents=True)
        fake_ttf.write_bytes(b"")

        # Forzamos la ruta con backslashes (como en Windows). El path real
        # bajo el alias sigue siendo accesible: Path acepta ambos.
        windows_style = str(fake_ttf).replace("/", "\\")
        # Verificamos que de verdad introducimos caracteres problemáticos.
        assert "\\U" in windows_style or "\\c" in windows_style or "\\m" in windows_style

        import droneplan_viz_app.theme_loader as tl
        monkeypatch.setattr(tl, "_resolve_font_path", lambda: windows_style)

        # Si no normalizásemos backslashes, esto reventaría con
        # json.JSONDecodeError: Invalid \escape.
        path = prepare_ui_theme()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)  # debe parsear sin error.
        # Y la ruta dentro del JSON debe estar normalizada.
        assert "\\" not in data["button"]["font"]["regular_path"]


class TestMonogramFontPath:
    def test_resuelve_el_ttf_empaquetado(self):
        p = monogram_font_path()
        assert p is not None
        assert Path(p).is_file()
        assert p.endswith("monogram.ttf")

    def test_ruta_normalizada_sin_backslashes(self):
        # Igual que para pygame_gui: forward slashes (válido también en Win).
        p = monogram_font_path()
        assert "\\" not in p

    def test_tolerante_a_fallo_devuelve_none(self, monkeypatch):
        # Si la resolución del TTF lanza, monogram_font_path() devuelve None
        # (la fuente fina es opcional para el render; no debe abortar).
        import droneplan_viz_app.theme_loader as tl

        def _boom() -> str:
            raise FileNotFoundError("simulado: TTF ausente")

        monkeypatch.setattr(tl, "_resolve_font_path", _boom)
        assert tl.monogram_font_path() is None
