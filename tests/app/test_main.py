"""Tests del módulo droneplan_viz_app.main.

Verifican el parsing CLI. NO invocan main() entero porque eso arrancaría
el loop infinito de DroneplanVizApp.run(); para los tests del loop ya
están los de test_app.py.

Sin pygame (todo es argparse).
"""
from __future__ import annotations

import argparse

import pytest

from droneplan_viz_app.main import _parse_window_size, build_parser


# ---------------------------------------------------------------------------
# _parse_window_size
# ---------------------------------------------------------------------------


class TestParseWindowSize:
    def test_formato_correcto(self):
        assert _parse_window_size("1280x800") == (1280, 800)
        assert _parse_window_size("1920x1080") == (1920, 1080)

    def test_sin_x_falla(self):
        with pytest.raises(argparse.ArgumentTypeError, match="WxH"):
            _parse_window_size("1280-800")

    def test_no_entero_falla(self):
        with pytest.raises(argparse.ArgumentTypeError, match="enteros"):
            _parse_window_size("1280xocho")


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.scenario == "demo"
        assert args.window_size == (1280, 800)

    def test_scenario_failure_demo(self):
        parser = build_parser()
        args = parser.parse_args(["--scenario", "failure_demo"])
        assert args.scenario == "failure_demo"

    def test_scenario_invalido_falla(self):
        parser = build_parser()
        with pytest.raises(SystemExit):  # argparse llama a sys.exit en error.
            parser.parse_args(["--scenario", "no_existe"])

    def test_window_size_personalizado(self):
        parser = build_parser()
        args = parser.parse_args(["--window-size", "1600x900"])
        assert args.window_size == (1600, 900)

    def test_window_size_invalido_falla(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--window-size", "muymalo"])

    def test_argumentos_combinados(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--scenario", "failure_demo", "--window-size", "1024x640"]
        )
        assert args.scenario == "failure_demo"
        assert args.window_size == (1024, 640)
