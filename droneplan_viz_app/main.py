"""Entry point CLI: comando `droneplan-viz` tras `pip install -e ".[app]"`.

Parsea argumentos de línea de comandos y arranca DroneplanVizApp.

Uso:
    droneplan-viz
    droneplan-viz --scenario failure_demo
    droneplan-viz --scenario demo --window-size 1600x900

Equivalente:
    python -m droneplan_viz_app
"""
from __future__ import annotations

import argparse
import sys

from droneplan_viz_app.app import DroneplanVizApp
from droneplan_viz_app.scenarios import SCENARIOS


def _parse_window_size(spec: str) -> tuple[int, int]:
    """Convierte 'WxH' en (W, H). Lanza argparse-friendly error si no es válido."""
    if "x" not in spec:
        raise argparse.ArgumentTypeError(
            f"formato esperado WxH (p.ej. 1280x800), recibido {spec!r}"
        )
    try:
        w_str, h_str = spec.split("x", 1)
        return (int(w_str), int(h_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"WxH no es un par de enteros válido: {spec!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser. Expuesto separadamente para los tests."""
    parser = argparse.ArgumentParser(
        prog="droneplan-viz",
        description=(
            "Visualizador interactivo del proyecto droneplan_viz. "
            "Reproduce un plan de logística humanitaria con drones "
            "(escenarios de demo construidos programáticamente)."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="demo",
        help="Escenario a cargar al arrancar (default: demo).",
    )
    parser.add_argument(
        "--window-size",
        type=_parse_window_size,
        default=(1280, 800),
        metavar="WxH",
        help="Tamaño inicial de la ventana, formato WxH (default: 1280x800).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI.

    Args:
        argv: argumentos sin el nombre del programa. None = leer
            sys.argv[1:] (comportamiento por defecto). Lo expone para
            los tests.

    Returns:
        código de salida (0 = éxito).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    app = DroneplanVizApp(
        scenario_name=args.scenario,
        window_size=args.window_size,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
