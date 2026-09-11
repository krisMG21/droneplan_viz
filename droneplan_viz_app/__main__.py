"""Entry point para `python -m droneplan_viz_app`.

Delegación trivial a main.main(); existe en paralelo al script
`droneplan-viz` definido en pyproject.toml para soportar ambos modos de
invocación (el módulo es útil en CI y debugging; el script es más
ergonómico para el usuario final).
"""
from droneplan_viz_app.main import main

if __name__ == "__main__":
    main()
