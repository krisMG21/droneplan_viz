"""Ejemplo D — ESCALA: reparto masivo con muchos elementos.

Genera, por programa, un escenario grande y lo SIMULA, para mostrar que la
fachada y el runtime aguantan una cantidad de elementos muy superior a la
de los ejemplos didácticos:

    1 depósito central + N casas (una persona y un paquete por casa)
    D drones repartiendo EN PARALELO.

El reparto se asigna round-robin: el dron d atiende las casas d, d+D,
d+2D, … Cada dron trabaja en su propia PISTA temporal (recoge en el
depósito, vuela a la casa, entrega, vuelve, repite), y todas las pistas
arrancan en t=0, así que los drones operan concurrentemente sobre recursos
disjuntos (drones, paquetes y personas distintos). El makespan refleja el
paralelismo: lo marca la pista más larga, no la suma de todas.

El plan es TEMPORAL (todas las acciones con inicio=), generado con bucles
Python: justo el tipo de plan que sería tediosísimo escribir a mano y que
la fachada hace trivial.

Por defecto solo simula e imprime métricas (incluido el tiempo de
simulación). Con --run abre además la ventana interactiva.

Ejecutar:
    python examples/escala_reparto_masivo.py                 # simula 24 casas, 6 drones
    python examples/escala_reparto_masivo.py --casas 40 --drones 8
    python examples/escala_reparto_masivo.py --run           # + ventana
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz  # noqa: E402

# Contenidos que se van rotando entre las casas.
_CONTENIDOS = ("medicina", "comida", "agua")

# Duraciones (segundos de plan) de cada fase.
_DUR_RECOGER = 5.0
_DUR_VOLAR = 8.0
_DUR_ENTREGAR = 5.0


def construir(n_casas: int, n_drones: int) -> DronePlanViz:
    viz = DronePlanViz()

    # --- Mundo ---
    viz.world.location("deposito")
    for c in _CONTENIDOS:
        viz.world.content(c)

    costes: dict[tuple[str, str], float] = {}
    for k in range(n_casas):
        casa = f"casa{k:02d}"
        contenido = _CONTENIDOS[k % len(_CONTENIDOS)]
        viz.world.location(casa)
        # Arista depósito<->casa: imprescindible para que el vuelo sea posible.
        costes[("deposito", casa)] = 6.0 + (k % 7)
        viz.world.person(f"vecino{k:02d}", at=casa, necesita=[contenido])
        viz.world.package(f"caja{k:02d}", contiene=contenido, at="deposito")
    viz.world.costes(costes, simetrico=True)

    # --- Agentes: D drones de dos brazos en el depósito ---
    for d in range(n_drones):
        viz.agents.drone(f"dron{d}", at="deposito")

    # --- Plan temporal por pistas (una por dron), todas desde t=0 ---
    for d in range(n_drones):
        casas_del_dron = list(range(d, n_casas, n_drones))
        t = 0.0
        for pos, k in enumerate(casas_del_dron):
            casa = f"casa{k:02d}"
            dron = f"dron{d}"
            caja = f"caja{k:02d}"
            vecino = f"vecino{k:02d}"

            viz.recoger(
                dron, caja=caja, brazo="izq",
                inicio=t, duracion=_DUR_RECOGER, id=f"recoger_d{d}_k{k:02d}",
            )
            t += _DUR_RECOGER
            viz.mover(
                dron, a=casa,
                inicio=t, duracion=_DUR_VOLAR, id=f"ir_d{d}_k{k:02d}",
            )
            t += _DUR_VOLAR
            viz.entregar(
                dron, caja=caja, a=vecino,
                inicio=t, duracion=_DUR_ENTREGAR, id=f"entregar_d{d}_k{k:02d}",
            )
            t += _DUR_ENTREGAR
            # Volver al depósito para la siguiente casa (salvo la última).
            if pos < len(casas_del_dron) - 1:
                viz.mover(
                    dron, a="deposito",
                    inicio=t, duracion=_DUR_VOLAR, id=f"volver_d{d}_k{k:02d}",
                )
                t += _DUR_VOLAR

    return viz


def _parse_int(flag: str, default: int) -> int:
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            print(f"valor inválido para {flag}; uso {default}")
    return default


def main() -> None:
    n_casas = _parse_int("--casas", 24)
    n_drones = _parse_int("--drones", 6)

    t0 = time.perf_counter()
    viz = construir(n_casas, n_drones)
    t_build_state = time.perf_counter()

    world, plan = viz.build()
    t_build_plan = time.perf_counter()

    res = viz.simular()
    t_sim = time.perf_counter()

    print("Ejemplo D — escala: reparto masivo")
    print("  Configuración:")
    print(f"    casas / drones        : {n_casas} / {n_drones}")
    print("  Tamaño del escenario:")
    print(f"    localizaciones        : {len(world.locations)}")
    print(f"    drones                : {len(world.drones)}")
    print(f"    paquetes              : {len(world.packages)}")
    print(f"    personas              : {len(world.persons)}")
    print(f"    aristas de coste      : {len(world.costs)}")
    print(f"    acciones del plan     : {len(plan.scheduled)}")
    print("  Resultado de la simulación:")
    print(f"    éxito                 : {res.succeeded}")
    print(f"    makespan              : {res.makespan}")
    print(f"    fallos                : {len(res.failures)}")
    print(f"    acciones aplicadas    : {res.final_metrics.action_count}")
    print(f"    snapshots de historia : {res.history_length}")
    print("  Tiempos (ms):")
    print(f"    declarar mundo+plan   : {(t_build_state - t0) * 1e3:.1f}")
    print(f"    build() (World+Plan)  : {(t_build_plan - t_build_state) * 1e3:.1f}")
    print(f"    simular() (runner)    : {(t_sim - t_build_plan) * 1e3:.1f}")

    if res.failures:
        print("  Primeros fallos:")
        for f in res.failures[:5]:
            print(f"    -> [{f.kind}] {f.reason}")

    if "--run" in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
