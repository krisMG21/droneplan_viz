"""Tests del render UNIFORME por entidad (render_world_at).

render_world_at sustituye al par (sample_with_neighbors + render_frame) en
demos y app. A diferencia del render por "protagonista único", interpola CADA
entidad sobre el span completo de su acción, lo que arregla de forma uniforme:

  - Concurrencia: varios drones moviéndose a la vez se animan sin
    teletransporte, incluso el drone "intercalado" cuyo flip de posición caía
    en una transición de duración 0.
  - Encadenado intra-loc: acciones consecutivas en la misma loc no vuelven al
    centro entre ellas (issue 1 de demo_sim).
  - Carrier: tras cargar, el dron NO pasa por el centro antes de agarrar el
    carrier y salir (issue 2).
  - Cámara: zoom=1, pan=(0,0) es píxel-idéntico al camino sin escalado.
"""

from __future__ import annotations

import numpy as np
import pygame
import pygame.surfarray as sa
import pytest

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.domain import (
    AtLocation,
    Content,
    Location,
    Package,
    Person,
    Transporter,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.render import Theme, Timeline, locate_entity_at, render_world_at
from droneplan_viz.render.painter import (
    _drone_segments,
    _scene_viewbox,
    compute_layout,
)
from droneplan_viz.render.sprite_manager import SpriteManager
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

SIZE = (900, 600)


def _surface() -> pygame.Surface:
    return pygame.Surface(SIZE)


def _identical(s1: pygame.Surface, s2: pygame.Surface) -> bool:
    return np.array_equal(sa.array3d(s1), sa.array3d(s2))


# ---------------------------------------------------------------------------
# Worlds / planes de prueba
# ---------------------------------------------------------------------------


def _world_two_drones() -> World:
    """Dos drones en deposito; ana en casa1, bob en casa2 (concurrencia)."""
    med = Content(id="medicina")
    com = Content(id="comida")
    locs = {
        "deposito": Location(id="deposito"),
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
    }
    drones = {
        "d1": Drone(id="d1", position="deposito", arms=(Arm(id="izq"),)),
        "d2": Drone(id="d2", position="deposito", arms=(Arm(id="izq"),)),
    }
    persons = {
        "ana": Person(id="ana", position="casa1", needs=(med,)),
        "bob": Person(id="bob", position="casa2", needs=(com,)),
    }
    packages = {
        "pkg_med": Package(id="pkg_med", contains=med, at=AtLocation(loc_id="deposito")),
        "pkg_com": Package(id="pkg_com", contains=com, at=AtLocation(loc_id="deposito")),
    }
    costs = {
        ("deposito", "casa1"): 6.0, ("casa1", "deposito"): 6.0,
        ("deposito", "casa2"): 8.0, ("casa2", "deposito"): 8.0,
    }
    return World(locations=locs, drones=drones, persons=persons,
                 packages=packages, contents={"medicina": med, "comida": com},
                 costs=costs)


def _plan_concurrent() -> Plan:
    """d1 y d2 actúan a la vez: PickUp@0, Move@2, Deliver al final."""
    def S(cmd, t):
        return ScheduledCommand(command=cmd, start_time=t)
    return Plan(scheduled=(
        S(PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med",
                 duration=2.0, command_id="d1_pk"), 0.0),
        S(Move(drone_id="d1", destination_id="casa1",
               duration=6.0, command_id="d1_mv"), 2.0),
        S(Deliver(drone_id="d1", package_id="pkg_med", person_id="ana",
                  duration=2.0, command_id="d1_dl"), 8.0),
        S(PickUp(drone_id="d2", arm_id="izq", package_id="pkg_com",
                 duration=2.0, command_id="d2_pk"), 0.0),
        S(Move(drone_id="d2", destination_id="casa2",
               duration=8.0, command_id="d2_mv"), 2.0),
        S(Deliver(drone_id="d2", package_id="pkg_com", person_id="bob",
                  duration=2.0, command_id="d2_dl"), 10.0),
    ))


def _world_carrier() -> World:
    """d1 en deposito con un carrier y dos paquetes; pacientes en hospital."""
    a = Content(id="a")
    b = Content(id="b")
    locs = {"deposito": Location(id="deposito"), "hospital": Location(id="hospital")}
    drones = {"d1": Drone(id="d1", position="deposito",
                          arms=(Arm(id="izq"),))}
    transporters = {"carrier": Transporter(id="carrier", position="deposito",
                                           capacity=4)}
    persons = {
        "p0": Person(id="p0", position="hospital", needs=(a,)),
        "p1": Person(id="p1", position="hospital", needs=(b,)),
    }
    packages = {
        "pkg0": Package(id="pkg0", contains=a, at=AtLocation(loc_id="deposito")),
        "pkg1": Package(id="pkg1", contains=b, at=AtLocation(loc_id="deposito")),
    }
    costs = {("deposito", "hospital"): 4.0, ("hospital", "deposito"): 4.0}
    return World(locations=locs, drones=drones, transporters=transporters,
                 persons=persons, packages=packages,
                 contents={"a": a, "b": b}, costs=costs)


def _plan_carrier() -> Plan:
    """Cargar 2 paquetes en el carrier, llevarlo y descargar/entregar."""
    sched = []
    t = 0.0

    def add(cmd, dt):
        nonlocal t
        sched.append(ScheduledCommand(command=cmd, start_time=t))
        t += dt
    for i in (0, 1):
        add(PickUp(drone_id="d1", arm_id="izq", package_id=f"pkg{i}",
                   duration=1.0, command_id=f"pk{i}"), 1.0)
        add(LoadIntoTransporter(drone_id="d1", package_id=f"pkg{i}",
                                transporter_id="carrier", duration=1.0,
                                command_id=f"ld{i}"), 1.0)
    add(MoveWithTransporter(drone_id="d1", transporter_id="carrier",
                            destination_id="hospital", duration=4.0,
                            command_id="mv"), 4.0)
    for i in (0, 1):
        add(UnloadFromTransporter(drone_id="d1", arm_id="izq", package_id=f"pkg{i}",
                                  transporter_id="carrier", duration=1.0,
                                  command_id=f"ul{i}"), 1.0)
        add(Deliver(drone_id="d1", package_id=f"pkg{i}", person_id=f"p{i}",
                    duration=1.0, command_id=f"dl{i}"), 1.0)
    return Plan(scheduled=tuple(sched))


def _timeline(world: World, plan: Plan) -> Timeline:
    runner = PlanRunner(world)
    runner.execute(plan)
    return Timeline(runner.history, theme=Theme.default())


# ---------------------------------------------------------------------------
# Concurrencia: sin teletransporte
# ---------------------------------------------------------------------------


class TestConcurrencia:
    def test_drone_intercalado_interpola_su_move_sin_teletransporte(self):
        # d2 es el drone "intercalado": su Move abarca varias transiciones y
        # su flip de posición caía en una de duración 0 → teletransporte. Con
        # el render por entidad debe interpolar suave sobre TODO su span.
        theme = Theme.default()
        tl = _timeline(_world_two_drones(), _plan_concurrent())
        segs = _drone_segments(tl)["d2"]
        mv = next(s for s in segs if s["kind"] == "move")
        assert mv["v1"] - mv["v0"] > 1.0  # el move ocupa duración real, no 0
        # Muestrear la X de d2 a lo largo de su move: estrictamente monótona.
        xs = []
        for k in range(1, 10):
            t = mv["v0"] + (mv["v1"] - mv["v0"]) * k / 10.0
            x, _y = locate_entity_at(tl, t, "d2", SIZE, theme=theme)
            xs.append(x)
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)), xs

    def test_ambos_drones_se_mueven_a_la_vez(self):
        # En el solape de los dos Move, AMBOS drones están fuera de su loc de
        # origen (deposito) simultáneamente: prueba de concurrencia real.
        theme = Theme.default()
        tl = _timeline(_world_two_drones(), _plan_concurrent())
        segs = _drone_segments(tl)
        mv1 = next(s for s in segs["d1"] if s["kind"] == "move")
        mv2 = next(s for s in segs["d2"] if s["kind"] == "move")
        lo = max(mv1["v0"], mv2["v0"]) + 0.5
        hi = min(mv1["v1"], mv2["v1"]) - 0.5
        assert hi > lo  # hay solape
        viewbox = _scene_viewbox(SIZE, theme)
        layout = compute_layout(
            tl.history.at(0).world, viewbox,
            spread_factor=theme.location_spread_factor,
        )
        dep = layout.location_position("deposito")
        t = (lo + hi) / 2.0
        p1 = locate_entity_at(tl, t, "d1", SIZE, theme=theme)
        p2 = locate_entity_at(tl, t, "d2", SIZE, theme=theme)
        # Ninguno está aún en deposito (ambos en vuelo).
        d1 = ((p1[0] - dep.x) ** 2 + (p1[1] - dep.y) ** 2) ** 0.5
        d2 = ((p2[0] - dep.x) ** 2 + (p2[1] - dep.y) ** 2) ** 0.5
        assert d1 > 20 and d2 > 20


# ---------------------------------------------------------------------------
# Encadenado intra-loc (issue 1) y carrier (issue 2): sin pasar por el centro
# ---------------------------------------------------------------------------


class TestSinCentro:
    def test_acciones_consecutivas_no_vuelven_al_centro(self):
        # Entre dos acciones de paquete consecutivas en la misma loc (un solo
        # drone), el dron NO debe volver al centro: se queda en su aproximación
        # y encadena directo a la siguiente. (La PRIMERA acción sí arranca del
        # centro: eso es correcto y no se comprueba aquí.)
        theme = Theme.default()
        tl = _timeline(_world_carrier(), _plan_carrier())
        segs = _drone_segments(tl)["d1"]
        act_idx = [i for i, s in enumerate(segs) if s["kind"] == "act"]
        viewbox = _scene_viewbox(SIZE, theme)
        layout = compute_layout(
            tl.history.at(0).world, viewbox,
            spread_factor=theme.location_spread_factor,
        )
        center = layout.drone_position("d1", theme)
        # Bordes entre cargas consecutivas (act0→act1, act1→act2, act2→act3):
        # ambos extremos son 'act' contiguos (sin Move en medio).
        checked = 0
        for k in range(len(act_idx) - 1):
            i_a, i_b = act_idx[k], act_idx[k + 1]
            if any(segs[j]["kind"] == "move" for j in range(i_a, i_b + 1)):
                continue  # hay un Move entre medias: no es encadenado intra-loc
            t = segs[i_a]["v1"]  # frontera entre las dos acciones
            x, y = locate_entity_at(tl, t, "d1", SIZE, theme=theme)
            dist = ((x - center.x) ** 2 + (y - center.y) ** 2) ** 0.5
            assert dist > 10, (k, dist)
            checked += 1
        assert checked >= 1  # nos aseguramos de haber comprobado algún borde

    def test_carga_a_move_no_pasa_por_el_centro(self):
        # issue 2: tras la última carga, el dron agarra el carrier y sale sin
        # volver al centro de la loc.
        theme = Theme.default()
        tl = _timeline(_world_carrier(), _plan_carrier())
        segs = _drone_segments(tl)["d1"]
        mv = next(s for s in segs if s["kind"] == "move")
        viewbox = _scene_viewbox(SIZE, theme)
        layout = compute_layout(
            tl.history.at(mv["i_start"]).world, viewbox,
            spread_factor=theme.location_spread_factor,
        )
        center = layout.drone_position("d1", theme)
        worst = 1e9
        for k in range(20):
            t = max(0.0, mv["v0"] - 1.0 + k * 0.08)
            x, y = locate_entity_at(tl, t, "d1", SIZE, theme=theme)
            worst = min(worst, ((x - center.x) ** 2 + (y - center.y) ** 2) ** 0.5)
        assert worst > 15, worst


# ---------------------------------------------------------------------------
# Robustez + cámara
# ---------------------------------------------------------------------------


class TestRobustezYCamara:
    @pytest.mark.parametrize("builder", [
        lambda: (_world_two_drones(), _plan_concurrent()),
        lambda: (_world_carrier(), _plan_carrier()),
    ])
    def test_renderiza_toda_la_timeline_sin_crash(self, builder):
        world, plan = builder()
        tl = _timeline(world, plan)
        theme = Theme.default()
        sm = SpriteManager(theme)
        surf = _surface()
        for k in range(41):
            t = tl.duration * k / 40.0
            render_world_at(surf, tl, t, theme=theme, sprite_manager=sm)

    def test_camara_identidad_es_pixel_identica(self):
        # zoom=1, pan=(0,0) debe ser idéntico a omitir los kwargs (fast-path).
        tl = _timeline(_world_carrier(), _plan_carrier())
        theme = Theme.default()
        sm = SpriteManager(theme)
        t = tl.duration * 0.5
        s1, s2 = _surface(), _surface()
        render_world_at(s1, tl, t, theme=theme, sprite_manager=sm)
        render_world_at(s2, tl, t, theme=theme, sprite_manager=sm,
                        zoom=1.0, pan=(0, 0))
        assert _identical(s1, s2)

    def test_camara_zoom_cambia_la_imagen(self):
        tl = _timeline(_world_carrier(), _plan_carrier())
        theme = Theme.default()
        sm = SpriteManager(theme)
        t = tl.duration * 0.5
        s1, s2 = _surface(), _surface()
        render_world_at(s1, tl, t, theme=theme, sprite_manager=sm)
        render_world_at(s2, tl, t, theme=theme, sprite_manager=sm,
                        zoom=1.8, pan=(10, -5))
        assert not _identical(s1, s2)
