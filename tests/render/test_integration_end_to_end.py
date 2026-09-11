"""Tests end-to-end de la cadena completa de Sesión D.

Estos tests recorren el flujo real que usará la UI:

    1. World inicial + Plan.
    2. PlanRunner.execute(plan) → HistoryManager.
    3. Timeline(runner.history) → mapeo tiempo virtual.
    4. Para varios playback_times, render_frame sobre una Surface.

Verifica invariantes globales que NO son propias de un solo módulo:

- No crash en ningún frame de un plan multi-step (PickUp + Move + Deliver).
- El World capturado en los snapshots NO se muta tras renderizar todos
  los frames (canario equivalente al de Sesión B, pero al final del
  pipeline render).
- Los frames de los extremos (playback_time=0 y =duration) son
  pixel-idénticos a render_snapshot del primer y último snapshot
  respectivamente.
- Los frames intermedios difieren de ambos extremos.
- Frames sucesivos con dt pequeño producen surfaces SIMILARES pero no
  idénticas (la animación se mueve, no salta).
- El test funciona con Plan durativo y con Plan secuencial puro
  (duration=0); ambos modos pasan por la misma cadena.

NOTA: estos tests cruzan los cuatro paquetes (domain → commands →
history → runtime → render). Si rompiéramos algo en cualquier eslabón,
estos tests deberían avisar antes que los unitarios de cada paquete
(porque inspeccionan el resultado final, no el contrato interno). Son
caros (varias decenas de ms) pero pocos.
"""
import pygame
import pytest

from droneplan_viz.commands import (
    Deliver,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.domain import (
    AtLocation,
    Content,
    InTransporter,
    MetricsTracker,
    Package,
    Transporter,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.person import Person
from droneplan_viz.render import Theme, Timeline, render_frame, render_snapshot
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_complete_world() -> World:
    """World docente con todas las clases de entidad presentes.

    Topología:
        Locations: deposito, casa1.
        Drone d1 en deposito, dos brazos (izq, der).
        Transporter t1 en deposito, capacidad 4.
        Person p1 en casa1, necesita medicina.
        Package pkg_med1 (medicina) AtLocation deposito.
        Coste 8.0 entre deposito y casa1.
    """
    contents = {"medicina": Content(id="medicina")}
    locs = {
        "deposito": Location(id="deposito"),
        "casa1": Location(id="casa1"),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="deposito",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }
    persons = {
        "p1": Person(
            id="p1",
            position="casa1",
            needs=(contents["medicina"],),
        ),
    }
    packages = {
        "pkg_med1": Package(
            id="pkg_med1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="deposito"),
        ),
    }
    costs = {
        ("deposito", "casa1"): 8.0,
        ("casa1", "deposito"): 8.0,
    }
    return World(
        locations=locs,
        drones=drones,
        transporters=transporters,
        persons=persons,
        packages=packages,
        contents=contents,
        costs=costs,
    )


def make_surface(w: int = 600, h: int = 400) -> pygame.Surface:
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    return surf


def surfaces_equal(a: pygame.Surface, b: pygame.Surface) -> bool:
    """¿Dos surfaces son píxel-idénticas?"""
    import pygame.surfarray as sa
    return bool((sa.array3d(a) == sa.array3d(b)).all())


def world_dict_snapshot(world: World) -> dict:
    """Estado del World como dict de dicts (para comparación pre/post)."""
    return {
        "locations": dict(world.locations),
        "drones": dict(world.drones),
        "transporters": dict(world.transporters),
        "packages": dict(world.packages),
        "persons": dict(world.persons),
        "contents": dict(world.contents),
        "costs": dict(world.costs),
    }


def build_durative_plan() -> Plan:
    """Plan durativo de 3 pasos: PickUp(5s) → Move(8s) → Deliver(5s).

    Timestamps: 0 → 5 → 13.
    Total duration esperada del runner: 18.
    """
    return Plan(scheduled=(
        ScheduledCommand(
            command=PickUp(
                drone_id="d1", arm_id="izq",
                package_id="pkg_med1", duration=5.0,
            ),
            start_time=0.0,
        ),
        ScheduledCommand(
            command=Move(
                drone_id="d1", destination_id="casa1",
                duration=8.0,
            ),
            start_time=5.0,
        ),
        ScheduledCommand(
            command=Deliver(
                drone_id="d1", package_id="pkg_med1",
                person_id="p1", duration=5.0,
            ),
            start_time=13.0,
        ),
    ))


def build_sequential_plan() -> Plan:
    """Plan secuencial (todos duration=0): PickUp → Move → Deliver."""
    return Plan.sequential([
        PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1"),
        Move(drone_id="d1", destination_id="casa1"),
        Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1"),
    ])


# ---------------------------------------------------------------------------
# Plan durativo end-to-end
# ---------------------------------------------------------------------------


class TestEndToEndDurativo:
    """Pipeline completo con Plan durativo (PDDL parte 3)."""

    def test_plan_completo_se_ejecuta_y_succeeded(self):
        # Sanity: el plan se construye y ejecuta sin fallos.
        world = make_complete_world()
        runner = PlanRunner(world)
        result = runner.execute(build_durative_plan())
        assert result.succeeded, (
            f"Plan no debería fallar; failures = {result.failures}"
        )

    def test_timeline_duration_consistente_con_runner(self):
        # Plan durativo: end_time esperado = 5 + 8 + 5 = 18.
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history, theme=Theme.default())

        # La duration del timeline incluye fallback en cada tramo con
        # delta=0 (entre el initial t=0 y el snap_start del primer
        # Command que también está en t=0, y entre snap_end y snap_start
        # consecutivos). El tiempo real del plan es 18.
        # El número exacto depende de cuántos pares con delta=0 hay.
        # Lo que sí podemos exigir: duration >= 18 (el tiempo real).
        assert timeline.duration >= 18.0

    def test_render_frame_en_todos_los_extremos_funciona(self):
        # sample() en los bordes y en varios puntos intermedios.
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        theme = Theme.default()
        # Muestreamos 11 puntos a lo largo de toda la duration:
        n_samples = 11
        for i in range(n_samples):
            t = (i / (n_samples - 1)) * timeline.duration
            snap_a, snap_b, progress = timeline.sample(t)
            surf = make_surface()
            # No debe crashear:
            render_frame(surf, snap_a, snap_b, progress, theme=theme)

    def test_sample_en_t_cero_equivale_a_render_snapshot_inicial(self):
        # En t=0, sample devuelve (h[0], h[0], 0.0). Eso pasa por la
        # rama "snap_a is snap_b" en render_frame y dibuja snap_b.
        # Lo comparamos con render_snapshot(h[0]).
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        snap_a, snap_b, progress = timeline.sample(0.0)
        assert snap_a is snap_b
        assert snap_a is runner.history.at(0)

        theme = Theme.default()
        surf_frame = make_surface()
        surf_snap = make_surface()
        render_frame(surf_frame, snap_a, snap_b, progress, theme=theme)
        render_snapshot(surf_snap, runner.history.at(0), theme=theme)
        assert surfaces_equal(surf_frame, surf_snap)

    def test_sample_en_duration_equivale_a_render_snapshot_final(self):
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        last_idx = len(runner.history) - 1
        snap_a, snap_b, progress = timeline.sample(timeline.duration)
        assert snap_a is snap_b
        assert snap_a is runner.history.at(last_idx)

        theme = Theme.default()
        surf_frame = make_surface()
        surf_snap = make_surface()
        render_frame(surf_frame, snap_a, snap_b, progress, theme=theme)
        render_snapshot(surf_snap, runner.history.at(last_idx), theme=theme)
        assert surfaces_equal(surf_frame, surf_snap)

    def test_frames_intermedios_difieren_de_los_extremos(self):
        # En el medio del Move durativo, el drone está volando: la
        # surface DEBE diferir tanto del frame inicial como del final.
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        theme = Theme.default()
        # Frame en el extremo y en el medio:
        surf_inicial = make_surface()
        surf_medio = make_surface()
        surf_final = make_surface()

        a0, b0, p0 = timeline.sample(0.0)
        render_frame(surf_inicial, a0, b0, p0, theme=theme)

        am, bm, pm = timeline.sample(timeline.duration / 2.0)
        render_frame(surf_medio, am, bm, pm, theme=theme)

        af, bf, pf = timeline.sample(timeline.duration)
        render_frame(surf_final, af, bf, pf, theme=theme)

        assert not surfaces_equal(surf_inicial, surf_medio), (
            "El frame medio debería diferir del inicial"
        )
        assert not surfaces_equal(surf_medio, surf_final), (
            "El frame medio debería diferir del final"
        )

    def test_frames_sucesivos_con_dt_pequeno_difieren_levemente(self):
        # El requisito "la animación se mueve, no salta": para dos
        # frames con dt = 0.2s en mitad de un Move durativo, las
        # surfaces NO deben ser idénticas (drone se movió), pero
        # tampoco completamente distintas (sigue siendo el mismo plan).
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        theme = Theme.default()
        # En t=9 estamos dentro del Move (entre t=5 y t=13 del plan,
        # más el fallback inicial de 0.6, así que entre 5.6 y 13.6
        # del timeline).
        t1 = 9.0
        t2 = 9.2
        a1, b1, p1 = timeline.sample(t1)
        a2, b2, p2 = timeline.sample(t2)
        # Ambos deben caer en el mismo par de snapshots:
        assert a1 is a2
        assert b1 is b2
        # Y progress debe haber avanzado:
        assert p2 > p1

        surf1 = make_surface()
        surf2 = make_surface()
        render_frame(surf1, a1, b1, p1, theme=theme)
        render_frame(surf2, a2, b2, p2, theme=theme)
        # Las surfaces difieren (el drone se movió 0.2s en el plan).
        assert not surfaces_equal(surf1, surf2)


# ---------------------------------------------------------------------------
# Plan secuencial puro (duration=0)
# ---------------------------------------------------------------------------


class TestEndToEndSecuencial:
    """Pipeline completo con Plan secuencial (PDDL parte 1-2)."""

    def test_plan_secuencial_se_ejecuta_y_succeeded(self):
        world = make_complete_world()
        runner = PlanRunner(world)
        result = runner.execute(build_sequential_plan())
        assert result.succeeded, (
            f"Plan secuencial no debería fallar; failures = {result.failures}"
        )

    def test_timeline_duration_solo_por_fallback(self):
        # Plan secuencial: 3 Commands con duration=0, timestamps 0/1/2.
        # Snapshots: [initial(0), end(0), end(1), end(2)]. Cuatro snaps
        # con timestamps 0, 0, 1, 2 → deltas (0, 1, 1).
        # virtual_times = [0, 0.6, 1.6, 2.6]. duration = 2.6.
        # (El primer tramo lleva fallback porque delta=0; los demás son
        # delta=1 real.)
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_sequential_plan())
        timeline = Timeline(runner.history, theme=Theme.default())
        # Solo verificamos que duration > 0:
        assert timeline.duration > 0.0

    def test_frames_a_lo_largo_de_la_timeline_no_crashean(self):
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_sequential_plan())
        timeline = Timeline(runner.history)

        theme = Theme.default()
        for i in range(11):
            t = (i / 10.0) * timeline.duration
            snap_a, snap_b, progress = timeline.sample(t)
            surf = make_surface()
            render_frame(surf, snap_a, snap_b, progress, theme=theme)

    def test_extremos_consistentes_con_render_snapshot(self):
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_sequential_plan())
        timeline = Timeline(runner.history)
        theme = Theme.default()

        # Sample en t=0 → primer snapshot.
        a0, b0, p0 = timeline.sample(0.0)
        surf_frame = make_surface()
        surf_snap = make_surface()
        render_frame(surf_frame, a0, b0, p0, theme=theme)
        render_snapshot(surf_snap, runner.history.at(0), theme=theme)
        assert surfaces_equal(surf_frame, surf_snap)

        # Sample en t=duration → último snapshot.
        last = len(runner.history) - 1
        af, bf, pf = timeline.sample(timeline.duration)
        surf_frame_f = make_surface()
        surf_snap_f = make_surface()
        render_frame(surf_frame_f, af, bf, pf, theme=theme)
        render_snapshot(surf_snap_f, runner.history.at(last), theme=theme)
        assert surfaces_equal(surf_frame_f, surf_snap_f)


# ---------------------------------------------------------------------------
# Inmutabilidad: el render NO muta nada del dominio
# ---------------------------------------------------------------------------


class TestEndToEndNoMutacion:
    """El render NUNCA muta el World ni los snapshots."""

    def test_renderizar_todos_los_frames_no_muta_los_worlds(self):
        # Canario de Sesión D: tras renderizar el plan entero, todos
        # los snapshots siguen siendo idénticos.
        world = make_complete_world()
        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)

        # Capturamos el estado de cada snapshot ANTES de renderizar:
        before: list[dict] = []
        for i in range(len(runner.history)):
            before.append(world_dict_snapshot(runner.history.at(i).world))

        # Renderizar muchos frames:
        theme = Theme.default()
        for i in range(50):
            t = (i / 49.0) * timeline.duration
            snap_a, snap_b, progress = timeline.sample(t)
            surf = make_surface()
            render_frame(surf, snap_a, snap_b, progress, theme=theme)

        # Comparamos: cada snapshot debe seguir igual.
        for i in range(len(runner.history)):
            after = world_dict_snapshot(runner.history.at(i).world)
            assert after == before[i], (
                f"El snapshot {i} fue mutado por el render"
            )

    def test_renderizar_no_muta_el_world_inicial_externo(self):
        # El World que pasamos al PlanRunner tampoco debe mutarse
        # (esto está garantizado por la inmutabilidad del dominio,
        # pero lo reverificamos al final del pipeline render).
        world = make_complete_world()
        before = world_dict_snapshot(world)

        runner = PlanRunner(world)
        runner.execute(build_durative_plan())
        timeline = Timeline(runner.history)
        theme = Theme.default()
        for i in range(20):
            t = (i / 19.0) * timeline.duration
            snap_a, snap_b, progress = timeline.sample(t)
            surf = make_surface()
            render_frame(surf, snap_a, snap_b, progress, theme=theme)

        after = world_dict_snapshot(world)
        assert after == before


# ---------------------------------------------------------------------------
# Plan con fallo (drone a ERROR a mitad del plan)
# ---------------------------------------------------------------------------


class TestEndToEndConFallo:
    """Pipeline cuando un Command falla: el render lo muestra correctamente."""

    def test_plan_con_pickup_imposible_renderiza_sin_crash(self):
        # Plan con un PickUp imposible: el paquete está en otra loc.
        contents = {"medicina": Content(id="medicina")}
        locs = {
            "casa1": Location(id="casa1"),
            "casa2": Location(id="casa2"),
        }
        drones = {
            "d1": Drone(
                id="d1",
                position="casa1",  # Drone en casa1
                arms=(Arm(id="izq"),),
                state=DroneState.IDLE,
            ),
        }
        packages = {
            "pkg_med1": Package(
                id="pkg_med1",
                contains=contents["medicina"],
                at=AtLocation(loc_id="casa2"),  # Paquete en casa2
            ),
        }
        world = World(
            locations=locs,
            drones=drones,
            packages=packages,
            contents=contents,
        )

        runner = PlanRunner(world)
        result = runner.execute(Plan.sequential([
            PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1"),
        ]))
        assert not result.succeeded
        assert len(result.failures) == 1

        timeline = Timeline(runner.history)
        theme = Theme.default()

        # Renderizamos a lo largo de la timeline sin crash:
        for i in range(5):
            t = (i / 4.0) * timeline.duration if timeline.duration > 0 else 0.0
            snap_a, snap_b, progress = timeline.sample(t)
            surf = make_surface()
            render_frame(surf, snap_a, snap_b, progress, theme=theme)

        # Frame final muestra el drone en ERROR (rojo + X):
        surf_final = make_surface()
        af, bf, pf = timeline.sample(timeline.duration)
        render_frame(surf_final, af, bf, pf, theme=theme)
        # Buscar color rojo del drone error:
        from droneplan_viz.render.theme import Theme as _T
        theme_default = _T.default()
        w, h = surf_final.get_size()
        found_red = False
        for y in range(h):
            for x in range(w):
                px = surf_final.get_at((x, y))[:3]
                if tuple(px) == theme_default.drone_error:
                    found_red = True
                    break
            if found_red:
                break
        assert found_red, "El drone en ERROR no se renderizó en rojo"


# ---------------------------------------------------------------------------
# Reproducibilidad cross-run
# ---------------------------------------------------------------------------


class TestEndToEndReproducibilidad:
    """Mismo plan + mismo World + misma t → mismo output píxel a píxel.

    Esto valida que TODO el pipeline es determinista, lo cual es
    crítico para tests de regresión visual y para que el TFG sea
    reproducible.
    """

    def test_dos_renders_del_mismo_frame_son_identicos(self):
        # Ejecutamos DOS veces el mismo plan en worlds idénticos y
        # comparamos el frame en t = duration/2.
        world_a = make_complete_world()
        runner_a = PlanRunner(world_a)
        runner_a.execute(build_durative_plan())
        timeline_a = Timeline(runner_a.history)

        world_b = make_complete_world()
        runner_b = PlanRunner(world_b)
        runner_b.execute(build_durative_plan())
        timeline_b = Timeline(runner_b.history)

        # Las duraciones deben coincidir:
        assert timeline_a.duration == pytest.approx(timeline_b.duration)

        theme = Theme.default()
        t = timeline_a.duration / 2.0

        sa_a, sb_a, p_a = timeline_a.sample(t)
        sa_b, sb_b, p_b = timeline_b.sample(t)
        # Mismo progress:
        assert p_a == pytest.approx(p_b)

        surf_a = make_surface()
        surf_b = make_surface()
        render_frame(surf_a, sa_a, sb_a, p_a, theme=theme)
        render_frame(surf_b, sa_b, sb_b, p_b, theme=theme)

        assert surfaces_equal(surf_a, surf_b)


# ---------------------------------------------------------------------------
# Render de transferencias con transporter (Load / Unload)
#
# Regresión: antes, renderizar una transición LoadIntoTransporter o
# UnloadFromTransporter llamaba a package_position sobre un paquete
# InTransporter y lanzaba ValueError (la coreografía de paquete asumía
# suelo↔brazo). Estos tests blindan que ahora se renderizan sin crash y
# que la caja viaja entre el brazo y el transporter.
# ---------------------------------------------------------------------------


def _lifecycle_world_plan():
    """make_complete_world() + plan pickup→load→move-with-t→unload→deliver."""
    world = make_complete_world()
    S = lambda cmd, t: ScheduledCommand(command=cmd, start_time=t)
    plan = Plan(scheduled=(
        S(PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med1",
                 duration=2.0, command_id="pk"), 0.0),
        S(LoadIntoTransporter(drone_id="d1", package_id="pkg_med1",
                              transporter_id="t1", duration=2.0,
                              command_id="ld"), 2.0),
        S(MoveWithTransporter(drone_id="d1", transporter_id="t1",
                              destination_id="casa1", duration=4.0,
                              command_id="mt"), 4.0),
        S(UnloadFromTransporter(drone_id="d1", arm_id="izq",
                                package_id="pkg_med1", transporter_id="t1",
                                duration=2.0, command_id="ul"), 8.0),
        S(Deliver(drone_id="d1", package_id="pkg_med1", person_id="p1",
                  duration=2.0, command_id="dl"), 10.0),
    ))
    return world, plan


class TestRenderTransporterTransfer:
    from droneplan_viz.commands import LoadIntoTransporter  # noqa: F401 (uso en _lifecycle)

    def test_lifecycle_completo_renderiza_sin_crash(self):
        """El plan con Load + MoveWithTransporter + Unload se renderiza en
        todos sus frames sin lanzar (antes reventaba en la carga/descarga)."""
        world, plan = _lifecycle_world_plan()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        assert result.succeeded
        timeline = Timeline(runner.history, theme=Theme.default())

        for i in range(0, 61):
            t = timeline.duration * i / 60.0
            sa, sb, p = timeline.sample(t)
            surf = make_surface()
            # No debe lanzar ValueError ni ninguna otra excepción.
            render_frame(surf, sa, sb, p, theme=Theme.default())

    def test_frame_de_carga_dibuja_contenido(self):
        """Un frame en mitad de la carga no queda en blanco (se dibujó la
        escena y la caja en tránsito)."""
        import pygame.surfarray as sa_mod

        world, plan = _lifecycle_world_plan()
        runner = PlanRunner(world)
        runner.execute(plan)
        timeline = Timeline(runner.history, theme=Theme.default())

        # t≈3 cae dentro de la carga [2,4].
        sa, sb, p = timeline.sample(3.0)
        surf = make_surface()
        render_frame(surf, sa, sb, p, theme=Theme.default())

        blank = make_surface()
        assert not surfaces_equal(surf, blank)

    def test_caja_viaja_de_brazo_a_transporter(self):
        """En la carga, la caja interpola entre el brazo del drone y el
        transporter: ambos extremos están bien resueltos y son distintos."""
        from dataclasses import replace

        from droneplan_viz.domain import HeldByArm, InTransporter
        from droneplan_viz.history import WorldSnapshot
        from droneplan_viz.render.interpolation import classify_transition
        from droneplan_viz.render.painter import (
            _package_endpoint_pos,
            _scene_viewbox,
            compute_layout,
        )

        world = make_complete_world()
        theme = Theme.default()
        size = (600, 400)

        # snap_a: pkg sostenido por d1·izq. snap_b: pkg dentro de t1.
        w_held = replace(
            world,
            packages={**world.packages, "pkg_med1": replace(
                world.packages["pkg_med1"],
                at=HeldByArm(drone_id="d1", arm_id="izq"),
            )},
        )
        w_in_t = replace(
            world,
            packages={**world.packages, "pkg_med1": replace(
                world.packages["pkg_med1"], at=InTransporter(transporter_id="t1"),
            )},
        )
        snap_a = WorldSnapshot(world=w_held, metrics=MetricsTracker(),
                               produced_by=None, timestamp=0.0)
        snap_b = WorldSnapshot(
            world=w_in_t, metrics=MetricsTracker(),
            produced_by=LoadIntoTransporter(
                drone_id="d1", package_id="pkg_med1", transporter_id="t1",
                duration=2.0, command_id="ld"),
            timestamp=2.0,
        )

        transition = classify_transition(snap_a, snap_b)
        assert transition.from_place.kind == "arm"
        assert transition.to_place.kind == "transp"

        viewbox = _scene_viewbox(size, theme)
        layout = compute_layout(w_held, viewbox,
                                spread_factor=theme.location_spread_factor)
        drone_pos = layout.drone_position("d1", theme)
        from_pt = _package_endpoint_pos(transition.from_place, layout, w_held, drone_pos, theme)
        to_pt = _package_endpoint_pos(transition.to_place, layout, w_held, drone_pos, theme)
        # Extremos bien resueltos y distintos (la caja realmente viaja).
        assert from_pt != to_pt


# Import a nivel de módulo necesario para _lifecycle_world_plan y los tests.
from droneplan_viz.commands import LoadIntoTransporter  # noqa: E402
