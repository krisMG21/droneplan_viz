"""Tests para droneplan_viz.render.painter.

Tests con pygame.Surface en memoria (SDL dummy via conftest).
Cubren render_snapshot y render_frame:

- render_snapshot sobre varios worlds (vacío, simple, rich).
- render_frame en los extremos (progress 0 y 1) equivale a render_snapshot.
- render_frame con TransitionDroneMove en progress=0.5: el drone está
  entre origen y destino, no en ninguno de ellos.
- render_frame con TransitionPackageMove en progress=0.5: el paquete
  está entre from_place y to_place.
- render_frame con TransitionStatic / TransitionFailure: dibuja snap_b.
- Smoke end-to-end: plan completo se renderiza sin crashear.
"""
import pygame
import pytest

from droneplan_viz.commands import Deliver, Move, PickUp
from droneplan_viz.domain import (
    AtLocation,
    Content,
    HeldByArm,
    MetricsTracker,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import Package
from droneplan_viz.domain.person import Person
from droneplan_viz.history import WorldSnapshot
from droneplan_viz.render import Theme, render_frame, render_snapshot
from droneplan_viz.render.geometry import Point, ViewBox
from droneplan_viz.render.layout import compute_layout
from droneplan_viz.render.painter import _scene_viewbox
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand


# ---------------------------------------------------------------------------
# Helpers (reutilizamos los del test_interpolation; los redefinimos aquí
# locales en lugar de hacer import cross-test por simplicidad)
# ---------------------------------------------------------------------------


def make_world_move() -> World:
    """World con d1 en casa1, casa2 a la que mover."""
    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
    }
    drones = {
        "d1": Drone(id="d1", position="casa1", arms=(), state=DroneState.IDLE),
    }
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(locations=locs, drones=drones, costs=costs)


def make_world_pickup_deliver() -> World:
    """World con d1 (dos brazos), pkg medicina en suelo y p1 que la necesita."""
    contents = {"medicina": Content(id="medicina")}
    locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
    drones = {
        "d1": Drone(
            id="d1",
            position="casa1",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
    }
    persons = {
        "p1": Person(id="p1", position="casa1", needs=(contents["medicina"],)),
    }
    packages = {
        "pkg_med1": Package(
            id="pkg_med1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="casa1"),
        ),
    }
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(
        locations=locs,
        drones=drones,
        persons=persons,
        packages=packages,
        contents=contents,
        costs=costs,
    )


def make_surface(w: int = 700, h: int = 500) -> pygame.Surface:
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    return surf


def color_at(surf: pygame.Surface, x: int, y: int) -> tuple[int, int, int]:
    r, g, b, _ = surf.get_at((x, y))
    return (r, g, b)


def find_color(
    surf: pygame.Surface,
    target_rgb: tuple[int, int, int],
    tol: int = 0,
) -> tuple[int, int] | None:
    """Recorre toda la Surface buscando un píxel del color objetivo.

    Devuelve la primera coordenada (x, y) que matchea o None si no
    aparece. Útil para asserts "el drone aparece en algún sitio".
    """
    w, h = surf.get_size()
    for y in range(h):
        for x in range(w):
            px = color_at(surf, x, y)
            if all(abs(px[i] - target_rgb[i]) <= tol for i in range(3)):
                return (x, y)
    return None


def count_color(
    surf: pygame.Surface,
    target_rgb: tuple[int, int, int],
    tol: int = 0,
) -> int:
    """Cuántos píxeles de la Surface matchean target_rgb (con tolerancia)."""
    w, h = surf.get_size()
    count = 0
    for y in range(h):
        for x in range(w):
            px = color_at(surf, x, y)
            if all(abs(px[i] - target_rgb[i]) <= tol for i in range(3)):
                count += 1
    return count


def find_color_near(
    surf: pygame.Surface,
    target_rgb: tuple[int, int, int],
    center: tuple[int, int],
    half_box: int = 10,
    tol: int = 0,
) -> bool:
    cx, cy = center
    w, h = surf.get_size()
    for dx in range(-half_box, half_box + 1):
        for dy in range(-half_box, half_box + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                px = color_at(surf, x, y)
                if all(abs(px[i] - target_rgb[i]) <= tol for i in range(3)):
                    return True
    return False


# ---------------------------------------------------------------------------
# render_snapshot
# ---------------------------------------------------------------------------


class TestRenderSnapshotBasico:
    """render_snapshot smoke + verificación de capas dibujadas."""

    def test_world_vacio_solo_pinta_fondo(self, empty_world):
        surf = make_surface()
        theme = Theme.default()
        snap = WorldSnapshot(
            world=empty_world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        render_snapshot(surf, snap, theme=theme)
        # Toda la Surface es background:
        assert color_at(surf, 0, 0) == theme.background
        assert color_at(surf, 100, 100) == theme.background
        assert color_at(surf, 200, 200) == theme.background

    def test_world_simple_pinta_drone_y_locations(self):
        surf = make_surface()
        theme = Theme.default()
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        render_snapshot(surf, snap, theme=theme)

        # Color de drone IDLE debe aparecer en algún sitio:
        assert find_color(surf, theme.drone_idle) is not None
        # Color de casa también:
        casa_fill = theme.loc_fill_neutral
        assert find_color(surf, casa_fill) is not None

    def test_world_pickup_deliver_pinta_paquete_y_persona(self):
        surf = make_surface()
        theme = Theme.default()
        world = make_world_pickup_deliver()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        render_snapshot(surf, snap, theme=theme)

        # La persona se pinta (sin sprite_manager → círculo primitivo):
        assert find_color(surf, theme.person_fill) is not None
        # Paquete de medicina → color medicina presente (color canónico):
        from droneplan_viz.render.theme import color_for_content
        medicina_color = color_for_content("medicina", theme)
        assert find_color(surf, medicina_color) is not None

    def test_theme_none_usa_default(self):
        surf = make_surface()
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        # No pasar theme:
        render_snapshot(surf, snap)
        # Background del default:
        assert color_at(surf, 0, 0) == Theme.default().background

    def test_surface_pequena_no_crashea(self):
        # Caso límite: surface tan pequeña que el viewbox después del
        # padding es degenerado.
        surf = pygame.Surface((50, 50))
        theme = Theme.default()
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        # No verifica nada sobre el resultado; solo que no crashee.
        render_snapshot(surf, snap, theme=theme)


class TestRenderSnapshotConFallo:
    """render_snapshot con drone en ERROR muestra el rojo + X."""

    def test_drone_en_error_se_renderiza_rojo(self):
        # Construir un world con un drone DIRECTAMENTE en ERROR.
        locs = {"casa1": Location(id="casa1")}
        drones = {
            "d1": Drone(
                id="d1",
                position="casa1",
                arms=(),
                state=DroneState.ERROR,
            ),
        }
        world = World(locations=locs, drones=drones)
        surf = make_surface()
        theme = Theme.default()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        render_snapshot(surf, snap, theme=theme)
        # El color rojo del drone error debe aparecer en algún sitio:
        assert find_color(surf, theme.drone_error) is not None
        # La X blanca también:
        assert find_color(surf, theme.error_x) is not None


# ---------------------------------------------------------------------------
# render_frame: extremos
# ---------------------------------------------------------------------------


class TestRenderFrameExtremos:
    """En progress 0 / 1 / fuera de rango, equivale a render_snapshot."""

    def test_progress_cero_equivale_a_render_snapshot_a(self):
        # Construir dos snapshots de un Move y comparar.
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)  # snap_start
        snap_b = runner.history.at(2)  # snap_end

        theme = Theme.default()
        surf_frame = make_surface()
        surf_snap = make_surface()

        render_frame(surf_frame, snap_a, snap_b, progress=0.0, theme=theme)
        render_snapshot(surf_snap, snap_a, theme=theme)

        # Las dos surfaces deben ser idénticas píxel a píxel:
        import pygame.surfarray as sa
        assert (sa.array3d(surf_frame) == sa.array3d(surf_snap)).all()

    def test_progress_uno_equivale_a_render_snapshot_b(self):
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)

        theme = Theme.default()
        surf_frame = make_surface()
        surf_snap = make_surface()

        render_frame(surf_frame, snap_a, snap_b, progress=1.0, theme=theme)
        render_snapshot(surf_snap, snap_b, theme=theme)

        import pygame.surfarray as sa
        assert (sa.array3d(surf_frame) == sa.array3d(surf_snap)).all()

    def test_progress_negativo_se_satura_a_cero(self):
        # Mismo test que progress=0, pero con valor fuera de rango.
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)

        theme = Theme.default()
        surf_neg = make_surface()
        surf_zero = make_surface()
        render_frame(surf_neg, snap_a, snap_b, progress=-0.5, theme=theme)
        render_frame(surf_zero, snap_a, snap_b, progress=0.0, theme=theme)

        import pygame.surfarray as sa
        assert (sa.array3d(surf_neg) == sa.array3d(surf_zero)).all()

    def test_progress_mayor_uno_se_satura(self):
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)

        theme = Theme.default()
        surf_big = make_surface()
        surf_one = make_surface()
        render_frame(surf_big, snap_a, snap_b, progress=2.5, theme=theme)
        render_frame(surf_one, snap_a, snap_b, progress=1.0, theme=theme)

        import pygame.surfarray as sa
        assert (sa.array3d(surf_big) == sa.array3d(surf_one)).all()


# ---------------------------------------------------------------------------
# render_frame: TransitionDroneMove
# ---------------------------------------------------------------------------


class TestRenderFrameDroneMove:
    """Frame intermedio durante un Move: drone entre origen y destino."""

    def _setup(self):
        """Devuelve (snap_a, snap_b, layout) para un Move durativo."""
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)  # snap_start
        snap_b = runner.history.at(2)  # snap_end

        # Calcular layout sobre el snap_a.world para conocer posiciones.
        # El viewbox debe coincidir con el tamaño de surface en el que se
        # pinta (make_surface=700x500 por defecto), para que las posiciones
        # del layout encajen con los píxeles que verificamos.
        theme = Theme.default()
        # El render reserva holgura para las decoraciones (ver
        # painter._scene_viewbox); el layout de referencia debe usar el MISMO
        # viewbox para que las posiciones encajen con los píxeles verificados.
        vb = _scene_viewbox((700, 500), theme)
        layout = compute_layout(
            snap_a.world, vb, spread_factor=theme.location_spread_factor
        )
        return snap_a, snap_b, layout, theme

    def test_a_medio_camino_drone_no_esta_en_origen_ni_destino(self):
        snap_a, snap_b, layout, theme = self._setup()
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.5, theme=theme)

        # En progress=0.5 el drone está aproximadamente equidistante
        # entre las posiciones VISUALES del drone en casa1 y casa2 (que
        # incluyen el centro del rombo + hover, no el centro del lienzo).
        from droneplan_viz.render.painter import _drone_position_in_loc
        drone_id = next(iter(snap_a.world.drones))
        d_from = _drone_position_in_loc(layout, snap_a.world, "casa1", drone_id, theme)
        d_to = _drone_position_in_loc(layout, snap_a.world, "casa2", drone_id, theme)
        midpoint = ((d_from.x + d_to.x) / 2, (d_from.y + d_to.y) / 2)

        # En el punto medio debe haber píxeles del color MOVING:
        # (el drone está en MOVING en snap_a.world).
        assert find_color_near(
            surf, theme.drone_moving,
            (int(midpoint[0]), int(midpoint[1])),
            half_box=int(theme.drone_radius + 2),
        )

    def test_a_medio_camino_drone_no_esta_en_origen(self):
        snap_a, snap_b, layout, theme = self._setup()
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.5, theme=theme)

        # En casa1 no debe haber color de drone moving. El píxel central
        # ahora ES el fill del círculo (la etiqueta se pinta debajo del
        # nodo desde el fix del Frente 1), pero comprobamos un píxel
        # desplazado hacia arriba que es inequívocamente fill, sin riesgo
        # de tocar ninguna entidad co-localizada cerca del centro.
        casa1 = layout.location_position("casa1")
        cx, cy = casa1.as_int_tuple()
        check_x = cx
        check_y = cy - theme.location_radius // 2
        assert color_at(surf, check_x, check_y) == theme.loc_fill_neutral

    def test_a_medio_camino_drone_no_esta_en_destino(self):
        snap_a, snap_b, layout, theme = self._setup()
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.5, theme=theme)

        casa2 = layout.location_position("casa2")
        cx, cy = casa2.as_int_tuple()
        # Píxel desplazado hacia arriba del centro: fill inequívoco.
        check_x = cx
        check_y = cy - theme.location_radius // 2
        assert color_at(surf, check_x, check_y) == theme.loc_fill_neutral

    def test_progress_avanza_drone_se_mueve(self):
        snap_a, snap_b, layout, theme = self._setup()

        # Render a progress=0.25 y a progress=0.75: la posición del
        # drone debe haberse desplazado hacia casa2.
        surf_25 = make_surface()
        surf_75 = make_surface()
        render_frame(surf_25, snap_a, snap_b, progress=0.25, theme=theme)
        render_frame(surf_75, snap_a, snap_b, progress=0.75, theme=theme)

        # Las dos surfaces NO deben ser iguales:
        import pygame.surfarray as sa
        assert not (sa.array3d(surf_25) == sa.array3d(surf_75)).all()


# ---------------------------------------------------------------------------
# render_frame: TransitionPackageMove
# ---------------------------------------------------------------------------


class TestRenderFramePackageMove:
    """Frame intermedio durante PickUp/Deliver: paquete entre from y to."""

    def _setup_pickup(self):
        world = make_world_pickup_deliver()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            PickUp(
                drone_id="d1", arm_id="izq",
                package_id="pkg_med1", duration=5.0,
            ),
        ]))
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        theme = Theme.default()
        vb = ViewBox.fit_into((400, 300), padding=theme.padding)
        layout = compute_layout(snap_a.world, vb)
        return snap_a, snap_b, layout, theme

    def test_pickup_caja_visible_en_posicion_de_enganche(self):
        # Modelo nuevo (sin vuelo): la caja NO viaja por el aire. Está en
        # posición fija de enganche (alineada con el dron, a la altura de
        # la capa box). Antes del midpoint se dibuja "en el suelo"; debe
        # verse. Camino primitiva (sin sprite_manager) → caja coloreada
        # por contenido.
        snap_a, snap_b, layout, theme = self._setup_pickup()
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.25, theme=theme)

        from droneplan_viz.render.theme import color_for_content
        med_color = color_for_content("medicina", theme)
        assert find_color(surf, med_color) is not None

    def test_pickup_caja_en_dron_tras_midpoint(self):
        # Tras el midpoint (progress >= 0.5) la caja es capa del dron, no
        # se dibuja aparte. En camino primitiva, el dron protagonista
        # dibuja la caja sobre el brazo. Sigue habiendo color medicina (la
        # caja existe), pero ya no en posición de suelo.
        snap_a, snap_b, layout, theme = self._setup_pickup()
        surf_before = make_surface()
        surf_after = make_surface()
        render_frame(surf_before, snap_a, snap_b, progress=0.25, theme=theme)
        render_frame(surf_after, snap_a, snap_b, progress=0.75, theme=theme)
        # El snap de propiedad en 0.5 cambia la representación → las dos
        # surfaces deben diferir.
        import pygame.surfarray as sa
        assert not (sa.array3d(surf_before) == sa.array3d(surf_after)).all()

    def test_pickup_progress_distintos_producen_surfaces_distintas(self):
        snap_a, snap_b, layout, theme = self._setup_pickup()
        surf_25 = make_surface()
        surf_75 = make_surface()
        render_frame(surf_25, snap_a, snap_b, progress=0.25, theme=theme)
        render_frame(surf_75, snap_a, snap_b, progress=0.75, theme=theme)

        import pygame.surfarray as sa
        assert not (sa.array3d(surf_25) == sa.array3d(surf_75)).all()

    def test_pickup_caja_no_aparece_dos_veces(self):
        # La caja no debe pintarse a la vez "en el suelo" Y "sobre el
        # dron". En cada frame es una cosa o la otra (snap en 0.5).
        # Contamos píxeles del color medicina: deben corresponder a UNA
        # sola caja, no a dos. Camino primitiva (caja coloreada).
        snap_a, snap_b, layout, theme = self._setup_pickup()
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.25, theme=theme)

        from droneplan_viz.render.theme import color_for_content
        med_color = color_for_content("medicina", theme)
        count = count_color(surf, med_color)
        # 1 caja pintada: ~package_size^2 = 196 (cuadrado relleno).
        # Si se pintara dos veces (suelo + dron) sería ~2x.
        assert count < 300, f"Demasiados píxeles medicina: {count}"
        assert count > 50, f"Faltan píxeles medicina: {count}"


# ---------------------------------------------------------------------------
# render_frame: TransitionStatic y TransitionFailure
# ---------------------------------------------------------------------------


class TestRenderFrameStaticYFailure:

    def test_static_dibuja_snap_b_directo(self):
        # snap_a y snap_b corresponden al MISMO snapshot (degenerate Static).
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        theme = Theme.default()
        surf_frame = make_surface()
        surf_snap = make_surface()
        render_frame(surf_frame, snap, snap, progress=0.5, theme=theme)
        render_snapshot(surf_snap, snap, theme=theme)

        import pygame.surfarray as sa
        assert (sa.array3d(surf_frame) == sa.array3d(surf_snap)).all()

    def test_failure_dibuja_snap_b_con_drone_rojo(self):
        # Plan que falla: Move a destino inexistente.
        locs = {"casa1": Location(id="casa1")}
        drones = {"d1": Drone(id="d1", position="casa1", arms=())}
        world = World(locations=locs, drones=drones)

        runner = PlanRunner(world)
        result = runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="fantasma"),
        ]))
        assert not result.succeeded

        snap_initial = runner.history.at(0)
        snap_fail = runner.history.at(1)

        theme = Theme.default()
        surf = make_surface()
        render_frame(surf, snap_initial, snap_fail, progress=0.5, theme=theme)

        # El drone en rojo debe aparecer:
        assert find_color(surf, theme.drone_error) is not None
        # La X también:
        assert find_color(surf, theme.error_x) is not None


# ---------------------------------------------------------------------------
# Integración end-to-end: plan completo
# ---------------------------------------------------------------------------


class TestIntegracionEndToEnd:
    """Plan multi-step + render de varios frames sin crash."""

    def test_plan_pickup_deliver_renderiza_todos_los_frames(self):
        world = make_world_pickup_deliver()
        runner = PlanRunner(world)
        runner.execute(Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(
                    drone_id="d1", arm_id="izq",
                    package_id="pkg_med1", duration=5.0,
                ),
                start_time=0.0,
            ),
            ScheduledCommand(
                command=Deliver(
                    drone_id="d1", package_id="pkg_med1",
                    person_id="p1", duration=5.0,
                ),
                start_time=5.0,
            ),
        )))

        # Render todos los snapshots como render_snapshot:
        theme = Theme.default()
        for i in range(len(runner.history)):
            surf = make_surface()
            render_snapshot(surf, runner.history.at(i), theme=theme)
            # No crash.

        # Render frames intermedios entre pares consecutivos:
        for i in range(len(runner.history) - 1):
            snap_a = runner.history.at(i)
            snap_b = runner.history.at(i + 1)
            for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
                surf = make_surface()
                render_frame(surf, snap_a, snap_b, progress=p, theme=theme)

    def test_render_snapshot_no_muta_el_world(self):
        # Verificación crítica: el render NO debe modificar el snapshot
        # ni el world subyacente. Tras render, el world sigue exactamente
        # igual.
        world = make_world_pickup_deliver()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        # Capturamos estado antes:
        drone_before = dict(world.drones)
        pkg_before = dict(world.packages)
        person_before = dict(world.persons)
        loc_before = dict(world.locations)

        surf = make_surface()
        render_snapshot(surf, snap, theme=Theme.default())

        # Estado después: idéntico.
        assert dict(world.drones) == drone_before
        assert dict(world.packages) == pkg_before
        assert dict(world.persons) == person_before
        assert dict(world.locations) == loc_before


# ---------------------------------------------------------------------------
# Theme override en render_*
# ---------------------------------------------------------------------------


class TestThemeOverride:
    """Cambiar el theme afecta al output visual."""

    def test_distinto_theme_distinto_output(self):
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        theme_default = Theme.default()
        theme_alt = Theme.default().with_overrides(background=(200, 200, 200))

        surf_a = make_surface()
        surf_b = make_surface()
        render_snapshot(surf_a, snap, theme=theme_default)
        render_snapshot(surf_b, snap, theme=theme_alt)

        # Backgrounds distintos:
        assert color_at(surf_a, 0, 0) == theme_default.background
        assert color_at(surf_b, 0, 0) == theme_alt.background
        assert color_at(surf_a, 0, 0) != color_at(surf_b, 0, 0)

    def test_differentiate_locations_false_unifica_colores(self):
        # Con differentiate_locations=False, una casa pintada con la
        # heurística (cálido) se pinta neutral.
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        theme_with = Theme.default().with_overrides(differentiate_locations=True)
        theme_without = Theme.default().with_overrides(differentiate_locations=False)

        surf_with = make_surface()
        surf_without = make_surface()
        render_snapshot(surf_with, snap, theme=theme_with)
        render_snapshot(surf_without, snap, theme=theme_without)

        # Con diferenciación: aparece loc_fill_house.
        assert find_color(surf_with, theme_with.loc_fill_house) is not None
        # Sin diferenciación: NO aparece loc_fill_house.
        assert find_color(surf_without, theme_without.loc_fill_house) is None
        # Sí aparece loc_fill_neutral:
        assert find_color(surf_without, theme_without.loc_fill_neutral) is not None


# ---------------------------------------------------------------------------
# Integración con SpriteManager (Frente 2, paso drone)
# ---------------------------------------------------------------------------


class TestRenderConSpriteManager:
    """render_snapshot/render_frame usan el sprite del drone si está disponible.

    Estrategia: generamos un PNG sintético de color magenta puro
    (255, 0, 255) — un color que NO está en la paleta del Theme — para
    el drone. Si tras renderizar con sprite_manager ese color aparece
    donde está el drone, es prueba de que se blitteó el sprite y NO la
    primitiva (que pintaría verde/azul/ámbar según el estado).
    """

    MAGENTA = (255, 0, 255)

    def _make_sprite_manager_with_drone(self, tmp_path, theme):
        from droneplan_viz.render.sprite_manager import SpriteManager
        # Modelo de capas: generamos el CUERPO base del dron (drone.png) a
        # su tamaño nativo (58x58) relleno de magenta. El compositor lo
        # usará como capa base; al componer y escalar 1:1 (drone_radius=29
        # → destino 58), el magenta llega intacto y aparece en pantalla.
        # No generamos cara ni objeto: basta el cuerpo para verificar que
        # el camino-sprite (composición) se usa en vez de la primitiva.
        body_w, body_h = theme.sprite_anchors.body_size
        surf = pygame.Surface((body_w, body_h), pygame.SRCALPHA)
        surf.fill((*self.MAGENTA, 255))
        pygame.image.save(surf, str(tmp_path / "drone.png"))
        return SpriteManager(theme)

    def test_render_snapshot_usa_sprite_si_disponible(self, tmp_path):
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = self._make_sprite_manager_with_drone(tmp_path, theme)
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        surf = make_surface()
        render_snapshot(surf, snap, theme=theme, sprite_manager=sm)
        # El magenta del sprite debe aparecer en algún sitio.
        assert find_color(surf, self.MAGENTA, tol=8) is not None

    def test_render_snapshot_sin_manager_no_usa_sprite(self, tmp_path):
        # Sin sprite_manager, NO aparece magenta (se usan primitivas).
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        # Generamos los PNG igualmente, pero NO pasamos el manager.
        self._make_sprite_manager_with_drone(tmp_path, theme)
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        surf = make_surface()
        render_snapshot(surf, snap, theme=theme)  # sin sprite_manager
        assert find_color(surf, self.MAGENTA) is None

    def test_manager_sin_assets_cae_a_primitiva(self, tmp_path):
        # Manager construido pero sin PNGs en el dir: render NO debe
        # crashear y NO debe aparecer magenta (cae a primitiva).
        from droneplan_viz.render.sprite_manager import SpriteManager
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = SpriteManager(theme)  # dir vacío
        world = make_world_move()
        snap = WorldSnapshot(
            world=world, metrics=MetricsTracker(),
            produced_by=None, timestamp=0.0,
        )
        surf = make_surface()
        render_snapshot(surf, snap, theme=theme, sprite_manager=sm)
        assert find_color(surf, self.MAGENTA) is None
        # El color del drone IDLE (primitiva) sí debe estar:
        assert find_color(surf, theme.drone_idle) is not None

    def test_render_frame_interpolado_usa_sprite(self, tmp_path):
        # Durante un Move interpolado, el drone en movimiento también debe
        # dibujarse con sprite.
        theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
        sm = self._make_sprite_manager_with_drone(tmp_path, theme)
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(Plan.sequential([
            Move(drone_id="d1", destination_id="casa2", duration=5.0),
        ]))
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.5, theme=theme, sprite_manager=sm)
        assert find_color(surf, self.MAGENTA, tol=8) is not None


# ---------------------------------------------------------------------------
# Regresión: la pila de cajas en reposo se pinta IGUAL en todos los caminos
# (estático y animados). Antes, las rutas animadas (_render_frame_drone_move,
# _render_frame_package_move, _render_world_at_plain) pintaban TODAS las cajas
# AtLocation —sin filtro de capacidad ni orden—, lo que cargaba las cajas que
# rebasan las 15 (sombras apiladas en el slot central) y rompía la oclusión.
# Hoy todas pasan por _draw_resting_packages.
# ---------------------------------------------------------------------------
from droneplan_viz.render import painter as _painter_mod  # noqa: E402
from droneplan_viz.render.layout import (  # noqa: E402
    _box_capacity,
    _box_depth_key,
    _packages_at_loc,
)
from droneplan_viz.render.painter import _draw_resting_packages  # noqa: E402


def _world_n_boxes(n: int, loc: str = "dep") -> World:
    content = Content(id="c")
    packages = {
        f"caja{i:02d}": Package(id=f"caja{i:02d}", contains=content,
                                at=AtLocation(loc_id=loc))
        for i in range(n)
    }
    return World(
        locations={loc: Location(id=loc)}, drones={}, transporters={},
        packages=packages, persons={}, contents={"c": content}, costs={},
    )


def _layout_for(world: World, theme: Theme):
    return compute_layout(world, _scene_viewbox((700, 500), theme))


class TestPilaCajasReposoCapacidadYorden:
    def _record_positions(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            _painter_mod, "draw_package",
            lambda surface, pos, cid, theme, **kw: calls.append((pos.x, pos.y)),
        )
        return calls

    def test_rebose_no_dibuja_mas_de_capacidad(self, monkeypatch):
        # 18 cajas en una loc -> solo se dibujan _box_capacity() (15).
        theme = Theme.default()
        world = _world_n_boxes(18)
        layout = _layout_for(world, theme)
        calls = self._record_positions(monkeypatch)
        _draw_resting_packages(make_surface(), world.packages, layout, theme, None)
        assert len(calls) == _box_capacity() == 15

    def test_orden_de_pintado_es_por_profundidad(self, monkeypatch):
        # El orden de las llamadas coincide con sorted(visible, _box_depth_key),
        # no con el orden de inserción del dict.
        theme = Theme.default()
        world = _world_n_boxes(15)
        layout = _layout_for(world, theme)
        calls = self._record_positions(monkeypatch)
        _draw_resting_packages(make_surface(), world.packages, layout, theme, None)
        ids = _packages_at_loc(world.packages)["dep"]
        expected_order = sorted(range(len(ids)), key=_box_depth_key)
        expected = [
            (layout.package_position(ids[i], theme).x,
             layout.package_position(ids[i], theme).y)
            for i in expected_order
        ]
        assert calls == expected

    def test_exclude_no_descuadra_indices(self, monkeypatch):
        # Excluir una caja (p.ej. la que se mueve) la salta SIN reindexar:
        # las demás conservan exactamente su posición de slot.
        theme = Theme.default()
        world = _world_n_boxes(15)
        layout = _layout_for(world, theme)
        ids = _packages_at_loc(world.packages)["dep"]
        excluded = ids[3]
        pos_de = {
            pid: (layout.package_position(pid, theme).x,
                  layout.package_position(pid, theme).y)
            for pid in ids
        }
        calls = self._record_positions(monkeypatch)
        _draw_resting_packages(make_surface(), world.packages, layout, theme,
                               None, exclude=(excluded,))
        assert len(calls) == 14
        assert pos_de[excluded] not in calls           # la excluida no se pinta
        for pid in ids:
            if pid != excluded:
                assert pos_de[pid] in calls            # las demás, en su slot


# ---------------------------------------------------------------------------
# Regresión: _draw_persons reparte los sets de sprite de persona (personN)
# entre las personas (round-robin por id ordenado) y usa la variante "caja"
# cuando la persona ya recibió su entrega (has_received no vacío).
# ---------------------------------------------------------------------------
from droneplan_viz.render.painter import _draw_persons  # noqa: E402


class _FakeSM:
    """SpriteManager mínimo: solo expone los sets de persona disponibles."""

    def __init__(self, bases):
        self._bases = bases

    def person_sprite_bases(self):
        return self._bases

    def get_person(self, base, *, box, size):
        return None  # fuerza primitiva; aquí solo medimos el reparto


class TestDrawPersonsRepartoYvariante:
    def _world(self, person_specs):
        # person_specs: lista de (loc_id, recibido: bool)
        content = Content(id="c")
        locs = {}
        persons = {}
        for i, (loc, recibido) in enumerate(person_specs):
            locs[loc] = Location(id=loc)
            persons[f"vec{i}"] = Person(
                id=f"vec{i}", position=loc,
                needs=() if recibido else (content,),
                has_received=(content,) if recibido else (),
            )
        return World(locations=locs, drones={}, transporters={}, packages={},
                     persons=persons, contents={"c": content}, costs={})

    def _record(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            _painter_mod, "draw_person",
            lambda surface, pos, pid, **kw: calls.append(
                (pid, kw.get("sprite_base"), kw.get("has_box"))
            ),
        )
        return calls

    def test_reparto_round_robin_por_id_ordenado(self, monkeypatch):
        theme = Theme.default()
        world = self._world([("l0", False), ("l1", False),
                             ("l2", False), ("l3", False)])
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        calls = self._record(monkeypatch)
        _draw_persons(make_surface(), world.persons, layout, theme,
                      _FakeSM(("person1", "person2")))
        bases = {pid: base for pid, base, _ in calls}
        assert bases == {"vec0": "person1", "vec1": "person2",
                         "vec2": "person1", "vec3": "person2"}

    def test_variante_box_si_recibido(self, monkeypatch):
        theme = Theme.default()
        world = self._world([("l0", False), ("l1", True)])
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        calls = self._record(monkeypatch)
        _draw_persons(make_surface(), world.persons, layout, theme,
                      _FakeSM(("person1",)))
        box = {pid: has_box for pid, _, has_box in calls}
        assert box == {"vec0": False, "vec1": True}

    def test_sin_sets_sprite_base_none(self, monkeypatch):
        theme = Theme.default()
        world = self._world([("l0", False)])
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        calls = self._record(monkeypatch)
        _draw_persons(make_surface(), world.persons, layout, theme, _FakeSM(()))
        assert calls[0][1] is None  # sprite_base None → painter cae a primitiva


# ---------------------------------------------------------------------------
# Entrega: la caja entregada NO se pinta en el suelo (la representa el sprite
# de la persona) y el destino de la coreografía es la PERSONA, no el suelo.
# ---------------------------------------------------------------------------
from droneplan_viz.render.painter import (  # noqa: E402
    _approach_target_pos,
    _delivered_package_ids,
    _drop_box_pos,
)
from droneplan_viz.render.interpolation import (  # noqa: E402
    PackagePlace,
    TransitionPackageMove,
)


class TestEntregaSinCajaDuplicada:
    def _content(self):
        return Content(id="medicina")

    def test_delivered_ids_detecta_entregada(self):
        c = self._content()
        packages = {"p0": Package(id="p0", contains=c, at=AtLocation(loc_id="casa"))}
        persons = {"v": Person(id="v", position="casa", needs=(), has_received=(c,))}
        assert _delivered_package_ids(packages, persons) == {"p0"}

    def test_no_entregada_no_se_marca(self):
        c = self._content()
        packages = {"p0": Package(id="p0", contains=c, at=AtLocation(loc_id="dep"))}
        persons = {"v": Person(id="v", position="casa", needs=(c,))}  # aún espera
        assert _delivered_package_ids(packages, persons) == set()

    def test_multiplicidad_solo_suprime_las_entregadas(self):
        c = self._content()
        # dos cajas del mismo contenido en la misma loc; la persona recibió UNA
        packages = {
            "p0": Package(id="p0", contains=c, at=AtLocation(loc_id="casa")),
            "p1": Package(id="p1", contains=c, at=AtLocation(loc_id="casa")),
        }
        persons = {"v": Person(id="v", position="casa", needs=(), has_received=(c,))}
        # solo se marca UNA (la primera por orden alfabético de id)
        assert _delivered_package_ids(packages, persons) == {"p0"}

    def test_resting_omite_la_entregada(self, monkeypatch):
        theme = Theme.default()
        c = self._content()
        world = World(
            locations={"casa": Location(id="casa")}, drones={}, transporters={},
            packages={"p0": Package(id="p0", contains=c, at=AtLocation(loc_id="casa"))},
            persons={"v": Person(id="v", position="casa", needs=(), has_received=(c,))},
            contents={"medicina": c}, costs={},
        )
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        calls = []
        monkeypatch.setattr(_painter_mod, "draw_package",
                            lambda *a, **k: calls.append(a))
        _draw_resting_packages(make_surface(), world.packages, layout, theme, None,
                               persons=world.persons)
        assert calls == []  # la caja entregada no se dibuja en el suelo

    def test_resting_sin_persons_dibuja_normal(self, monkeypatch):
        # Sin pasar persons, no se filtra nada (compatibilidad).
        theme = Theme.default()
        c = self._content()
        world = World(
            locations={"casa": Location(id="casa")}, drones={}, transporters={},
            packages={"p0": Package(id="p0", contains=c, at=AtLocation(loc_id="casa"))},
            persons={"v": Person(id="v", position="casa", needs=(), has_received=(c,))},
            contents={"medicina": c}, costs={},
        )
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        calls = []
        monkeypatch.setattr(_painter_mod, "draw_package",
                            lambda *a, **k: calls.append(a))
        _draw_resting_packages(make_surface(), world.packages, layout, theme, None)
        assert len(calls) == 1


class TestEntregaApuntaALaPersona:
    def _setup(self):
        theme = Theme.default()
        c = Content(id="medicina")
        world = World(
            locations={"casa": Location(id="casa")}, drones={}, transporters={},
            packages={"p0": Package(id="p0", contains=c,
                                    at=AtLocation(loc_id="casa"))},
            persons={"v": Person(id="v", position="casa", needs=(c,))},
            contents={"medicina": c}, costs={},
        )
        layout = compute_layout(world, _scene_viewbox((700, 500), theme))
        tr = TransitionPackageMove(
            drone_id="d", package_id="p0",
            from_place=PackagePlace(kind="arm", drone_id="d", arm_id="izq"),
            to_place=PackagePlace(kind="loc", loc_id="casa"),
            person_id="v",
        )
        return theme, world, layout, tr

    def test_drop_box_pos_es_la_persona(self):
        theme, world, layout, tr = self._setup()
        assert _drop_box_pos(layout, tr, theme) == layout.person_position("v", theme)

    def test_approach_target_es_la_persona(self):
        theme, world, layout, tr = self._setup()
        got = _approach_target_pos(layout, layout, world, tr, theme)
        assert got == layout.person_position("v", theme)

    def test_sin_person_id_no_apunta_a_persona(self):
        # Un drop sin person_id (no es Deliver) va al slot del suelo, no a nadie.
        theme, world, layout, _ = self._setup()
        tr = TransitionPackageMove(
            drone_id="d", package_id="p0",
            from_place=PackagePlace(kind="arm", drone_id="d", arm_id="izq"),
            to_place=PackagePlace(kind="loc", loc_id="casa"),
            person_id=None,
        )
        assert _drop_box_pos(layout, tr, theme) == layout.package_position("p0", theme)


class TestColocacionLlegadaSimultanea:
    """Dos drones que llegan a la MISMA loc en el mismo instante: el que se
    mueve apunta a su hueco FINAL del anillo (grupo asentado), no al centro,
    evitando el salto de recolocación."""

    def _historia(self):
        from droneplan_viz import DronePlanViz
        v = DronePlanViz()
        for L in ("n", "s", "hub"):
            v.world.location(L)
        v.world.costes({("n", "hub"): 10.0, ("s", "hub"): 10.0}, simetrico=True)
        v.world.content("m")
        v.world.person("a", at="hub", necesita=["m"])
        v.world.package("c", contiene="m", at="n")
        v.agents.drone("A", at="n")
        v.agents.drone("B", at="s")
        v.mover("A", a="hub", inicio=0.0, duracion=10.0, id="mA")
        v.mover("B", a="hub", inicio=0.0, duracion=10.0, id="mB")
        w, p = v.build()
        r = PlanRunner(w)
        r.execute(p)
        return r.history

    def _i_llega(self, h, drone_id, loc):
        for i in range(h.head_index + 1):
            if h.at(i).world.drones[drone_id].position == loc:
                return i
        raise AssertionError("el dron no llega")

    def test_grupo_asentado_incluye_ambos(self):
        from droneplan_viz.render.painter import _settled_colocated_at
        h = self._historia()
        i_a = self._i_llega(h, "A", "hub")  # A llega antes (B aún fuera)
        # El grupo ASENTADO en hub incluye a los dos, pese a que en ese
        # snapshot intermedio B todavía no figura en hub.
        assert _settled_colocated_at(h, i_a, "hub") == ["A", "B"]

    def test_destino_del_move_coincide_con_hueco_final(self):
        from droneplan_viz.render.painter import _settled_colocated_at
        from droneplan_viz.render.layout import compute_layout
        h = self._historia()
        theme = Theme.default()
        vb = _scene_viewbox((900, 640), theme)
        i_a = self._i_llega(h, "A", "hub")
        lay_start = compute_layout(
            h.at(i_a - 1).world, vb, spread_factor=theme.location_spread_factor
        )
        lay_final = compute_layout(
            h.at(h.head_index).world, vb, spread_factor=theme.location_spread_factor
        )
        final_A = lay_final.drone_position("A", theme)
        settled = _settled_colocated_at(h, i_a, "hub")
        d_to = lay_start.drone_position_for_colocation("hub", "A", settled, theme)
        # El destino de la animación coincide con el hueco final: sin salto.
        assert abs(d_to.x - final_A.x) < 1.0
        assert abs(d_to.y - final_A.y) < 1.0

    def test_reparto_masivo_6_drones_sin_snap_de_colocacion(self):
        """Regresión (reparto masivo): 6 drones que vuelven al depósito en el
        mismo instante no saltan. El grupo ASENTADO se usa de forma coherente
        en el Move (destino), en el idle y en la acción, así que ningún dron
        salta >15 px entre frames contiguos durante la co-localización."""
        import importlib.util
        import math
        from pathlib import Path
        import pygame
        from droneplan_viz.render import Timeline, render_world_at
        from droneplan_viz.render.sprite_manager import SpriteManager
        import droneplan_viz.render.painter as _P

        ejp = Path(__file__).resolve().parents[2] / "examples" / "escala_reparto_masivo.py"
        _sp = importlib.util.spec_from_file_location("_ej_masivo", ejp)
        ej = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(ej)
        viz = ej.construir(n_casas=12, n_drones=6)
        w, p = viz.build()
        r = PlanRunner(w)
        r.execute(p)
        theme = Theme.default()
        tl = Timeline(r.history, theme=theme)
        sm = SpriteManager(theme)

        cur = {"t": 0.0}
        rec: dict[str, list] = {}
        orig = _P.draw_drone

        def spy(surface, position, drone_id, *a, **k):
            rec.setdefault(drone_id, []).append((cur["t"], position.x, position.y))
            return orig(surface, position, drone_id, *a, **k)

        _P.draw_drone = spy
        try:
            n = int(tl.duration / 0.1)
            for i in range(n + 1):
                cur["t"] = i * 0.1
                s = pygame.Surface((640, 480))
                s.fill(theme.background)
                render_world_at(s, tl, cur["t"], theme=theme,
                                sprite_manager=sm, anim_time=0.4)
        finally:
            _P.draw_drone = orig

        snaps = sum(
            1
            for d in rec
            for (a, xa, ya), (b, xb, yb) in zip(rec[d], rec[d][1:])
            if math.hypot(xb - xa, yb - ya) > 15 and b - a < 0.15
        )
        assert snaps == 0, f"hubo {snaps} saltos de colocación >15px"
