"""Tests de la cámara (zoom + pan) y de locate_entity en el painter.

ZOOM UNIFORME (ver memoria_sesion_d.md §16): a diferencia de la versión
previa, el zoom escala TODO junto (sprites, espaciado del grafo, offsets,
texto, aristas) como un visor estándar. La implementación renderiza la
escena a zoom=1 sobre una surface intermedia y luego escala esa imagen con
pygame.transform.scale (nearest-neighbor) + pan (painter.Estrategia B).

Cobertura:
  - Invariante de no-regresión: zoom=1.0, pan=(0,0) es píxel-idéntico al
    render sin esos kwargs (primitivas, sprites, frames, coreografía).
  - Zoom UNIFORME: a zoom=2 los sprites (y el resto de la escena) se DUPLICAN
    de tamaño; a zoom<1 se reducen. El centro del zoom es el centro de la
    surface; el pan desplaza en píxeles tras el zoom.
        NOTA: la clase TestZoomEscalaUniforme.test_zoom_2x_duplica_tamano_del_sprite
        SUSTITUYE al antiguo test_zoom_no_cambia_tamano_del_drone (que blindaba
        "el zoom NO escala los sprites"). Ahora se afirma su contrario.
  - Saturación del zoom al rango [ZOOM_MIN, ZOOM_MAX] (escala absoluta; ahora [0.3, 2.5]).
  - locate_entity: contrato "posición a zoom=1, pan=0" para los 5 tipos de
    entidad, con ValueError/KeyError donde corresponde.
  - Click-to-focus: el pan calculado desde locate_entity centra la entidad,
    tanto a zoom=1 como a zoom>1.
"""
from __future__ import annotations

import os

import numpy as np
import pygame
import pygame.surfarray as sa
import pytest

from droneplan_viz.commands import Deliver, Move, PickUp
from droneplan_viz.domain import (
    Arm,
    AtLocation,
    Content,
    Drone,
    DroneState,
    Location,
    MetricsTracker,
    Package,
    Person,
    World,
)
from droneplan_viz.history import WorldSnapshot
from droneplan_viz.render import (
    Theme,
    ZOOM_MAX,
    ZOOM_MIN,
    locate_entity,
    locate_entity_interpolated,
    render_frame,
    render_snapshot,
)
from droneplan_viz.render.painter import _clamp_zoom
from droneplan_viz.render.sprite_manager import SpriteManager
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAGENTA = (255, 0, 255)


def make_surface(w: int = 700, h: int = 500) -> pygame.Surface:
    """Surface opaca del tamaño por defecto de los tests de cámara."""
    return pygame.Surface((w, h))


def _snap(world: World) -> WorldSnapshot:
    """Envuelve un World en un WorldSnapshot mínimo para render."""
    return WorldSnapshot(
        world=world, metrics=MetricsTracker(), produced_by=None, timestamp=0.0
    )


def _identical(s1: pygame.Surface, s2: pygame.Surface) -> bool:
    """True si dos surfaces son píxel-idénticas."""
    return np.array_equal(sa.array3d(s1), sa.array3d(s2))


def _magenta_bbox(surface: pygame.Surface, tol: int = 12):
    """Bounding box (minx, miny, maxx, maxy) de los píxeles magenta, o None.

    sa.array3d indexa [x][y][rgb] (primer eje = x), así que xs son columnas
    (coordenada X de pantalla) e ys son filas (coordenada Y).
    """
    a = sa.array3d(surface).astype(int)
    mask = (
        (np.abs(a[:, :, 0] - MAGENTA[0]) <= tol)
        & (np.abs(a[:, :, 1] - MAGENTA[1]) <= tol)
        & (np.abs(a[:, :, 2] - MAGENTA[2]) <= tol)
    )
    xs, ys = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _bbox_size(b) -> tuple[int, int]:
    return (b[2] - b[0], b[3] - b[1])


def _has_color_near(
    surface: pygame.Surface,
    center: tuple[float, float],
    half_box: int = 70,
    target: tuple[int, int, int] = MAGENTA,
    tol: int = 12,
) -> bool:
    """True si hay píxeles `target` dentro de un cuadrado alrededor de center."""
    a = sa.array3d(surface).astype(int)
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1 = max(0, cx - half_box), min(a.shape[0], cx + half_box)
    y0, y1 = max(0, cy - half_box), min(a.shape[1], cy + half_box)
    region = a[x0:x1, y0:y1]
    mask = (
        (np.abs(region[:, :, 0] - target[0]) <= tol)
        & (np.abs(region[:, :, 1] - target[1]) <= tol)
        & (np.abs(region[:, :, 2] - target[2]) <= tol)
    )
    return bool(mask.any())


def _make_magenta_drone(tmp_path) -> tuple[Theme, SpriteManager]:
    """Crea un sprite 'drone' magenta del tamaño del cuerpo base y devuelve
    (theme, sprite_manager) que lo cargan. Permite medir el sprite del dron
    por color en los renders (los demás caen a primitiva)."""
    theme = Theme.default().with_overrides(sprite_dir=str(tmp_path))
    bw, bh = theme.sprite_anchors.body_size
    surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
    surf.fill((*MAGENTA, 255))
    pygame.image.save(surf, os.path.join(str(tmp_path), "drone.png"))
    return theme, SpriteManager(theme)


def _world_drone_centered() -> World:
    """World con UNA location (queda en el centro de la surface) y un dron.

    El dron flota sobre el centro de la location (offset suelo + hover), así
    que su posición está ~60px por ENCIMA del centro de la surface: queda
    fuera-de-centro lo justo para validar el mapeo de posición del zoom, pero
    on-screen a zoom 2 sobre una surface 700×500.
    """
    return World(
        locations={"centro": Location(id="centro")},
        drones={
            "d1": Drone(id="d1", position="centro", arms=(), state=DroneState.IDLE)
        },
    )


def make_world_move() -> World:
    """d1 en casa1, casa2 a la que mover (para frames interpolados)."""
    locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
    drones = {"d1": Drone(id="d1", position="casa1", arms=(), state=DroneState.IDLE)}
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(locations=locs, drones=drones, costs=costs)


def make_world_pickup_deliver() -> World:
    """d1 (dos brazos), pkg_med1 en suelo y p1 que la necesita."""
    contents = {"medicina": Content(id="medicina")}
    locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
    drones = {
        "d1": Drone(
            id="d1",
            position="casa1",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        )
    }
    persons = {"p1": Person(id="p1", position="casa1", needs=(contents["medicina"],))}
    packages = {
        "pkg_med1": Package(
            id="pkg_med1", contains=contents["medicina"], at=AtLocation(loc_id="casa1")
        )
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


# ---------------------------------------------------------------------------
# Invariante de no-regresión: defaults == sin cámara
# ---------------------------------------------------------------------------


class TestInvarianteNoRegresion:
    """zoom=1.0, pan=(0,0) produce EXACTAMENTE el mismo render que omitirlos."""

    S = (700, 500)

    def test_snapshot_primitivas_invariante(self, rich_world):
        snap = _snap(rich_world)
        s1, s2 = make_surface(), make_surface()
        render_snapshot(s1, snap)
        render_snapshot(s2, snap, zoom=1.0, pan=(0, 0))
        assert _identical(s1, s2)

    def test_snapshot_sprites_invariante(self, rich_world):
        snap = _snap(rich_world)
        sm = SpriteManager(Theme.default())
        s1, s2 = make_surface(), make_surface()
        render_snapshot(s1, snap, sprite_manager=sm)
        render_snapshot(s2, snap, sprite_manager=sm, zoom=1.0, pan=(0, 0))
        assert _identical(s1, s2)

    def test_frame_varios_progress_invariante(self):
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(
            Plan.sequential([Move(drone_id="d1", destination_id="casa2", duration=5.0)])
        )
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            s1, s2 = make_surface(), make_surface()
            render_frame(s1, snap_a, snap_b, progress=p)
            render_frame(s2, snap_a, snap_b, progress=p, zoom=1.0, pan=(0, 0))
            assert _identical(s1, s2), f"divergencia en progress={p}"

    def test_frame_pickup_deliver_invariante(self):
        world = make_world_pickup_deliver()
        runner = PlanRunner(world)
        runner.execute(
            Plan(
                scheduled=(
                    ScheduledCommand(
                        command=PickUp(
                            drone_id="d1",
                            arm_id="izq",
                            package_id="pkg_med1",
                            duration=5.0,
                        ),
                        start_time=0.0,
                    ),
                    ScheduledCommand(
                        command=Deliver(
                            drone_id="d1",
                            package_id="pkg_med1",
                            person_id="p1",
                            duration=5.0,
                        ),
                        start_time=5.0,
                    ),
                )
            )
        )
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        for p in (0.3, 0.6):
            s1, s2 = make_surface(), make_surface()
            render_frame(s1, snap_a, snap_b, progress=p)
            render_frame(s2, snap_a, snap_b, progress=p, zoom=1.0, pan=(0, 0))
            assert _identical(s1, s2), f"divergencia en progress={p}"


# ---------------------------------------------------------------------------
# Zoom UNIFORME: escala tamaño Y posición; pan desplaza
# ---------------------------------------------------------------------------


class TestZoomEscalaUniforme:
    """El zoom escala TODO uniformemente alrededor del centro de la surface."""

    S = (700, 500)

    def _render(self, tmp_path, zoom, pan=(0, 0)):
        theme, sm = _make_magenta_drone(tmp_path)
        surf = make_surface(*self.S)
        render_snapshot(
            surf, _snap(_world_drone_centered()), theme=theme,
            sprite_manager=sm, zoom=zoom, pan=pan,
        )
        return surf

    def test_zoom_2x_duplica_tamano_del_sprite(self, tmp_path):
        """A zoom=2 el sprite del dron mide el DOBLE.

        SUSTITUYE al antiguo `test_zoom_no_cambia_tamano_del_drone`: aquel
        blindaba que el zoom NO escalaba el sprite; con zoom uniforme el
        sprite SÍ escala con el zoom.
        """
        b1 = _magenta_bbox(self._render(tmp_path, 1.0))
        b2 = _magenta_bbox(self._render(tmp_path, 2.0))
        assert b1 is not None and b2 is not None
        w1, h1 = _bbox_size(b1)
        w2, h2 = _bbox_size(b2)
        assert 1.85 <= w2 / w1 <= 2.15, f"ratio ancho {w2/w1}"
        assert 1.85 <= h2 / h1 <= 2.15, f"ratio alto {h2/h1}"

    def test_zoom_min_reduce_tamano_del_sprite(self, tmp_path):
        """A zoom=ZOOM_MIN (0.3) el sprite del dron se ve reducido a ~ese factor."""
        b1 = _magenta_bbox(self._render(tmp_path, 1.0))
        bz = _magenta_bbox(self._render(tmp_path, ZOOM_MIN))
        assert b1 is not None and bz is not None
        w1, h1 = _bbox_size(b1)
        wz, hz = _bbox_size(bz)
        assert ZOOM_MIN - 0.12 <= wz / w1 <= ZOOM_MIN + 0.12, f"ratio ancho {wz/w1}"
        assert ZOOM_MIN - 0.12 <= hz / h1 <= ZOOM_MIN + 0.12, f"ratio alto {hz/h1}"

    def test_zoom_escala_posicion_centrado_en_surface(self, tmp_path):
        """El centro del sprite se mapea por c + (M1 - c) * zoom.

        Verifica que el zoom se centra en el centro de la surface: un punto
        a M1 (medido a zoom=1) aparece a c + (M1-c)*zoom a otro zoom.
        """
        cx, cy = self.S[0] / 2.0, self.S[1] / 2.0
        m1 = _bbox_center(_magenta_bbox(self._render(tmp_path, 1.0)))
        m2 = _bbox_center(_magenta_bbox(self._render(tmp_path, 2.0)))
        pred = (cx + (m1[0] - cx) * 2.0, cy + (m1[1] - cy) * 2.0)
        assert abs(m2[0] - pred[0]) <= 3.0, f"x {m2[0]} vs {pred[0]}"
        assert abs(m2[1] - pred[1]) <= 3.0, f"y {m2[1]} vs {pred[1]}"

    def test_pan_desplaza_sin_reescalar(self, tmp_path):
        """A zoom=1, el pan desplaza el sprite en píxeles SIN cambiar su tamaño."""
        b1 = _magenta_bbox(self._render(tmp_path, 1.0))
        bp = _magenta_bbox(self._render(tmp_path, 1.0, pan=(60, 40)))
        m1, mp = _bbox_center(b1), _bbox_center(bp)
        assert abs(mp[0] - (m1[0] + 60)) <= 3.0, f"x {mp[0]} vs {m1[0]+60}"
        assert abs(mp[1] - (m1[1] + 40)) <= 3.0, f"y {mp[1]} vs {m1[1]+40}"
        # tamaño igual (±2 px por redondeo nearest-neighbor)
        w1, h1 = _bbox_size(b1)
        wp, hp = _bbox_size(bp)
        assert abs(wp - w1) <= 2 and abs(hp - h1) <= 2

    def test_combinacion_zoom_y_pan(self, tmp_path):
        """zoom=2 + pan=(40,0): posición = c + (M1-c)*2 + pan; tamaño x2."""
        cx, cy = self.S[0] / 2.0, self.S[1] / 2.0
        b1 = _magenta_bbox(self._render(tmp_path, 1.0))
        bc = _magenta_bbox(self._render(tmp_path, 2.0, pan=(40, 0)))
        m1, mc = _bbox_center(b1), _bbox_center(bc)
        pred = (cx + (m1[0] - cx) * 2.0 + 40, cy + (m1[1] - cy) * 2.0 + 0)
        assert abs(mc[0] - pred[0]) <= 3.0, f"x {mc[0]} vs {pred[0]}"
        assert abs(mc[1] - pred[1]) <= 3.0, f"y {mc[1]} vs {pred[1]}"
        w1, _ = _bbox_size(b1)
        wc, _ = _bbox_size(bc)
        assert 1.85 <= wc / w1 <= 2.15

    def test_zoom_max_amplia_sin_lanzar(self, tmp_path):
        """A zoom=ZOOM_MAX (2.5) el render no lanza y el sprite sigue presente."""
        surf = self._render(tmp_path, ZOOM_MAX)
        assert _magenta_bbox(surf) is not None


# ---------------------------------------------------------------------------
# Saturación (clamp) del zoom
# ---------------------------------------------------------------------------


class TestClampZoom:
    """El zoom se satura silenciosamente a [ZOOM_MIN, ZOOM_MAX]."""

    S = (700, 500)

    def test_clamp_zoom_directo(self):
        assert _clamp_zoom(10.0) == ZOOM_MAX
        assert _clamp_zoom(0.01) == ZOOM_MIN
        assert _clamp_zoom(2.0) == 2.0

    def test_clamp_superior_via_render(self):
        snap = _snap(_world_drone_centered())
        s_hi, s_max = make_surface(), make_surface()
        render_snapshot(s_hi, snap, zoom=10.0)
        render_snapshot(s_max, snap, zoom=ZOOM_MAX)
        assert _identical(s_hi, s_max)

    def test_clamp_inferior_via_render(self):
        snap = _snap(_world_drone_centered())
        s_lo, s_min = make_surface(), make_surface()
        render_snapshot(s_lo, snap, zoom=0.01)
        render_snapshot(s_min, snap, zoom=ZOOM_MIN)
        assert _identical(s_lo, s_min)


# ---------------------------------------------------------------------------
# locate_entity: contrato "posición a zoom=1, pan=0"
# ---------------------------------------------------------------------------


class TestLocateEntity:
    """locate_entity resuelve los 5 tipos de entidad sin aplicar cámara."""

    S = (700, 500)

    def test_locate_drone_devuelve_par_de_ints(self, rich_world):
        p = locate_entity(rich_world, "d1", self.S)
        assert isinstance(p, tuple) and len(p) == 2
        assert all(isinstance(c, int) for c in p)

    def test_locate_location_devuelve_par_de_ints(self, rich_world):
        p = locate_entity(rich_world, "casa1", self.S)
        assert isinstance(p, tuple) and len(p) == 2
        assert all(isinstance(c, int) for c in p)

    def test_locate_transporter(self, rich_world):
        p = locate_entity(rich_world, "t1", self.S)
        assert len(p) == 2 and all(isinstance(c, int) for c in p)

    def test_locate_person(self, rich_world):
        p = locate_entity(rich_world, "p1", self.S)
        assert len(p) == 2 and all(isinstance(c, int) for c in p)

    def test_locate_package_at_location_no_lanza(self, rich_world):
        # pkg_med1 está AtLocation(deposito) en rich_world.
        p = locate_entity(rich_world, "pkg_med1", self.S)
        assert len(p) == 2 and all(isinstance(c, int) for c in p)

    def test_locate_drone_por_encima_de_su_location(self, rich_world):
        # d1 está en 'deposito'; el dron flota POR ENCIMA del centro de su
        # location (y menor = más arriba en pantalla).
        loc_id = rich_world.drones["d1"].position
        drone_p = locate_entity(rich_world, "d1", self.S)
        loc_p = locate_entity(rich_world, loc_id, self.S)
        assert drone_p[1] < loc_p[1]

    def test_locate_package_held_lanza_value_error(self, package_held_by_arm_world):
        with pytest.raises(ValueError):
            locate_entity(package_held_by_arm_world, "pkg1", self.S)

    def test_locate_package_in_transporter_lanza_value_error(
        self, package_in_transporter_world
    ):
        with pytest.raises(ValueError):
            locate_entity(package_in_transporter_world, "pkg1", self.S)

    def test_locate_id_inexistente_lanza_key_error(self, rich_world):
        with pytest.raises(KeyError):
            locate_entity(rich_world, "no_existe", self.S)

    def test_locate_no_aplica_camara(self, rich_world):
        # El contrato es "como si zoom=1, pan=0": debe coincidir con la
        # posición que el render canónico (sin cámara) usaría para el dron.
        # Usa el MISMO viewbox que el render (con holgura de decoraciones).
        from droneplan_viz.render.layout import compute_layout
        from droneplan_viz.render.painter import _scene_viewbox

        theme = Theme.default()
        layout = compute_layout(
            rich_world,
            _scene_viewbox(self.S, theme),
            spread_factor=theme.location_spread_factor,
        )
        expected = layout.drone_position("d1", theme).as_int_tuple()
        assert locate_entity(rich_world, "d1", self.S) == expected


# ---------------------------------------------------------------------------
# Click-to-focus: el pan derivado de locate_entity centra la entidad
# ---------------------------------------------------------------------------


class TestClickToFocus:
    """El pan calculado desde locate_entity centra la entidad en la surface."""

    S = (700, 500)

    def test_focus_zoom_1_centra_la_entidad(self, tmp_path):
        theme, sm = _make_magenta_drone(tmp_path)
        world = _world_drone_centered()
        cx, cy = self.S[0] / 2.0, self.S[1] / 2.0
        lx, ly = locate_entity(world, "d1", self.S, theme=theme)
        # A zoom=1: pan = centro - posición.
        pan = (round(cx - lx), round(cy - ly))
        surf = make_surface(*self.S)
        render_snapshot(
            surf, _snap(world), theme=theme, sprite_manager=sm, zoom=1.0, pan=pan
        )
        assert _has_color_near(surf, (cx, cy), half_box=40)

    def test_focus_con_zoom_2_centra_la_entidad(self, tmp_path):
        """Con zoom=2, pan = (c - L) * zoom centra la entidad (fórmula E2)."""
        theme, sm = _make_magenta_drone(tmp_path)
        world = _world_drone_centered()
        cx, cy = self.S[0] / 2.0, self.S[1] / 2.0
        lx, ly = locate_entity(world, "d1", self.S, theme=theme)
        z = 2.0
        pan = (round((cx - lx) * z), round((cy - ly) * z))
        surf = make_surface(*self.S)
        render_snapshot(
            surf, _snap(world), theme=theme, sprite_manager=sm, zoom=z, pan=pan
        )
        # El sprite a zoom 2 mide ~115px: half_box holgado.
        assert _has_color_near(surf, (cx, cy), half_box=70)


# ---------------------------------------------------------------------------
# Coreografía con cámara: no debe romper con zoom/pan activos
# ---------------------------------------------------------------------------


class TestCoreografiaConCamara:
    """Las rutas interpoladas (Move, PickUp/Deliver) renderizan con cámara."""

    def test_move_con_zoom_y_pan_no_lanza(self):
        world = make_world_move()
        runner = PlanRunner(world)
        runner.execute(
            Plan.sequential([Move(drone_id="d1", destination_id="casa2", duration=5.0)])
        )
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        surf = make_surface()
        # No debe lanzar y debe dibujar algo distinto del fondo.
        render_frame(surf, snap_a, snap_b, progress=0.5, zoom=2.0, pan=(30, -20))
        assert sa.array3d(surf).std() > 0

    def test_pickup_deliver_con_zoom_no_lanza(self):
        world = make_world_pickup_deliver()
        runner = PlanRunner(world)
        runner.execute(
            Plan(
                scheduled=(
                    ScheduledCommand(
                        command=PickUp(
                            drone_id="d1",
                            arm_id="izq",
                            package_id="pkg_med1",
                            duration=5.0,
                        ),
                        start_time=0.0,
                    ),
                    ScheduledCommand(
                        command=Deliver(
                            drone_id="d1",
                            package_id="pkg_med1",
                            person_id="p1",
                            duration=5.0,
                        ),
                        start_time=5.0,
                    ),
                )
            )
        )
        snap_a = runner.history.at(1)
        snap_b = runner.history.at(2)
        surf = make_surface()
        render_frame(surf, snap_a, snap_b, progress=0.5, zoom=2.0, pan=(10, 10))
        assert sa.array3d(surf).std() > 0


# ---------------------------------------------------------------------------
# Regresión: la vista por defecto no recorta decoraciones (dron flotante,
# sprite de location) en los bordes superior/inferior.
# ---------------------------------------------------------------------------


def _content_margins(surface: pygame.Surface, theme: Theme):
    """(izq, der, arriba, abajo) píxeles libres entre el contenido y el borde."""
    a = sa.array3d(surface).astype(int)
    bg = np.array(theme.background)
    mask = np.abs(a - bg).sum(axis=2) > 12
    xs, ys = np.where(mask)
    w, h = surface.get_size()
    return xs.min(), w - 1 - xs.max(), ys.min(), h - 1 - ys.max()


class TestNoRecorteVistaDefault:
    """El layout reserva holgura para las decoraciones que sobresalen del
    centro de una location (dron que flota arriba, sprite/etiqueta abajo), de
    modo que la vista por defecto (zoom=1) no las recorta arriba ni abajo.
    """

    def _circular_con_drones(self, n: int) -> World:
        ids = "abcdef"[:n]
        return World(
            locations={c: Location(id=c) for c in ids},
            drones={
                "d1": Drone(id="d1", position=ids[0], arms=(), state=DroneState.IDLE),
                "d2": Drone(id="d2", position=ids[-1], arms=(), state=DroneState.IDLE),
            },
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("S", [(700, 500), (900, 600), (500, 400)])
    def test_sin_recorte_vertical(self, n, S):
        theme = Theme.default()
        surf = pygame.Surface(S)
        render_snapshot(surf, _snap(self._circular_con_drones(n)), theme=theme)
        _, _, arriba, abajo = _content_margins(surf, theme)
        assert arriba > 0, f"recorte SUPERIOR con n={n}, S={S} (dron flotante)"
        assert abajo > 0, f"recorte INFERIOR con n={n}, S={S} (sprite/etiqueta)"

    def test_dron_superior_visible_completo(self):
        """El sprite del dron de la location más alta no toca el borde de
        arriba (queda holgura), a diferencia del bug reportado."""
        S = (700, 500)
        theme = Theme.default()
        surf = pygame.Surface(S)
        render_snapshot(surf, _snap(self._circular_con_drones(3)), theme=theme)
        _, _, arriba, _ = _content_margins(surf, theme)
        assert arriba >= 5, f"holgura superior insuficiente: {arriba}px"





# ---------------------------------------------------------------------------
# location_spread_factor: separación de locations (layout circular) con cap
# de seguridad que impide recortes en cualquier tamaño de surface.
# ---------------------------------------------------------------------------


class TestSpreadFactor:
    """El factor de separación del layout circular escala la distancia entre
    locations, pero está capado para no rebasar la región segura de
    decoraciones: ningún factor recorta, en ningún tamaño de surface.
    """

    def _circular(self, n: int = 3) -> World:
        ids = "abcdef"[:n]
        return World(
            locations={c: Location(id=c) for c in ids},
            drones={
                "d1": Drone(id="d1", position=ids[0], arms=(), state=DroneState.IDLE),
                "d2": Drone(id="d2", position=ids[-1], arms=(), state=DroneState.IDLE),
            },
        )

    def test_factor_mayor_separa_mas(self):
        """0.50 separa ~1.25x respecto a 0.40 (layout circular)."""
        from droneplan_viz.render.layout import compute_layout
        from droneplan_viz.render.painter import _scene_viewbox

        theme = Theme.default()
        S = (900, 600)
        vb = _scene_viewbox(S, theme)
        w = self._circular(3)

        def span(f):
            lay = compute_layout(w, vb, spread_factor=f)
            ys = [lay.location_position(l).y for l in w.locations]
            return max(ys) - min(ys)

        ratio = span(0.50) / span(0.40)
        assert 1.2 <= ratio <= 1.3, f"ratio de separación {ratio}"

    @pytest.mark.parametrize("S", [(700, 500), (900, 600), (1400, 1000)])
    def test_factor_alto_no_recorta(self, S):
        """Un factor por encima de 0.50 (que sin cap recortaría, y peor en
        surfaces grandes) NO recorta gracias al cap."""
        theme = Theme.default().with_overrides(location_spread_factor=0.85)
        surf = pygame.Surface(S)
        render_snapshot(surf, _snap(self._circular(3)), theme=theme)
        _, _, arriba, abajo = _content_margins(surf, theme)
        assert arriba > 0 and abajo > 0, f"recorta con factor 0.85 en {S}"

    def test_factor_por_encima_de_05_satura(self):
        """Factores >0.50 saturan al borde de la región segura: 0.60 y 0.90
        producen el mismo render (el cap los iguala)."""
        S = (900, 600)
        w = self._circular(3)
        s060 = pygame.Surface(S)
        s090 = pygame.Surface(S)
        render_snapshot(
            s060, _snap(w), theme=Theme.default().with_overrides(location_spread_factor=0.60)
        )
        render_snapshot(
            s090, _snap(w), theme=Theme.default().with_overrides(location_spread_factor=0.90)
        )
        assert _identical(s060, s090)


class TestInmutabilidadConCamara:
    """Renderizar con zoom/pan no muta el estado del mundo."""

    def test_render_con_camara_no_muta_world(self, rich_world):
        snap = _snap(rich_world)
        pos_antes = rich_world.drones["d1"].position
        at_antes = rich_world.packages["pkg_med1"].at
        surf = make_surface()
        render_snapshot(surf, snap, zoom=2.5, pan=(40, 40))
        assert rich_world.drones["d1"].position == pos_antes
        assert rich_world.packages["pkg_med1"].at == at_antes


# ---------------------------------------------------------------------------
# locate_entity_interpolated: posición durante una transición (seguimiento)
# ---------------------------------------------------------------------------


def _world_drone_at(loc: str) -> World:
    """Mundo de dos locations con d1 posado en `loc`."""
    locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
    drones = {"d1": Drone(id="d1", position=loc, arms=(), state=DroneState.IDLE)}
    costs = {("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0}
    return World(locations=locs, drones=drones, costs=costs)


class TestLocateEntityInterpolated:
    SIZE = (900, 600)

    def setup_method(self):
        self.theme = Theme.default()
        # Transición DroneMove REAL: ambos snapshots producidos por el mismo
        # Move (d1: casa1 → casa2). classify_transition lo reconoce como
        # TransitionDroneMove y el localizador interpola la posición del drone.
        move = Move(drone_id="d1", destination_id="casa2",
                    duration=5.0, command_id="mv")
        self.a = WorldSnapshot(world=_world_drone_at("casa1"),
                               metrics=MetricsTracker(), produced_by=move,
                               timestamp=0.0)
        self.b = WorldSnapshot(world=_world_drone_at("casa2"),
                               metrics=MetricsTracker(), produced_by=move,
                               timestamp=5.0)
        self.pa = locate_entity(self.a.world, "d1", self.SIZE, theme=self.theme)
        self.pb = locate_entity(self.b.world, "d1", self.SIZE, theme=self.theme)

    def test_endpoints_difieren_en_el_escenario(self):
        # Precondición del test: el drone realmente se mueve.
        assert self.pa != self.pb

    def test_progress_0_coincide_con_snap_a(self):
        p0 = locate_entity_interpolated(self.a, self.b, 0.0, "d1", self.SIZE, theme=self.theme)
        assert p0 == self.pa

    def test_progress_1_coincide_con_snap_b(self):
        p1 = locate_entity_interpolated(self.a, self.b, 1.0, "d1", self.SIZE, theme=self.theme)
        assert p1 == self.pb

    def test_progress_intermedio_queda_entre_los_extremos(self):
        pm = locate_entity_interpolated(self.a, self.b, 0.5, "d1", self.SIZE, theme=self.theme)
        lo_y, hi_y = sorted((self.pa[1], self.pb[1]))
        assert lo_y < pm[1] < hi_y       # estrictamente entre origen y destino
        assert pm != self.pa and pm != self.pb

    def test_satura_progress_fuera_de_rango(self):
        assert locate_entity_interpolated(self.a, self.b, -1.0, "d1", self.SIZE, theme=self.theme) == self.pa
        assert locate_entity_interpolated(self.a, self.b, 2.0, "d1", self.SIZE, theme=self.theme) == self.pb

    def test_entidad_quieta_equivale_a_locate_estatico(self):
        # Una location no se mueve entre snapshots → misma posición en todo
        # progress, igual a locate_entity(snap_a).
        est = locate_entity(self.a.world, "casa1", self.SIZE, theme=self.theme)
        for p in (0.0, 0.3, 0.5, 1.0):
            assert locate_entity_interpolated(self.a, self.b, p, "casa1", self.SIZE, theme=self.theme) == est

    def test_id_inexistente_propaga_keyerror(self):
        with pytest.raises(KeyError):
            locate_entity_interpolated(self.a, self.b, 0.5, "fantasma", self.SIZE, theme=self.theme)


# ---------------------------------------------------------------------------
# Seguimiento COMPLETO: cualquier objeto móvil a lo largo del replay
# ---------------------------------------------------------------------------


class TestSeguimientoCompleto:
    """locate_entity_interpolated sigue a drones, transporters y paquetes
    en todos sus placements y a lo largo de todas las acciones."""

    SIZE = (900, 600)

    def _snap(self, world, produced_by, ts):
        return WorldSnapshot(world=world, metrics=MetricsTracker(),
                             produced_by=produced_by, timestamp=ts)

    def test_paquete_sostenido_sigue_al_drone_en_move(self):
        import math

        from droneplan_viz.commands import Move
        from droneplan_viz.domain import (
            Arm, Content, Drone, HeldByArm, Location, Package, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="medicina")

        def w(loc):
            return World(
                locations={"casa1": Location(id="casa1"), "casa2": Location(id="casa2")},
                drones={"d1": Drone(id="d1", position=loc, arms=(Arm(id="izq"),),
                                    state=DroneState.IDLE)},
                packages={"pkg": Package(id="pkg", contains=med,
                                         at=HeldByArm(drone_id="d1", arm_id="izq"))},
                contents={"medicina": med},
                costs={("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0},
            )

        mv = Move(drone_id="d1", destination_id="casa2", duration=5.0, command_id="mv")
        a = self._snap(w("casa1"), mv, 0.0)
        b = self._snap(w("casa2"), mv, 5.0)

        dr = locate_entity_interpolated(a, b, 0.5, "d1", self.SIZE, theme=Theme.default())
        pk = locate_entity_interpolated(a, b, 0.5, "pkg", self.SIZE, theme=Theme.default())
        # El paquete viaja en el brazo del drone → muy cerca de él.
        assert math.hypot(pk[0] - dr[0], pk[1] - dr[1]) < 60
        # Y se ha desplazado con el drone entre el inicio y el fin.
        pk0 = locate_entity_interpolated(a, b, 0.0, "pkg", self.SIZE, theme=Theme.default())
        pk1 = locate_entity_interpolated(a, b, 1.0, "pkg", self.SIZE, theme=Theme.default())
        assert pk0 != pk1

    def test_paquete_en_transporter_sigue_al_arrastre(self):
        import math

        from droneplan_viz.commands import MoveWithTransporter
        from droneplan_viz.domain import (
            Arm, Content, Drone, InTransporter, Location, Package, Transporter, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="medicina")

        def w(loc):
            return World(
                locations={"casa1": Location(id="casa1"), "casa2": Location(id="casa2")},
                drones={"d1": Drone(id="d1", position=loc,
                                    arms=(Arm(id="izq"), Arm(id="der")),
                                    state=DroneState.IDLE)},
                transporters={"t1": Transporter(id="t1", position=loc, capacity=4)},
                packages={"pkg": Package(id="pkg", contains=med,
                                         at=InTransporter(transporter_id="t1"))},
                contents={"medicina": med},
                costs={("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0},
            )

        mvt = MoveWithTransporter(drone_id="d1", transporter_id="t1",
                                  destination_id="casa2", duration=5.0, command_id="mvt")
        a = self._snap(w("casa1"), mvt, 0.0)
        b = self._snap(w("casa2"), mvt, 5.0)

        tr = locate_entity_interpolated(a, b, 0.5, "t1", self.SIZE, theme=Theme.default())
        pk = locate_entity_interpolated(a, b, 0.5, "pkg", self.SIZE, theme=Theme.default())
        # El paquete viaja DENTRO del transporter → pegado a él.
        assert math.hypot(pk[0] - tr[0], pk[1] - tr[1]) < 60
        # El transporter (arrastrado) se desplazó.
        tr0 = locate_entity_interpolated(a, b, 0.0, "t1", self.SIZE, theme=Theme.default())
        tr1 = locate_entity_interpolated(a, b, 1.0, "t1", self.SIZE, theme=Theme.default())
        assert tr0 != tr1

    def test_drone_se_mueve_en_coreografia_de_pickup(self):
        from droneplan_viz.commands import PickUp
        from droneplan_viz.domain import (
            Arm, Content, Drone, HeldByArm, AtLocation, Location, Package, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="medicina")
        locs = {"casa1": Location(id="casa1")}

        def w(at):
            return World(
                locations=locs,
                drones={"d1": Drone(id="d1", position="casa1",
                                    arms=(Arm(id="izq"),),
                                    state=DroneState.IDLE)},
                packages={"pkg": Package(id="pkg", contains=med, at=at)},
                contents={"medicina": med},
                costs={},
            )

        pk_cmd = PickUp(drone_id="d1", arm_id="izq", package_id="pkg",
                        duration=4.0, command_id="pk")
        a = self._snap(w(AtLocation(loc_id="casa1")), pk_cmd, 0.0)
        b = self._snap(w(HeldByArm(drone_id="d1", arm_id="izq")), pk_cmd, 4.0)

        # Durante la coreografía el drone baja a la caja: su posición a mitad
        # del primer step difiere del centro (no se queda estático).
        centro = locate_entity_interpolated(a, b, 0.0, "d1", self.SIZE, theme=Theme.default())
        approach = locate_entity_interpolated(a, b, 0.25, "d1", self.SIZE, theme=Theme.default())
        assert centro != approach

    def test_caja_en_pickup_va_del_suelo_al_brazo(self):
        import math

        from droneplan_viz.commands import PickUp
        from droneplan_viz.domain import (
            Arm, Content, Drone, HeldByArm, AtLocation, Location, Package, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="medicina")

        def w(at):
            return World(
                locations={"casa1": Location(id="casa1")},
                drones={"d1": Drone(id="d1", position="casa1",
                                    arms=(Arm(id="izq"),), state=DroneState.IDLE)},
                packages={"pkg": Package(id="pkg", contains=med, at=at)},
                contents={"medicina": med},
                costs={},
            )

        pk_cmd = PickUp(drone_id="d1", arm_id="izq", package_id="pkg",
                        duration=4.0, command_id="pk")
        a = self._snap(w(AtLocation(loc_id="casa1")), pk_cmd, 0.0)
        b = self._snap(w(HeldByArm(drone_id="d1", arm_id="izq")), pk_cmd, 4.0)

        # Al final la caja está sobre el drone (en su brazo).
        dr_fin = locate_entity_interpolated(a, b, 1.0, "d1", self.SIZE, theme=Theme.default())
        pk_fin = locate_entity_interpolated(a, b, 1.0, "pkg", self.SIZE, theme=Theme.default())
        assert math.hypot(pk_fin[0] - dr_fin[0], pk_fin[1] - dr_fin[1]) < 60
        # Y la caja cambió de posición entre el suelo (inicio) y el brazo (fin).
        pk_ini = locate_entity_interpolated(a, b, 0.0, "pkg", self.SIZE, theme=Theme.default())
        assert pk_ini != pk_fin

    def test_drone_se_acerca_al_carrier_en_load(self):
        import math

        from droneplan_viz.commands import LoadIntoTransporter
        from droneplan_viz.domain import (
            Arm, Content, Drone, HeldByArm, InTransporter, Location,
            Package, Transporter, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="m")

        def w(at):
            return World(
                locations={"dep": Location(id="dep")},
                drones={"d1": Drone(id="d1", position="dep",
                                    arms=(Arm(id="izq"),), state=DroneState.IDLE)},
                transporters={"car": Transporter(id="car", position="dep", capacity=3)},
                packages={"pkg": Package(id="pkg", contains=med, at=at)},
                contents={"m": med}, costs={},
            )

        cmd = LoadIntoTransporter(drone_id="d1", package_id="pkg",
                                  transporter_id="car", duration=2.0, command_id="ld")
        a = self._snap(w(HeldByArm(drone_id="d1", arm_id="izq")), cmd, 0.0)
        b = self._snap(w(InTransporter(transporter_id="car")), cmd, 2.0)

        # El dron se DESPLAZA del centro hacia el carrier (coreografía, no
        # estático como en la versión anterior de glide recto).
        centro = locate_entity_interpolated(a, b, 0.0, "d1", self.SIZE, theme=Theme.default())
        approach = locate_entity_interpolated(a, b, 0.25, "d1", self.SIZE, theme=Theme.default())
        assert centro != approach
        # Tras soltarla, la caja se sitúa SOBRE el carrier (oculta dentro),
        # no en un slot de suelo: el seguimiento la lleva al carrier.
        car = locate_entity_interpolated(a, b, 1.0, "car", self.SIZE, theme=Theme.default())
        pk = locate_entity_interpolated(a, b, 1.0, "pkg", self.SIZE, theme=Theme.default())
        assert math.hypot(pk[0] - car[0], pk[1] - car[1]) < 50

    def test_drone_se_acerca_al_carrier_en_unload(self):
        import math

        from droneplan_viz.commands import UnloadFromTransporter
        from droneplan_viz.domain import (
            Arm, Content, Drone, HeldByArm, InTransporter, Location,
            Package, Transporter, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="m")

        def w(at):
            return World(
                locations={"dep": Location(id="dep")},
                drones={"d1": Drone(id="d1", position="dep",
                                    arms=(Arm(id="izq"),), state=DroneState.IDLE)},
                transporters={"car": Transporter(id="car", position="dep", capacity=3)},
                packages={"pkg": Package(id="pkg", contains=med, at=at)},
                contents={"m": med}, costs={},
            )

        cmd = UnloadFromTransporter(drone_id="d1", arm_id="izq", package_id="pkg",
                                    transporter_id="car", duration=2.0, command_id="ul")
        a = self._snap(w(InTransporter(transporter_id="car")), cmd, 0.0)
        b = self._snap(w(HeldByArm(drone_id="d1", arm_id="izq")), cmd, 2.0)

        # El dron se aproxima al carrier para sacar la caja.
        centro = locate_entity_interpolated(a, b, 0.0, "d1", self.SIZE, theme=Theme.default())
        approach = locate_entity_interpolated(a, b, 0.25, "d1", self.SIZE, theme=Theme.default())
        assert centro != approach
        # Al inicio la caja está en el carrier; al final, en el brazo del dron.
        car = locate_entity_interpolated(a, b, 0.0, "car", self.SIZE, theme=Theme.default())
        pk_ini = locate_entity_interpolated(a, b, 0.0, "pkg", self.SIZE, theme=Theme.default())
        assert math.hypot(pk_ini[0] - car[0], pk_ini[1] - car[1]) < 50
        dr_fin = locate_entity_interpolated(a, b, 1.0, "d1", self.SIZE, theme=Theme.default())
        pk_fin = locate_entity_interpolated(a, b, 1.0, "pkg", self.SIZE, theme=Theme.default())
        assert math.hypot(pk_fin[0] - dr_fin[0], pk_fin[1] - dr_fin[1]) < 60

    def test_carrier_se_agarra_con_aproximacion_en_move(self):
        import math

        from droneplan_viz.commands import MoveWithTransporter
        from droneplan_viz.domain import (
            Arm, Content, Drone, InTransporter, Location, Package, Transporter, World,
        )
        from droneplan_viz.domain.drone_state import DroneState

        med = Content(id="m")

        def w(loc):
            return World(
                locations={"a": Location(id="a"), "b": Location(id="b")},
                drones={"d1": Drone(id="d1", position=loc,
                                    arms=(Arm(id="izq"),), state=DroneState.IDLE)},
                transporters={"car": Transporter(id="car", position=loc, capacity=3)},
                packages={"pkg": Package(id="pkg", contains=med,
                                         at=InTransporter(transporter_id="car"))},
                contents={"m": med},
                costs={("a", "b"): 5.0, ("b", "a"): 5.0},
            )

        cmd = MoveWithTransporter(drone_id="d1", transporter_id="car",
                                  destination_id="b", duration=5.0, command_id="mvt")
        a = self._snap(w("a"), cmd, 0.0)
        b = self._snap(w("b"), cmd, 5.0)

        # El carrier NO se teletransporta a los brazos del dron al arrancar:
        # en p≈0 sigue en su posición de reposo (no salta).
        car0 = locate_entity_interpolated(a, b, 0.0, "car", self.SIZE, theme=Theme.default())
        car_eps = locate_entity_interpolated(a, b, 0.02, "car", self.SIZE, theme=Theme.default())
        assert math.hypot(car0[0] - car_eps[0], car0[1] - car_eps[1]) < 5
        # En la fase de transporte el carrier va justo DEBAJO del dron.
        d_mid = locate_entity_interpolated(a, b, 0.5, "d1", self.SIZE, theme=Theme.default())
        c_mid = locate_entity_interpolated(a, b, 0.5, "car", self.SIZE, theme=Theme.default())
        assert c_mid[0] == d_mid[0]
        assert c_mid[1] > d_mid[1]
        # El paquete dentro acompaña al carrier.
        p_mid = locate_entity_interpolated(a, b, 0.5, "pkg", self.SIZE, theme=Theme.default())
        assert math.hypot(p_mid[0] - c_mid[0], p_mid[1] - c_mid[1]) < 5

    def test_acciones_consecutivas_no_vuelven_al_centro(self):
        # Pipeline real: dos PickUp consecutivas del mismo dron en la misma
        # loc (con el snapshot Static "enter-interacting" intercalado). La 2ª
        # acción NO debe arrancar desde el centro (encadena desde la
        # aproximación de la 1ª); sin contexto de vecinos, sí arranca del
        # centro (comportamiento clásico).
        import math

        from droneplan_viz.commands import PickUp
        from droneplan_viz.domain import (
            Arm, AtLocation, Content, Drone, Location, Package, World,
        )
        from droneplan_viz.domain.drone_state import DroneState
        from droneplan_viz.render import Timeline
        from droneplan_viz.render.interpolation import (
            classify_transition, TransitionPackageMove,
        )
        from droneplan_viz.render.layout import compute_layout
        from droneplan_viz.render.painter import _scene_viewbox
        from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

        th = Theme.default()
        med = Content(id="m")
        world = World(
            locations={"dep": Location(id="dep")},
            drones={"d1": Drone(id="d1", position="dep",
                                arms=(Arm(id="izq"), Arm(id="der")),
                                state=DroneState.IDLE)},
            transporters={},
            packages={
                "pa": Package(id="pa", contains=med, at=AtLocation(loc_id="dep")),
                "pb": Package(id="pb", contains=med, at=AtLocation(loc_id="dep")),
            },
            contents={"m": med}, costs={},
        )
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pa",
                               duration=2.0, command_id="c1"),
                start_time=0.0),
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="der", package_id="pb",
                               duration=2.0, command_id="c2"),
                start_time=2.0),
        ))
        runner = PlanRunner(world)
        runner.execute(plan)
        tl = Timeline(runner.history, theme=th)

        lay = compute_layout(world, _scene_viewbox(self.SIZE, th),
                             spread_factor=th.location_spread_factor)
        ctr = lay.drone_position("d1", th).as_int_tuple()

        found = False
        for k in range(400):
            t = tl.duration * k / 399.0
            prev_action, a, b, next_action, p = tl.sample_with_neighbors(t)
            tr = classify_transition(a, b)
            if isinstance(tr, TransitionPackageMove) and tr.package_id == "pb" and p < 0.05:
                con = locate_entity_interpolated(
                    a, b, p, "d1", self.SIZE, theme=th,
                    prev_action=prev_action, next_action=next_action,
                )
                sin = locate_entity_interpolated(a, b, p, "d1", self.SIZE, theme=th)
                # Con vecinos: encadena (lejos del centro).
                assert math.hypot(con[0] - ctr[0], con[1] - ctr[1]) > 5
                # Sin vecinos: arranca del centro (clásico).
                assert math.hypot(sin[0] - ctr[0], sin[1] - ctr[1]) < 5
                found = True
                break
        assert found, "no se halló el inicio de la 2ª PickUp"

    def test_pausa_entre_steps_encadenados_colapsa(self):
        # La pausa (Static enter-interacting) ENTRE dos steps encadenados del
        # mismo dron en la misma loc se COLAPSA a una duración ínfima: el dron
        # fluye de un step al siguiente sin pausa. En cambio, el primer step
        # (tras el snapshot inicial) conserva el fallback completo, que separa
        # acciones completas.
        from droneplan_viz.commands import PickUp
        from droneplan_viz.domain import (
            Arm, AtLocation, Content, Drone, Location, Package, World,
        )
        from droneplan_viz.domain.drone_state import DroneState
        from droneplan_viz.render import Timeline
        from droneplan_viz.render.interpolation import (
            classify_transition, TransitionPackageMove, TransitionStatic,
        )
        from droneplan_viz.render.painter import _is_static_only_entering_interacting
        from droneplan_viz.render.timeline import _CHAINED_DWELL_EPSILON
        from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

        th = Theme.default()
        med = Content(id="m")
        world = World(
            locations={"dep": Location(id="dep")},
            drones={"d1": Drone(id="d1", position="dep",
                                arms=(Arm(id="izq"), Arm(id="der")),
                                state=DroneState.IDLE)},
            transporters={},
            packages={
                "pa": Package(id="pa", contains=med, at=AtLocation(loc_id="dep")),
                "pb": Package(id="pb", contains=med, at=AtLocation(loc_id="dep")),
            },
            contents={"m": med}, costs={},
        )
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pa",
                               duration=2.0, command_id="c1"),
                start_time=0.0),
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="der", package_id="pb",
                               duration=2.0, command_id="c2"),
                start_time=2.0),
        ))
        runner = PlanRunner(world)
        runner.execute(plan)
        tl = Timeline(runner.history, theme=th)
        st = tl.snapshot_times
        hist = runner.history

        chained_found = False
        for i in range(1, len(hist)):
            sa, sb = hist.at(i - 1), hist.at(i)
            tr = classify_transition(sa, sb)
            if not (
                isinstance(tr, TransitionStatic)
                and _is_static_only_entering_interacting(sa, sb)
            ):
                continue
            dur = st[i] - st[i - 1]
            prev_pm = (
                i >= 2
                and isinstance(classify_transition(hist.at(i - 2), hist.at(i - 1)),
                               TransitionPackageMove)
            )
            next_pm = (
                i <= len(hist) - 2
                and isinstance(classify_transition(hist.at(i), hist.at(i + 1)),
                               TransitionPackageMove)
            )
            if prev_pm and next_pm:
                # Pausa entre steps encadenados → colapsada.
                assert dur == pytest.approx(_CHAINED_DWELL_EPSILON)
                chained_found = True
            else:
                # Primer step (no encadenado) → fallback completo.
                assert dur == pytest.approx(th.fallback_step_duration)
        assert chained_found, "no se halló una pausa entre steps encadenados"


class TestPausaEnterInteractingNoSaltaAlCentro:
    """Durante la ESPERA (Static enter-interacting) previa a una acción, si
    encadena con la acción anterior del mismo dron, el seguimiento/render
    deja al dron en su aproximación (donde lo dejó la coreografía previa), NO
    en el centro. Se prueba llamando directamente sobre los snapshots de la
    pausa (sin sampler), para no depender de su duración virtual.
    """
    SIZE = (900, 600)

    def test_dron_no_salta_al_centro_en_la_pausa(self):
        import math

        from droneplan_viz.domain import Arm, AtLocation, Content, Drone, Location, World
        from droneplan_viz.domain.drone_state import DroneState
        from droneplan_viz.render import Theme, Timeline
        from droneplan_viz.render.interpolation import (
            classify_transition, TransitionPackageMove, TransitionStatic,
        )
        from droneplan_viz.render.layout import compute_layout
        from droneplan_viz.render.painter import (
            _is_static_only_entering_interacting, _scene_viewbox,
        )

        th = Theme.default()
        med = Content(id="m")
        world = World(
            locations={"dep": Location(id="dep")},
            drones={"d1": Drone(id="d1", position="dep",
                                arms=(Arm(id="izq"), Arm(id="der")),
                                state=DroneState.IDLE)},
            transporters={},
            packages={
                "pa": Package(id="pa", contains=med, at=AtLocation(loc_id="dep")),
                "pb": Package(id="pb", contains=med, at=AtLocation(loc_id="dep")),
            },
            contents={"m": med}, costs={},
        )
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="izq", package_id="pa",
                               duration=2.0, command_id="c1"), start_time=0.0),
            ScheduledCommand(
                command=PickUp(drone_id="d1", arm_id="der", package_id="pb",
                               duration=2.0, command_id="c2"), start_time=2.0),
        ))
        runner = PlanRunner(world)
        runner.execute(plan)
        hist = runner.history

        # Localizar la pausa encadenada (enter-interacting flanqueada por dos
        # PackageMove de d1) directamente en el historial.
        target = None
        for i in range(2, len(hist) - 1):
            a, b = hist.at(i - 1), hist.at(i)
            tr = classify_transition(a, b)
            if not (isinstance(tr, TransitionStatic)
                    and _is_static_only_entering_interacting(a, b)):
                continue
            tp = classify_transition(hist.at(i - 2), hist.at(i - 1))
            tn = classify_transition(hist.at(i), hist.at(i + 1))
            if isinstance(tp, TransitionPackageMove) and isinstance(tn, TransitionPackageMove):
                target = i
                break
        assert target is not None, "no se halló la pausa encadenada"

        a, b = hist.at(target - 1), hist.at(target)
        prev_action = (hist.at(target - 2), hist.at(target - 1))
        next_action = (hist.at(target), hist.at(target + 1))

        lay = compute_layout(a.world, _scene_viewbox(self.SIZE, th),
                             spread_factor=th.location_spread_factor)
        ctr = lay.drone_position("d1", th).as_int_tuple()

        con = locate_entity_interpolated(
            a, b, 0.5, "d1", self.SIZE, theme=th,
            prev_action=prev_action, next_action=next_action)
        sin = locate_entity_interpolated(a, b, 0.5, "d1", self.SIZE, theme=th)

        # Con contexto de vecinos: NO está en el centro (sigue en su aprox.).
        assert math.hypot(con[0] - ctr[0], con[1] - ctr[1]) > 5
        # Sin contexto (clásico): la pausa lo situaba en el centro.
        assert math.hypot(sin[0] - ctr[0], sin[1] - ctr[1]) < 5
