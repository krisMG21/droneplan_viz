"""Tests para droneplan_viz.render.layout.

Todos sin pygame. Verifican:
- compute_layout en los tres caminos: todas con position_screen,
  ninguna/algunas sin, world vacío.
- Determinismo del layout circular (orden alfabético, no de inserción).
- WorldLayout.location_position / drone_position / transporter_position /
  person_position / package_position.
- Co-localización: drones en anillo determinista.
- _ring_offset: caso 1 entidad al centro, caso N en anillo equiespaciado.
- Errores documentados: package_position lanza ValueError si no AtLocation.
"""
import math
from types import MappingProxyType

import pytest

from droneplan_viz.domain import Content, Location, World
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.package import AtLocation, Package
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.render.geometry import Point, ViewBox
from droneplan_viz.render.layout import (
    WorldLayout,
    _BOX_FRONT,
    _BOX_GAP,
    _BOX_LAYERS,
    _CARRIER_MAX,
    _CARRIER_PER_LAYER,
    _LOCATION_FOOTPRINT,
    _box_capacity,
    _box_depth_key,
    _box_slot,
    _carrier_depth_key,
    _carrier_slot,
    _ellipse_would_crowd,
    _has_graph_edges,
    _layout_force_directed,
    _layout_sunflower,
    _min_pairwise_distance,
    _ring_offset,
    compute_layout,
)
from droneplan_viz.render.theme import Theme


# ---------------------------------------------------------------------------
# compute_layout: caso World vacío
# ---------------------------------------------------------------------------


class TestComputeLayoutVacio:
    """World sin locations: WorldLayout válido con mapping vacío."""

    def test_world_vacio_produce_layout_sin_locations(self, empty_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(empty_world, viewbox)
        assert dict(layout.location_positions) == {}

    def test_world_vacio_mantiene_viewbox(self, empty_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(empty_world, viewbox)
        assert layout.viewbox == viewbox

    def test_world_vacio_consultar_loc_lanza_keyerror(self, empty_world):
        layout = compute_layout(empty_world, ViewBox.fit_into((800, 600)))
        with pytest.raises(KeyError):
            layout.location_position("inexistente")


# ---------------------------------------------------------------------------
# compute_layout: caso "todas con position_screen"
# ---------------------------------------------------------------------------


class TestComputeLayoutConPositionScreen:
    """Si TODAS las locs tienen position_screen, se respeta con mapeo afín."""

    def test_tres_locs_con_screen_caben_en_viewbox(
        self, three_locations_with_screen
    ):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_with_screen, viewbox)
        for loc_id in ["a", "b", "c"]:
            p = layout.location_position(loc_id)
            assert viewbox.contains(p), f"{loc_id} en {p} fuera de {viewbox}"

    def test_preserva_aspect_ratio(self, three_locations_with_screen):
        # bbox de origen: 100x100, ratio 1:1.
        # Si lo metemos en un viewbox 800x600, el ratio del proyecto
        # debe seguir siendo 1:1 (escala uniforme).
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_with_screen, viewbox)
        a = layout.location_position("a")
        b = layout.location_position("b")
        c = layout.location_position("c")
        # 'a' y 'b' están alineadas horizontalmente en el origen (y=0)
        # y separadas 100 unidades en x. 'c' está a (50, 100): debajo
        # del punto medio de a-b.
        # Verificamos que la distancia horizontal a-b sea 100*scale, y
        # la vertical de c respecto a a-b sea 100*scale (mismo scale).
        ab_horiz = b.x - a.x
        c_vert = c.y - a.y
        assert ab_horiz > 0
        assert c_vert > 0
        # Mismo factor de escala (1:1 preservado):
        assert ab_horiz == pytest.approx(c_vert, rel=1e-6)

    def test_centrado_en_viewbox(self, three_locations_with_screen):
        # El bbox proyectado debe quedar centrado dentro del viewbox.
        # Cogemos el centroide de las tres locs proyectadas y debe estar
        # cerca del centro del viewbox.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_with_screen, viewbox)
        a = layout.location_position("a")
        b = layout.location_position("b")
        c = layout.location_position("c")
        # bbox proyectado de origen es {a, b, c} con bbox real:
        proj_min_x = min(a.x, b.x, c.x)
        proj_max_x = max(a.x, b.x, c.x)
        proj_min_y = min(a.y, b.y, c.y)
        proj_max_y = max(a.y, b.y, c.y)
        proj_center_x = (proj_min_x + proj_max_x) / 2.0
        proj_center_y = (proj_min_y + proj_max_y) / 2.0
        assert proj_center_x == pytest.approx(viewbox.center.x, abs=2.0)
        assert proj_center_y == pytest.approx(viewbox.center.y, abs=2.0)

    def test_orden_relativo_se_preserva(self, three_locations_with_screen):
        # En el origen, 'a' está a la izquierda de 'b', y 'c' está abajo.
        # Tras la proyección, el orden relativo se conserva.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_with_screen, viewbox)
        a = layout.location_position("a")
        b = layout.location_position("b")
        c = layout.location_position("c")
        assert a.x < b.x          # a a la izquierda de b
        assert c.y > max(a.y, b.y)  # c por debajo de a y b


class TestComputeLayoutScreenCasosLimite:
    """Casos límite del mapeo afín."""

    def test_una_sola_location_con_screen_al_centro(self):
        world = World(locations={"unica": Location(id="unica", position_screen=(50, 50))})
        viewbox = ViewBox.fit_into((400, 300))
        layout = compute_layout(world, viewbox)
        assert layout.location_position("unica") == viewbox.center

    def test_todas_misma_position_screen_se_apilan_al_centro(self):
        # Caso degenerado: bbox de origen es un punto.
        world = World(
            locations={
                "a": Location(id="a", position_screen=(10, 10)),
                "b": Location(id="b", position_screen=(10, 10)),
                "c": Location(id="c", position_screen=(10, 10)),
            }
        )
        viewbox = ViewBox.fit_into((400, 300))
        layout = compute_layout(world, viewbox)
        center = viewbox.center
        assert layout.location_position("a") == center
        assert layout.location_position("b") == center
        assert layout.location_position("c") == center

    def test_locs_alineadas_horizontalmente(self):
        # bbox de altura 0: todas en una línea horizontal.
        world = World(
            locations={
                "a": Location(id="a", position_screen=(0, 50)),
                "b": Location(id="b", position_screen=(100, 50)),
            }
        )
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(world, viewbox)
        a = layout.location_position("a")
        b = layout.location_position("b")
        # Misma y, distinta x:
        assert a.y == pytest.approx(b.y)
        assert a.x < b.x

    def test_locs_alineadas_verticalmente(self):
        world = World(
            locations={
                "a": Location(id="a", position_screen=(50, 0)),
                "b": Location(id="b", position_screen=(50, 100)),
            }
        )
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(world, viewbox)
        a = layout.location_position("a")
        b = layout.location_position("b")
        assert a.x == pytest.approx(b.x)
        assert a.y < b.y


# ---------------------------------------------------------------------------
# compute_layout: caso "alguna None" → circular
# ---------------------------------------------------------------------------


class TestComputeLayoutCircular:
    """Si alguna position_screen es None, se ignoran todas y se va a circular."""

    def test_tres_locs_sin_screen_van_a_circular(
        self, three_locations_circular
    ):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        # Las tres posiciones existen:
        for loc_id in ["casa1", "manzana", "zorro"]:
            assert loc_id in layout.location_positions

    def test_circular_es_determinista_por_id_no_por_orden_insercion(
        self, three_locations_circular
    ):
        # El fixture inserta en orden zorro, casa1, manzana.
        # El layout debe ordenar alfabéticamente: casa1 (i=0, ángulo -90°),
        # manzana (i=1, ángulo +30°), zorro (i=2, ángulo +150°).
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)

        # casa1 es el primer alfabético: ángulo -π/2 → x=center.x, y < center.y
        casa1 = layout.location_position("casa1")
        center = viewbox.center
        assert casa1.x == pytest.approx(center.x, abs=1e-6)
        assert casa1.y < center.y  # arriba del centro

    def test_circular_locs_en_elipse_5_a_4(
        self, three_locations_circular
    ):
        # El layout es elíptico (proporción 5:4 igual que el lienzo del
        # sprite isométrico). Verificamos que (x/rx)^2 + (y/ry)^2 ≈ 1
        # para cada loc, donde rx:ry = 5:4.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        center = viewbox.center
        # Semiejes según el algoritmo: ry = min(0.4*h, 0.4*w/(5/4));
        # rx = ry * 5/4.
        ratio = 5.0 / 4.0
        half_w = viewbox.width * 0.40
        half_h = viewbox.height * 0.40
        radius_y = min(half_h, half_w / ratio)
        radius_x = radius_y * ratio
        for loc_id in ["casa1", "manzana", "zorro"]:
            p = layout.location_position(loc_id)
            dx = (p.x - center.x) / radius_x
            dy = (p.y - center.y) / radius_y
            # Cada punto debe estar sobre la elipse: dx² + dy² ≈ 1.
            assert (dx * dx + dy * dy) == pytest.approx(1.0, abs=1e-6)

    def test_circular_equiespaciado(self, three_locations_circular):
        # 3 locs → ángulos separados 120° entre sí. En el layout elíptico
        # los ÁNGULOS del parámetro siguen siendo equiespaciados (aunque
        # las distancias al centro varíen según la posición en la elipse).
        # Comprobamos el ángulo paramétrico, no atan2 directo.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        center = viewbox.center
        ratio = 5.0 / 4.0
        half_w = viewbox.width * 0.40
        half_h = viewbox.height * 0.40
        radius_y = min(half_h, half_w / ratio)
        radius_x = radius_y * ratio
        angles = []
        for loc_id in ["casa1", "manzana", "zorro"]:
            p = layout.location_position(loc_id)
            # Recuperamos el ángulo paramétrico: cos(t) = dx/rx, sin(t) = dy/ry
            cos_t = (p.x - center.x) / radius_x
            sin_t = (p.y - center.y) / radius_y
            angles.append(math.atan2(sin_t, cos_t))
        diff_01 = (angles[1] - angles[0]) % (2 * math.pi)
        diff_12 = (angles[2] - angles[1]) % (2 * math.pi)
        assert diff_01 == pytest.approx(2 * math.pi / 3, abs=1e-6)
        assert diff_12 == pytest.approx(2 * math.pi / 3, abs=1e-6)

    def test_mixed_screen_world_va_a_circular(self, mixed_screen_world):
        # Dos locs con screen, una sin: debe caer al layout elíptico y
        # descartar las screens existentes. Verifica que las tres están
        # sobre la elipse 5:4.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(mixed_screen_world, viewbox)
        center = viewbox.center
        ratio = 5.0 / 4.0
        half_w = viewbox.width * 0.40
        half_h = viewbox.height * 0.40
        radius_y = min(half_h, half_w / ratio)
        radius_x = radius_y * ratio
        for loc_id in ["con_coords_1", "con_coords_2", "sin_coords"]:
            p = layout.location_position(loc_id)
            dx = (p.x - center.x) / radius_x
            dy = (p.y - center.y) / radius_y
            assert (dx * dx + dy * dy) == pytest.approx(1.0, abs=1e-6)


class TestComputeLayoutCircularCasoUna:
    """Una sola Location sin position_screen va al centro."""

    def test_una_sola_loc_al_centro(self, single_location_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(single_location_world, viewbox)
        assert layout.location_position("solo_aqui") == viewbox.center


# ---------------------------------------------------------------------------
# WorldLayout: inmutabilidad
# ---------------------------------------------------------------------------


class TestWorldLayoutInmutabilidad:
    """El mapping location_positions debe ser de solo lectura."""

    def test_mapping_es_mappingproxy(self, three_locations_circular):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        assert isinstance(layout.location_positions, MappingProxyType)

    def test_no_se_puede_mutar(self, three_locations_circular):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        with pytest.raises(TypeError):
            layout.location_positions["nuevo"] = Point(0.0, 0.0)  # type: ignore

    def test_layout_frozen(self, three_locations_circular):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(three_locations_circular, viewbox)
        with pytest.raises((AttributeError, TypeError)):
            layout.viewbox = ViewBox(0, 0, 100, 100)  # type: ignore


# ---------------------------------------------------------------------------
# drone_position: lookup dinámico + co-localización
# ---------------------------------------------------------------------------


class TestDronePosition:
    """Posición del drone derivada de world.drones[id].position + co-loc."""

    def test_un_solo_drone_al_centro_de_su_loc(self, rich_world):
        # d2 está en casa1, solo.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        # El drone vuela estacionario por drone_hover_y_offset sobre el
        # centro LÓGICO (rombo). Para 1 solo drone, sin lift adicional.
        ground = layout._location_ground_center("casa1", theme)
        expected = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        d2_pos = layout.drone_position("d2", theme)
        assert d2_pos == expected

    def test_drone_inexistente_lanza_keyerror(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        with pytest.raises(KeyError):
            layout.drone_position("fantasma", Theme.default())

    def test_dos_drones_colocalizados_no_se_superponen(
        self, two_drones_same_location
    ):
        # Ambos en 'centro' → deben acabar en posiciones distintas.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default()
        a_pos = layout.drone_position("drone_a", theme)
        b_pos = layout.drone_position("drone_b", theme)
        assert a_pos != b_pos

    def test_co_localizados_equidistan_del_centro(self, two_drones_same_location):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default()
        # Centro LÓGICO + hover (donde flota el dron) es la referencia
        # para el anillo. Adicionalmente el anillo se eleva por
        # drone_colocation_y_lift cuando hay 2+ drones.
        ground = layout._location_ground_center("centro", theme)
        center = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        ring_center_y = center.y + theme.drone_colocation_y_lift
        a_pos = layout.drone_position("drone_a", theme)
        b_pos = layout.drone_position("drone_b", theme)
        dist_a = math.hypot(a_pos.x - center.x, a_pos.y - ring_center_y)
        dist_b = math.hypot(b_pos.x - center.x, b_pos.y - ring_center_y)
        assert dist_a == pytest.approx(theme.drone_colocation_offset, rel=1e-6)
        assert dist_b == pytest.approx(theme.drone_colocation_offset, rel=1e-6)

    def test_co_localizados_diametralmente_opuestos(self, two_drones_same_location):
        # Con 2 entidades en anillo, deben estar a 180° entre sí.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default()
        # Centro LÓGICO + hover: ver test anterior.
        ground = layout._location_ground_center("centro", theme)
        center = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        a = layout.drone_position("drone_a", theme)
        b = layout.drone_position("drone_b", theme)
        ring_center_y = center.y + theme.drone_colocation_y_lift
        sum_x = (a.x - center.x) + (b.x - center.x)
        sum_y = (a.y - ring_center_y) + (b.y - ring_center_y)
        assert sum_x == pytest.approx(0.0, abs=1e-6)
        assert sum_y == pytest.approx(0.0, abs=1e-6)


class TestDronePositionLift:
    """Aplicación del drone_colocation_y_lift cuando hay anillo."""

    def test_un_solo_drone_no_aplica_lift(self, rich_world):
        # d2 está solo en casa1: lift NO se aplica, va al centro de la loc.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        # Con 1 solo drone, su posición = centro lógico + hover (sin lift).
        ground = layout._location_ground_center("casa1", theme)
        expected = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        d2_pos = layout.drone_position("d2", theme)
        assert d2_pos == expected

    def test_dos_drones_colocalizados_aplica_lift(
        self, two_drones_same_location
    ):
        # drone_a y drone_b ambos en centro: anillo desplazado hacia arriba.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default()
        # Centro LÓGICO + hover: desde aquí se aplica el lift.
        ground = layout._location_ground_center("centro", theme)
        center = Point(ground.x, ground.y + theme.drone_hover_y_offset)

        a = layout.drone_position("drone_a", theme)
        b = layout.drone_position("drone_b", theme)
        # Centro del anillo está en (center.x, center.y + y_lift).
        # Los dos drones están a 180° respecto a ese centro_anillo,
        # así que su promedio en y es exactamente center.y + y_lift.
        avg_y = (a.y + b.y) / 2.0
        expected_ring_center_y = center.y + theme.drone_colocation_y_lift
        assert avg_y == pytest.approx(expected_ring_center_y, abs=1e-6)

    def test_lift_negativo_eleva_anillo_en_pantalla(
        self, two_drones_same_location
    ):
        # Con y_lift por defecto (-10), el anillo se eleva
        # (en coords pantalla, y decreciente = arriba).
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default()
        center = layout.location_position("centro")
        a = layout.drone_position("drone_a", theme)
        b = layout.drone_position("drone_b", theme)
        # El promedio en y debe ser MENOR que center.y (más arriba).
        assert (a.y + b.y) / 2.0 < center.y

    def test_lift_cero_recupera_comportamiento_anterior(
        self, two_drones_same_location
    ):
        # Si y_lift=0, el centro del anillo coincide con el centro LÓGICO
        # de la loc (rombo isométrico). Comportamiento previo al sprite.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme = Theme.default().with_overrides(drone_colocation_y_lift=0)
        # Con y_lift=0, el centro del anillo coincide con el hover_center.
        ground = layout._location_ground_center("centro", theme)
        center = Point(ground.x, ground.y + theme.drone_hover_y_offset)
        a = layout.drone_position("drone_a", theme)
        b = layout.drone_position("drone_b", theme)
        # Promedio coincide con center.y.
        assert (a.y + b.y) / 2.0 == pytest.approx(center.y, abs=1e-6)

    def test_lift_grande_separa_mas_drones_de_paquetes(
        self, two_drones_same_location
    ):
        # Smoke: cambiar el lift cambia las posiciones.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(two_drones_same_location, viewbox)
        theme_small = Theme.default().with_overrides(drone_colocation_y_lift=-5)
        theme_large = Theme.default().with_overrides(drone_colocation_y_lift=-30)
        a_small = layout.drone_position("drone_a", theme_small)
        a_large = layout.drone_position("drone_a", theme_large)
        # Mayor lift → drone más arriba en pantalla:
        assert a_large.y < a_small.y

    def test_lift_no_afecta_drones_solo_en_loc_incluso_si_loc_tiene_paquetes(
        self, rich_world
    ):
        # rich_world: d2 está solo en casa1, no se aplica lift.
        # Hay paquetes en otras locs pero no en casa1: caso anti-lift.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme_with_lift = Theme.default().with_overrides(drone_colocation_y_lift=-50)
        theme_without = Theme.default().with_overrides(drone_colocation_y_lift=0)
        # Como d2 está SOLO en casa1, ambos themes producen la misma pos.
        pos_with = layout.drone_position("d2", theme_with_lift)
        pos_without = layout.drone_position("d2", theme_without)
        assert pos_with == pos_without


class TestDronePositionRegresionPaqueteCubierto:
    """Regresión: drone co-localizado no debe tapar COMPLETAMENTE paquetes.

    Bug histórico descubierto en el demo de Sesión D (Paso 8): cuando
    d2 estaba IDLE en casa1 y d1 acababa de entregar pkg_med1 allí
    también, el círculo del drone tapaba completamente al cuadrado del
    paquete entregado (drones y cajas centrados en el mismo punto).

    Tras el rediseño con sprite isométrico de location, el usuario aceptó
    que los drones puedan salir del rombo hábil y solaparse PARCIALMENTE
    con las cajas (el rombo es prioritariamente para cajas/carriers). Lo
    que sigue siendo inaceptable es el solape TOTAL (caja invisible
    dentro del círculo del drone), que es el bug histórico. Verificamos
    que al menos un drone+caja están "claramente separados" (centro de
    la caja fuera del radio del drone).
    """

    def test_drones_no_tapan_completamente_paquetes_libres_en_misma_loc(self):
        # Construir un World con 2 drones en casa1 y 1 paquete libre
        # AtLocation casa1.
        contents = {"medicina": Content(id="medicina")}
        locs = {"casa1": Location(id="casa1"), "casa2": Location(id="casa2")}
        drones = {
            "a": Drone(id="a", position="casa1", arms=(Arm(id="izq"),),
                       state=DroneState.IDLE),
            "b": Drone(id="b", position="casa1", arms=(Arm(id="izq"),),
                       state=DroneState.IDLE),
        }
        packages = {
            "pkg1": Package(
                id="pkg1",
                contains=contents["medicina"],
                at=AtLocation(loc_id="casa1"),
            ),
        }
        world_co = World(
            locations=locs, drones=drones, packages=packages,
            contents=contents,
        )

        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(world_co, viewbox)
        theme = Theme.default()
        pkg_pos = layout.package_position("pkg1", theme)
        a_pos = layout.drone_position("a", theme)
        b_pos = layout.drone_position("b", theme)

        # Distancia mínima: el centro de la caja NO debe estar DENTRO del
        # círculo de ningún drone (drone_radius). Eso garantiza que se
        # ve al menos parte de la caja. Solape parcial está permitido
        # bajo el nuevo criterio (drones pueden salir del rombo).
        min_clearance = theme.drone_radius
        d_a = math.hypot(pkg_pos.x - a_pos.x, pkg_pos.y - a_pos.y)
        d_b = math.hypot(pkg_pos.x - b_pos.x, pkg_pos.y - b_pos.y)
        assert d_a > min_clearance, (
            f"Drone a ({a_pos}) cubre el centro del paquete ({pkg_pos}): "
            f"distancia {d_a:.1f}, mínimo {min_clearance:.1f}"
        )
        assert d_b > min_clearance, (
            f"Drone b ({b_pos}) cubre el centro del paquete ({pkg_pos}): "
            f"distancia {d_b:.1f}, mínimo {min_clearance:.1f}"
        )


# ---------------------------------------------------------------------------
# transporter_position y person_position
# ---------------------------------------------------------------------------


class TestTransporterPosition:
    def test_un_transp_a_la_izquierda_del_centro(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        deposito_center = layout._location_ground_center("deposito", theme)
        t1_pos = layout.transporter_position("t1", theme)
        assert t1_pos.x < deposito_center.x  # a la izquierda
        assert t1_pos.y == pytest.approx(deposito_center.y)  # alineado

    def test_inexistente_lanza_keyerror(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        with pytest.raises(KeyError):
            layout.transporter_position("inexistente", Theme.default())


class TestPersonPosition:
    def test_persona_a_la_derecha_del_centro(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        casa1_center = layout._location_ground_center("casa1", theme)
        p1_pos = layout.person_position("p1", theme)
        assert p1_pos.x > casa1_center.x  # a la derecha
        assert p1_pos.y == pytest.approx(casa1_center.y)  # alineado

    def test_inexistente_lanza_keyerror(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        with pytest.raises(KeyError):
            layout.person_position("alguien", Theme.default())


# ---------------------------------------------------------------------------
# package_position: solo AtLocation
# ---------------------------------------------------------------------------


class TestPackagePosition:
    def test_paquete_atlocation_se_dibuja_cerca_de_su_loc(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        deposito_center = layout.location_position("deposito")
        pkg_pos = layout.package_position("pkg_med1", theme)
        # No exigimos x exacto (puede haber offset por co-localización),
        # pero debe estar dentro de un radio razonable:
        dist = math.hypot(pkg_pos.x - deposito_center.x, pkg_pos.y - deposito_center.y)
        assert dist < theme.location_radius + theme.package_size

    def test_paquete_heldbyarm_lanza_valueerror(self, package_held_by_arm_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(package_held_by_arm_world, viewbox)
        with pytest.raises(ValueError, match="HeldByArm"):
            layout.package_position("pkg1", Theme.default())

    def test_paquete_intransporter_lanza_valueerror(self, package_in_transporter_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(package_in_transporter_world, viewbox)
        with pytest.raises(ValueError, match="InTransporter"):
            layout.package_position("pkg1", Theme.default())

    def test_paquete_inexistente_lanza_keyerror(self, rich_world):
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        with pytest.raises(KeyError):
            layout.package_position("nope", Theme.default())

    def test_varios_paquetes_libres_en_misma_loc_no_se_superponen(self, rich_world):
        # En el rich_world, pkg_med1 y pkg_water1 están AtLocation deposito.
        viewbox = ViewBox.fit_into((800, 600))
        layout = compute_layout(rich_world, viewbox)
        theme = Theme.default()
        pos_med = layout.package_position("pkg_med1", theme)
        pos_water = layout.package_position("pkg_water1", theme)
        assert pos_med != pos_water


class TestApiladoCompacto:
    """Apilado compacto en la zona hábil: cajas en pirámide truncada (3 abajo +
    2 arriba) en la mitad derecha, con capas detrás y sistema tipo lista; los
    transportadores en diagonal arriba-izquierda, como mucho 3 visibles."""

    VIEWBOX = ViewBox.fit_into((800, 600))

    def _world_cajas(self, ids, loc="base"):
        locs = {loc: Location(id=loc, position_screen=(400, 300))}
        pkgs = {
            pid: Package(id=pid, contains=Content(id="c"), at=AtLocation(loc_id=loc))
            for pid in ids
        }
        return World(locations=locs, packages=pkgs)

    def _pos(self, ids, pid, loc="base"):
        lay = compute_layout(self._world_cajas(ids, loc), self.VIEWBOX)
        return lay.package_position(pid, Theme.default())

    def test_front_es_3_abajo_y_2_arriba(self):
        theme = Theme.default()
        step = theme.package_size + _BOX_GAP
        s = [_box_slot(i, theme) for i in range(5)]
        # fila de 3 abajo (misma y), separadas un paso
        assert s[0][1] == s[1][1] == s[2][1]
        assert s[1][0] - s[0][0] == pytest.approx(step)
        assert s[2][0] - s[1][0] == pytest.approx(step)
        # fila de 2 encima (y menor = arriba), en los valles (±step/2)
        assert s[3][1] < s[0][1] and s[4][1] < s[0][1]
        assert s[3][0] == pytest.approx(-step / 2)
        assert s[4][0] == pytest.approx(step / 2)

    def test_separacion_entre_cajas(self):
        theme = Theme.default()
        s = [_box_slot(i, theme) for i in range(2)]
        gap = (s[1][0] - s[0][0]) - theme.package_size
        assert gap == pytest.approx(_BOX_GAP)
        assert _BOX_GAP == pytest.approx(0.0)  # cajas pegadas (0px de separación)

    def test_capas_detras_arriba_derecha(self):
        theme = Theme.default()
        front0 = _box_slot(0, theme)             # capa 0, slot 0
        back0 = _box_slot(_BOX_FRONT, theme)     # capa 1, mismo slot
        assert back0[0] > front0[0]  # más a la DERECHA (espejo del carrier)
        assert back0[1] < front0[1]  # más arriba (al fondo)

    def test_rebose_en_caja_central_ultima_capa(self):
        theme = Theme.default()
        cap = _box_capacity()
        # caja central (fila inferior, centro) de la última capa
        central = _box_slot((_BOX_LAYERS - 1) * _BOX_FRONT + 1, theme)
        # todas las cajas que rebasan comparten esa posición
        assert _box_slot(cap, theme) == central
        assert _box_slot(cap + 7, theme) == central

    def test_orden_pintado_por_capas(self):
        # El orden de dibujo va atrás→adelante POR CAPAS (igual que el carrier):
        # la última caja (mayor clave) está en la capa frontal (capa 0) y en la
        # fila base; las capas traseras y la fila de encima se pintan antes.
        cap = _box_capacity()
        last = max(range(cap), key=_box_depth_key)
        assert last // _BOX_FRONT == 0          # capa frontal (la 0)
        assert last % _BOX_FRONT < 3            # fila base (delante), no la de encima
        # una capa más al fondo se dibuja antes (clave menor)
        assert _box_depth_key(_BOX_FRONT) < _box_depth_key(0)
        # dentro de una capa, la fila de encima va antes que la base
        assert _box_depth_key(3) < _box_depth_key(0)

    def test_pila_en_mitad_derecha(self):
        lay = compute_layout(self._world_cajas(["a", "b", "c"]), self.VIEWBOX)
        theme = Theme.default()
        gc = lay._location_ground_center("base", theme)
        assert lay.package_position("b", theme).x > gc.x

    def test_lista_compacta_al_extraer(self):
        # En [a,b,c], b es idx1; al extraer 'a', en [b,c] b pasa a idx0 y ocupa
        # el slot que tenía 'a' (sustitución tipo lista).
        pa_full = self._pos(["a", "b", "c"], "a")
        pb_full = self._pos(["a", "b", "c"], "b")
        pb_after = self._pos(["b", "c"], "b")
        assert pb_after != pb_full   # b se movió
        assert pb_after == pa_full   # b ocupa el slot 0 (el de 'a')

    def test_dos_cajas_no_se_superponen(self):
        assert self._pos(["a", "b"], "a") != self._pos(["a", "b"], "b")

    def _world_carriers(self, n, loc="base"):
        locs = {loc: Location(id=loc, position_screen=(400, 300))}
        trs = {
            f"t{i}": Transporter(id=f"t{i}", position=loc, capacity=4)
            for i in range(n)
        }
        return World(locations=locs, transporters=trs)

    def test_carriers_primera_capa_diagonal_arriba_izquierda(self):
        lay = compute_layout(self._world_carriers(3), self.VIEWBOX)
        theme = Theme.default()
        gc = lay._location_ground_center("base", theme)
        p0 = lay.transporter_position("t0", theme)
        p1 = lay.transporter_position("t1", theme)
        p2 = lay.transporter_position("t2", theme)
        assert p0.x < gc.x                       # el 1º a la izquierda
        assert p1.x < p0.x and p1.y < p0.y       # 2º arriba-izquierda
        assert p2.x < p1.x and p2.y < p1.y       # 3º más arriba-izquierda

    def test_carriers_segunda_capa_arriba_y_derecha(self):
        lay = compute_layout(self._world_carriers(6), self.VIEWBOX)
        theme = Theme.default()
        p0 = lay.transporter_position("t0", theme)  # 1ª capa, 1º
        p3 = lay.transporter_position("t3", theme)  # 2ª capa, 1º
        # la 2ª capa se desplaza ARRIBA y a la DERECHA respecto a la 1ª
        assert p3.x > p0.x   # a la derecha
        assert p3.y < p0.y   # arriba

    def test_como_mucho_6_carriers(self):
        lay = compute_layout(self._world_carriers(8), self.VIEWBOX)
        theme = Theme.default()
        p5 = lay.transporter_position("t5", theme)
        assert lay.transporter_position("t6", theme) == p5  # 7º oculto tras el 6º
        assert lay.transporter_position("t7", theme) == p5  # 8º también
        assert _CARRIER_MAX == 6 == _CARRIER_PER_LAYER * 2

    def test_carrier_depth_key_atras_adelante(self):
        # 2ª capa (detrás) antes que la 1ª; dentro de capa, el más alejado antes
        assert _carrier_depth_key(_CARRIER_PER_LAYER) < _carrier_depth_key(0)
        assert _carrier_depth_key(2) < _carrier_depth_key(0)


# ---------------------------------------------------------------------------
# _ring_offset: utilidad interna
# ---------------------------------------------------------------------------


class TestRingOffset:
    """Distribución en anillo determinista."""

    def test_un_solo_id_va_al_centro(self):
        center = Point(100.0, 200.0)
        result = _ring_offset(
            center=center,
            entity_id="solo",
            colocated_ids=["solo"],
            radius=10.0,
        )
        assert result == center

    def test_dos_ids_opuestos(self):
        center = Point(0.0, 0.0)
        a = _ring_offset(center, "a", ["a", "b"], radius=10.0)
        b = _ring_offset(center, "b", ["a", "b"], radius=10.0)
        assert a.x + b.x == pytest.approx(0.0)
        assert a.y + b.y == pytest.approx(0.0)

    def test_tres_ids_equiespaciados(self):
        center = Point(0.0, 0.0)
        positions = [
            _ring_offset(center, eid, ["a", "b", "c"], radius=10.0)
            for eid in ["a", "b", "c"]
        ]
        # Distancia entre cualquier par debe ser la misma (lado del
        # triángulo equilátero inscrito = r * √3).
        d_ab = math.hypot(positions[0].x - positions[1].x, positions[0].y - positions[1].y)
        d_bc = math.hypot(positions[1].x - positions[2].x, positions[1].y - positions[2].y)
        d_ac = math.hypot(positions[0].x - positions[2].x, positions[0].y - positions[2].y)
        assert d_ab == pytest.approx(d_bc, rel=1e-6)
        assert d_bc == pytest.approx(d_ac, rel=1e-6)

    def test_radio_se_respeta(self):
        center = Point(0.0, 0.0)
        ids = ["a", "b", "c", "d"]
        for eid in ids:
            p = _ring_offset(center, eid, ids, radius=15.0)
            dist = math.hypot(p.x, p.y)
            assert dist == pytest.approx(15.0, rel=1e-6)

    def test_id_no_presente_lanza(self):
        with pytest.raises(ValueError, match="no aparece"):
            _ring_offset(
                center=Point(0.0, 0.0),
                entity_id="fantasma",
                colocated_ids=["a", "b"],
                radius=10.0,
            )

    def test_orden_determinista(self):
        # Mismas entradas → mismas salidas, siempre.
        center = Point(100.0, 100.0)
        ids = ["alpha", "beta", "gamma"]
        a1 = _ring_offset(center, "beta", ids, radius=20.0)
        a2 = _ring_offset(center, "beta", ids, radius=20.0)
        assert a1 == a2

    def test_primera_entidad_arriba(self):
        # La primera entidad del orden debe quedar arriba (angle = -π/2).
        center = Point(0.0, 0.0)
        first = _ring_offset(center, "alpha", ["alpha", "beta"], radius=10.0)
        # angle = -π/2 → x ~ 0, y ~ -radius (negativo en coords pantalla = arriba).
        assert first.x == pytest.approx(0.0, abs=1e-6)
        assert first.y == pytest.approx(-10.0, rel=1e-6)


def _locs_sin_screen(n: int) -> list:
    """n Locations sin position_screen (fuerzan el layout automático)."""
    return [Location(id=f"loc{i:03d}") for i in range(n)]


class TestLayoutMuchasLocations:
    """Reparto girasol (phyllotaxis) y reducción de tamaño cuando hay muchas
    locations, de modo que se repartan en 2D y no se solapen."""

    VIEWBOX = ViewBox(40.0, 60.0, 1020.0, 600.0)
    SPREAD = 0.50

    def _world(self, n: int) -> World:
        return World(locations={loc.id: loc for loc in _locs_sin_screen(n)})

    def test_pocas_locations_conservan_elipse_y_escala_1(self):
        # Con pocas locations no apelotona: layout circular y scale=1.0
        # (salida nativa, byte-idéntica al comportamiento histórico).
        assert not _ellipse_would_crowd(8, self.VIEWBOX, self.SPREAD)
        lay = compute_layout(self._world(8), self.VIEWBOX, spread_factor=self.SPREAD)
        assert lay.scale == 1.0

    def test_muchas_locations_activan_girasol_y_reducen_escala(self):
        # Con muchas locations: la elipse apelotonaría, así que se usa girasol
        # y se reduce el tamaño de dibujo (scale < 1).
        assert _ellipse_would_crowd(40, self.VIEWBOX, self.SPREAD)
        lay = compute_layout(self._world(40), self.VIEWBOX, spread_factor=self.SPREAD)
        assert lay.scale < 1.0

    def test_invariante_sin_solape_en_un_rango_de_n(self):
        # Para todo N, la huella REDUCIDA debe caber en la separación mínima
        # real entre locations: huella·scale <= min-dist. Es la garantía de
        # "no se solapan".
        for n in (16, 25, 40, 61, 81, 101):
            lay = compute_layout(
                self._world(n), self.VIEWBOX, spread_factor=self.SPREAD
            )
            min_dist = _min_pairwise_distance(list(lay.location_positions.values()))
            assert _LOCATION_FOOTPRINT * lay.scale <= min_dist + 0.5, (
                f"n={n}: huella·scale={_LOCATION_FOOTPRINT*lay.scale:.1f} "
                f"> min-dist={min_dist:.1f} (solape)"
            )

    def test_escala_degrada_monotona_con_n(self):
        # A más locations, menor tamaño de dibujo (o igual, nunca mayor).
        scales = []
        for n in (25, 40, 61, 81, 101):
            lay = compute_layout(
                self._world(n), self.VIEWBOX, spread_factor=self.SPREAD
            )
            scales.append(lay.scale)
        assert all(scales[i] >= scales[i + 1] for i in range(len(scales) - 1))

    def test_todas_las_locations_dentro_del_viewbox(self):
        # El girasol reparte DENTRO del viewbox: nada se sale del frame (el
        # "zoom out" es de tamaños, no de posiciones).
        lay = compute_layout(self._world(61), self.VIEWBOX, spread_factor=self.SPREAD)
        for pos in lay.location_positions.values():
            assert self.VIEWBOX.left <= pos.x <= self.VIEWBOX.right
            assert self.VIEWBOX.top <= pos.y <= self.VIEWBOX.bottom

    def test_girasol_determinista(self):
        # Mismas entradas → mismas posiciones y misma escala.
        a = compute_layout(self._world(50), self.VIEWBOX, spread_factor=self.SPREAD)
        b = compute_layout(self._world(50), self.VIEWBOX, spread_factor=self.SPREAD)
        assert a.location_positions == b.location_positions
        assert a.scale == b.scale


class TestLayoutPorGrafo:
    """Layout por grafo (híbrido radial + fuerzas) cuando hay World.costs:
    distancias ∝ coste y hubs al centro. Se activa con N>=13 y aristas."""

    VIEWBOX = ViewBox(40.0, 60.0, 1020.0, 600.0)
    SPREAD = 0.50

    def _star(self, costs: list[float]) -> World:
        """Estrella: 'hub' conectado a cada hoja con el coste dado (simétrico)."""
        locs = {"hub": Location(id="hub")}
        cost_map: dict[tuple[str, str], float] = {}
        for i, c in enumerate(costs):
            lid = f"l{i:03d}"
            locs[lid] = Location(id=lid)
            cost_map[("hub", lid)] = float(c)
            cost_map[(lid, "hub")] = float(c)
        return World(locations=locs, costs=cost_map)

    def _dist(self, lay, a, b) -> float:
        pa, pb = lay.location_position(a), lay.location_position(b)
        return math.hypot(pa.x - pb.x, pa.y - pb.y)

    def test_dispatcher_usa_grafo_cuando_hay_aristas(self):
        # Con costes y N>=13, el dispatcher NO debe usar el girasol.
        world = self._star([8.0] * 16)
        assert _has_graph_edges(list(world.locations.values()), world)
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        sun = _layout_sunflower(
            list(world.locations.values()), self.VIEWBOX, self.SPREAD
        )
        assert lay.location_positions != sun  # camino distinto al girasol

    def test_sin_aristas_no_usa_grafo(self):
        # N>=13 pero sin costes → girasol (no grafo).
        world = World(locations={loc.id: loc for loc in _locs_sin_screen(16)})
        assert not _has_graph_edges(list(world.locations.values()), world)
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        sun = _layout_sunflower(
            list(world.locations.values()), self.VIEWBOX, self.SPREAD
        )
        assert lay.location_positions == sun

    def test_hub_al_centro(self):
        # El nodo muy conectado (hub) queda cerca del centro del viewbox.
        world = self._star([6.0, 7.0, 8.0, 9.0, 10.0] * 4)  # 20 hojas
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        hub = lay.location_position("hub")
        c = self.VIEWBOX.center
        dev = math.hypot(hub.x - c.x, hub.y - c.y)
        assert dev < 0.15 * min(self.VIEWBOX.width, self.VIEWBOX.height)

    def test_distancia_proporcional_al_coste(self):
        # Hojas de coste alto deben quedar claramente más lejos del hub que
        # las de coste bajo (distancia ∝ coste).
        costs = [5.0] * 8 + [15.0] * 8
        world = self._star(costs)
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        cerca = [self._dist(lay, "hub", f"l{i:03d}") for i in range(8)]
        lejos = [self._dist(lay, "hub", f"l{i:03d}") for i in range(8, 16)]
        assert max(cerca) < min(lejos)  # separación nítida entre grupos
        # y la razón media de distancias sigue a la razón de costes (3×), holgado
        assert (sum(lejos) / 8) > 2.0 * (sum(cerca) / 8)

    def test_determinista(self):
        world = self._star([6.0, 9.0, 12.0] * 5)  # 15 hojas
        a = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        b = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        assert a.location_positions == b.location_positions
        assert a.scale == b.scale

    def test_dentro_del_viewbox(self):
        world = self._star([6.0, 8.0, 10.0, 12.0] * 5)  # 20 hojas
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        for pos in lay.location_positions.values():
            assert self.VIEWBOX.left <= pos.x <= self.VIEWBOX.right
            assert self.VIEWBOX.top <= pos.y <= self.VIEWBOX.bottom

    def test_scale_no_solape(self):
        # El factor de encaje garantiza huella·scale <= separación mínima real.
        world = self._star([6.0, 8.0, 10.0, 12.0] * 6)  # 24 hojas
        lay = compute_layout(world, self.VIEWBOX, spread_factor=self.SPREAD)
        min_dist = _min_pairwise_distance(list(lay.location_positions.values()))
        assert _LOCATION_FOOTPRINT * lay.scale <= min_dist + 1e-6


class TestThemeScaled:
    """Theme.scaled: reduce SOLO geometría, preserva el resto, y es idéntico
    en factor>=1 (camino por defecto byte-idéntico)."""

    def test_factor_1_devuelve_self(self):
        t = Theme.default()
        assert t.scaled(1.0) is t
        assert t.scaled(2.0) is t

    def test_reduce_geometria(self):
        t = Theme.default()
        s = t.scaled(0.5)
        assert s.drone_radius < t.drone_radius
        assert s.location_sprite_native_size[0] < t.location_sprite_native_size[0]
        assert abs(s.drone_hover_y_offset) < abs(t.drone_hover_y_offset)
        assert s.sprite_anchors.body_size[0] < t.sprite_anchors.body_size[0]

    def test_preserva_no_geometria(self):
        t = Theme.default()
        s = t.scaled(0.4)
        assert s.background == t.background
        assert s.location_spread_factor == t.location_spread_factor
        assert s.drone_move_max_tilt_deg == t.drone_move_max_tilt_deg
        assert s.location_sprite_scale == t.location_sprite_scale

    def test_fuentes_no_bajan_de_legible(self):
        # Las fuentes no se reducen por debajo de un mínimo legible.
        s = Theme.default().scaled(0.1)
        assert s.font_size_label >= 8
        assert s.font_size_id >= 8
