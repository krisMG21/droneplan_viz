"""Tests de World.

Cubren tres bloques principales:

1. Construcción y defaults (mundo vacío, mundo poblado).
2. Inmutabilidad estructural: ni se pueden reasignar los campos, ni se
   pueden mutar los dicts envueltos.
3. Helpers de consulta: packages_at, packages_in_transporter,
   package_held_by. Comportamiento correcto, incluyendo casos vacíos.

NO probamos reglas del dominio aquí: ni existencia de referencias, ni
cardinalidad, ni capacidad. Todo eso es del Validator.
"""

import pytest
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

from droneplan_viz.domain.content import Content
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import (
    AtLocation,
    HeldByArm,
    InTransporter,
    Package,
)
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.domain.world import World


# ---------------------------------------------------------------------------
# Fixtures: un mundo realista pequeño
# ---------------------------------------------------------------------------


@pytest.fixture
def medicina() -> Content:
    return Content("medicina")


@pytest.fixture
def comida() -> Content:
    return Content("comida")


@pytest.fixture
def mundo_pequeno(medicina, comida) -> World:
    """Mundo con dos localizaciones, un drone, un transporter, dos
    personas, y tres paquetes en distintos estados:

        caja1: libre en deposito
        caja2: sostenida por el brazo izq de dron1
        caja3: cargada en t1
    """
    locations = {
        "deposito": Location("deposito"),
        "casa1": Location("casa1"),
    }
    drones = {"dron1": Drone(id="dron1", position="deposito")}
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4)
    }
    persons = {
        "p1": Person(id="p1", position="casa1", needs=(medicina,)),
        "p2": Person(id="p2", position="casa1", needs=(comida,)),
    }
    contents = {"medicina": medicina, "comida": comida}
    packages = {
        "caja1": Package(id="caja1", contains=medicina, at=AtLocation("deposito")),
        "caja2": Package(
            id="caja2",
            contains=medicina,
            at=HeldByArm(drone_id="dron1", arm_id="izq"),
        ),
        "caja3": Package(
            id="caja3", contains=comida, at=InTransporter("t1")
        ),
    }
    costs = {
        ("deposito", "casa1"): 66.0,
        ("casa1", "deposito"): 66.0,
    }
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        packages=packages,
        persons=persons,
        contents=contents,
        costs=costs,
        metric="minimize total-time",
    )


# ---------------------------------------------------------------------------
# Construcción y defaults
# ---------------------------------------------------------------------------


class TestWorldConstruccion:
    def test_world_vacio(self):
        w = World()
        assert len(w.locations) == 0
        assert len(w.drones) == 0
        assert len(w.transporters) == 0
        assert len(w.packages) == 0
        assert len(w.persons) == 0
        assert len(w.contents) == 0
        assert len(w.costs) == 0
        assert w.metric is None

    def test_world_poblado(self, mundo_pequeno):
        w = mundo_pequeno
        assert len(w.locations) == 2
        assert len(w.drones) == 1
        assert len(w.transporters) == 1
        assert len(w.packages) == 3
        assert len(w.persons) == 2
        assert len(w.contents) == 2
        assert len(w.costs) == 2
        assert w.metric == "minimize total-time"

    def test_acceso_por_id(self, mundo_pequeno):
        assert mundo_pequeno.drones["dron1"].position == "deposito"
        assert mundo_pequeno.packages["caja1"].id == "caja1"
        assert mundo_pequeno.costs[("deposito", "casa1")] == 66.0

    def test_iteracion_sobre_valores(self, mundo_pequeno):
        ids = sorted(p.id for p in mundo_pequeno.packages.values())
        assert ids == ["caja1", "caja2", "caja3"]


# ---------------------------------------------------------------------------
# Inmutabilidad estructural (la decisión 2 hecha contrato)
# ---------------------------------------------------------------------------


class TestWorldInmutabilidad:
    def test_no_se_pueden_reasignar_campos(self, mundo_pequeno):
        with pytest.raises(FrozenInstanceError):
            mundo_pequeno.drones = {}  # type: ignore[misc]

    def test_no_se_pueden_mutar_dicts_internos(self, mundo_pequeno):
        # El núcleo de la decisión 2: ni siquiera los dicts se pueden
        # mutar desde fuera. MappingProxyType lanza TypeError ante
        # cualquier intento de asignación.
        with pytest.raises(TypeError):
            mundo_pequeno.drones["dron2"] = Drone(  # type: ignore[index]
                id="dron2", position="casa1"
            )

    def test_no_se_puede_eliminar_de_dicts_internos(self, mundo_pequeno):
        with pytest.raises(TypeError):
            del mundo_pequeno.packages["caja1"]  # type: ignore[misc]

    def test_no_se_puede_mutar_costs(self, mundo_pequeno):
        with pytest.raises(TypeError):
            mundo_pequeno.costs[("a", "b")] = 1.0  # type: ignore[index]

    def test_los_dicts_son_mappingproxy(self, mundo_pequeno):
        # Verificación de la implementación. Si en el futuro cambiamos
        # la estrategia (p. ej. a frozendict), este test es el primero
        # que se actualiza.
        assert isinstance(mundo_pequeno.drones, MappingProxyType)
        assert isinstance(mundo_pequeno.costs, MappingProxyType)


# ---------------------------------------------------------------------------
# replace: crear un World nuevo es el único modo legítimo de "cambiar"
# ---------------------------------------------------------------------------


class TestWorldReplace:
    def test_replace_construye_nuevo_world(self, mundo_pequeno):
        nuevo_drone = Drone(id="dron1", position="casa1")
        nuevos_drones = {**mundo_pequeno.drones, "dron1": nuevo_drone}
        w2 = replace(mundo_pequeno, drones=nuevos_drones)
        assert w2 is not mundo_pequeno
        assert w2.drones["dron1"].position == "casa1"

    def test_replace_no_muta_el_original(self, mundo_pequeno):
        # Garantía clave para el Memento: los snapshots anteriores
        # quedan intactos al avanzar la simulación.
        nuevos_drones = {
            **mundo_pequeno.drones,
            "dron1": Drone(id="dron1", position="casa1"),
        }
        _ = replace(mundo_pequeno, drones=nuevos_drones)
        assert mundo_pequeno.drones["dron1"].position == "deposito"

    def test_replace_envuelve_el_dict_nuevo(self, mundo_pequeno):
        # El World nuevo también debe envolver sus dicts; si alguien
        # mutara el dict pasado a replace después, el W2 no debería
        # verse afectado.
        d = {**mundo_pequeno.drones, "dron2": Drone(id="dron2", position="casa1")}
        w2 = replace(mundo_pequeno, drones=d)
        # Mutamos el dict de origen.
        d["dron3"] = Drone(id="dron3", position="deposito")
        # El World no se ve afectado.
        assert "dron3" not in w2.drones


# ---------------------------------------------------------------------------
# Igualdad por valor
# ---------------------------------------------------------------------------


class TestWorldIgualdad:
    def test_dos_worlds_vacios_iguales(self):
        assert World() == World()

    def test_mismos_datos_iguales(self):
        loc = Location("deposito")
        a = World(locations={"deposito": loc})
        b = World(locations={"deposito": loc})
        assert a == b

    def test_datos_distintos_no_iguales(self):
        a = World(locations={"deposito": Location("deposito")})
        b = World(locations={"deposito": Location("deposito"),
                             "casa1": Location("casa1")})
        assert a != b


# ---------------------------------------------------------------------------
# Helpers de consulta: packages_at
# ---------------------------------------------------------------------------


class TestPackagesAt:
    def test_devuelve_solo_los_libres_en_esa_localizacion(self, mundo_pequeno):
        en_deposito = mundo_pequeno.packages_at("deposito")
        ids = sorted(p.id for p in en_deposito)
        # caja1 está libre en deposito. caja2 está en brazo. caja3 en t1.
        # Solo caja1 cumple.
        assert ids == ["caja1"]

    def test_localizacion_sin_paquetes_libres(self, mundo_pequeno):
        assert mundo_pequeno.packages_at("casa1") == []

    def test_localizacion_inexistente_devuelve_vacia(self, mundo_pequeno):
        # No es responsabilidad del helper validar la existencia de la
        # localización. Devuelve lista vacía, el Validator se ocupa de
        # rechazar referencias a localizaciones inexistentes.
        assert mundo_pequeno.packages_at("inexistente") == []

    def test_world_vacio_lista_vacia(self):
        assert World().packages_at("loc") == []


# ---------------------------------------------------------------------------
# Helpers de consulta: packages_in_transporter
# ---------------------------------------------------------------------------


class TestPackagesInTransporter:
    def test_contenido_del_transporter(self, mundo_pequeno):
        contenido = mundo_pequeno.packages_in_transporter("t1")
        ids = [p.id for p in contenido]
        assert ids == ["caja3"]

    def test_transporter_vacio(self, mundo_pequeno):
        # No hay t2 en el mundo, pero el helper no valida existencia.
        assert mundo_pequeno.packages_in_transporter("t2") == []

    def test_world_vacio_lista_vacia(self):
        assert World().packages_in_transporter("t1") == []


# ---------------------------------------------------------------------------
# Helpers de consulta: package_held_by
# ---------------------------------------------------------------------------


class TestPackageHeldBy:
    def test_brazo_con_paquete(self, mundo_pequeno):
        p = mundo_pequeno.package_held_by("dron1", "izq")
        assert p is not None
        assert p.id == "caja2"

    def test_brazo_libre(self, mundo_pequeno):
        # El brazo "der" de dron1 no aparece en ningún Package.at.
        assert mundo_pequeno.package_held_by("dron1", "der") is None

    def test_drone_inexistente(self, mundo_pequeno):
        # Helper no valida: devuelve None porque no encuentra match.
        assert mundo_pequeno.package_held_by("inexistente", "izq") is None

    def test_world_vacio(self):
        assert World().package_held_by("dron1", "izq") is None


# ---------------------------------------------------------------------------
# El World no valida coherencia referencial: documentado por test
# ---------------------------------------------------------------------------


class TestWorldNoValidaCoherencia:
    """El World admite estados estructuralmente coherentes pero
    semánticamente inválidos. La validación es responsabilidad del
    Validator y de los builders. Documentamos por test los casos
    relevantes para que la decisión sea explícita."""

    def test_drone_con_position_a_loc_que_no_existe(self):
        # El World se construye sin protestar. El Validator detectará
        # que dron1 está en una localización que no existe.
        w = World(
            locations={"deposito": Location("deposito")},
            drones={"d1": Drone(id="d1", position="loc_fantasma")},
        )
        assert w.drones["d1"].position == "loc_fantasma"

    def test_package_referenciando_drone_inexistente(self, medicina):
        # caja referenciada a un brazo de un drone que no existe.
        w = World(
            packages={
                "caja1": Package(
                    id="caja1",
                    contains=medicina,
                    at=HeldByArm("dron_fantasma", "izq"),
                )
            }
        )
        # El World no se queja. El Validator sí.
        assert "caja1" in w.packages
