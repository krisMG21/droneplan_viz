"""Tests de Package y de los subtipos de PackageLocation.

La pieza fundamental a verificar es que el tipo suma funciona como se
espera: cada subtipo se construye con sus campos, son distinguibles entre
sí (isinstance, match), y el campo `at` de Package acepta cualquiera de
los tres. También cubrimos inmutabilidad y la imposibilidad estructural
de representar estados ilegales (no se puede construir un Package "en dos
sitios a la vez").
"""

import pytest
from dataclasses import FrozenInstanceError, replace

from droneplan_viz.domain.content import Content
from droneplan_viz.domain.package import (
    AtLocation,
    HeldByArm,
    InTransporter,
    Package,
    PackageLocation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def medicina() -> Content:
    return Content("medicina")


@pytest.fixture
def caja_en_deposito(medicina) -> Package:
    return Package(id="caja1", contains=medicina, at=AtLocation("deposito"))


# ---------------------------------------------------------------------------
# Subtipos de PackageLocation
# ---------------------------------------------------------------------------


class TestAtLocation:
    def test_construccion(self):
        a = AtLocation(loc_id="deposito")
        assert a.loc_id == "deposito"

    def test_inmutable(self):
        a = AtLocation(loc_id="deposito")
        with pytest.raises(FrozenInstanceError):
            a.loc_id = "casa1"  # type: ignore[misc]

    def test_igualdad_por_valor(self):
        assert AtLocation("deposito") == AtLocation("deposito")
        assert AtLocation("deposito") != AtLocation("casa1")


class TestHeldByArm:
    def test_construccion(self):
        h = HeldByArm(drone_id="dron1", arm_id="izq")
        assert h.drone_id == "dron1"
        assert h.arm_id == "izq"

    def test_inmutable(self):
        h = HeldByArm(drone_id="dron1", arm_id="izq")
        with pytest.raises(FrozenInstanceError):
            h.arm_id = "der"  # type: ignore[misc]

    def test_igualdad_por_valor(self):
        # Hace falta coincidir en ambos campos: mismo drone Y mismo brazo.
        # El brazo "izq" de dron1 y el brazo "izq" de dron2 son brazos
        # físicamente distintos.
        assert HeldByArm("dron1", "izq") == HeldByArm("dron1", "izq")
        assert HeldByArm("dron1", "izq") != HeldByArm("dron2", "izq")
        assert HeldByArm("dron1", "izq") != HeldByArm("dron1", "der")


class TestInTransporter:
    def test_construccion(self):
        i = InTransporter(transporter_id="t1")
        assert i.transporter_id == "t1"

    def test_inmutable(self):
        i = InTransporter(transporter_id="t1")
        with pytest.raises(FrozenInstanceError):
            i.transporter_id = "t2"  # type: ignore[misc]


class TestSubtiposSonDistinguibles:
    """Garantía clave del tagged union: cada subtipo es identificable
    por su tipo. Esto permite que el Validator y los handlers hagan
    match exhaustivo sin ambigüedad."""

    def test_isinstance_discrimina(self):
        ubicaciones: list[PackageLocation] = [
            AtLocation("deposito"),
            HeldByArm("dron1", "izq"),
            InTransporter("t1"),
        ]
        assert isinstance(ubicaciones[0], AtLocation)
        assert isinstance(ubicaciones[1], HeldByArm)
        assert isinstance(ubicaciones[2], InTransporter)
        assert not isinstance(ubicaciones[0], HeldByArm)
        assert not isinstance(ubicaciones[0], InTransporter)

    def test_match_exhaustivo(self):
        # Patrón típico de uso desde el Validator o un handler.
        def describir(loc: PackageLocation) -> str:
            match loc:
                case AtLocation(loc_id=l):
                    return f"libre en {l}"
                case HeldByArm(drone_id=d, arm_id=a):
                    return f"sujeto por {d}.{a}"
                case InTransporter(transporter_id=t):
                    return f"dentro de {t}"

        assert describir(AtLocation("deposito")) == "libre en deposito"
        assert describir(HeldByArm("dron1", "izq")) == "sujeto por dron1.izq"
        assert describir(InTransporter("t1")) == "dentro de t1"


# ---------------------------------------------------------------------------
# Package: construcción y composición con los subtipos
# ---------------------------------------------------------------------------


class TestPackageConstruccion:
    def test_libre_en_localizacion(self, medicina):
        p = Package(id="caja1", contains=medicina, at=AtLocation("deposito"))
        assert p.id == "caja1"
        assert p.contains == medicina
        assert p.at == AtLocation("deposito")

    def test_en_brazo_de_drone(self, medicina):
        p = Package(
            id="caja1", contains=medicina, at=HeldByArm("dron1", "izq")
        )
        assert isinstance(p.at, HeldByArm)
        assert p.at.drone_id == "dron1"
        assert p.at.arm_id == "izq"

    def test_dentro_de_transporter(self, medicina):
        p = Package(
            id="caja1", contains=medicina, at=InTransporter("t1")
        )
        assert isinstance(p.at, InTransporter)
        assert p.at.transporter_id == "t1"


# ---------------------------------------------------------------------------
# Package: inmutabilidad
# ---------------------------------------------------------------------------


class TestPackageInmutabilidad:
    def test_no_se_puede_reasignar_id(self, caja_en_deposito):
        with pytest.raises(FrozenInstanceError):
            caja_en_deposito.id = "otra"  # type: ignore[misc]

    def test_no_se_puede_reasignar_contains(self, caja_en_deposito):
        # Un paquete no cambia de categoría.
        with pytest.raises(FrozenInstanceError):
            caja_en_deposito.contains = Content("comida")  # type: ignore[misc]

    def test_no_se_puede_reasignar_at(self, caja_en_deposito):
        with pytest.raises(FrozenInstanceError):
            caja_en_deposito.at = HeldByArm("dron1", "izq")  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self, caja_en_deposito):
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            caja_en_deposito.peso = 5.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Transiciones funcionales con replace
# ---------------------------------------------------------------------------


class TestPackageTransiciones:
    """Los handlers de comandos (sesión B) construirán nuevos Package con
    dataclasses.replace para reflejar las transiciones. Aquí solo
    documentamos que ese patrón funciona y no rompe nada."""

    def test_replace_devuelve_nueva_instancia(self, caja_en_deposito):
        nueva = replace(caja_en_deposito, at=HeldByArm("dron1", "izq"))
        assert nueva is not caja_en_deposito
        assert nueva.at == HeldByArm("dron1", "izq")

    def test_replace_no_muta_la_original(self, caja_en_deposito):
        # Crítico para los snapshots del Memento.
        original_at = caja_en_deposito.at
        _ = replace(caja_en_deposito, at=HeldByArm("dron1", "izq"))
        assert caja_en_deposito.at == original_at

    def test_replace_preserva_id_y_contenido(self, caja_en_deposito):
        nueva = replace(caja_en_deposito, at=InTransporter("t1"))
        assert nueva.id == caja_en_deposito.id
        assert nueva.contains == caja_en_deposito.contains

    def test_ciclo_completo_de_transiciones(self, medicina):
        # Simulamos la vida de una caja a lo largo de un plan típico:
        # depósito -> brazo -> transporter -> brazo -> casa entregada.
        p = Package(id="caja1", contains=medicina, at=AtLocation("deposito"))
        p = replace(p, at=HeldByArm("dron1", "izq"))
        p = replace(p, at=InTransporter("t1"))
        p = replace(p, at=HeldByArm("dron1", "der"))
        p = replace(p, at=AtLocation("casa1"))
        assert p.at == AtLocation("casa1")
        assert p.id == "caja1"
        assert p.contains == medicina


# ---------------------------------------------------------------------------
# El tipo prohíbe estados imposibles
# ---------------------------------------------------------------------------


class TestEstadosImposibles:
    """Documentamos por test que ciertos estados ilegales que serían
    representables con tres Optional independientes NO lo son con el
    tagged union. Es la demostración del 'make illegal states unrepresentable'."""

    def test_imposible_construir_sin_at(self, medicina):
        # `at` es obligatorio: no hay default. Un Package siempre está
        # en algún sitio.
        with pytest.raises(TypeError):
            Package(id="caja1", contains=medicina)  # type: ignore[call-arg]

    def test_at_siempre_es_uno_de_los_tres_subtipos(self, caja_en_deposito):
        # No es un test de runtime check (no lo añadimos), sino una
        # declaración de la invariante estructural: si tienes un Package
        # válido, su `at` es necesariamente uno de los tres subtipos.
        assert isinstance(
            caja_en_deposito.at, (AtLocation, HeldByArm, InTransporter)
        )
