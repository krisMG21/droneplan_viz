"""Tests de droneplan_viz.facade.references.

Cubren las dos capas del módulo:

  1. resolve_id: la resolución string-vs-referencia (decisión 2 del
     diseño). Un str se interpreta como id; un objeto con `.id` cede su
     id; cualquier otra cosa (o un id vacío) es FacadeError con mensaje
     que explica las dos formas válidas.

  2. Los seis resolvedores validados: resuelven el id y exigen que esté
     declarado en la tabla correspondiente, lanzando el Unknown<Tipo>Error
     adecuado en caso contrario, propagando `contexto`.

Se usan tanto dobles ligeros (objetos con `.id`) como entidades reales del
dominio, para probar que la integración con domain/ funciona de verdad.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from droneplan_viz.domain import (
    Content,
    Drone,
    Location,
    Package,
    Person,
    Transporter,
)
from droneplan_viz.domain.package import AtLocation
from droneplan_viz.facade.errors import (
    FacadeError,
    UnknownContentError,
    UnknownDroneError,
    UnknownLocationError,
    UnknownPackageError,
    UnknownPersonError,
    UnknownTransporterError,
)
from droneplan_viz.facade.references import (
    resolve_content_id,
    resolve_drone_id,
    resolve_id,
    resolve_location_id,
    resolve_package_id,
    resolve_person_id,
    resolve_transporter_id,
)


@dataclass
class _Doble:
    """Objeto mínimo con `.id`, para probar la resolución por referencia
    sin acoplar el test a una entidad de dominio concreta."""

    id: str


# ===========================================================================
# resolve_id: núcleo string-vs-referencia
# ===========================================================================


class TestResolveIdString:
    def test_str_se_devuelve_tal_cual(self):
        assert resolve_id("dron1") == "dron1"

    def test_str_con_id_realista(self):
        assert resolve_id("pkg_med1") == "pkg_med1"

    def test_str_vacio_es_facade_error(self):
        with pytest.raises(FacadeError):
            resolve_id("")

    def test_str_en_blanco_es_facade_error(self):
        with pytest.raises(FacadeError):
            resolve_id("   ")


class TestResolveIdReferencia:
    def test_objeto_con_id_cede_su_id(self):
        assert resolve_id(_Doble("casa1")) == "casa1"

    def test_str_y_referencia_son_equivalentes(self):
        """El corazón de la decisión 2: mismas dos formas, mismo id."""
        d = _Doble("dron7")
        assert resolve_id("dron7") == resolve_id(d) == "dron7"

    @pytest.mark.parametrize(
        "entidad, esperado",
        [
            (Location(id="deposito"), "deposito"),
            (Content(id="medicina"), "medicina"),
            (Person(id="p1", position="casa1"), "p1"),
            (
                Package(
                    id="caja1",
                    contains=Content(id="agua"),
                    at=AtLocation(loc_id="deposito"),
                ),
                "caja1",
            ),
            (Drone(id="d1", position="deposito"), "d1"),
            (Transporter(id="t1", position="deposito", capacity=4), "t1"),
        ],
    )
    def test_entidades_reales_del_dominio(self, entidad, esperado):
        """Integración real: cualquier entidad de dominio expone `.id`."""
        assert resolve_id(entidad) == esperado


class TestResolveIdInvalido:
    @pytest.mark.parametrize("x", [None, 5, 3.14, ["casa1"], object()])
    def test_tipos_no_resolubles_son_facade_error(self, x):
        with pytest.raises(FacadeError):
            resolve_id(x)

    def test_objeto_con_id_no_str_es_facade_error(self):
        with pytest.raises(FacadeError):
            resolve_id(_Doble(id=123))  # type: ignore[arg-type]

    def test_objeto_con_id_vacio_es_facade_error(self):
        with pytest.raises(FacadeError):
            resolve_id(_Doble(id=""))

    def test_mensaje_explica_las_dos_formas(self):
        with pytest.raises(FacadeError) as exc:
            resolve_id(42)
        msg = str(exc.value)
        assert "str" in msg
        assert "builder" in msg


# ===========================================================================
# Resolvedores validados: tabla de aciertos
# ===========================================================================

# Cada entrada: (resolvedor, tabla_de_ejemplo, id_presente, error_de_miss)
_RESOLVERS = [
    (
        resolve_location_id,
        {"casa1": Location(id="casa1")},
        "casa1",
        UnknownLocationError,
    ),
    (
        resolve_content_id,
        {"medicina": Content(id="medicina")},
        "medicina",
        UnknownContentError,
    ),
    (
        resolve_person_id,
        {"p1": Person(id="p1", position="casa1")},
        "p1",
        UnknownPersonError,
    ),
    (
        resolve_package_id,
        {
            "caja1": Package(
                id="caja1",
                contains=Content(id="agua"),
                at=AtLocation(loc_id="deposito"),
            )
        },
        "caja1",
        UnknownPackageError,
    ),
    (
        resolve_drone_id,
        {"d1": Drone(id="d1", position="deposito")},
        "d1",
        UnknownDroneError,
    ),
    (
        resolve_transporter_id,
        {"t1": Transporter(id="t1", position="deposito", capacity=4)},
        "t1",
        UnknownTransporterError,
    ),
]


class TestResolversAcierto:
    @pytest.mark.parametrize("resolver, tabla, presente, _err", _RESOLVERS)
    def test_str_declarado_se_resuelve(self, resolver, tabla, presente, _err):
        assert resolver(presente, tabla) == presente

    @pytest.mark.parametrize("resolver, tabla, presente, _err", _RESOLVERS)
    def test_referencia_declarada_se_resuelve(
        self, resolver, tabla, presente, _err
    ):
        # La propia entidad almacenada en la tabla, pasada por referencia.
        entidad = tabla[presente]
        assert resolver(entidad, tabla) == presente


class TestResolversMiss:
    @pytest.mark.parametrize("resolver, tabla, _presente, error", _RESOLVERS)
    def test_id_no_declarado_lanza_su_error(
        self, resolver, tabla, _presente, error
    ):
        with pytest.raises(error):
            resolver("inexistente_xyz", tabla)

    @pytest.mark.parametrize("resolver, tabla, _presente, error", _RESOLVERS)
    def test_el_error_es_facade_error(
        self, resolver, tabla, _presente, error
    ):
        """Un solo except FacadeError atrapa cualquier miss."""
        with pytest.raises(FacadeError):
            resolver("inexistente_xyz", tabla)

    @pytest.mark.parametrize("resolver, _tabla, _presente, error", _RESOLVERS)
    def test_miss_sobre_tabla_vacia(
        self, resolver, _tabla, _presente, error
    ):
        with pytest.raises(error):
            resolver("lo_que_sea", {})


class TestResolversContexto:
    def test_contexto_se_propaga_al_error(self):
        with pytest.raises(UnknownLocationError) as exc:
            resolve_location_id(
                "casa9", {}, contexto="al declarar la persona 'p1'"
            )
        assert exc.value.contexto == "al declarar la persona 'p1'"
        assert "al declarar la persona 'p1'" in str(exc.value)

    def test_el_error_guarda_el_id_ofensor(self):
        with pytest.raises(UnknownDroneError) as exc:
            resolve_drone_id("dronX", {})
        assert exc.value.drone_id == "dronX"

    def test_sin_contexto_el_error_usa_generico(self):
        with pytest.raises(UnknownTransporterError) as exc:
            resolve_transporter_id("tZ", {})
        assert exc.value.contexto is None
        assert "aquí" in str(exc.value)
