"""Tests de droneplan_viz.facade.world_builder.

Verifican que WorldBuilder:

  1. Crea entidades de dominio correctas y las deja en el _FacadeState
     compartido (mutación por referencia).
  2. Devuelve la entidad creada como handle reutilizable por referencia.
  3. Valida construcción de forma eager y localizada: id vacío/duplicado,
     referencias a entidades no declaradas, con el error y el `contexto`
     correctos.
  4. Expande costes simétricos antes de guardarlos y NO exige aristas de
     coste (son informativas).

Lógica pura sobre domain/ + facade/: sin pygame.
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import Content, Location, Person
from droneplan_viz.domain.package import AtLocation, Package
from droneplan_viz.facade.errors import (
    DuplicateIdError,
    FacadeError,
    UnknownContentError,
    UnknownLocationError,
)
from droneplan_viz.facade.state import _FacadeState
from droneplan_viz.facade.world_builder import WorldBuilder


@pytest.fixture
def state() -> _FacadeState:
    return _FacadeState()


@pytest.fixture
def wb(state: _FacadeState) -> WorldBuilder:
    return WorldBuilder(state)


# ===========================================================================
# location
# ===========================================================================


class TestLocation:
    def test_crea_y_guarda_en_state(self, wb, state):
        loc = wb.location("deposito")
        assert isinstance(loc, Location)
        assert loc.id == "deposito"
        assert state.locations["deposito"] is loc

    def test_at_screen_opcional_por_defecto_none(self, wb):
        loc = wb.location("casa1")
        assert loc.position_screen is None

    def test_at_screen_se_respeta(self, wb):
        loc = wb.location("casa1", at_screen=(400, 300))
        assert loc.position_screen == (400, 300)

    def test_id_duplicado_lanza(self, wb):
        wb.location("casa1")
        with pytest.raises(DuplicateIdError):
            wb.location("casa1")

    def test_id_vacio_lanza(self, wb):
        with pytest.raises(FacadeError):
            wb.location("")

    def test_id_no_str_lanza(self, wb):
        with pytest.raises(FacadeError):
            wb.location(123)  # type: ignore[arg-type]


# ===========================================================================
# content
# ===========================================================================


class TestContent:
    def test_crea_y_guarda(self, wb, state):
        c = wb.content("medicina")
        assert isinstance(c, Content)
        assert c.id == "medicina"
        assert state.contents["medicina"] is c

    def test_duplicado_lanza(self, wb):
        wb.content("agua")
        with pytest.raises(DuplicateIdError):
            wb.content("agua")


# ===========================================================================
# person
# ===========================================================================


class TestPerson:
    def test_crea_con_location_declarada(self, wb, state):
        wb.location("casa1")
        p = wb.person("p1", at="casa1")
        assert isinstance(p, Person)
        assert p.position == "casa1"
        assert p.needs == ()
        assert state.persons["p1"] is p

    def test_acepta_referencia_de_location(self, wb):
        casa1 = wb.location("casa1")
        p = wb.person("p1", at=casa1)  # por referencia, no por str
        assert p.position == "casa1"

    def test_necesita_resuelve_contenidos(self, wb):
        wb.location("casa1")
        wb.content("medicina")
        wb.content("comida")
        p = wb.person("p1", at="casa1", necesita=["medicina", "comida"])
        assert p.needs == (Content(id="medicina"), Content(id="comida"))

    def test_necesita_acepta_referencias_content(self, wb):
        wb.location("casa1")
        med = wb.content("medicina")
        p = wb.person("p1", at="casa1", necesita=[med])
        assert p.needs == (med,)

    def test_at_no_declarada_lanza_unknown_location_con_contexto(self, wb):
        with pytest.raises(UnknownLocationError) as exc:
            wb.person("p1", at="casa9")
        assert exc.value.loc_id == "casa9"
        assert "al declarar la persona 'p1'" in str(exc.value)
        assert "ANTES" in str(exc.value)

    def test_necesita_contenido_no_declarado_lanza(self, wb):
        wb.location("casa1")
        with pytest.raises(UnknownContentError) as exc:
            wb.person("p1", at="casa1", necesita=["medicinaX"])
        assert exc.value.content_id == "medicinaX"
        assert "al declarar la persona 'p1'" in str(exc.value)

    def test_duplicado_lanza(self, wb):
        wb.location("casa1")
        wb.person("p1", at="casa1")
        with pytest.raises(DuplicateIdError):
            wb.person("p1", at="casa1")


# ===========================================================================
# package
# ===========================================================================


class TestPackage:
    def test_crea_con_content_y_location(self, wb, state):
        wb.location("deposito")
        wb.content("medicina")
        pkg = wb.package("caja1", contiene="medicina", at="deposito")
        assert isinstance(pkg, Package)
        assert pkg.contains == Content(id="medicina")
        assert pkg.at == AtLocation(loc_id="deposito")
        assert state.packages["caja1"] is pkg

    def test_content_no_declarado_lanza(self, wb):
        wb.location("deposito")
        with pytest.raises(UnknownContentError):
            wb.package("caja1", contiene="medicinaX", at="deposito")

    def test_location_no_declarada_lanza(self, wb):
        wb.content("medicina")
        with pytest.raises(UnknownLocationError) as exc:
            wb.package("caja1", contiene="medicina", at="deposito9")
        assert "al declarar la caja 'caja1'" in str(exc.value)

    def test_duplicado_lanza(self, wb):
        wb.location("deposito")
        wb.content("medicina")
        wb.package("caja1", contiene="medicina", at="deposito")
        with pytest.raises(DuplicateIdError):
            wb.package("caja1", contiene="medicina", at="deposito")


# ===========================================================================
# costes
# ===========================================================================


class TestCostes:
    def test_asimetrico_guarda_una_direccion(self, wb, state):
        wb.location("deposito")
        wb.location("casa1")
        wb.costes({("deposito", "casa1"): 66})
        assert state.costs == {("deposito", "casa1"): 66.0}

    def test_simetrico_expande_ambas_direcciones(self, wb, state):
        wb.location("deposito")
        wb.location("casa1")
        wb.costes({("deposito", "casa1"): 66}, simetrico=True)
        assert state.costs == {
            ("deposito", "casa1"): 66.0,
            ("casa1", "deposito"): 66.0,
        }

    def test_coerce_a_float(self, wb, state):
        wb.location("a")
        wb.location("b")
        wb.costes({("a", "b"): 5})
        assert isinstance(state.costs[("a", "b")], float)

    def test_acumulativo_entre_llamadas(self, wb, state):
        wb.location("a")
        wb.location("b")
        wb.location("c")
        wb.costes({("a", "b"): 5})
        wb.costes({("b", "c"): 7})
        assert state.costs == {("a", "b"): 5.0, ("b", "c"): 7.0}

    def test_endpoint_no_declarado_lanza(self, wb):
        wb.location("deposito")
        with pytest.raises(UnknownLocationError) as exc:
            wb.costes({("deposito", "casa9"): 10})
        assert exc.value.loc_id == "casa9"

    def test_clave_malformada_lanza(self, wb):
        wb.location("a")
        with pytest.raises(FacadeError):
            wb.costes({"no_es_par": 5})  # type: ignore[dict-item]

    def test_valor_no_numerico_lanza(self, wb):
        wb.location("a")
        wb.location("b")
        with pytest.raises(FacadeError):
            wb.costes({("a", "b"): "barato"})  # type: ignore[dict-item]


# ===========================================================================
# Mutación compartida: dos builders sobre el mismo state se ven
# ===========================================================================


class TestEstadoCompartido:
    def test_dos_builders_mismo_state_comparten_datos(self, state):
        wb1 = WorldBuilder(state)
        wb2 = WorldBuilder(state)
        wb1.location("casa1")
        # wb2 ve la location creada por wb1 porque comparten _FacadeState.
        p = wb2.person("p1", at="casa1")
        assert p.position == "casa1"
