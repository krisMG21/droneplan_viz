"""Tests de Person.

Cubre construcción con y sin needs/has_received iniciales, inmutabilidad
estricta, y el contrato dual de receive(): añade a has_received y retira
una ocurrencia de needs. Particular atención al comportamiento de
duplicados, que el PDDL canónico no aprovecha pero el modelo sí permite
para extensibilidad.
"""

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.domain.content import Content
from droneplan_viz.domain.person import Person


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def medicina() -> Content:
    return Content("medicina")


@pytest.fixture
def comida() -> Content:
    return Content("comida")


@pytest.fixture
def persona_necesita_medicina(medicina) -> Person:
    return Person(id="p1", position="casa1", needs=(medicina,))


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------


class TestPersonConstruccion:
    def test_se_construye_con_id_y_posicion(self):
        p = Person(id="persona1", position="casa1")
        assert p.id == "persona1"
        assert p.position == "casa1"

    def test_needs_y_has_received_por_defecto_vacios(self):
        p = Person(id="persona1", position="casa1")
        assert p.needs == ()
        assert p.has_received == ()

    def test_se_puede_construir_con_needs_iniciales(self, medicina, comida):
        p = Person(id="p1", position="casa1", needs=(medicina, comida))
        assert p.needs == (medicina, comida)
        assert p.has_received == ()

    def test_se_puede_construir_con_historico_inicial(self, medicina):
        # Útil para reconstruir estado desde un snapshot.
        p = Person(id="p1", position="casa1", has_received=(medicina,))
        assert p.has_received == (medicina,)


# ---------------------------------------------------------------------------
# Inmutabilidad
# ---------------------------------------------------------------------------


class TestPersonInmutabilidad:
    def test_no_se_puede_reasignar_id(self, persona_necesita_medicina):
        with pytest.raises(FrozenInstanceError):
            persona_necesita_medicina.id = "otra"  # type: ignore[misc]

    def test_no_se_puede_reasignar_posicion(self, persona_necesita_medicina):
        with pytest.raises(FrozenInstanceError):
            persona_necesita_medicina.position = "casa2"  # type: ignore[misc]

    def test_no_se_puede_reasignar_needs(self, persona_necesita_medicina):
        with pytest.raises(FrozenInstanceError):
            persona_necesita_medicina.needs = ()  # type: ignore[misc]

    def test_no_se_puede_reasignar_has_received(
        self, persona_necesita_medicina, medicina
    ):
        with pytest.raises(FrozenInstanceError):
            persona_necesita_medicina.has_received = (medicina,)  # type: ignore[misc]

    def test_needs_es_tupla_no_lista(self, persona_necesita_medicina):
        assert isinstance(persona_necesita_medicina.needs, tuple)
        assert not hasattr(persona_necesita_medicina.needs, "append")

    def test_has_received_es_tupla_no_lista(self):
        p = Person(id="p1", position="casa1")
        assert isinstance(p.has_received, tuple)
        assert not hasattr(p.has_received, "append")

    def test_no_se_pueden_anadir_atributos_nuevos(self, persona_necesita_medicina):
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            persona_necesita_medicina.edad = 30  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# receive(): el contrato dual
# ---------------------------------------------------------------------------


class TestReceiveDevuelveNuevaInstancia:
    def test_no_muta_la_original(self, persona_necesita_medicina, medicina):
        original = persona_necesita_medicina
        _ = original.receive(medicina)
        assert original.needs == (medicina,)
        assert original.has_received == ()

    def test_devuelve_objeto_distinto(self, persona_necesita_medicina, medicina):
        nueva = persona_necesita_medicina.receive(medicina)
        assert nueva is not persona_necesita_medicina

    def test_preserva_id_y_posicion(self, persona_necesita_medicina, medicina):
        nueva = persona_necesita_medicina.receive(medicina)
        assert nueva.id == persona_necesita_medicina.id
        assert nueva.position == persona_necesita_medicina.position


class TestReceiveActualizaHasReceived:
    def test_anade_al_final(self, medicina, comida):
        p = Person(id="p1", position="casa1", needs=(medicina, comida))
        p = p.receive(medicina)
        p = p.receive(comida)
        assert p.has_received == (medicina, comida)


class TestReceiveRetiraDeNeeds:
    def test_retira_una_ocurrencia_existente(
        self, persona_necesita_medicina, medicina
    ):
        nueva = persona_necesita_medicina.receive(medicina)
        assert nueva.needs == ()

    def test_retira_la_primera_ocurrencia_cuando_hay_duplicados(
        self, medicina
    ):
        # Si una persona necesita medicina dos veces, recibir UNA caja
        # de medicina debe dejar UNA pendiente, no cero.
        p = Person(id="p1", position="casa1", needs=(medicina, medicina))
        p = p.receive(medicina)
        assert p.needs == (medicina,)
        assert p.has_received == (medicina,)

    def test_preserva_orden_de_los_demas(self, medicina, comida):
        p = Person(id="p1", position="casa1", needs=(medicina, comida, medicina))
        # Recibe una medicina: retira la PRIMERA medicina, comida queda
        # intacta en su posición relativa al resto.
        p = p.receive(medicina)
        assert p.needs == (comida, medicina)

    def test_si_no_estaba_en_needs_queda_igual(self, comida):
        # Caso defensivo: el Validator habría rechazado esta entrega
        # antes (regla "necesita"), pero documentamos que receive() no
        # rompe ni inventa: si no está, no toca needs.
        p = Person(id="p1", position="casa1", needs=())
        nueva = p.receive(comida)
        assert nueva.needs == ()
        assert nueva.has_received == (comida,)


class TestReceiveCicloCompleto:
    """Documenta el ciclo de vida típico de una Person: nace con sus
    necesidades, las va satisfaciendo, termina con needs vacío y un
    has_received que es la traza completa de lo recibido."""

    def test_satisface_todas_sus_necesidades(self, medicina, comida):
        p = Person(id="p1", position="casa1", needs=(medicina, comida))
        assert p.needs != ()
        p = p.receive(medicina)
        assert p.needs == (comida,)
        p = p.receive(comida)
        assert p.needs == ()
        assert p.has_received == (medicina, comida)


# ---------------------------------------------------------------------------
# Igualdad por valor (todos los campos)
# ---------------------------------------------------------------------------


class TestPersonIgualdad:
    """No sobrescribimos __eq__. Razón: a diferencia de Location,
    donde la posición en pantalla es metadato visual descartable, en
    Person needs y has_received son estado lógico. Dos Person con
    mismo id pero distinto estado están en momentos distintos de la
    simulación, no son equivalentes."""

    def test_misma_persona_mismo_estado_son_iguales(self, medicina):
        a = Person(id="p1", position="casa1", needs=(medicina,))
        b = Person(id="p1", position="casa1", needs=(medicina,))
        assert a == b

    def test_mismo_id_distinto_needs_no_iguales(self, medicina):
        a = Person(id="p1", position="casa1", needs=(medicina,))
        b = Person(id="p1", position="casa1", needs=())
        assert a != b

    def test_mismo_id_distinto_historico_no_iguales(self, medicina):
        a = Person(id="p1", position="casa1")
        b = Person(id="p1", position="casa1", has_received=(medicina,))
        assert a != b
