"""Tests de Transporter.

Análogos a los de Drone pero más simples: el Transporter es dato puro,
sin FSM ni componentes anidados. Verificamos construcción, inmutabilidad,
y por test explícito documentamos que la regla de capacidad y el rango
de capacity no se aplican en construcción (son responsabilidad del
Validator y los builders, coherente con el resto del dominio).
"""

import pytest
from dataclasses import FrozenInstanceError, replace

from droneplan_viz.domain.transporter import Transporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transporter_basico() -> Transporter:
    return Transporter(id="t1", position="deposito", capacity=4)


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------


class TestTransporterConstruccion:
    def test_se_construye_con_id_position_capacity(self, transporter_basico):
        assert transporter_basico.id == "t1"
        assert transporter_basico.position == "deposito"
        assert transporter_basico.capacity == 4

    def test_id_position_capacity_son_obligatorios(self):
        # Ningún campo tiene default. No tendría sentido un transporter
        # "flotante" sin posición o sin capacidad declarada.
        with pytest.raises(TypeError):
            Transporter(id="t1")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Transporter(id="t1", position="deposito")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Inmutabilidad
# ---------------------------------------------------------------------------


class TestTransporterInmutabilidad:
    def test_no_se_puede_reasignar_id(self, transporter_basico):
        with pytest.raises(FrozenInstanceError):
            transporter_basico.id = "t2"  # type: ignore[misc]

    def test_no_se_puede_reasignar_position(self, transporter_basico):
        with pytest.raises(FrozenInstanceError):
            transporter_basico.position = "casa1"  # type: ignore[misc]

    def test_no_se_puede_reasignar_capacity(self, transporter_basico):
        # La capacidad es propiedad estructural del vehículo. No varía a
        # lo largo de la simulación.
        with pytest.raises(FrozenInstanceError):
            transporter_basico.capacity = 8  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self, transporter_basico):
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            transporter_basico.peso_max = 200  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Movimiento: solo el Transporter cambia de posición; lo que contiene
# vive en Package.at y no se toca aquí.
# ---------------------------------------------------------------------------


class TestTransporterMovimiento:
    def test_replace_position(self, transporter_basico):
        movido = replace(transporter_basico, position="casa1")
        assert movido.position == "casa1"
        # Original intacta.
        assert transporter_basico.position == "deposito"

    def test_replace_preserva_id_y_capacity(self, transporter_basico):
        movido = replace(transporter_basico, position="casa1")
        assert movido.id == transporter_basico.id
        assert movido.capacity == transporter_basico.capacity


# ---------------------------------------------------------------------------
# Igualdad por valor
# ---------------------------------------------------------------------------


class TestTransporterIgualdad:
    """Mismo criterio que Drone y Person: igualdad por todos los campos.
    Dos transporter con mismo id pero distinta posición están en momentos
    distintos de la simulación."""

    def test_misma_configuracion_iguales(self):
        a = Transporter(id="t1", position="deposito", capacity=4)
        b = Transporter(id="t1", position="deposito", capacity=4)
        assert a == b

    def test_mismo_id_distinta_posicion_no_iguales(self):
        a = Transporter(id="t1", position="deposito", capacity=4)
        b = Transporter(id="t1", position="casa1", capacity=4)
        assert a != b


# ---------------------------------------------------------------------------
# La validación NO ocurre en construcción
# ---------------------------------------------------------------------------


class TestTransporterNoValidaEnConstruccion:
    """Documentación ejecutable: el tipo no impone reglas semánticas.
    El Validator y los builders se ocupan."""

    def test_capacity_cero_se_construye(self):
        # Conceptualmente raro pero estructuralmente válido. El builder
        # del facade rechazará capacities no positivas; el tipo no.
        t = Transporter(id="t1", position="deposito", capacity=0)
        assert t.capacity == 0

    def test_capacity_negativa_se_construye(self):
        # Otra incoherencia que captura el Validator/builder, no el tipo.
        t = Transporter(id="t1", position="deposito", capacity=-3)
        assert t.capacity == -3

    def test_position_que_no_existe_en_world_se_construye(self):
        # El Transporter no conoce el World. La verificación de existencia
        # de la Location la hace el Validator.
        t = Transporter(id="t1", position="inexistente", capacity=4)
        assert t.position == "inexistente"

    def test_capacity_excedida_no_se_comprueba_aqui(self):
        # No es expresable en este test (no tenemos packages todavía),
        # pero documentamos por test conceptual: la regla "número de
        # packages con InTransporter(t1) <= t1.capacity" la aplica el
        # Validator, no este tipo. El Transporter ni siquiera conoce
        # los packages.
        t = Transporter(id="t1", position="deposito", capacity=2)
        # No hay forma de "construir un transporter sobrecargado" porque
        # el transporter no almacena su contenido. La sobrecarga solo se
        # puede crear vía relaciones en Package.at.
        assert t.capacity == 2
