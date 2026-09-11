"""Tests de Arm.

Arm es estructuralmente trivial (un único campo id), pero su semántica
no lo es: el id es único dentro de su drone, no globalmente. Los tests
documentan esa convención.
"""

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.domain.arm import Arm


class TestArmConstruccion:
    def test_se_construye_con_id(self):
        a = Arm(id="izq")
        assert a.id == "izq"

    def test_id_posicional(self):
        assert Arm("der").id == "der"


class TestArmInmutabilidad:
    def test_no_se_puede_reasignar_id(self):
        a = Arm(id="izq")
        with pytest.raises(FrozenInstanceError):
            a.id = "der"  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self):
        a = Arm(id="izq")
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            a.alcance = 1.5  # type: ignore[attr-defined]


class TestArmIgualdad:
    def test_mismo_id_iguales(self):
        # OJO: dos Arm con el mismo id se consideran iguales como dataclass,
        # aunque semánticamente representen brazos de drones distintos. Esto
        # es correcto: el Arm aislado no sabe a qué drone pertenece. La
        # unicidad global vive en el par (drone_id, arm_id), que el código
        # consumidor (HeldByArm, Drone) gestiona.
        assert Arm("izq") == Arm("izq")

    def test_distinto_id_distintos(self):
        assert Arm("izq") != Arm("der")
