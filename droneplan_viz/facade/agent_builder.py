"""AgentBuilder: la cara fluida para declarar los agentes del mundo.

Traduce

    viz.agents.drone("dron1", at="deposito")                 # 2 brazos
    viz.agents.drone("dron2", at="deposito", arms=["unico"]) # monobrazo
    viz.agents.drone("explorador", at="deposito", arms=[])   # sin brazos
    viz.agents.transporter("t1", capacidad=4, at="deposito")

a Drones y Transporters inmutables del dominio, que deja en el
`_FacadeState` compartido. Como WorldBuilder, es eager y valida
construcción de forma localizada; la posición inicial (`at`) debe ser una
localización ya declarada.

No valida física: que un brazo esté libre, que el transportador tenga hueco
o que el agente pueda moverse es responsabilidad del PlanRunner. Aquí solo
se comprueba que lo declarado es coherente (id único, brazos con nombres
válidos y distintos, posición declarada, capacidad sensata).
"""
from __future__ import annotations

from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.facade.errors import DuplicateIdError, FacadeError
from droneplan_viz.facade.references import resolve_location_id
from droneplan_viz.facade.state import _FacadeState
from droneplan_viz.facade.world_builder import _require_nonempty_id


class AgentBuilder:
    """Builder fluido de drones y transportadores. Muta `_FacadeState`."""

    __slots__ = ("_state",)

    def __init__(self, state: _FacadeState) -> None:
        """Recibe por referencia el mismo acumulador que WorldBuilder y
        DronePlanViz. No copia: muta el objeto compartido."""
        self._state = state

    # -----------------------------------------------------------------
    # Drone
    # -----------------------------------------------------------------
    def drone(
        self,
        id: str,
        *,
        at: str,
        arms: list[str] | None = None,
    ) -> Drone:
        """Declara un dron en una localización.

        Sobre `arms` (distinción sutil, documentada a propósito):
            - arms=None (no pasar nada)  -> default ["izq", "der"]. El 90 %
              de los escenarios docentes usan drones de dos brazos con esos
              nombres.
            - arms=["unico"]              -> dron monobrazo con ese nombre.
            - arms=[]                     -> dron explorador SIN brazos
              (válido; no podrá recoger ni entregar, solo moverse/explorar).

        Args:
            id: identificador simbólico único.
            at: localización inicial (str id o la Location devuelta por
                world.location()). Debe estar declarada ANTES.
            arms: nombres de los brazos. Ver la distinción de arriba.

        Returns:
            El Drone creado, en estado IDLE, utilizable como referencia.

        Raises:
            DuplicateIdError: si ya existe un dron con ese id.
            UnknownLocationError: si `at` no está declarada.
            FacadeError: id vacío/no-str, o nombres de brazo vacíos o
                repetidos dentro del mismo dron.
        """
        drone_id = _require_nonempty_id(id, que="el dron")
        if drone_id in self._state.drones:
            raise DuplicateIdError(drone_id, categoria="un dron")
        contexto = f"al declarar el dron '{drone_id}'"
        loc_id = resolve_location_id(
            at, self._state.locations, contexto=contexto
        )
        # Patrón seguro contra el mutable default compartido entre llamadas.
        if arms is None:
            arms = ["izq", "der"]
        vistos: set[str] = set()
        brazos: list[Arm] = []
        for nombre in arms:
            if not isinstance(nombre, str) or not nombre.strip():
                raise FacadeError(
                    f"los nombres de brazo del dron '{drone_id}' deben ser "
                    f"textos no vacíos; recibí {nombre!r}"
                )
            if nombre in vistos:
                raise FacadeError(
                    f"el dron '{drone_id}' tiene dos brazos llamados "
                    f"'{nombre}'; los nombres de brazo deben ser distintos "
                    f"dentro de un mismo dron"
                )
            vistos.add(nombre)
            brazos.append(Arm(id=nombre))
        drone = Drone(
            id=drone_id,
            position=loc_id,
            arms=tuple(brazos),
            state=DroneState.IDLE,
        )
        self._state.drones[drone_id] = drone
        return drone

    # -----------------------------------------------------------------
    # Transporter
    # -----------------------------------------------------------------
    def transporter(
        self,
        id: str,
        *,
        capacidad: int,
        at: str,
    ) -> Transporter:
        """Declara un transportador en una localización.

        Args:
            id: identificador simbólico único.
            capacidad: número máximo de paquetes que puede contener. Es un
                parámetro físico relevante y OBLIGATORIO (sin default): debe
                ser un entero >= 1. La fachada no replica el patrón PDDL de
                objetos `num` enlazados por (siguiente); es un entero simple.
            at: localización inicial (str id o Location). Debe estar
                declarada ANTES.

        Returns:
            El Transporter creado, utilizable como referencia.

        Raises:
            DuplicateIdError: si ya existe un transportador con ese id.
            UnknownLocationError: si `at` no está declarada.
            FacadeError: id vacío/no-str, o capacidad no entera o < 1.
        """
        transporter_id = _require_nonempty_id(id, que="el transportador")
        if transporter_id in self._state.transporters:
            raise DuplicateIdError(
                transporter_id, categoria="un transportador"
            )
        # bool es subclase de int: lo excluimos explícitamente para que
        # transporter(..., capacidad=True) no pase por un entero "1".
        if not isinstance(capacidad, int) or isinstance(capacidad, bool):
            raise FacadeError(
                f"la capacidad del transportador '{transporter_id}' debe ser "
                f"un número entero; recibí {capacidad!r}"
            )
        if capacidad < 1:
            raise FacadeError(
                f"la capacidad del transportador '{transporter_id}' debe ser "
                f">= 1; recibí {capacidad}"
            )
        contexto = f"al declarar el transportador '{transporter_id}'"
        loc_id = resolve_location_id(
            at, self._state.locations, contexto=contexto
        )
        transporter = Transporter(
            id=transporter_id, position=loc_id, capacity=capacidad
        )
        self._state.transporters[transporter_id] = transporter
        return transporter
