"""WorldBuilder: la cara fluida para construir la topología del mundo.

Traduce las llamadas ergonómicas del alumno

    viz.world.location("deposito")
    viz.world.location("casa1", at_screen=(400, 300))
    viz.world.content("medicina")
    viz.world.person("persona1", at="casa1", necesita=["medicina"])
    viz.world.package("caja1", contiene="medicina", at="deposito")
    viz.world.costes({("deposito", "casa1"): 66}, simetrico=True)

a entidades inmutables del dominio (Location, Content, Person, Package) que
deja, ya construidas, en el `_FacadeState` compartido. No produce el World:
solo llena el acumulador. `build()` (en facade.py) ensambla el World a
partir de él.

Construcción EAGER con errores localizados: cada referencia se valida
contra lo ya declarado en el momento de la llamada. Si una persona se
ancla a una localización inexistente, el error salta aquí, citando la
persona concreta y la llamada que lo arregla. Esto obliga (a propósito) a
un orden topológico: primero la infraestructura (locations, contents),
luego las entidades que la referencian (persons, packages), luego los
costes. Es el mismo orden mental que el alumno tiene al leer la sección
:init de su .pddl.

Qué valida y qué no:
    - Valida CONSTRUCCIÓN: id no vacío, id no duplicado, referencias a
      entidades declaradas (location de una persona/paquete, content de un
      paquete o de las necesidades, endpoints de los costes).
    - NO valida FÍSICA del plan (eso es del PlanRunner). En particular, las
      aristas de coste NO son solo informativas: en el modelo de grafo del
      dominio, un coste declarado entre dos localizaciones ES lo que las
      hace adyacentes/transitables. Un Move hacia una localización sin
      arista de coste declarada FALLA en el runner ("la arista no es
      transitable"). La fachada no puede comprobarlo en tiempo de
      construcción (depende de la posición del dron a esa altura del plan),
      así que ese fallo lo reporta el PlanRunner, no este builder. Práctica:
      declara con costes() todas las aristas por las que tus drones vuelan;
      el helper `simetrico=True` evita declararlas dos veces.
"""
from __future__ import annotations

from droneplan_viz.domain.content import Content
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import AtLocation, Package
from droneplan_viz.domain.person import Person
from droneplan_viz.facade.errors import DuplicateIdError, FacadeError
from droneplan_viz.facade.references import (
    resolve_content_id,
    resolve_location_id,
)
from droneplan_viz.facade.state import _FacadeState


def _require_nonempty_id(id_: object, *, que: str) -> str:
    """Exige que `id_` sea un str no vacío para DECLARAR una entidad.

    A diferencia de resolve_id (que también acepta objetos), al declarar
    una entidad nueva el id tiene que venir como string: no tiene sentido
    declarar `viz.world.location(otra_location)`.
    """
    if not isinstance(id_, str) or not id_.strip():
        raise FacadeError(
            f"el id de {que} debe ser un texto no vacío; recibí {id_!r}"
        )
    return id_


class WorldBuilder:
    """Builder fluido de la topología del mundo. Muta `_FacadeState`."""

    __slots__ = ("_state",)

    def __init__(self, state: _FacadeState) -> None:
        """Recibe por referencia el acumulador compartido con DronePlanViz
        y AgentBuilder. No copia: muta el mismo objeto."""
        self._state = state

    # -----------------------------------------------------------------
    # Localizaciones y contenidos (infraestructura: declarar primero)
    # -----------------------------------------------------------------
    def location(
        self, id: str, *, at_screen: tuple[int, int] | None = None
    ) -> Location:
        """Declara una localización (nodo del grafo).

        Args:
            id: identificador simbólico único (p. ej. "deposito", "casa1").
            at_screen: posición opcional en píxeles. Si se omite, el layout
                automático del render decide dónde dibujarla. El espacio es
                un GRAFO, no una cuadrícula: at_screen es solo una pista
                visual, no afecta a la lógica.

        Returns:
            La Location creada (utilizable como referencia en otras
            llamadas, p. ej. `at=casa1`).

        Raises:
            DuplicateIdError: si ya existe una localización con ese id.
            FacadeError: si el id es vacío o no es un str.
        """
        loc_id = _require_nonempty_id(id, que="la localización")
        if loc_id in self._state.locations:
            raise DuplicateIdError(loc_id, categoria="una localización")
        loc = Location(id=loc_id, position_screen=at_screen)
        self._state.locations[loc_id] = loc
        return loc

    def content(self, id: str) -> Content:
        """Declara una categoría de contenido transportable.

        Args:
            id: identificador simbólico (p. ej. "medicina", "comida",
                "agua"), replicando los símbolos del PDDL.

        Returns:
            El Content creado.

        Raises:
            DuplicateIdError: si ya existe un contenido con ese id.
            FacadeError: si el id es vacío o no es un str.
        """
        content_id = _require_nonempty_id(id, que="el contenido")
        if content_id in self._state.contents:
            raise DuplicateIdError(content_id, categoria="un contenido")
        content = Content(id=content_id)
        self._state.contents[content_id] = content
        return content

    # -----------------------------------------------------------------
    # Entidades que referencian la infraestructura
    # -----------------------------------------------------------------
    def person(
        self,
        id: str,
        *,
        at: str | Location,
        necesita: list[str | Content] | None = None,
    ) -> Person:
        """Declara una persona receptora en una localización.

        Args:
            id: identificador simbólico único.
            at: localización donde vive (str id o la Location devuelta por
                location()). Debe estar declarada ANTES.
            necesita: lista de contenidos que la persona necesita (ids o
                Content). Cada uno debe estar declarado. Por defecto, nada.

        Returns:
            La Person creada.

        Raises:
            DuplicateIdError: si ya existe una persona con ese id.
            UnknownLocationError: si `at` no está declarada.
            UnknownContentError: si algún contenido de `necesita` no está
                declarado.
            FacadeError: si el id es vacío o no es un str.
        """
        person_id = _require_nonempty_id(id, que="la persona")
        if person_id in self._state.persons:
            raise DuplicateIdError(person_id, categoria="una persona")
        contexto = f"al declarar la persona '{person_id}'"
        loc_id = resolve_location_id(
            at, self._state.locations, contexto=contexto
        )
        needs: tuple[Content, ...] = ()
        if necesita is not None:
            resolved: list[Content] = []
            for ref in necesita:
                cid = resolve_content_id(
                    ref, self._state.contents, contexto=contexto
                )
                resolved.append(self._state.contents[cid])
            needs = tuple(resolved)
        person = Person(id=person_id, position=loc_id, needs=needs)
        self._state.persons[person_id] = person
        return person

    def package(
        self,
        id: str,
        *,
        contiene: str | Content,
        at: str | Location,
    ) -> Package:
        """Declara un paquete (caja) libre en una localización.

        Args:
            id: identificador simbólico único.
            contiene: contenido que lleva dentro (str id o Content). Debe
                estar declarado.
            at: localización donde está depositado (str id o Location).
                Debe estar declarada. El paquete arranca libre en el suelo;
                las acciones del plan lo recogerán/moverán.

        Returns:
            El Package creado, con su ubicación inicial AtLocation(at).

        Raises:
            DuplicateIdError: si ya existe un paquete con ese id.
            UnknownContentError: si `contiene` no está declarado.
            UnknownLocationError: si `at` no está declarada.
            FacadeError: si el id es vacío o no es un str.
        """
        package_id = _require_nonempty_id(id, que="la caja")
        if package_id in self._state.packages:
            raise DuplicateIdError(package_id, categoria="una caja")
        contexto = f"al declarar la caja '{package_id}'"
        content_id = resolve_content_id(
            contiene, self._state.contents, contexto=contexto
        )
        loc_id = resolve_location_id(
            at, self._state.locations, contexto=contexto
        )
        package = Package(
            id=package_id,
            contains=self._state.contents[content_id],
            at=AtLocation(loc_id=loc_id),
        )
        self._state.packages[package_id] = package
        return package

    # -----------------------------------------------------------------
    # Costes del grafo (opcionales)
    # -----------------------------------------------------------------
    def costes(
        self,
        costs: dict[tuple[str, str], float],
        *,
        simetrico: bool = False,
    ) -> None:
        """Declara costes de aristas entre localizaciones.

        Args:
            costs: dict de (origen, destino) -> coste. Cada extremo debe
                ser una localización declarada.
            simetrico: si True, cada arista (o, d) se expande también a
                (d, o) con el mismo coste ANTES de almacenarla. El World no
                conoce el concepto de simetría; recibe el dict ya aplanado.

        Es acumulativo: varias llamadas a costes() se combinan. Si se repite
        un par, gana la última declaración.

        Raises:
            UnknownLocationError: si algún extremo no está declarado.
            FacadeError: si una clave no es un par o un valor no es numérico.

        Nota IMPORTANTE: en el modelo de grafo del dominio, declarar un coste
        entre dos localizaciones es lo que crea la ARISTA transitable entre
        ellas (la adyacencia es implícita por la existencia del coste).
        Además de alimentar la métrica de coste, por tanto, costes() define
        por dónde se puede volar: un Move hacia una localización con la que
        el origen no comparte arista declarada FALLA en el runner ("la arista
        no es transitable"). En la práctica, declara aquí toda arista por la
        que algún dron vaya a moverse; con `simetrico=True` te ahorras
        declarar ida y vuelta. mover() solo comprueba en construcción que el
        destino EXISTA; que la arista sea transitable lo verifica el
        PlanRunner al ejecutar (no se puede saber antes: depende de dónde
        esté el dron en ese punto del plan).
        """
        contexto = "al declarar los costes"
        for clave, valor in costs.items():
            if (
                not isinstance(clave, tuple)
                or len(clave) != 2
                or not all(isinstance(p, str) for p in clave)
            ):
                raise FacadeError(
                    f"cada clave de costes debe ser un par "
                    f"(origen, destino) de localizaciones; recibí {clave!r}"
                )
            if not isinstance(valor, (int, float)) or isinstance(valor, bool):
                raise FacadeError(
                    f"el coste de {clave!r} debe ser un número; recibí "
                    f"{valor!r}"
                )
            origen, destino = clave
            o_id = resolve_location_id(
                origen, self._state.locations, contexto=contexto
            )
            d_id = resolve_location_id(
                destino, self._state.locations, contexto=contexto
            )
            self._state.costs[(o_id, d_id)] = float(valor)
            if simetrico:
                self._state.costs[(d_id, o_id)] = float(valor)
