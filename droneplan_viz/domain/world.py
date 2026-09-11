"""World: agregador inmutable de todas las entidades del dominio.

Contiene las colecciones de entidades indexadas por id, el grafo de
costes entre localizaciones, y la métrica declarada del problema PDDL.
Es la "foto" completa del mundo en un instante; el HistoryManager
 hará deepcopy de World para sus snapshots.

Decisiones de modelado:

1. Diccionarios indexados por id (lookups O(1)). El Validator hace
   lookups por id constantemente; el coste lineal de recorrer todas
   las entidades para buscar una sería incompatible con escenarios
   con docenas de elementos.

2. Inmutabilidad estructural real, no por convención. Los dicts se
   envuelven en types.MappingProxyType al construir el World. Eso
   impide world.drones["d1"] = ... desde fuera. Las firmas exponen
   Mapping (interfaz de solo lectura), no dict, para que esa intención
   esté en el tipo. Esta garantía es la que hace seguros los snapshots
   del Memento: una vez creado el World, nadie puede mutarlo.

3. costs ya expandido. El builder del facade traduce
   "costes(d, simétrico=True)" a un dict con ambas direcciones antes
   de pasarlo al World. El World no conoce el concepto de simetría;
   solo el dict final aplanado.

4. metric es texto descriptivo, no expresión evaluable. Refleja la
   métrica del .pddl (p. ej. "minimize total-time") para que la UI la
   muestre. Sin parser, sin evaluación (decisión 8 del diseño).

5. Helpers de consulta limitados a recorridos estructurales que el
   Validator usa repetidamente. No incluyen validación ni reglas;
   solo proyecciones de los datos.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from droneplan_viz.domain.content import Content
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import HeldByArm, InTransporter, Package, AtLocation
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter


def _freeze(d: Mapping) -> MappingProxyType:
    """Envuelve un mapping mutable en una vista de solo lectura.

    El proxy no copia los datos: refleja el dict original. Como el World
    solo recibe dicts construidos por el llamante y nunca los expone
    fuera del proxy, en la práctica funciona como inmutable. El
    HistoryManager hará deepcopy del World entero para sus snapshots,
    rompiendo cualquier eventual referencia al dict original.
    """
    return MappingProxyType(dict(d))


@dataclass(frozen=True, slots=True)
class World:
    """Estado completo del dominio en un instante.

    Atributos:
        locations: nodos del grafo, indexados por loc_id.
        drones: drones del mundo, indexados por drone_id.
        transporters: transporters, indexados por transporter_id.
        packages: paquetes, indexados por package_id.
        persons: personas receptoras, indexadas por person_id.
        contents: categorías de carga, indexadas por content_id.
        costs: coste de cada arista del grafo, indexado por par
            ordenado (loc_origen, loc_destino). Si el grafo es no
            dirigido en algún caso, el builder se encarga de añadir
            ambas direcciones al dict.
        metric: texto descriptivo de la métrica del problema PDDL.
            None si no se declara.
    """

    locations: Mapping[str, Location] = field(default_factory=dict)
    drones: Mapping[str, Drone] = field(default_factory=dict)
    transporters: Mapping[str, Transporter] = field(default_factory=dict)
    packages: Mapping[str, Package] = field(default_factory=dict)
    persons: Mapping[str, Person] = field(default_factory=dict)
    contents: Mapping[str, Content] = field(default_factory=dict)
    costs: Mapping[tuple[str, str], float] = field(default_factory=dict)
    metric: str | None = None

    def __post_init__(self) -> None:
        # Envolvemos cada mapping en una vista de solo lectura. Como el
        # dataclass es frozen, hay que usar object.__setattr__ para
        # sustituir los campos por sus versiones envueltas. Este es el
        # patrón canónico para inmutabilidad estructural profunda en
        # dataclasses frozen.
        object.__setattr__(self, "locations", _freeze(self.locations))
        object.__setattr__(self, "drones", _freeze(self.drones))
        object.__setattr__(self, "transporters", _freeze(self.transporters))
        object.__setattr__(self, "packages", _freeze(self.packages))
        object.__setattr__(self, "persons", _freeze(self.persons))
        object.__setattr__(self, "contents", _freeze(self.contents))
        object.__setattr__(self, "costs", _freeze(self.costs))

    # -----------------------------------------------------------------
    # Helpers de consulta estructural
    #
    # Estos métodos NO validan nada y NO aplican reglas del dominio.
    # Solo proyectan los datos para que el Validator (y eventualmente
    # el renderer) no repitan el mismo recorrido en cada caso de uso.
    # -----------------------------------------------------------------

    def packages_at(self, loc_id: str) -> list[Package]:
        """Paquetes libres en una localización (no en brazo ni transporter).

        Recorre todos los paquetes y filtra los que tienen
        AtLocation(loc_id). Para escenarios docentes (decenas de
        paquetes) el coste lineal es irrelevante.
        """
        return [
            p
            for p in self.packages.values()
            if isinstance(p.at, AtLocation) and p.at.loc_id == loc_id
        ]

    def packages_in_transporter(self, transporter_id: str) -> list[Package]:
        """Paquetes actualmente cargados en un transporter.

        Es la consulta que reemplaza al hipotético Transporter.content.
        Coherente con la decisión de "una sola fuente de verdad":
        la relación vive en Package.at, y aquí se proyecta.
        """
        return [
            p
            for p in self.packages.values()
            if isinstance(p.at, InTransporter)
            and p.at.transporter_id == transporter_id
        ]

    def package_held_by(self, drone_id: str, arm_id: str) -> Package | None:
        """Paquete sostenido por un brazo concreto, o None si el brazo
        está libre.

        Es invariante del dominio que como mucho un paquete está en un
        brazo dado; si encontramos varios sería un bug del Validator o
        de los handlers, no de este método. Devolvemos el primero por
        seguridad.
        """
        for p in self.packages.values():
            if (
                isinstance(p.at, HeldByArm)
                and p.at.drone_id == drone_id
                and p.at.arm_id == arm_id
            ):
                return p
        return None
