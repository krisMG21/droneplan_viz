"""Location: nodo del grafo de localizaciones del mundo.

El espacio en este dominio es un grafo (decisión 9 del diseño), no una
cuadrícula. Cada Location es un nodo, y los costes entre nodos viven en
World.costs como diccionario de aristas. Una Location desconoce a sus
vecinos: la topología es responsabilidad del World.

La posición en pantalla es opcional. Si el usuario no la proporciona, el
render usará un layout automático (force-directed). Si la proporciona vía
builder con at_screen=(x, y), el render la respeta.

Aunque el dataclass es frozen, sobrescribimos __eq__ y __hash__ para que la
identidad dependa solo del id. Razón: la posición en pantalla es metadato
visual; cambiar dónde se dibuja casa1 no la convierte en otra casa. Esto
permite además guardar Locations en sets y dicts indexándolas por id sin
sorpresas si algún día queremos reposicionarlas (creando una nueva instancia
con el mismo id) sin invalidar las referencias semánticas.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True, eq=False)
class Location:
    """Nodo del grafo de localizaciones.

    Atributos:
        id: identificador simbólico único en el mundo (p. ej. "deposito",
            "casa1"). Replica los símbolos del PDDL.
        position_screen: posición opcional en píxeles para el render. None
            significa que el layout automático decidirá dónde colocarla.
    """

    id: str
    position_screen: tuple[int, int] | None = field(default=None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Location):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
