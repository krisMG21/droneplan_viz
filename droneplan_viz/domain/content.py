"""Content: categoría simbólica de la carga transportable.

Modela el tipo de contenido que un Package contiene y que una Person puede
recibir: "medicina", "comida", "agua", etc. Coincide con los símbolos
atómicos que aparecen en los .pddl de la asignatura.

Es dato puro inmutable. No se valida nada aquí; las reglas (id no vacío,
no duplicado en el mundo) viven en Validator y en los builders del facade.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Content:
    """Categoría de contenido transportable.

    Atributos:
        id: identificador simbólico único en el mundo. Por convención en
            minúsculas y sin espacios, replicando los símbolos del PDDL
            (p. ej. "medicina", "comida").

    La igualdad y el hash son por todos los campos, pero al haber un único
    campo equivale a igualdad por id. Esto es lo deseable: dos Content con
    el mismo id son la misma categoría.
    """

    id: str
