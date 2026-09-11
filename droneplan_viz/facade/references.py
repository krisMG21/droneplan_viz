"""Resolución de identificadores: string-vs-referencia y validación.

Dos servicios, en capas:

1. `resolve_id(x)` — el núcleo. Convierte lo que el alumno escribe en un
   id-string canónico. Acepta indistintamente:
       - un str (que ES el id), o
       - el objeto que devolvió un builder (un Location, Drone, Package…),
         del que extrae su `.id`.
   Esto materializa la decisión 2 del diseño: `viz.mover("dron1", ...)` y
   `viz.mover(dron1, ...)` (donde `dron1` es lo que devolvió
   `agents.drone(...)`) son ambas válidas y equivalentes. No mira el
   estado del mundo; solo traduce a string.

2. Los resolvedores VALIDADOS — `resolve_location_id`, `resolve_content_id`,
   `resolve_person_id`, `resolve_package_id`, `resolve_drone_id`,
   `resolve_transporter_id`. Cada uno hace `resolve_id` y además comprueba
   que el id resultante esté DECLARADO en la tabla correspondiente del
   estado en construcción. Si no, lanza el `Unknown<Tipo>Error` de
   errors.py, con su mensaje didáctico (que cita la llamada del builder
   que lo arregla y enfatiza el "ANTES").

Por qué los resolvedores reciben el MAPPING y no `_FacadeState`:
    Mantiene este módulo desacoplado del acumulador de la fachada (que se
    define más adelante) y trivialmente testeable con dicts planos. El
    builder pasará `self._state.locations` (y hermanas) en su sitio; "la
    validación contra _state" se cumple igual, solo que la dependencia
    fluye hacia dentro (el builder conoce references, no al revés).

Lo que este módulo NO hace:
    - No detecta referencias cruzadas (pasar una persona donde se espera
      una localización en el parámetro `a` de mover/entregar). Esa guarda
      necesita conocer el método y el parámetro implicados y vive en las
      acciones de la fachada (facade.py), que sí tienen ese contexto y
      lanzan `WrongReferenceKindError` apoyándose en estos resolvedores.
    - No valida brazos: existencia del brazo en un dron requiere el objeto
      Drone resuelto y la lanza la acción correspondiente (UnknownArmError).
    - No valida nada físico (ocupación, co-localización, capacidad). Eso es
      del PlanRunner.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from droneplan_viz.facade.errors import (
    FacadeError,
    UnknownContentError,
    UnknownDroneError,
    UnknownLocationError,
    UnknownPackageError,
    UnknownPersonError,
    UnknownTransporterError,
)


def resolve_id(x: object) -> str:
    """Traduce `x` a su id-string canónico.

    Args:
        x: un str (interpretado como el id directamente) o un objeto que
            expone un atributo `.id` de tipo str (típicamente lo que
            devolvió un builder de la fachada).

    Returns:
        El id como str.

    Raises:
        FacadeError: si `x` no es ni un str no vacío ni un objeto con un
            `.id` str no vacío. El mensaje explica las dos formas válidas
            de identificar una entidad.

    Notas:
        - No consulta el estado del mundo: solo resuelve el tipo. La
          comprobación de "existe en el escenario" la hacen los
          resolvedores validados de más abajo.
        - Rechaza ids vacíos o en blanco: son casi siempre un error de
          tecleo y conviene atajarlos en el único punto por el que pasan
          todas las referencias.
    """
    if isinstance(x, str):
        if not x.strip():
            raise FacadeError(
                "un identificador no puede ser una cadena vacía; pásame el "
                "id de la entidad (p. ej. 'casa1') o el objeto que devolvió "
                "el builder"
            )
        return x

    candidate = getattr(x, "id", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate

    raise FacadeError(
        f"no sé cómo interpretar {x!r} como identificador; pásame un str "
        f"(el id de la entidad, p. ej. 'dron1') o el objeto que te devolvió "
        f"el builder (p. ej. lo que retornó viz.agents.drone(...))"
    )


def _resolve_in_table(
    x: object,
    table: Mapping[str, object],
    miss_error: Callable[..., FacadeError],
    *,
    contexto: str | None,
) -> str:
    """Resuelve `x` a id y exige que esté declarado en `table`.

    Helper común a los seis resolvedores públicos. Extrae el id con
    resolve_id y, si no está en la tabla, lo eleva con la fábrica de error
    correspondiente (que comparte la firma `(id, *, contexto=None)` en
    toda la familia "no declarado").
    """
    ref_id = resolve_id(x)
    if ref_id in table:
        return ref_id
    raise miss_error(ref_id, contexto=contexto)


def resolve_location_id(
    x: object,
    locations: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de una localización DECLARADA, o UnknownLocationError."""
    return _resolve_in_table(
        x, locations, UnknownLocationError, contexto=contexto
    )


def resolve_content_id(
    x: object,
    contents: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de un contenido DECLARADO, o UnknownContentError."""
    return _resolve_in_table(
        x, contents, UnknownContentError, contexto=contexto
    )


def resolve_person_id(
    x: object,
    persons: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de una persona DECLARADA, o UnknownPersonError."""
    return _resolve_in_table(
        x, persons, UnknownPersonError, contexto=contexto
    )


def resolve_package_id(
    x: object,
    packages: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de un paquete DECLARADO, o UnknownPackageError."""
    return _resolve_in_table(
        x, packages, UnknownPackageError, contexto=contexto
    )


def resolve_drone_id(
    x: object,
    drones: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de un dron DECLARADO, o UnknownDroneError."""
    return _resolve_in_table(
        x, drones, UnknownDroneError, contexto=contexto
    )


def resolve_transporter_id(
    x: object,
    transporters: Mapping[str, object],
    *,
    contexto: str | None = None,
) -> str:
    """Id de un transportador DECLARADO, o UnknownTransporterError."""
    return _resolve_in_table(
        x, transporters, UnknownTransporterError, contexto=contexto
    )


__all__ = [
    "resolve_id",
    "resolve_location_id",
    "resolve_content_id",
    "resolve_person_id",
    "resolve_package_id",
    "resolve_drone_id",
    "resolve_transporter_id",
]
