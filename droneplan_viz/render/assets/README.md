# Sprites del render

Coloca aquí los `.png` de las entidades. El `SpriteManager` los carga,
escala y cachea automáticamente. Si un archivo no está, el render cae a la
primitiva geométrica correspondiente (no hace falta entregar todos).

Los nombres y claves están declarados en `droneplan_viz/render/theme.py`
(`_DEFAULT_SPRITE_FILES`). El dron NO es un único sprite por estado, sino un
sistema **por capas componibles**: un cuerpo base + una cara intercambiable
(+ objeto agarrado + carrier).

## Dron (capas)

Cuerpos:

- `drone.png` — cuerpo base, sin cara incrustada (se le superpone una cara).
- `drone_interacting.png` — cuerpo con los brazos extendidos (trae su cara
  propia).

Caras (se superponen sobre `drone.png`):

- Direccionales en movimiento, 6 sectores: `face_N.png`, `face_S.png`,
  `face_NE.png`, `face_NW.png`, `face_SE.png`, `face_SW.png`.
- De estado: `face.png` (reposo a la espera), `face_idle.png` (ejecución
  terminada), `face_error1.png` y `face_error2.png` (error; el render las
  ALTERNA para que la cara de fallo parpadee).
- Reservadas para mensajes futuros: `face_talking1..3.png`.

Objeto agarrado y carrier:

- `box.png` — caja que el dron sostiene.
- `carrier.png` (= nivel 0%) y `carrier1.png`…`carrier5.png` (20%…100% de
  llenado).

## Personas (sets `personN` con variante de entrega)

Las personas usan **sets de sprite autodetectados**: el `SpriteManager`
ESCANEA esta carpeta buscando `personN.png` y reparte los sets encontrados
entre las personas del problema (round-robin por id ordenado; se repiten si
hay más personas que sets). Cada set tiene dos variantes:

- `personN.png` — persona EN ESPERA (aún no ha recibido su caja).
- `personN_box.png` — persona que YA recibió su entrega (sostiene la caja).
  El render cambia a esta variante automáticamente al completarse el Deliver.
  Si un set no tiene `_box`, la persona no cambia de sprite (cae a la base).

Ejemplo: `person1.png`, `person1_box.png`, `person2.png`, `person2_box.png`
→ dos sets que alternan entre las personas. Si no hay ningún `personN`, se usa
el genérico `person.png`; si tampoco, el círculo primitivo (sin flags).

> Estos sprites NO tienen por qué ser cuadrados: el `SpriteManager` los escala
> PRESERVANDO la proporción (un 20×35 no se achata). No se dibuja ningún anillo
> alrededor de las personas: el estado espera/entrega lo transmite el sprite.

## Resto de entidades (sprite base simple)

- `package.png`, `transporter.png`, `location.png`.

> **Outline de las cajas por tipo de contenido**: `box.png` es monocromo (la
> misma caja la sostiene el dron, está en el suelo o la lleva el carrier). Para
> distinguir el contenido en el suelo, el render dibuja un borde del color del
> tipo de contenido alrededor de la caja. El color sale de la paleta del
> escenario (`DronePlanViz.colores_contenido()`, reproducible) o de la
> estrategia del theme; el grosor es `Theme.package_outline_width` (0 lo
> desactiva).

## Recomendaciones

- Formato PNG con canal alfa (RGBA). El fondo transparente evita cajas
  cuadradas sobre el lienzo.
- Resolución cómoda: 2x o 3x el tamaño en pantalla; el `SpriteManager`
  reescala con `smoothscale` al tamaño que pide el `Theme`.

> Nota: los iconos de los botones de transporte y del inventario del HUD son
> aparte y viven en `droneplan_viz_app/assets/hud_icons/`
> (`btn_*_normal/pressed.png`, `inv_*.png`, `inv_chevron_*.png`).