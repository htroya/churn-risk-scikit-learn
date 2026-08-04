# Evaluación y gobierno del riesgo de abandono

Sistema analítico para ordenar revisiones de retención mediante probabilidades de abandono. Incluye una línea base, regresión logística, validación temporal progresiva, evaluación de calibración, selección de umbral limitada por capacidad, análisis por segmento, detección de deriva y registro verificable de la versión evaluada.

> **Uso responsable:** los datos incluidos son deterministas y no contienen clientes ni resultados empresariales. Las salidas orientan una revisión humana; no autorizan decisiones, contactos ni exclusiones automáticas.

**Sitio del portafolio:** https://htroya.github.io/telecom-customer-360/

| Curva ROC | Matriz de confusión |
|---|---|
| ![Curva ROC](docs/results/roc_curve.svg) | ![Matriz de confusión](docs/results/confusion_matrix.svg) |

## Qué resuelve

Un puntaje no es una política de retención. Para convertirlo en una lista revisable se debe conocer cuántos clientes pueden contactarse, qué valor se espera conservar, cuántos abandonos quedan fuera, cómo cambia la calibración y si el desempeño se deteriora en algún segmento.

La solución separa tres decisiones: estimar una probabilidad, evaluar si esa probabilidad es confiable y escoger un umbral sujeto a capacidad y costo. Cada etapa deja archivos tabulares y metadatos que permiten repetir el análisis y explicar por qué se seleccionó una configuración.

## Capacidades implementadas

| Capacidad | Implementación | Evidencia en el repositorio |
|---|---|---|
| Línea base | `DummyClassifier` para exigir mejora frente a una referencia sencilla | [`src/train.py`](src/train.py) |
| Modelo interpretable | estandarización y regresión logística con pesos balanceados | pipeline de entrenamiento |
| Evaluación de discriminación | ROC-AUC, precision, recall, F1 y curvas completas | [`docs/results/`](docs/results) |
| Calibración | Brier y error esperado de calibración por intervalos | [`src/model_governance.py`](src/model_governance.py) |
| Umbral por operación | capacidad máxima, costo de contacto y valor retenido | `select_capacity_threshold` |
| Validación temporal | particiones progresivas que impiden entrenar con observaciones futuras | `rolling_time_splits` |
| Deriva | PSI sobre la distribución de probabilidades | `population_stability_index` |
| Segmentos | precision, recall, selección, Brier y brecha de recall | `segment_performance` |
| Registro de versión | manifiesto, propietario, estado de decisión y huella SHA-256 | `model_manifest.json` |
| Comparación | evaluación del modelo vigente y el candidato con las mismas reglas | `governance_report.json` |

## Arquitectura de decisión

```text
Datos de comportamiento y servicio
                │
                ▼
       contrato + controles de datos
                │
      ┌─────────┴──────────┐
      ▼                    ▼
línea base        regresión logística
      └─────────┬──────────┘
                ▼
 discriminación + calibración + validación temporal
                │
      ┌─────────┴──────────┐
      ▼                    ▼
deriva y segmentos   capacidad y valor esperado
      └─────────┬──────────┘
                ▼
     umbral propuesto + manifiesto
                │
                ▼
          revisión y aprobación humana
```

El [diagrama de gobierno](docs/governance_architecture.md) describe responsabilidades, entradas, criterios de revisión y seguimiento posterior.

## Datos y variables

La [ficha de datos](docs/data_card.md) documenta origen, tipos, rangos y limitaciones. El conjunto incluye variables que pueden explicarse a un equipo de negocio:

- antigüedad del cliente;
- cargo mensual;
- cantidad de tickets;
- pagos tardíos;
- nivel de uso;
- tipo de contrato;
- segmento utilizado para diagnóstico.

La etiqueta se genera mediante una relación controlada para ejercitar el flujo completo. No representa la tasa de abandono de una compañía y no debe usarse para estimar resultados comerciales.

## Entrenamiento y evaluación

El flujo base reserva el 25% de observaciones para prueba mediante una partición estratificada. Compara el modelo con una línea base y mantiene el umbral fuera del entrenamiento para que el intercambio entre recall, precision y carga de revisión quede explícito.

La evaluación ampliada añade:

1. particiones temporales progresivas;
2. métricas de discriminación para modelo vigente y candidato;
3. Brier y error esperado de calibración;
4. PSI contra una distribución de referencia;
5. métricas por segmento y brecha máxima de recall;
6. selección de umbral bajo una capacidad máxima;
7. estimación de valor neto según costo de contacto y valor retenido;
8. manifiesto con huella del informe y prohibición de acción automática.

## Política de umbral

Hay dos rutas documentadas:

- el entrenamiento base busca puntos que cumplen el recall objetivo y, entre ellos, favorece mayor precision;
- el módulo de gobierno limita la cantidad seleccionada y maximiza el valor neto esperado, usando recall y precision como criterios de desempate.

Esta separación evita presentar un corte estadístico como si fuera una decisión comercial. Capacidad, costos, valor, horizonte y aprobación deben definirse con los responsables del proceso.

## Métricas

| Métrica | Pregunta que responde | Precaución |
|---|---|---|
| ROC-AUC | ¿el modelo ordena positivos por encima de negativos? | no informa la carga de contacto |
| Precision | ¿qué proporción de seleccionados abandona? | depende del umbral y la prevalencia |
| Recall | ¿qué proporción de abandonos se identifica? | puede crecer a costa de más contactos |
| F1 | ¿cómo se equilibran precision y recall? | asigna el mismo peso a ambos errores |
| Brier | ¿qué tan cercanas son las probabilidades al resultado? | combina calibración y discriminación |
| ECE | ¿qué diferencia hay entre confianza y frecuencia observada? | depende de la definición de intervalos |
| PSI | ¿cambió la distribución de puntuaciones? | no explica la causa del cambio |
| Valor neto esperado | ¿qué retorno se estima bajo costo y capacidad? | exige supuestos comerciales aprobados |

## Ejecución local

Requiere Python 3.11 o posterior.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.train --rows 4000 --seed 42 --target-recall 0.80
```

Para actualizar los resultados versionados sin guardar el modelo binario:

```powershell
.venv\Scripts\python -m src.train `
  --output docs/results `
  --rows 4000 `
  --seed 42 `
  --target-recall 0.80 `
  --skip-model
```

## Artefactos de evaluación

| Archivo | Contenido |
|---|---|
| [`metrics.json`](docs/results/metrics.json) | métricas, prevalencia, umbral y matriz de confusión |
| [`model_comparison.csv`](docs/results/model_comparison.csv) | comparación entre línea base y modelo |
| [`threshold_analysis.csv`](docs/results/threshold_analysis.csv) | precision y recall para cada umbral evaluado |
| [`test_predictions.csv`](docs/results/test_predictions.csv) | etiqueta, probabilidad y predicción del conjunto de prueba |
| [`roc_curve.csv`](docs/results/roc_curve.csv) | pares FPR/TPR y umbrales |
| [`precision_recall_curve.csv`](docs/results/precision_recall_curve.csv) | puntos de la curva precision-recall |
| [`feature_importance.csv`](docs/results/feature_importance.csv) | coeficientes estandarizados |
| [`threshold_rationale.md`](docs/results/threshold_rationale.md) | criterio aplicado al corte |
| [`business_interpretation.md`](docs/results/business_interpretation.md) | traducción de errores a carga de revisión |

`evaluate_governance` genera además `governance_report.json`, `model_manifest.json` y `segment_performance.csv` para el ciclo de comparación y aprobación.

## Lectura de resultados

Los coeficientes muestran asociaciones dentro de la fuente controlada. Su signo indica el efecto sobre los *log-odds* estimados después de estandarizar las variables; no establece causalidad ni garantiza que una intervención cambie el resultado.

La matriz de confusión debe leerse junto con la capacidad disponible:

- falsos positivos: revisiones que consumen capacidad sin observar abandono;
- falsos negativos: casos de abandono que quedan fuera del corte;
- verdaderos positivos: casos identificados, no retenciones conseguidas;
- verdaderos negativos: clientes no seleccionados que permanecen.

## Gobierno y controles

El manifiesto registra propietario de la decisión, nombre del modelo, versión, fecha, huella del informe y estado de automatización. El diseño establece que una puntuación no desencadena acciones por sí sola.

Antes de promover una versión deben revisarse:

- definición del evento y horizonte de predicción;
- separación temporal entre entrenamiento y evaluación;
- mejora frente a la línea base;
- calibración global y por grupos relevantes;
- PSI y cambios en las variables de entrada;
- capacidad, costo, valor y aceptación de ofertas;
- brechas de desempeño y riesgos de trato desigual;
- trazabilidad de datos, código, parámetros y aprobación.

## Pruebas y CI

```powershell
.venv\Scripts\python -m pytest -q
```

La suite comprueba:

- rangos, determinismo y parámetros inválidos;
- generación de todos los artefactos de evaluación;
- cumplimiento del recall objetivo cuando existe un punto viable;
- mejora del modelo frente al clasificador de referencia;
- Brier, ECE y PSI bajo escenarios conocidos;
- umbral que no excede la capacidad;
- valor neto esperado positivo en el escenario controlado;
- métricas por segmento y archivos de gobierno;
- rechazo de razones de capacidad fuera de rango.

GitHub Actions ejecuta las pruebas ante cambios en la rama o en una solicitud de integración.

## Operación y seguimiento

| Situación | Señal | Respuesta prevista |
|---|---|---|
| deriva de puntuaciones | PSI supera el criterio acordado | revisar datos, mezcla de clientes y calibración |
| deterioro de calibración | Brier o ECE aumentan | recalibrar y repetir validación temporal |
| capacidad reducida | seleccionados exceden cupo | recalcular el umbral, no truncar sin registrar |
| brecha entre segmentos | diferencia de recall relevante | analizar datos, error y política antes de aprobar |
| candidato no mejora | criterio de comparación fallido | conservar la versión vigente |
| cambio de informe | huella SHA-256 distinta | registrar una versión y aprobación nuevas |

## Decisiones de diseño

- La regresión logística ofrece una referencia interpretable y probabilidades que pueden calibrarse.
- El escalado se mantiene dentro del pipeline para evitar contaminación entre entrenamiento y prueba.
- La línea base impide aceptar complejidad que no aporta mejora medible.
- El umbral se gestiona como decisión operativa y no como parámetro oculto del modelo.
- Las curvas y predicciones se publican como tablas para permitir una revisión independiente.
- Las métricas de segmento son señales de diagnóstico; no definen por sí solas una política de equidad.
- La huella del informe enlaza la versión registrada con el contenido evaluado.

## Estructura del repositorio

```text
src/train.py                    datos, entrenamiento, evaluación y gráficos
src/model_governance.py         calibración, deriva, capacidad y registro
docs/data_card.md               contrato y límites del conjunto de datos
docs/model_card.md              propósito, métricas, riesgos y controles
docs/governance_architecture.md ciclo de revisión y responsabilidades
docs/results/                   tablas, gráficos e interpretación
tests/                          pruebas estadísticas y de gobierno
.github/workflows/              verificación continua
```

## Alcance y evolución

La fuente controlada produce relaciones más estables que las de un sistema operativo. Antes de utilizar datos corporativos deben resolverse etiquetas tardías, campañas previas, censura, cambios de producto, consentimiento, retención y calidad por período.

Las ampliaciones previstas son:

1. conectar resultados de contacto y permanencia para estimar valor incremental;
2. incorporar calibración por ventana y alertas con criterios aprobados;
3. añadir explicaciones locales revisables por caso;
4. evaluar sesgo de selección y políticas bajo capacidad cambiante;
5. establecer aprobación, reversión y seguimiento posterior a cada versión.

## Licencia

Código disponible bajo [MIT](LICENSE). Los datos incluidos se usan para validación segura sin exposición de información confidencial.
