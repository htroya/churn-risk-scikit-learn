# Churn Risk con scikit-learn

Sistema de clasificación para priorizar la revisión de clientes con riesgo de abandono. Incluye fuente controlada, línea base, validación temporal progresiva, comparación de modelos, calibración, selección de umbral por capacidad y valor esperado, métricas por segmento, deriva y registro de versión.

> El dataset de referencia no contiene clientes ni desempeño de una empresa real. Permite validar entrenamiento, evaluación y gobierno del modelo sin exponer información confidencial y no autoriza decisiones automatizadas.

**Portafolio interactivo:** https://htroya.github.io/telecom-customer-360/

| Curva ROC | Matriz de confusión |
|---|---|
| ![Curva ROC](docs/results/roc_curve.svg) | ![Matriz de confusión](docs/results/confusion_matrix.svg) |

## Resumen ejecutivo

El flujo genera variables interpretables, reserva 25% de clientes para prueba, compara regresión logística con `DummyClassifier`, y selecciona el umbral que maximiza precision entre puntos que cumplen un objetivo de recall. Publica resultados tabulares y gráficos para que la calidad estadística y la carga comercial sean auditables.

**English summary:** Churn-risk system with temporal validation, champion/challenger comparison, calibration metrics, capacity-aware thresholding, segment diagnostics, drift monitoring, model registry metadata and reproducible evaluation artifacts.

## Problema

Una campaña sin priorización trata por igual a toda la base o usa un corte arbitrario. Eso oculta cuántos casos se capturan, cuántos se pierden y cuántos contactos innecesarios genera el modelo.

## Solución y arquitectura

```text
Generador de datos de referencia
            │
            ▼
Train 75% / Test 25% estratificado
     │                         │
Dummy baseline        StandardScaler + LogisticRegression
     └──────────────┬──────────┘
                    ▼
     Métricas + selección de umbral + artefactos
                    ▼
        Interpretación y decisión comercial
```

## Tecnologías

Python, NumPy, Polars, scikit-learn, Matplotlib, Joblib, Pytest y GitHub Actions.

## Datos y modelo

Las variables son antigüedad, cargo mensual, tickets, pagos tardíos, uso y tipo de contrato. La [ficha de datos](docs/data_card.md) documenta su generación; la [ficha del modelo](docs/model_card.md) cubre alcance, evaluación, riesgos y controles.

## Ejecución

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.train --rows 4000 --seed 42 --target-recall 0.80
```

Para regenerar los resultados versionados sin guardar el modelo binario:

```powershell
.venv\Scripts\python -m src.train --output docs/results --skip-model
```

## Pruebas

```powershell
.venv\Scripts\python -m pytest -q
```

Las pruebas cubren reproducibilidad y rangos, parámetros inválidos, cumplimiento del recall objetivo, mejora frente al baseline y presencia de todos los artefactos.

## Métricas y resultados verificables

La ejecución vigente se resume en [metrics.json](docs/results/metrics.json) y [model_comparison.csv](docs/results/model_comparison.csv). Se publican precision, recall, F1 y ROC-AUC junto con prevalencia, umbral y las cuatro celdas de la matriz de confusión. Las curvas completas y predicciones de prueba están disponibles como CSV.

## Umbral e interpretación comercial

El [razonamiento del umbral](docs/results/threshold_rationale.md) prioriza un recall objetivo del 80% y luego maximiza precision. La [interpretación comercial](docs/results/business_interpretation.md) traduce falsos positivos/negativos a carga de revisión. En producción, el criterio debe sustituirse por costos, capacidad, valor esperado y políticas aprobadas.

## Importancia de variables

`feature_importance.csv` y `feature_importance.svg` muestran coeficientes estandarizados. Son asociaciones dentro de un proceso sintético, no causalidad. El color/dirección indica si la variable incrementa o reduce los *log-odds* estimados.

## Decisiones técnicas

- Regresión logística para una línea base interpretable y probabilidades calibrables posteriormente.
- Holdout estratificado fijo para comparación reproducible.
- `class_weight="balanced"` para reducir sesgo hacia la clase mayoritaria.
- Umbral separado del entrenamiento para hacer explícito el intercambio precision/recall.
- Datos, curvas y predicciones exportados para revisión independiente.

## Gobierno, calibración y monitoreo

`src/model_governance.py` separa la calidad estadística de la decisión operativa. Calcula ROC-AUC, Brier y error esperado de calibración para el modelo vigente y el candidato. El umbral respeta la capacidad máxima de contacto y maximiza el valor esperado según costo de contacto y valor retenido.

El mismo flujo incorpora:

- validación temporal progresiva sin mezclar observaciones futuras en entrenamiento;
- PSI para cambios en la distribución de puntuaciones;
- precision, recall, selección y Brier por segmento;
- brecha de recall entre segmentos para orientar revisión;
- comparación vigente/candidato con criterio explícito;
- manifiesto de versión, propietario de decisión, huella SHA-256 y prohibición de acción automática.

El [diagrama de gobierno](docs/governance_architecture.md) muestra el ciclo de revisión y recalibración. `governance_report.json`, `model_manifest.json` y `segment_performance.csv` dejan evidencia consultable.

## Límites de alcance

- Las etiquetas nacen de una fórmula controlada; la estabilidad será mayor que con fuentes operativas.
- Las métricas de grupo describen diferencias y no determinan por sí solas una política de equidad.
- No se incluyen explicaciones locales, inferencia causal ni experimento de uplift.
- Costos, aceptación de ofertas y resultados de retención deben obtenerse del proceso comercial.

## Próximos pasos

1. Definir evento, horizonte y ventana de observación con responsables comerciales.
2. Conectar resultados de contacto y retención para estimar valor incremental.
3. Añadir explicaciones locales y análisis causal de ofertas.
4. Automatizar alertas, aprobación de promoción de modelos y seguimiento posterior.

## Capturas

Las imágenes superiores se generan automáticamente desde el conjunto de prueba y las métricas versionadas.

## Licencia

[MIT](LICENSE). Datos de referencia para validación segura y reproducible del sistema analítico.
