# Churn Risk Decision System

Sistema completo de entrenamiento y scoring para priorizar clientes con riesgo
de abandono. Incluye generación reproducible de datos, selección comparativa de
modelos, optimización de umbral orientada a retención, explicabilidad,
versionado de artefactos, baseline de deriva y scoring por lotes.

> Los datos son sintéticos. El score es apoyo para revisión humana y no una
> decisión automatizada de alto impacto.

## Capacidades

- Variables numéricas y categóricas con relaciones de negocio interpretables.
- Preprocesamiento con imputación, estandarización y codificación categórica.
- Comparación mediante validación cruzada de regresión logística, Random Forest
  e HistGradientBoosting.
- Selección por ROC-AUC y umbral operativo sujeto a recall mínimo.
- Métricas holdout: ROC-AUC, recall, precision, F1, Brier score y lift@10%.
- Importancia por permutación independiente del tipo de modelo.
- Bundle versionado para scoring y segmentación `low/medium/high/critical`.
- Model card, baseline estadístico y manifiesto SHA-256 de artefactos.
- CI con lint y pruebas de entrenamiento y scoring.

## Arquitectura

```mermaid
flowchart LR
    D[Datos de clientes] --> V[Contrato y validación]
    V --> P[Preprocesamiento]
    P --> CV[Selección con CV]
    CV --> T[Modelo campeón]
    T --> TH[Optimización de umbral]
    TH --> A[Bundle + métricas + model card]
    A --> S[Scoring por lotes]
    S --> R[Segmentos de riesgo]
    A --> M[Baseline para monitoreo]
```

## Ejecución

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m src.train --rows 8000 --minimum-recall 0.78
.venv/Scripts/python -m src.score --rows 1000
.venv/Scripts/pytest -q
```

## Artefactos

| Archivo | Uso |
|---|---|
| `churn_model.joblib` | Pipeline, umbral, esquema y versión |
| `model_leaderboard.csv` | Comparación reproducible de candidatos |
| `metrics.json` | Métricas de aceptación |
| `feature_importance.csv` | Drivers globales |
| `training_baseline.json` | Línea base para deriva |
| `model_card.md` | Uso, limitaciones y gobierno |
| `manifest.json` | Integridad SHA-256 |

## Paso a producción

Una implantación real requiere datos representativos, revisión de sesgo por
segmento, calibración temporal, monitoreo de deriva y outcomes, registro en el
catálogo de modelos, aprobación del dueño de negocio y política de
reentrenamiento.
