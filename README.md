# Predicción de riesgo de churn con scikit-learn

## Problema presentado

La empresa reaccionaba después de perder clientes y no contaba con una priorización objetiva para orientar campañas de retención.

## Solución

El MVP prepara datos con Polars, entrena un modelo de clasificación reproducible con scikit-learn, ajusta el umbral para priorizar *recall* y publica métricas, artefacto del modelo e importancia de variables.

```text
Datos de clientes -> Polars -> Train/Test -> scikit-learn -> Métricas + Modelo + Drivers
```

## Tecnologías

Python, Polars, scikit-learn, NumPy, Joblib y Pytest.

## Ejecución

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m src.train
.venv/Scripts/pytest -q
```

## Resultado y conclusión

La ejecución genera `recall`, `precision` y `ROC-AUC`, además de un ranking de factores de riesgo. El resultado permite segmentar clientes prioritarios y convertir el modelo en una acción comercial verificable.
