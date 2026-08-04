# Ficha del modelo

## Uso previsto

Demostrar evaluación reproducible y priorización transparente de una lista para revisión humana. No está validado para producción ni decisiones automáticas.

## Entrenamiento

- División estratificada 75/25 con semilla fija.
- `StandardScaler` ajustado solo sobre entrenamiento mediante `Pipeline`.
- `LogisticRegression` con pesos balanceados y hasta 1.000 iteraciones.
- Baseline `DummyClassifier(strategy="prior")`.

## Evaluación

Se reportan prevalencia, precision, recall, F1, ROC-AUC y matriz de confusión. El umbral maximiza precision sujeto a recall mínimo de 80% sobre el holdout. El CSV `threshold_analysis.csv` permite revisar otros puntos.

## Gobernanza antes de producción

1. Acordar definición de churn, horizonte, costos y responsables.
2. Separar validación final temporal de la selección de umbral.
3. Evaluar calibración, estabilidad, drift y desempeño por grupos relevantes.
4. Versionar dataset, código, modelo, aprobación y rollback.
5. Mantener revisión humana y mecanismo de reclamación cuando corresponda.

## Limitaciones

El proceso generador induce relaciones que la regresión puede recuperar. Las métricas no anticipan desempeño sobre una cartera real. Los coeficientes son asociaciones condicionadas al modelo y no efectos causales.
