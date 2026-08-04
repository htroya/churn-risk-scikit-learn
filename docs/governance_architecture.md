# Arquitectura de evaluación y gobierno

```mermaid
flowchart LR
    A[Variables con fecha de observación] --> B[Validación temporal progresiva]
    B --> C[Modelo vigente]
    B --> D[Modelo candidato]
    C --> E[Discriminación y calibración]
    D --> E
    E --> F[Selección con capacidad y valor esperado]
    F --> G[Desempeño por segmento]
    G --> H[Registro de versión y huella]
    H --> I[Monitoreo PSI, Brier, ECE y brechas]
    I --> J{Revisión humana}
    J -->|Aprobar| K[Campaña controlada]
    J -->|Recalibrar| B
```

El umbral se decide fuera del entrenamiento. Se limita la cantidad de contactos y se calcula el valor esperado con costo de contacto y valor retenido. El reporte compara el modelo vigente con un candidato, cuantifica calibración y deriva, y separa el análisis por segmento. El manifiesto impide interpretar la puntuación como autorización automática.

