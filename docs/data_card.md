# Ficha del dataset sintético

## Propósito

Probar de extremo a extremo un flujo de riesgo de churn sin exponer información personal ni confidencial.

## Generación

`build_dataset(rows, seed)` usa `numpy.random.default_rng`. La etiqueta se muestrea desde una probabilidad logística que combina variables y ruido. Una misma semilla produce exactamente el mismo dataset.

| Variable | Rango/generación | Interpretación sintética |
|---|---|---|
| `tenure_months` | 1–120 | Antigüedad. |
| `monthly_charge` | 18–160 | Cargo mensual. |
| `support_tickets` | Poisson limitado 0–8 | Fricción de soporte. |
| `late_payments` | Poisson limitado 0–6 | Incumplimientos recientes. |
| `service_usage` | Beta escalada 0–100 | Índice de uso. |
| `contract_score` | 0, 0.5 o 1 | Mayor valor representa compromiso más largo. |
| `churn` | Bernoulli | Etiqueta sintética. |

## Exclusiones y riesgos

No contiene identidad, ubicación, características protegidas, comportamiento real ni consentimiento. La ausencia de atributos sensibles no elimina sesgos por proxies en un dataset real. No es apropiado para inferir causalidad o evaluar impacto de ofertas.
