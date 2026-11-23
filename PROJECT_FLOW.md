# AGROGUARD AI project flow

```mermaid
flowchart TD
A[Research question and scope] --> B[Acquire benchmark and regional data]
B --> C[Validate data, units and quality]
C --> D[EDA and feature analysis]
D --> E[Stratified 60/20/20 split]
E --> F[Six-model 5-fold CV comparison]
F --> G[RF and Extra Trees tuning]
G --> H[Select by CV Macro F1]
H --> I[Probability calibration]
I --> J[Conformal calibration split]
J --> K[Untouched test evaluation]
K --> L[Anomaly and stability experiments]
L --> M[Explainability and reports]
M --> N[Save model artifacts]
N --> O[FastAPI backend]
O --> P[JavaScript frontend]
P --> Q[Integration and testing]
Q --> R[Deployment and URL]
```

## Team allocation (maximum five groups)

1. Data and agricultural research: dataset, units, data audit, EDA, regional evidence.
2. ML experimentation: six models, cross-validation, tuning, evaluation.
3. Trust and explainability: calibration, conformal sets, anomaly detection, stability, feature importance.
4. Backend: FastAPI, validation, inference orchestration, logging, API tests.
5. Frontend and deployment: dashboard, warnings, results, responsive design, hosting, demonstration.

## Research contribution

The proposed contribution is an integrated trust-aware decision pipeline: calibrated class probabilities, conformal prediction sets, multivariate anomaly screening, perturbation-based stability analysis, and an explicit review/abstention state.

Do not claim that this is a new ML algorithm. The novelty is the integrated system design and its evaluation on appropriate data.
