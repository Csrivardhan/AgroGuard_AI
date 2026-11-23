# AGROGUARD AI — ML-first starter

Put the Kaggle CSV at `data/Crop_recommendation.csv`.

Required columns:
`N, P, K, temperature, humidity, ph, rainfall, label`

Install on Windows:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train:
```bash
python src/train.py
```

Predict:
```bash
python src/predict.py
```

The pipeline uses 60% training, 20% conformal calibration and 20% final testing.
Six classifiers are compared using stratified 5-fold CV on training data. Random Forest
and Extra Trees are tuned only on training data. The winner is selected using CV Macro F1.
Probability calibration uses CV on training data. A separate calibration partition estimates
a split-conformal LAC threshold. The final test partition is not used for model selection.

Important limitations:
- Conformal coverage is marginal and relies on exchangeability; distribution shift can invalidate it.
- Calibrated model probability is not the probability of crop success.
- Training min/max values are not agronomic limits.
- Isolation Forest flags are not proof of sensor error.
- Benchmark performance does not establish local field suitability.
