#!/usr/bin/env python3
"""
XGBoost prediction script for olive oil price forecasting.
Loads trained models and generates predictions for all horizons.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='[XGB-Predict] %(message)s')
logger = logging.getLogger(__name__)

HORIZONS = [4, 13, 26, 52]
FEATURE_NAMES = [
    'baseValue',
    'lag_1w', 'lag_2w', 'lag_4w', 'lag_8w', 'lag_12w', 'lag_26w', 'lag_52w',
    'rolling_mean_4w', 'rolling_mean_8w', 'rolling_mean_13w', 'rolling_mean_26w',
    'rolling_std_4w', 'rolling_std_8w', 'rolling_std_13w', 'rolling_std_26w',
    'rolling_vol_4w', 'rolling_vol_8w', 'rolling_vol_13w', 'rolling_vol_26w',
    'ret_1w', 'ret_4w', 'ret_13w', 'ret_26w',
    'week_sin', 'week_cos', 'month_sin', 'month_cos',
    'crop_year_progress', 'is_harvest_season',
]

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def load_model(series_key: str, horizon: int) -> tuple:
    """Load trained model and metadata."""
    model_path = os.path.join(ARTIFACTS_DIR, f'{series_key}_h{horizon}w.pkl')
    meta_path = os.path.join(ARTIFACTS_DIR, f'{series_key}_h{horizon}w_meta.json')

    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return model, meta


def prepare_features(features_dict: Dict[str, float]) -> np.ndarray:
    """Convert feature dictionary to numpy array in correct order."""
    feature_vector = []
    for name in FEATURE_NAMES:
        value = features_dict.get(name, 0.0)
        if value is None or np.isnan(value):
            value = 0.0
        feature_vector.append(value)
    return np.array([feature_vector])


def predict_horizon(model: Any, features: np.ndarray) -> float:
    """Generate prediction for a single horizon."""
    if model is None:
        return 0.0

    if isinstance(model, dict) and model.get('type') == 'linear':
        coeffs = np.array(model['coeffs'])
        return float(features @ coeffs)

    if hasattr(model, 'predict'):
        return float(model.predict(features)[0])

    return 0.0


def predict_all_horizons(series_key: str, features_dict: Dict[str, float]) -> Dict[str, Any]:
    """Generate predictions for all horizons."""
    features = prepare_features(features_dict)
    base_value = features_dict.get('baseValue', 4000)

    predictions = []

    for horizon in HORIZONS:
        model, meta = load_model(series_key, horizon)

        if model is None:
            predictions.append({
                'horizon': horizon,
                'predictedLogReturn': 0.0,
                'predictedPrice': base_value,
                'confidence': 0.0,
                'featureImportances': {},
                'error': 'Model not found',
            })
            continue

        log_return = predict_horizon(model, features)
        log_return = np.clip(log_return, -0.5, 0.5)

        predicted_price = base_value * np.exp(log_return)

        importances = meta.get('importances', {})
        metrics = meta.get('metrics', {})
        confidence = min(1.0, metrics.get('directional_accuracy', 0.5) * 1.5)

        predictions.append({
            'horizon': horizon,
            'predictedLogReturn': float(log_return),
            'predictedPrice': float(predicted_price),
            'confidence': float(confidence),
            'featureImportances': importances,
            'modelType': meta.get('model_type', 'unknown'),
            'trainedAt': meta.get('trained_at', ''),
        })

    return {
        'seriesKey': series_key,
        'predictedAt': datetime.now().isoformat(),
        'baseValue': base_value,
        'predictions': predictions,
    }


def main():
    """CLI interface for predictions."""
    if len(sys.argv) < 2:
        print("Usage: python predict_xgb.py <series_key> [features_json]")
        sys.exit(1)

    series_key = sys.argv[1]

    if len(sys.argv) > 2:
        features_dict = json.loads(sys.argv[2])
    else:
        features_dict = {'baseValue': 4000}

    result = predict_all_horizons(series_key, features_dict)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
