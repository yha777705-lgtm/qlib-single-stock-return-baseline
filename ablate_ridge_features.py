from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base = Path(__file__).resolve().parent
inputs = {
    'split_summary': base / 'reports/split_summary.json',
    'baseline_report': base / 'reports/baseline_validation.json',
    'train': base / 'data/splits/train.csv',
    'validation': base / 'data/splits/validation.csv',
}
meta = json.loads(inputs['split_summary'].read_text(encoding='utf-8'))
reference = json.loads(inputs['baseline_report'].read_text(encoding='utf-8'))
features = meta['features']
target = meta['target']
assert features == ['return_1d', 'return_5d', 'volume_ratio_5d']
assert reference['features'] == features
assert reference['evaluation_set'] == 'validation'
assert reference['test_evaluated'] is False
assert reference['ridge_alpha'] == 1.0

parts = {
    name: pd.read_csv(inputs[name], parse_dates=['feature_date', 'target_date'])
    for name in ('train', 'validation')
}
for name, part in parts.items():
    assert len(part) == meta['splits'][name]['rows']
    assert part['feature_date'].is_unique
    assert (part['target_date'] > part['feature_date']).all()
    assert np.isfinite(part[features + [target]].to_numpy(dtype=float)).all()
train, valid = parts['train'], parts['validation']
valid_start = pd.Timestamp(meta['validation_start'])
test_start = pd.Timestamp(meta['test_start'])
assert (train['feature_date'] < valid_start).all()
assert (train['target_date'] < valid_start).all()
assert (valid['feature_date'] >= valid_start).all()
assert (valid['feature_date'] < test_start).all()
assert (valid['target_date'] < test_start).all()
assert train['target_date'].max() < valid['feature_date'].min()

# All four feature configurations are fixed before fitting any model.
configurations = {'ALL_FEATURES': features.copy()}
configurations.update({
    'WITHOUT_' + removed.upper(): [name for name in features if name != removed]
    for removed in features
})


def scores(actual, predicted):
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    assert np.isfinite(error).all()
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
    }


zero = scores(valid[target], np.zeros(len(valid)))
assert zero['rmse'] > 0
for metric in ('mae', 'rmse'):
    assert np.isclose(zero[metric], reference['models']['ZERO'][metric], rtol=0, atol=1e-12)

results = {}
predictions = valid[['feature_date', 'target_date', target]].copy()
predictions = predictions.rename(columns={target: 'actual_return'})
for name, selected_features in configurations.items():
    model = make_pipeline(StandardScaler(), Ridge(alpha=reference['ridge_alpha']))
    # The scaler and regression are both fitted on training data only.
    model.fit(train[selected_features], train[target])
    predicted = model.predict(valid[selected_features])
    predictions[name] = predicted
    results[name] = {
        'features': selected_features,
        'removed_feature': next((f for f in features if f not in selected_features), None),
        'train_scores': scores(train[target], model.predict(train[selected_features])),
        'validation_scores': scores(valid[target], predicted),
    }

full_scores = results['ALL_FEATURES']['validation_scores']
for metric in ('mae', 'rmse'):
    assert np.isclose(full_scores[metric], reference['models']['RIDGE'][metric], rtol=0, atol=1e-12)
for result in results.values():
    rmse = result['validation_scores']['rmse']
    result['validation_rmse_gain_vs_zero_pct'] = (zero['rmse'] - rmse) / zero['rmse'] * 100
    result['validation_rmse_gain_vs_full_pct'] = (full_scores['rmse'] - rmse) / full_scores['rmse'] * 100

payload = {
    'experiment': 'fixed Ridge leave-one-feature-out validation comparison',
    'evaluation_set': 'validation',
    'test_evaluated': False,
    'new_training_performed': True,
    'hyperparameter_search_performed': False,
    'final_model_selected': False,
    'uncertainty_assessed': False,
    'training_rows': len(train),
    'validation_rows': len(valid),
    'ridge_alpha': reference['ridge_alpha'],
    'preprocessing': 'StandardScaler fitted on training rows separately for each configuration',
    'predefined_trial_count': len(configurations),
    'metric_units': 'fractional return; relative RMSE gains in percent',
    'interpretation': 'Exploratory validation comparison; not causal feature importance or proof of stable predictive value.',
    'input_sha256': {
        str(path.relative_to(base)).replace(chr(92), '/'): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs.values()
    },
    'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
    'zero_validation_scores': zero,
    'results': results,
}
report = base / 'reports/ridge_feature_ablation.json'
report.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
predictions.to_csv(base / 'reports/ridge_feature_ablation_predictions.csv', index=False)
print('CONFIGURATION  VALIDATION_MAE_PP  VALIDATION_RMSE_PP  RMSE_GAIN_VS_ZERO_PCT')
print(f'ZERO  {zero["mae"] * 100:.6f}  {zero["rmse"] * 100:.6f}  +0.0000%')
for name, result in results.items():
    score = result['validation_scores']
    gain = result['validation_rmse_gain_vs_zero_pct']
    print(f'{name}  {score["mae"] * 100:.6f}  {score["rmse"] * 100:.6f}  {gain:+.4f}%')
print('FULL_FEATURE_BASELINE_MATCH: True')
print('TEST_EVALUATED: False')
print('REPORT:', report)
