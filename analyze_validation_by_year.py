from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
reports = base / 'reports'
keys = ['feature_date', 'target_date']
models = ['ZERO', 'TRAIN_MEAN', 'RIDGE', 'LIGHTGBM_FIXED']
input_names = [
    'split_summary.json', 'baseline_validation.json',
    'lightgbm_validation.json', 'baseline_validation_predictions.csv',
    'lightgbm_validation_predictions.csv',
]
meta, baseline_report, lgb_report = [
    json.loads((reports / name).read_text(encoding='utf-8'))
    for name in input_names[:3]
]
for report in (baseline_report, lgb_report):
    assert report['evaluation_set'] == 'validation'
    assert report['test_evaluated'] is False

baseline = pd.read_csv(reports / input_names[3], parse_dates=keys)
lightgbm = pd.read_csv(reports / input_names[4], parse_dates=keys)
assert not baseline[keys].duplicated().any()
assert not lightgbm[keys].duplicated().any()
lightgbm = lightgbm.rename(columns={
    'actual_return': 'lightgbm_actual', 'prediction': 'LIGHTGBM_FIXED',
})
data = baseline.merge(
    lightgbm, on=keys, how='outer', validate='one_to_one', indicator=True,
)
assert data['_merge'].eq('both').all(), 'Prediction dates do not align.'
assert len(data) == meta['splits']['validation']['rows']
assert not data.isna().any().any()
assert np.array_equal(data['actual_return'], data['lightgbm_actual'])
assert data['feature_date'].is_unique
assert data['feature_date'].ge(pd.Timestamp(meta['validation_start'])).all()
assert data['feature_date'].lt(pd.Timestamp(meta['test_start'])).all()
assert data['target_date'].lt(pd.Timestamp(meta['test_start'])).all()
assert data['target_date'].gt(data['feature_date']).all()
assert np.isfinite(data[['actual_return'] + models].to_numpy()).all()
assert data['ZERO'].eq(0.0).all()
data = data.sort_values(keys).reset_index(drop=True)

def summarize(part):
    actual = part['actual_return'].to_numpy(dtype=float)
    zero_rmse = float(np.sqrt(np.mean(actual ** 2)))
    assert zero_rmse > 0
    scores = {}
    for name in models:
        error = actual - part[name].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        scores[name] = {
            'mae': float(np.mean(np.abs(error))),
            'rmse': rmse,
            'rmse_gain_vs_zero_pct': (zero_rmse - rmse) / zero_rmse * 100,
        }
    return {
        'rows': len(part),
        'feature_start': str(part['feature_date'].min().date()),
        'feature_end': str(part['feature_date'].max().date()),
        'models': scores,
    }

overall = summarize(data)
for name in models:
    for metric in ('mae', 'rmse'):
        assert np.isclose(
            overall['models'][name][metric],
            lgb_report['validation_comparison'][name][metric],
            rtol=0, atol=1e-12,
        ), 'Overall metrics disagree with the existing report.'

by_year = {
    str(int(year)): summarize(part)
    for year, part in data.groupby(data['feature_date'].dt.year, sort=True)
}
assert sum(part['rows'] for part in by_year.values()) == len(data)
payload = {
    'evaluation_set': 'validation',
    'grouping': 'calendar year of feature_date',
    'new_training_performed': False,
    'model_selection_performed': False,
    'test_evaluated': False,
    'uncertainty_assessed': False,
    'metric_units': 'fractional return; relative RMSE gain in percent',
    'positive_gain_means': 'lower RMSE than ZERO on the same subset',
    'interpretation': 'Descriptive subgroup results; no significance or causal claim.',
    'input_sha256': {
        'reports/' + name: hashlib.sha256((reports / name).read_bytes()).hexdigest()
        for name in input_names
    },
    'overall': overall,
    'by_year': by_year,
}
output = reports / 'validation_by_year.json'
output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print('YEAR  ROWS  ZERO_RMSE_PP  LIGHTGBM_RMSE_PP  LIGHTGBM_GAIN_PCT')
for year, result in by_year.items():
    zero = result['models']['ZERO']['rmse'] * 100
    lgb = result['models']['LIGHTGBM_FIXED']
    rmse = lgb['rmse'] * 100
    gain = lgb['rmse_gain_vs_zero_pct']
    print(f'{year}  {result["rows"]:4d}  {zero:.6f}  {rmse:.6f}  {gain:+.4f}%')
print('OVERALL_METRICS_MATCH: True')
print('NEW_TRAINING_PERFORMED: False')
print('TEST_EVALUATED: False')
print('REPORT:', output)
