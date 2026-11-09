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
    'training_data': base / 'data/splits/train.csv',
    'split_metadata': base / 'reports/split_summary.json',
    'baseline_metadata': base / 'reports/baseline_validation.json',
}
meta = json.loads(inputs['split_metadata'].read_text(encoding='utf-8'))
reference = json.loads(inputs['baseline_metadata'].read_text(encoding='utf-8'))
features, target = meta['features'], meta['target']
assert features == ['return_1d', 'return_5d', 'volume_ratio_5d']
assert reference['features'] == features
assert reference['ridge_alpha'] == 1.0
data = pd.read_csv(inputs['training_data'], parse_dates=['feature_date', 'target_date'])
data = data.sort_values('feature_date').reset_index(drop=True)
assert len(data) == meta['splits']['train']['rows']
assert data['feature_date'].is_unique
assert (data['feature_date'] < pd.Timestamp(meta['validation_start'])).all()
assert (data['target_date'] < pd.Timestamp(meta['validation_start'])).all()
assert (data['target_date'] > data['feature_date']).all()
assert np.isfinite(data[features + [target]].to_numpy(dtype=float)).all()


def scores(actual, predicted):
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    assert np.isfinite(error).all()
    return {'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2)))}


folds, prediction_parts = {}, []
# Predefined expanding windows, entirely inside the original 2010-2016 training period.
for year in (2015, 2016):
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    train_candidates = data.loc[data['feature_date'] < start]
    valid_candidates = data.loc[(data['feature_date'] >= start) & (data['feature_date'] < end)]
    train = train_candidates.loc[train_candidates['target_date'] < start].copy()
    valid = valid_candidates.loc[valid_candidates['target_date'] < end].copy()
    assert not train.empty and not valid.empty
    assert train['target_date'].max() < valid['feature_date'].min()
    assert valid['target_date'].max() < end

    # Each fold creates and fits a new scaler and Ridge on that fold's training rows only.
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=reference['ridge_alpha']))
    pipeline.fit(train[features], train[target])
    scaler = pipeline.named_steps['standardscaler']
    assert int(scaler.n_samples_seen_) == len(train)
    np.testing.assert_allclose(
        scaler.mean_, train[features].to_numpy(dtype=float).mean(axis=0), rtol=1e-12, atol=1e-12,
    )
    predicted = pipeline.predict(valid[features])
    zero = scores(valid[target], np.zeros(len(valid)))
    ridge = scores(valid[target], predicted)
    assert zero['rmse'] > 0
    folds[str(year)] = {
        'train_rows': len(train),
        'train_feature_start': str(train['feature_date'].min().date()),
        'train_feature_end': str(train['feature_date'].max().date()),
        'last_training_target_date': str(train['target_date'].max().date()),
        'validation_rows': len(valid),
        'validation_feature_start': str(valid['feature_date'].min().date()),
        'validation_feature_end': str(valid['feature_date'].max().date()),
        'last_validation_target_date': str(valid['target_date'].max().date()),
        'training_boundary_rows_removed': len(train_candidates) - len(train),
        'validation_boundary_rows_removed': len(valid_candidates) - len(valid),
        'scaler_fit_rows': int(scaler.n_samples_seen_),
        'scaler_mean_by_feature': dict(zip(features, map(float, scaler.mean_))),
        'scaler_scale_by_feature': dict(zip(features, map(float, scaler.scale_))),
        'zero_scores': zero,
        'ridge_scores': ridge,
        'ridge_rmse_gain_vs_zero_pct': (zero['rmse'] - ridge['rmse']) / zero['rmse'] * 100,
    }
    part = valid[['feature_date', 'target_date', target]].rename(columns={target: 'actual_return'}).copy()
    part['validation_year'] = year
    part['ZERO'], part['RIDGE'] = 0.0, predicted
    prediction_parts.append(part)

predictions = pd.concat(prediction_parts, ignore_index=True)
assert predictions['feature_date'].is_unique
assert len(predictions) == sum(fold['validation_rows'] for fold in folds.values())
pooled_zero = scores(predictions['actual_return'], predictions['ZERO'])
pooled_ridge = scores(predictions['actual_return'], predictions['RIDGE'])
payload = {
    'experiment': 'two predefined expanding-window Ridge folds within original training period',
    'validation_years': [2015, 2016],
    'features': features,
    'ridge_alpha': reference['ridge_alpha'],
    'scaling': 'fresh StandardScaler per fold, fitted on that fold training rows only',
    'boundary_rule': 'purge training targets crossing the fold start and validation targets crossing the fold end',
    'new_training_performed': True,
    'hyperparameter_search_performed': False,
    'final_model_selected': False,
    'outer_2017_2018_validation_scored_in_this_run': False,
    'test_evaluated': False,
    'uncertainty_assessed': False,
    'interpretation': 'Exploratory historical training-period diagnostics; not an untouched final test or proof of stable superiority.',
    'metric_units': 'fractional return; relative RMSE gain in percent',
    'input_sha256': {
        str(path.relative_to(base)).replace(chr(92), '/'): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs.values()
    },
    'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
    'folds': folds,
    'pooled_out_of_fold': {
        'rows': len(predictions),
        'aggregation': 'score all concatenated non-overlapping fold predictions, not the mean of fold RMSE',
        'zero_scores': pooled_zero,
        'ridge_scores': pooled_ridge,
        'ridge_rmse_gain_vs_zero_pct': (pooled_zero['rmse'] - pooled_ridge['rmse']) / pooled_zero['rmse'] * 100,
    },
}
report = base / 'reports/ridge_time_validation.json'
report.write_text(json.dumps(payload, indent=2, allow_nan=False) + chr(10), encoding='utf-8')
predictions.to_csv(base / 'reports/ridge_time_validation_predictions.csv', index=False)
print('YEAR TRAIN_ROWS VALID_ROWS ZERO_RMSE_PP RIDGE_RMSE_PP RIDGE_GAIN_PCT')
for year, fold in folds.items():
    print(year, fold['train_rows'], fold['validation_rows'],
          format(fold['zero_scores']['rmse'] * 100, '.6f'),
          format(fold['ridge_scores']['rmse'] * 100, '.6f'),
          format(fold['ridge_rmse_gain_vs_zero_pct'], '+.4f') + '%')
print('POOLED', len(predictions),
      format(pooled_zero['rmse'] * 100, '.6f'),
      format(pooled_ridge['rmse'] * 100, '.6f'),
      format(payload['pooled_out_of_fold']['ridge_rmse_gain_vs_zero_pct'], '+.4f') + '%')
print('FOLD_SCALERS_TRAIN_ONLY: True')
print('TEST_EVALUATED: False')
print('REPORT:', report)
