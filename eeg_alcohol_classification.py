import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def bandpower(x, sf=256, band=(4, 8), window_sec=None):
    if window_sec is None:
        nperseg = min(256, len(x))
    else:
        nperseg = int(window_sec * sf)

    f, Pxx = signal.welch(x, sf, nperseg=nperseg)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    # numpy 2.4+ では trapz は trapezoid に置き換えられた
    return np.trapezoid(Pxx[idx_band], f[idx_band])


def extract_channel_features(data, sf=256):
    x = np.asarray(data, dtype=float)
    out = {
        "mean": np.nanmean(x),
        "std": np.nanstd(x),
        "min": np.nanmin(x),
        "max": np.nanmax(x),
        "ptp": np.nanmax(x) - np.nanmin(x),
        "skew": stats.skew(x, nan_policy='omit'),
        "kurtosis": stats.kurtosis(x, nan_policy='omit'),
        "median": np.nanmedian(x),
        "quartile_1": np.nanpercentile(x, 25),
        "quartile_3": np.nanpercentile(x, 75),
    }

    # 周波数バンド特徴量
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    total_power = bandpower(x, sf, (1, 45)) + 1e-9

    for name, b in bands.items():
        bp = bandpower(x, sf, b)
        out[f'{name}_power'] = bp
        out[f'{name}_rel_power'] = bp / total_power

    return out


def extract_features_for_file(file_path, sf=256):
    df = pd.read_csv(file_path)
    base = {
        'file': Path(file_path).name,
        'subject': df['subject identifier'].iloc[0],
        'label': 1 if df['subject identifier'].iloc[0] == 'a' else 0,
        'matching_condition': df['matching condition'].iloc[0],
        'trial_number': df['trial number'].iloc[0],
    }

    sensor_groups = df.groupby('sensor position')['sensor value']
    for sensor, series in sensor_groups:
        fdict = extract_channel_features(series.values, sf=sf)
        for k, v in fdict.items():
            base[f'{sensor}_{k}'] = v

    # 必要なら全センサー平均等
    sensor_feats = [c for c in base.keys() if c.endswith('_mean') and c != 'mean']
    if sensor_feats:
        means = [base[c] for c in sensor_feats if np.isfinite(base[c])]
        if means:
            base['mean_of_channel_means'] = np.mean(means)
    return base


def load_dataset(root_dir):
    files = sorted(glob.glob(os.path.join(root_dir, 'Data*.csv')))
    records = []
    for fp in files:
        try:
            rec = extract_features_for_file(fp)
        except Exception as e:
            print('warning', fp, e)
            continue
        records.append(rec)

    return pd.DataFrame(records)


def build_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    report_dict = classification_report(y_test, y_pred, target_names=['control', 'alcohol'], output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    roc = roc_auc_score(y_test, y_proba)

    print('--- RandomForest Test Results ---')
    print(classification_report(y_test, y_pred, target_names=['control', 'alcohol']))
    print('confusion_matrix:')
    print(conf_mat)
    print('roc_auc:', roc)

    # 交差検証
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = cv_scores.mean()
    print('5-fold cv accuracy:', cv_scores, cv_mean)

    # 特徴量重要度
    feature_importances = pipe.named_steps['clf'].feature_importances_
    ft = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

    print('\n--- Top 20 features ---')
    print(ft.head(20))

    metrics = {
        'test_report': report_dict,
        'confusion_matrix': conf_mat,
        'roc_auc': float(roc),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_mean),
    }

    return pipe, ft, metrics


def main():
    root_train = 'archive/SMNI_CMI_TRAIN'
    root_test = 'archive/SMNI_CMI_TEST'

    print('Loading training set from', root_train)
    df_train = load_dataset(root_train)
    print('Loaded', len(df_train), 'records.')

    print('統計値: label distribution')
    print(df_train['label'].value_counts())

    X = df_train.drop(columns=['file', 'subject', 'matching_condition', 'trial_number', 'label'])
    y = df_train['label']

    model, feature_importance, metrics = build_and_evaluate(X, y)

    output_dir = Path('output_results')
    output_dir.mkdir(exist_ok=True)

    df_train.to_csv(output_dir / 'train_features.csv', index=False)
    feature_importance.to_csv(output_dir / 'feature_importance.csv', header=['importance'])

    with open(output_dir / 'train_metrics.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print('\nSaved training outputs to', output_dir)

    if os.path.isdir(root_test):
        print('\nLoading test set from', root_test)
        df_test = load_dataset(root_test)
        if len(df_test) > 0:
            X_test = df_test.drop(columns=['file', 'subject', 'matching_condition', 'trial_number', 'label'])
            y_test = df_test['label']
            y_pred = model.predict(X_test)
            print('--- Test set results ---')
            print(classification_report(y_test, y_pred, target_names=['control', 'alcohol']))
            print('confusion_matrix:')
            print(confusion_matrix(y_test, y_pred))

            test_report = classification_report(y_test, y_pred, target_names=['control', 'alcohol'], output_dict=True)
            test_conf = confusion_matrix(y_test, y_pred).tolist()

            if 'metrics' in locals():
                metrics['test_report'] = test_report
                metrics['test_confusion_matrix'] = test_conf

            with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
                import json
                json.dump({'test_report': test_report, 'test_confusion_matrix': test_conf}, f, indent=2, ensure_ascii=False)

            df_test.to_csv(output_dir / 'test_features.csv', index=False)
            print('Saved test outputs to', output_dir)


if __name__ == '__main__':
    main()
