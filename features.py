# features.py
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal

# 周波数帯域パワーを計算する関数
def bandpower(x, sf=256, band=(4, 8), window_sec=None):
    if window_sec is None:
        nperseg = min(256, len(x))
    else:
        nperseg = int(window_sec * sf)

    f, Pxx = signal.welch(x, sf, nperseg=nperseg)

    idx_band = np.logical_and(f >= band[0], f <= band[1])

    # NumPy のバージョン違い対応
    try:
        trapezoid_fn = np.trapezoid
    except AttributeError:
        trapezoid_fn = np.trapz

    return trapezoid_fn(Pxx[idx_band], f[idx_band])


# 1チャンネルの特徴量を抽出
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


# 1ファイルの特徴量をまとめる
def extract_features_for_file(file_path, sf=256):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 空白除去

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

    # 全チャンネル平均
    sensor_feats = [c for c in base.keys() if c.endswith('_mean') and c != 'mean']
    if sensor_feats:
        means = [base[c] for c in sensor_feats if np.isfinite(base[c])]
        if means:
            base['mean_of_channel_means'] = np.mean(means)

    return base


# ディレクトリ内すべての CSV を読み込み特徴量化
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

    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()  # 念のため空白除去
    return df