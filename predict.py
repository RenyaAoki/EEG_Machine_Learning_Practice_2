# predict.py
import joblib
import pandas as pd
from features import extract_features_for_file, load_dataset

# 学習済みモデルを読み込む
model = joblib.load('model.pkl')

# 新しいデータが入っているディレクトリ
new_data_dir = 'archive/SMNI_CMI_NEW'  # ← 新しいデータの場所に変更してください

# ディレクトリ内の CSV を特徴量化
df_new = load_dataset(new_data_dir)

# 学習時と同じ列だけ使う（余計な列は除外）
feature_columns = [c for c in df_new.columns if c not in ['file','subject','matching_condition','trial_number','label']]
X_new = df_new[feature_columns]

# 予測
y_pred = model.predict(X_new)
y_prob = model.predict_proba(X_new)[:, 1]  # 1: アルコールラベルの確率

# 結果を DataFrame にまとめる
df_result = df_new[['file', 'subject', 'matching_condition', 'trial_number']].copy()
df_result['pred_label'] = y_pred
df_result['pred_prob'] = y_prob

# CSV に保存
output_file = 'predicted_results.csv'
df_result.to_csv(output_file, index=False)
print(f'予測完了、結果を {output_file} に保存しました')
