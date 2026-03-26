# train.py
import joblib
from features import load_dataset
from model_utils import build_and_evaluate

root_train = 'archive/SMNI_CMI_TRAIN'
df_train = load_dataset(root_train)

# 列名の空白除去
df_train.columns = df_train.columns.str.strip()

X = df_train.drop(columns=['file','subject','matching_condition','trial_number','label'])
y = df_train['label']

model, feature_importance, metrics = build_and_evaluate(X, y)

joblib.dump(model, 'model.pkl')
print("学習完了、model.pklに保存されました")