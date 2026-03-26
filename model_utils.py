# model_utils.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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