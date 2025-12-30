import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc

def train_logistic_regression(X_train, y_train):
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        return lr
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
        raise

def train_xgboost(X_train, y_train, n_estimators=200, max_depth=5, learning_rate=0.1):
    try:
        scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:,1]
        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(y_test, y_pred)
        return {
            'f1_score': f1,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise
