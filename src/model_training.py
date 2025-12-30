# src/model_training.py
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr

def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]
    
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    return {"f1_score": f1, "confusion_matrix": cm, "auc_pr": pr_auc}
