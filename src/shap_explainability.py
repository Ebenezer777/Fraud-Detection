# src/shap_explainability.py
import shap

def explain_model(model, X_test, y_test, shap_values=None):
    """
    Generate SHAP explanations:
    - Force plots for TP, FP, FN
    - Summary bar plot
    - Decision plot
    """
    # Initialize JS
    shap.initjs()
    
    # Compute SHAP values if not provided
    if shap_values is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.TreeExplainer(model)
    
    # Masks for TP, FP, FN
    y_pred = model.predict(X_test)
    tp_idx = X_test[(y_test==1) & (y_pred==1)].index
    fp_idx = X_test[(y_test==0) & (y_pred==1)].index
    fn_idx = X_test[(y_test==1) & (y_pred==0)].index
    
    # Force plots (first example from each)
    if len(tp_idx) > 0:
        display(shap.force_plot(explainer.expected_value, shap_values[tp_idx[0]], X_test.loc[tp_idx[0]]))
    if len(fp_idx) > 0:
        display(shap.force_plot(explainer.expected_value, shap_values[fp_idx[0]], X_test.loc[fp_idx[0]]))
    if len(fn_idx) > 0:
        display(shap.force_plot(explainer.expected_value, shap_values[fn_idx[0]], X_test.loc[fn_idx[0]]))
    
    # Summary and decision plots
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=10)
    shap.decision_plot(explainer.expected_value, shap_values[:10], X_test.iloc[:10])
