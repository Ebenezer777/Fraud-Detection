import shap
from IPython.display import display

def plot_shap_values(explainer, shap_values, X_test, y_test, y_pred):
    try:
        shap.initjs()
        # Masks for True Positive, False Positive, False Negative
        tp_mask = (y_test==1) & (y_pred==1)
        fp_mask = (y_test==0) & (y_pred==1)
        fn_mask = (y_test==1) & (y_pred==0)

        for mask, label in zip([tp_mask, fp_mask, fn_mask], ['TP', 'FP', 'FN']):
            if mask.any():
                pos = mask[mask].index[0]
                display(shap.force_plot(explainer.expected_value, shap_values[pos], X_test.iloc[pos,:], matplotlib=False))
    except Exception as e:
        print(f"Error plotting SHAP values: {e}")
        raise

def shap_summary_plot(shap_values, X_test):
    try:
        shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=10)
    except Exception as e:
        print(f"Error generating SHAP summary plot: {e}")
        raise
