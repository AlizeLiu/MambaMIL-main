"""
Evaluation utilities for classification: ROC curves, confusion matrices, and artifact saving.
All visualization uses matplotlib only (no seaborn dependency).
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, \
    precision_score, recall_score, f1_score, balanced_accuracy_score


def compute_binary_roc(y_true, y_score):
    """Compute FPR, TPR, thresholds for binary ROC.
    
    Returns:
        fpr, tpr, thresholds (np arrays), auc (float, may be nan if single class)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Edge case: single class in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        warnings.warn(
            f"compute_binary_roc: y_true contains only one class ({unique_classes}). "
            f"AUC is undefined, returning nan.",
            UserWarning
        )
        nan_arr = np.array([0.0, 1.0])
        return nan_arr, nan_arr, np.array([0.0]), float('nan')
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, thresholds, auc


def plot_fold_roc(fpr, tpr, auc, fold_idx, save_path, title_prefix="Test"):
    """Plot ROC curve for a single fold."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, color='blue', lw=2,
            label=f'Fold {fold_idx} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title_prefix} ROC Curve - Fold {fold_idx}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_roc(fold_fprs, fold_tprs, fold_aucs, save_path, title_prefix="Test"):
    """Plot mean ROC curve across folds with individual fold curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    for i, (fpr, tpr, auc) in enumerate(zip(fold_fprs, fold_tprs, fold_aucs)):
        if np.isnan(auc):
            continue
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC = {auc:.4f})')
    
    valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
    if valid_aucs:
        mean_auc = np.mean(valid_aucs)
        std_auc = np.std(valid_aucs)
        all_fpr = np.unique(np.concatenate([fpr for fpr, auc in zip(fold_fprs, fold_aucs) if not np.isnan(auc)]))
        mean_tpr = np.zeros_like(all_fpr)
        valid_count = 0
        for fpr, tpr, auc in zip(fold_fprs, fold_tprs, fold_aucs):
            if np.isnan(auc):
                continue
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            valid_count += 1
        mean_tpr /= valid_count
        ax.plot(all_fpr, mean_tpr, color='blue', lw=2,
                label=f'Mean (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title_prefix} ROC Curves - Mean Across Folds')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_confusion_metrics(y_true, y_pred, class_names=None):
    """Compute confusion matrix and comprehensive medical classification metrics.
    
    For binary classification:
        sensitivity, specificity, precision, recall, F1, balanced_accuracy,
        support_pos, support_neg, confusion_matrix raw and normalized.
    
    For multiclass:
        macro_precision, macro_recall, macro_f1, balanced_accuracy, per_class_support.
    
    Returns:
        dict with all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        warnings.warn(
            f"compute_confusion_metrics: y_true contains only one class ({unique_true}). "
            f"Metrics may be ill-defined.",
            UserWarning
        )
    
    result = {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'labels': [int(l) for l in labels],
    }
    
    if len(labels) == 2:
        # Binary classification: detailed metrics
        # Assume labels[0]=negative, labels[1]=positive
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result.update({
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support_pos': int(tp + fn),
            'support_neg': int(tn + fp),
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
        })
    else:
        # Multiclass: macro averages
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        per_class_support = {}
        for i, label in enumerate(labels):
            per_class_support[f'class_{int(label)}'] = int(cm[i].sum())
        
        result.update({
            'macro_precision': float(macro_prec),
            'macro_recall': float(macro_rec),
            'macro_f1': float(macro_f1),
            'per_class_support': per_class_support,
        })
    
    return result


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix as annotated heatmap using matplotlib."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if class_names is None:
        class_names = [str(l) for l in labels]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_fold_eval_artifacts(results_dir, fold_idx, split_name, y_true, y_pred, y_prob,
                              class_names=None, plot_roc_flag=False, plot_confusion_flag=False,
                              slide_ids=None, case_ids=None, fold_num=None):
    """Save evaluation artifacts for a single fold and split (test or val).
    
    Args:
        results_dir: base results directory
        fold_idx: fold index
        split_name: 'test' or 'val'
        y_true: ground truth labels
        y_pred: predicted labels
        y_prob: predicted probabilities (shape [N, n_classes])
        class_names: optional list of class names
        plot_roc_flag: whether to save ROC plot
        plot_confusion_flag: whether to save confusion matrix plot
        slide_ids: optional list of slide IDs
        case_ids: optional list of case IDs
        fold_num: fold number for CSV column
    
    Returns:
        dict with metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    n_samples = len(y_true)
    
    artifact_dir = os.path.join(results_dir, 'eval_artifacts', f'fold_{fold_idx}')
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save predictions CSV with slide_id/case_id
    pred_data = {
        'slide_index': list(range(n_samples)),
    }
    
    # Add slide_id if available
    if slide_ids is not None:
        assert len(slide_ids) == n_samples, \
            f"slide_ids length ({len(slide_ids)}) != n_samples ({n_samples})"
        pred_data['slide_id'] = list(slide_ids)
    
    # Add case_id if available
    if case_ids is not None:
        assert len(case_ids) == n_samples, \
            f"case_ids length ({len(case_ids)}) != n_samples ({n_samples})"
        pred_data['case_id'] = list(case_ids)
    
    pred_data['label'] = [int(t) for t in y_true]
    pred_data['pred'] = [int(p) for p in y_pred]
    
    for c in range(n_classes):
        pred_data[f'prob_{c}'] = y_prob[:, c].tolist()
    
    pred_data['fold'] = [fold_num if fold_num is not None else fold_idx] * n_samples
    pred_data['split'] = [split_name] * n_samples
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = os.path.join(artifact_dir, f'{split_name}_predictions.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    
    # Compute metrics
    metrics = compute_confusion_metrics(y_true, y_pred, class_names=class_names)
    
    # ROC for binary classification
    if n_classes == 2:
        try:
            fpr, tpr, thresholds, auc = compute_binary_roc(y_true, y_prob[:, 1])
        except Exception as e:
            warnings.warn(f"AUC computation failed: {e}")
            fpr, tpr, thresholds, auc = np.array([0, 1]), np.array([0, 1]), np.array([0]), float('nan')
        metrics['auc'] = auc
        metrics['fpr'] = fpr.tolist() if isinstance(fpr, np.ndarray) else fpr
        metrics['tpr'] = tpr.tolist() if isinstance(tpr, np.ndarray) else tpr
        
        if plot_roc_flag and not np.isnan(auc):
            roc_path = os.path.join(artifact_dir, f'{split_name}_roc.png')
            plot_fold_roc(fpr, tpr, auc, fold_idx, roc_path,
                          title_prefix=f"{split_name.capitalize()}")
    else:
        metrics['auc'] = float('nan')
    
    # Confusion matrix plot
    if plot_confusion_flag:
        cm_path = os.path.join(artifact_dir, f'{split_name}_confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, class_names=class_names,
                              save_path=cm_path,
                              title=f'{split_name.capitalize()} Confusion Matrix - Fold {fold_idx}')
    
    # Save metrics JSON
    metrics_json = {k: v for k, v in metrics.items() if k not in ('fpr', 'tpr')}
    metrics_json_path = os.path.join(artifact_dir, f'{split_name}_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return metrics


def save_summary_metrics(fold_metrics_list, results_dir, split_name='test'):
    """Aggregate fold-level metrics into a summary CSV.
    
    Args:
        fold_metrics_list: list of metric dicts from save_fold_eval_artifacts
        results_dir: base results directory
        split_name: 'test' or 'val'
    """
    summary_dir = os.path.join(results_dir, 'eval_artifacts')
    os.makedirs(summary_dir, exist_ok=True)
    
    rows = []
    for fold_idx, m in enumerate(fold_metrics_list):
        row = {'fold': fold_idx}
        for k, v in m.items():
            if k in ('fpr', 'tpr', 'confusion_matrix', 'confusion_matrix_normalized', 
                     'labels', 'per_class_support'):
                continue
            row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add mean row
    numeric_cols = [c for c in df.columns if c != 'fold' and df[c].dtype in ['float64', 'int64']]
    mean_row = {'fold': 'mean'}
    std_row = {'fold': 'std'}
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        mean_row[col] = vals.mean()
        std_row[col] = vals.std()
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    csv_path = os.path.join(summary_dir, f'{split_name}_summary_metrics.csv')
    df.to_csv(csv_path, index=False)
    return csv_path
