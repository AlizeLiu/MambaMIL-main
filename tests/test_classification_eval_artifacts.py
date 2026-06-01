"""
Unit tests for classification eval artifacts (ROC, confusion matrix, predictions).
Run: python tests/test_classification_eval_artifacts.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import warnings

from utils.eval_utils import (
    compute_binary_roc, plot_fold_roc, plot_mean_roc,
    compute_confusion_metrics, plot_confusion_matrix,
    save_fold_eval_artifacts,
)


def test_compute_binary_roc_basic():
    """Test basic ROC computation with synthetic data."""
    print("=== compute_binary_roc Basic Test ===")
    
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
    
    fpr, tpr, thresholds, auc = compute_binary_roc(y_true, y_score)
    
    assert len(fpr) > 0, "FPR should be non-empty"
    assert len(tpr) > 0, "TPR should be non-empty"
    assert len(fpr) == len(tpr), "FPR and TPR should have same length"
    assert 0.0 <= auc <= 1.0, f"AUC should be in [0,1], got {auc}"
    assert auc > 0.9, f"Perfect separation should give AUC > 0.9, got {auc}"
    
    print(f"  AUC = {auc:.4f} ✓")
    print("  PASSED ✓\n")


def test_compute_binary_roc_single_class():
    """Test ROC with single-class y_true returns nan with warning."""
    print("=== compute_binary_roc Single Class Test ===")
    
    # All class 0
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fpr, tpr, thresholds, auc = compute_binary_roc(y_true, y_score)
        
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert "single class" in str(w[0].message).lower() or "one class" in str(w[0].message).lower()
    
    assert np.isnan(auc), f"AUC should be nan for single class, got {auc}"
    
    print(f"  AUC = {auc} (expected nan) ✓")
    print("  PASSED ✓\n")


def test_plot_fold_roc():
    """Test that fold ROC plot is saved."""
    print("=== plot_fold_roc Test ===")
    
    tmpdir = tempfile.mkdtemp()
    try:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
        fpr, tpr, _, auc = compute_binary_roc(y_true, y_score)
        
        save_path = os.path.join(tmpdir, 'fold_0_roc.png')
        plot_fold_roc(fpr, tpr, auc, 0, save_path)
        
        assert os.path.exists(save_path), f"ROC plot not saved: {save_path}"
        assert os.path.getsize(save_path) > 0, "ROC plot is empty"
        
        print(f"  Saved: {save_path} ({os.path.getsize(save_path)} bytes) ✓")
    finally:
        shutil.rmtree(tmpdir)
    
    print("  PASSED ✓\n")


def test_plot_mean_roc():
    """Test mean ROC across multiple folds."""
    print("=== plot_mean_roc Test ===")
    
    tmpdir = tempfile.mkdtemp()
    try:
        fold_fprs = []
        fold_tprs = []
        fold_aucs = []
        
        for fold in range(5):
            np.random.seed(fold)
            y_true = np.random.randint(0, 2, 20)
            y_score = np.random.rand(20)
            fpr, tpr, _, auc = compute_binary_roc(y_true, y_score)
            fold_fprs.append(fpr)
            fold_tprs.append(tpr)
            fold_aucs.append(auc)
        
        save_path = os.path.join(tmpdir, 'mean_roc.png')
        plot_mean_roc(fold_fprs, fold_tprs, fold_aucs, save_path)
        
        assert os.path.exists(save_path), f"Mean ROC not saved: {save_path}"
        assert os.path.getsize(save_path) > 0, "Mean ROC is empty"
        
        print(f"  Saved: {save_path} ({os.path.getsize(save_path)} bytes) ✓")
    finally:
        shutil.rmtree(tmpdir)
    
    print("  PASSED ✓\n")


def test_compute_confusion_metrics():
    """Test confusion metrics computation."""
    print("=== compute_confusion_metrics Test ===")
    
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    
    metrics = compute_confusion_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics, "Missing 'accuracy'"
    assert 'precision' in metrics, "Missing 'precision'"
    assert 'recall' in metrics, "Missing 'recall'"
    assert 'f1' in metrics, "Missing 'f1'"
    assert 'confusion_matrix' in metrics, "Missing 'confusion_matrix'"
    assert 'labels' in metrics, "Missing 'labels'"
    
    # accuracy: 4/6 correct
    assert abs(metrics['accuracy'] - 4/6) < 1e-6, f"Accuracy should be ~0.6667, got {metrics['accuracy']}"
    
    # confusion matrix should be 2x2
    cm = np.array(metrics['confusion_matrix'])
    assert cm.shape == (2, 2), f"CM shape should be (2,2), got {cm.shape}"
    
    print(f"  Accuracy = {metrics['accuracy']:.4f}")
    print(f"  Precision = {metrics['precision']:.4f}")
    print(f"  Recall = {metrics['recall']:.4f}")
    print(f"  F1 = {metrics['f1']:.4f}")
    print(f"  CM = {metrics['confusion_matrix']}")
    print("  PASSED ✓\n")


def test_compute_confusion_metrics_single_class():
    """Test confusion metrics with single-class y_true."""
    print("=== compute_confusion_metrics Single Class Test ===")
    
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = compute_confusion_metrics(y_true, y_pred)
        
        assert len(w) >= 1, f"Expected at least 1 warning, got {len(w)}"
    
    assert metrics['accuracy'] == 1.0, f"Accuracy should be 1.0, got {metrics['accuracy']}"
    
    print(f"  Accuracy = {metrics['accuracy']:.4f} ✓")
    print("  PASSED ✓\n")


def test_plot_confusion_matrix():
    """Test confusion matrix plot."""
    print("=== plot_confusion_matrix Test ===")
    
    tmpdir = tempfile.mkdtemp()
    try:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        
        save_path = os.path.join(tmpdir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, class_names=['LUAD', 'LUSC'],
                              save_path=save_path)
        
        assert os.path.exists(save_path), f"CM plot not saved: {save_path}"
        assert os.path.getsize(save_path) > 0, "CM plot is empty"
        
        print(f"  Saved: {save_path} ({os.path.getsize(save_path)} bytes) ✓")
    finally:
        shutil.rmtree(tmpdir)
    
    print("  PASSED ✓\n")


def test_save_fold_eval_artifacts():
    """Test full eval artifact saving pipeline."""
    print("=== save_fold_eval_artifacts Test ===")
    
    tmpdir = tempfile.mkdtemp()
    try:
        np.random.seed(42)
        n = 20
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_prob = np.random.rand(n, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # normalize
        
        metrics = save_fold_eval_artifacts(
            tmpdir, 0, 'test', y_true, y_pred, y_prob,
            plot_roc_flag=True, plot_confusion_flag=True,
        )
        
        fold_dir = os.path.join(tmpdir, 'eval_artifacts', 'fold_0')
        
        # Check predictions CSV
        pred_csv = os.path.join(fold_dir, 'test_predictions.csv')
        assert os.path.exists(pred_csv), f"Missing: {pred_csv}"
        pred_df = pd.read_csv(pred_csv)
        assert len(pred_df) == n, f"Expected {n} rows, got {len(pred_df)}"
        assert 'y_true' in pred_df.columns
        assert 'y_pred' in pred_df.columns
        assert 'prob_class_0' in pred_df.columns
        assert 'prob_class_1' in pred_df.columns
        
        # Check metrics JSON
        metrics_json = os.path.join(fold_dir, 'test_metrics.json')
        assert os.path.exists(metrics_json), f"Missing: {metrics_json}"
        with open(metrics_json) as f:
            loaded_metrics = json.load(f)
        assert 'accuracy' in loaded_metrics
        assert 'auc' in loaded_metrics
        
        # Check ROC plot
        roc_path = os.path.join(fold_dir, 'test_roc.png')
        assert os.path.exists(roc_path), f"Missing: {roc_path}"
        
        # Check confusion matrix plot
        cm_path = os.path.join(fold_dir, 'test_confusion_matrix.png')
        assert os.path.exists(cm_path), f"Missing: {cm_path}"
        
        print(f"  All artifacts saved in: {fold_dir}")
        print(f"  Predictions CSV: {os.path.getsize(pred_csv)} bytes")
        print(f"  Metrics JSON: {os.path.getsize(metrics_json)} bytes")
        print(f"  ROC plot: {os.path.getsize(roc_path)} bytes")
        print(f"  CM plot: {os.path.getsize(cm_path)} bytes")
        print(f"  AUC = {loaded_metrics['auc']:.4f}")
        print("  PASSED ✓\n")
    finally:
        shutil.rmtree(tmpdir)


def test_save_fold_eval_artifacts_no_plots():
    """Test eval artifact saving without plots."""
    print("=== save_fold_eval_artifacts (no plots) Test ===")
    
    tmpdir = tempfile.mkdtemp()
    try:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
        
        metrics = save_fold_eval_artifacts(
            tmpdir, 0, 'val', y_true, y_pred, y_prob,
            plot_roc_flag=False, plot_confusion_flag=False,
        )
        
        fold_dir = os.path.join(tmpdir, 'eval_artifacts', 'fold_0')
        
        # CSV and JSON should exist
        assert os.path.exists(os.path.join(fold_dir, 'val_predictions.csv'))
        assert os.path.exists(os.path.join(fold_dir, 'val_metrics.json'))
        
        # Plots should NOT exist
        assert not os.path.exists(os.path.join(fold_dir, 'val_roc.png'))
        assert not os.path.exists(os.path.join(fold_dir, 'val_confusion_matrix.png'))
        
        print("  CSV + JSON saved, no plots ✓")
        print("  PASSED ✓\n")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("=" * 60)
    print("Classification Eval Artifacts Unit Tests")
    print("=" * 60)
    print()
    
    test_compute_binary_roc_basic()
    test_compute_binary_roc_single_class()
    test_plot_fold_roc()
    test_plot_mean_roc()
    test_compute_confusion_metrics()
    test_compute_confusion_metrics_single_class()
    test_plot_confusion_matrix()
    test_save_fold_eval_artifacts()
    test_save_fold_eval_artifacts_no_plots()
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
