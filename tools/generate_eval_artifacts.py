#!/usr/bin/env python3
"""Generate eval artifacts from existing checkpoints without retraining."""
import os
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_utils import save_fold_eval_artifacts, save_summary_metrics, plot_mean_roc, compute_binary_roc


def load_model_and_predict(checkpoint_path, dataset, args):
    """Load model from checkpoint and run predictions on dataset."""
    from models.MambaMIL import MambaMIL
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = MambaMIL(
        in_dim=args.get('in_dim', 1024),
        n_classes=args.get('n_classes', 2),
        dropout=0.0,
        act='gelu',
        survival=False,
        hidden_dim=args.get('hidden_dim', 256),
        local_layers=args.get('local_layers', 1),
        global_layers=args.get('global_layers', 1),
        pool_size=args.get('pool_size', 50),
        use_atp_pool=not args.get('disable_atp_pool', False),
        pool_mode=args.get('pool_mode', 'avg'),
        diffusion_steps=args.get('diffusion_steps', 0),
        K_init=args.get('K_init', 2.5),
        atp_dt=args.get('atp_dt', 0.1),
        norm_type=args.get('norm_type', 'mean'),
        tau_init=args.get('tau_init', 2.0),
        gamma_init=args.get('gamma_init', 0.0),
        attn_type=args.get('attn_type', 'simple'),
        attn_dim=args.get('attn_dim', 128),
    )
    
    # Load checkpoint with key remapping
    state_dict = torch.load(checkpoint_path, map_location=device)
    remapped = {}
    needs_remap = False
    for k, v in state_dict.items():
        if k.startswith('attention.0.') or k.startswith('attention.2.'):
            remapped[k.replace('attention.', 'attention.attn.', 1)] = v
            needs_remap = True
        else:
            remapped[k] = v
    if needs_remap:
        state_dict = remapped
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Run predictions
    all_probs = []
    all_labels = []
    all_preds = []
    
    for idx in range(len(dataset)):
        data, label = dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.unsqueeze(0).to(device)
        with torch.no_grad():
            _, Y_prob, Y_hat, _, _ = model(data)
        all_probs.append(Y_prob.cpu().numpy().squeeze())
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        all_preds.append(Y_hat.item() if isinstance(Y_hat, torch.Tensor) else Y_hat)
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--split_dir', required=True)
    parser.add_argument('--data_root_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--backbone', default='uni')
    parser.add_argument('--patch_size', default='512')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--attn_type', default='simple')
    parser.add_argument('--attn_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--pool_mode', default='avg')
    parser.add_argument('--max_seq_len', type=int, default=999999)
    parser.add_argument('--feature_subdir', default='pt_files')
    parser.add_argument('--features_already_hilbert', action='store_true')
    parser.add_argument('--use_hilbert_index', action='store_true')
    parser.add_argument('--sampling_mode', default='random_points')
    parser.add_argument('--order_mode', default='keep')
    parser.add_argument('--disable_atp_pool', action='store_true')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'eval_artifacts')
    
    from dataset.dataset_generic import Generic_MIL_Dataset
    
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=1,
        print_info=True,
        label_dict={'LUAD': 0, 'LUSC': 1},
        patient_strat=False,
        ignore=[],
    )
    
    # Load split
    split_csv = os.path.join(args.split_dir, f'splits_{args.fold}.csv')
    dataset.update_split(split_csv)
    
    model_args = {
        'in_dim': 1024, 'n_classes': 2,
        'hidden_dim': args.hidden_dim, 'local_layers': 1, 'global_layers': 1,
        'pool_size': args.pool_size, 'pool_mode': args.pool_mode,
        'diffusion_steps': 0, 'K_init': 2.5, 'atp_dt': 0.1, 'norm_type': 'mean',
        'tau_init': 2.0, 'gamma_init': 0.0,
        'attn_type': args.attn_type, 'attn_dim': args.attn_dim,
        'disable_atp_pool': args.disable_atp_pool,
    }
    
    # Get slide info
    slide_ids = dataset.slide_data['slide_id'].tolist()
    case_ids = dataset.slide_data.get('case_id', pd.Series(['']*len(slide_ids))).tolist()
    
    # Run predictions on test split
    print(f"Running predictions on fold {args.fold} test split...")
    y_true, y_pred, y_prob = load_model_and_predict(
        args.checkpoint, dataset, model_args
    )
    
    # Save artifacts
    metrics = save_fold_eval_artifacts(
        args.output_dir, args.fold, 'test',
        y_true, y_pred, y_prob,
        class_names=['LUAD', 'LUSC'],
        plot_roc_flag=True, plot_confusion_flag=True,
        slide_ids=slide_ids[:len(y_true)],
        case_ids=case_ids[:len(y_true)],
        fold_num=args.fold,
    )
    
    print(f"\nResults:")
    print(f"  AUC: {metrics.get('auc', 'N/A')}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  Sensitivity: {metrics.get('sensitivity', 'N/A')}")
    print(f"  Specificity: {metrics.get('specificity', 'N/A')}")
    print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A')}")
    print(f"  F1: {metrics.get('f1', 'N/A')}")
    print(f"\nArtifacts saved to: {args.output_dir}/fold_{args.fold}/")


if __name__ == '__main__':
    main()
