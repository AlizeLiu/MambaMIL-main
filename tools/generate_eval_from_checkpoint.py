#!/usr/bin/env python3
"""
Generate eval artifacts from existing checkpoints.
Loads checkpoint, runs inference on test/val splits, saves ROC/confusion/metrics.
"""
import os, sys, argparse, json, warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model(args):
    from models.MambaMIL import MambaMIL
    return MambaMIL(
        in_dim=args.in_dim, n_classes=args.n_classes,
        dropout=0.0, act='gelu', survival=False,
        hidden_dim=args.hidden_dim,
        local_layers=args.local_layers, global_layers=args.global_layers,
        pool_size=args.pool_size, use_atp_pool=not args.disable_atp_pool,
        pool_mode=args.pool_mode, diffusion_steps=args.diffusion_steps,
        K_init=args.K_init, atp_dt=args.atp_dt, norm_type=args.norm_type,
        tau_init=args.tau_init, gamma_init=args.gamma_init,
        attn_type=args.attn_type, attn_dim=args.attn_dim,
    )


def load_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('attention.0.') or k.startswith('attention.2.'):
            remapped[k.replace('attention.', 'attention.attn.', 1)] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=True)
    return model


def run_inference(model, dataset, device):
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    slide_ids, case_ids = [], []
    
    for idx in range(len(dataset)):
        data, label = dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.unsqueeze(0).to(device)
        with torch.no_grad():
            _, Y_prob, Y_hat, _, _ = model(data)
        all_probs.append(Y_prob.cpu().numpy().squeeze())
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else int(label))
        all_preds.append(Y_hat.item() if isinstance(Y_hat, torch.Tensor) else int(Y_hat))
        # Get slide/case id from dataset
        sid = dataset.slide_data.iloc[idx].get('slide_id', f'slide_{idx}')
        cid = dataset.slide_data.iloc[idx].get('case_id', '')
        slide_ids.append(sid)
        case_ids.append(cid)
    
    return (np.array(all_labels), np.array(all_preds), np.array(all_probs),
            slide_ids, case_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--split_dir', required=True)
    parser.add_argument('--data_root_dir', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--backbone', default='uni')
    parser.add_argument('--patch_size', default='512')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=1)
    parser.add_argument('--global_layers', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--pool_mode', default='avg')
    parser.add_argument('--diffusion_steps', type=int, default=0)
    parser.add_argument('--K_init', type=float, default=2.5)
    parser.add_argument('--atp_dt', type=float, default=0.1)
    parser.add_argument('--norm_type', default='mean')
    parser.add_argument('--tau_init', type=float, default=2.0)
    parser.add_argument('--gamma_init', type=float, default=0.0)
    parser.add_argument('--attn_type', default='simple')
    parser.add_argument('--attn_dim', type=int, default=128)
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--disable_atp_pool', action='store_true')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build and load model
    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    print(f"Model loaded from {args.checkpoint}")
    
    # Load dataset
    from dataset.dataset_generic import Generic_MIL_Dataset
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path, data_dir=args.data_root_dir,
        shuffle=False, seed=1, print_info=False,
        label_dict={'LUAD': 0, 'LUSC': 1},
        patient_strat=False, ignore=[],
    )
    
    # Get splits using return_splits
    split_csv = os.path.join(args.split_dir, f'splits_{args.fold}.csv')
    _, _, test_dataset = dataset.return_splits(
        backbone=args.backbone, patch_size=args.patch_size,
        from_id=False, csv_path=split_csv
    )
    
    print(f"Test split: {len(test_dataset)} samples")
    
    # Run inference on test split
    y_true, y_pred, y_prob, slide_ids, case_ids = run_inference(model, test_dataset, device)
    print(f"  {len(y_true)} samples, LUAD={sum(y_true==0)}, LUSC={sum(y_true==1)}")
    
    # Save artifacts
    from utils.eval_utils import save_fold_eval_artifacts
    metrics = save_fold_eval_artifacts(
        args.output_dir, args.fold, 'test',
        y_true, y_pred, y_prob,
        class_names=['LUAD', 'LUSC'],
        plot_roc_flag=True, plot_confusion_flag=True,
        slide_ids=slide_ids, case_ids=case_ids, fold_num=args.fold,
    )
    
    print(f"\nMetrics:")
    for k in ['auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1', 'precision', 'recall']:
        v = metrics.get(k, 'N/A')
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSaved to: {args.output_dir}/fold_{args.fold}/")


if __name__ == '__main__':
    main()
