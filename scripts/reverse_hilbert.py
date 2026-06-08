#!/usr/bin/env python3
"""将 Hilbert 排序的特征反排序为原始顺序"""
import torch
import os
import numpy as np

src_dir = "/home/a255372639/LUSC_LUAD/ihg_data/pt_files/uni"
idx_dir = "/home/a255372639/LUSC_LUAD/SCAD_example/hilbert"
dst_dir = "/home/a255372639/LUSC_LUAD/ihg_data/pt_files_raw_order"

os.makedirs(dst_dir, exist_ok=True)

files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pt')])
print(f"处理 {len(files)} 个文件...")

done = 0
skipped = 0
for fname in files:
    slide_id = fname.replace('.pt', '')
    idx_file = os.path.join(idx_dir, f"{slide_id}_hilbert.pt")
    
    if not os.path.exists(idx_file):
        skipped += 1
        continue
    
    feat_hilbert = torch.load(os.path.join(src_dir, fname), map_location='cpu')
    hilbert_idx = torch.load(idx_file, map_location='cpu')
    
    if isinstance(hilbert_idx, torch.Tensor):
        hilbert_idx = hilbert_idx.numpy()
    hilbert_idx = hilbert_idx.astype(int)
    
    # 反排序
    reverse_idx = np.argsort(hilbert_idx)
    feat_raw = feat_hilbert[reverse_idx]
    
    torch.save(feat_raw, os.path.join(dst_dir, fname))
    done += 1
    
    if done % 200 == 0:
        print(f"  {done}/{len(files)}")

print(f"\n完成! 成功={done}, 跳过={skipped}")
print(f"保存在: {dst_dir}")
print(f"文件数: {len(os.listdir(dst_dir))}")
