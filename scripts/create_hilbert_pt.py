"""
对 features_uni/pt_files 中的特征按 Hilbert 索引重排，
生成 Hilbert 排序后的 pt 文件到 hilbert_pt/ 目录。
用法: python scripts/create_hilbert_pt.py
"""
import os
import torch
from tqdm import tqdm

DATA_ROOT = "/home/a255372639/TCGA-LUAD/LUAD_example"
PT_DIR = os.path.join(DATA_ROOT, "features_uni", "pt_files")
HILBERT_IDX_DIR = os.path.join(DATA_ROOT, "hilbert")  # 索引文件 (int64)
OUT_DIR = os.path.join(DATA_ROOT, "hilbert_pt")        # 排序后的 pt 文件

os.makedirs(OUT_DIR, exist_ok=True)

pt_files = [f for f in os.listdir(PT_DIR) if f.endswith('.pt')]
print(f"共 {len(pt_files)} 个 pt 文件, 输出到 {OUT_DIR}")

skipped = 0
for fname in tqdm(pt_files, desc="Hilbert 排序"):
    slide_id = fname.replace('.pt', '')
    out_path = os.path.join(OUT_DIR, fname)
    
    if os.path.exists(out_path):
        skipped += 1
        continue
    
    pt_path = os.path.join(PT_DIR, fname)
    idx_path = os.path.join(HILBERT_IDX_DIR, f"{slide_id}_hilbert.pt")
    
    if not os.path.exists(idx_path):
        print(f"[WARNING] 无 Hilbert 索引: {slide_id}")
        continue
    
    features = torch.load(pt_path)           # [N, 1024]
    hilbert_idx = torch.load(idx_path).long() # [N]
    
    assert features.shape[0] == hilbert_idx.shape[0], \
        f"Shape mismatch: {slide_id} features={features.shape[0]} idx={hilbert_idx.shape[0]}"
    
    features_sorted = features[hilbert_idx]
    torch.save(features_sorted, out_path)

print(f"完成! 跳过 {skipped} 个已存在文件")
print(f"输出目录: {OUT_DIR}")
print(f"文件数: {len(os.listdir(OUT_DIR))}")
