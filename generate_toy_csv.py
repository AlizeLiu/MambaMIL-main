import os
import pandas as pd
import numpy as np

# ==========================================
#  ！！！流程测试脚本！！！ 不可用于正式数据集生成，仅供流程测试使用！！！
# ==========================================
# 指向存放特征的父目录
DATA_ROOT = r"J:\Work\CLAM-master\toy_test\feature_univ1"
PT_DIR = os.path.join(DATA_ROOT, "pt_files")

# MambaMIL 要求的输出路径
CSV_SAVE_PATH = "dataset_csv/toy_survival.csv"
SPLIT_SAVE_DIR = "splits/toy_split"
SPLIT_SAVE_PATH = os.path.join(SPLIT_SAVE_DIR, "splits_0.csv")


# ==========================================

def generate_toy_dataset():
    # 1. 创建输出的文件夹目录
    os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)

    # 2. 自动扫描文件夹，获取所有特征文件名
    if not os.path.exists(PT_DIR):
        print(f"[Error] 找不到特征文件夹: {PT_DIR}")
        return

    pt_files = [f for f in os.listdir(PT_DIR) if f.endswith('.pt')]
    slide_ids = [f.replace('.pt', '') for f in pt_files]

    if len(slide_ids) == 0:
        print(f"[Error] {PT_DIR} 目录下没有找到任何 .pt 文件！")
        return

    print(f"🔍 扫描完毕！共找到 {len(slide_ids)} 个切片特征: {slide_ids}")

    # 3. 构造伪造的临床生存期数据 (CSV)
    np.random.seed(42)  # 固定随机种子，保证每次生成的存活期一样
    df_dataset = pd.DataFrame({
        'case_id': [f"patient_{i + 1}" for i in range(len(slide_ids))],
        'slide_id': slide_ids,
        # 随机生成 5.0 到 60.0 个月的生存期
        'survival_months': np.random.uniform(5.0, 60.0, len(slide_ids)).round(1),
        # 随机生成是否删失 (0或1)
        'censorship': np.random.choice([0, 1], len(slide_ids))
    })

    df_dataset.to_csv(CSV_SAVE_PATH, index=False)
    print(f"✅ 成功生成数据集表格: {CSV_SAVE_PATH}")

    # 4. 构造交叉验证划分文件 (splits_0.csv)
    # 💡 核心技巧：因为是 Toy Test 只有几张图，为了防止验证集为空报错，
    # 强行把所有图都放进 train, val, test 中保证连通性测试顺利通过。
    df_split = pd.DataFrame({
        'train': pd.Series(slide_ids),
        'val': pd.Series(slide_ids),
        'test': pd.Series(slide_ids)
    })

    df_split.to_csv(SPLIT_SAVE_PATH, index=False)
    print(f"✅ 成功生成数据划分表格: {SPLIT_SAVE_PATH}")
    print("\n🎉 准备就绪！请运行以下命令启动训练测试：")
    print(
        f'python main_survival.py --data_root_dir "{DATA_ROOT}" --csv_path "{CSV_SAVE_PATH}" --split_dir "{SPLIT_SAVE_DIR}" --model_type mamba_mil --max_epochs 2')


if __name__ == "__main__":
    generate_toy_dataset()