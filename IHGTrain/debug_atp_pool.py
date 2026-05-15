"""
ATP-Pool 诊断脚本：检查扩散系数是否退化
用真实数据或模拟数据跑一遍 ATPPool，打印关键统计量
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.AtpPool import ATPPool

def diagnose_atp_pool():
    print("=" * 60)
    print("ATP-Pool 扩散诊断")
    print("=" * 60)
    
    # ========== 1. 模拟数据测试 ==========
    print("\n--- 测试 1: 模拟数据 (hidden_dim=256) ---")
    
    for K_init in [0.5, 1.0, 2.0, 5.0, 10.0]:
        atp = ATPPool(dim=256, pool_size=100, K_init=K_init, diffusion_steps=2, dt=0.1)
        
        # 模拟 Mamba 输出：batch=1, seq_len=1000, hidden_dim=256
        # 加入局部相关性（模拟真实 WSI 特征）
        x = torch.randn(1, 1000, 256)
        # 让相邻 token 有一定相似性（加入平滑）
        for i in range(1, 1000):
            x[0, i] = x[0, i] * 0.3 + x[0, i-1] * 0.7
        
        # Hook 来捕获中间值
        captured = {}
        original_fn = atp._perona_malik_diffusion_1d
        
        def hook_fn(x_in, captured=captured):
            for step_idx in range(atp.diffusion_steps):
                grad_right = torch.zeros_like(x_in)
                grad_right[:, :-1, :] = x_in[:, 1:, :] - x_in[:, :-1, :]
                
                grad_left = torch.zeros_like(x_in)
                grad_left[:, 1:, :] = x_in[:, :-1, :] - x_in[:, 1:, :]
                
                norm_right_sq = torch.mean(grad_right ** 2, dim=-1, keepdim=True)
                norm_left_sq = torch.mean(grad_left ** 2, dim=-1, keepdim=True)
                
                K = torch.clamp(atp.K, min=1e-3)
                c_right = torch.exp(-norm_right_sq / (K ** 2 + 1e-6))
                c_left = torch.exp(-norm_left_sq / (K ** 2 + 1e-6))
                
                captured['norm_right_sq'] = norm_right_sq
                captured['c_right'] = c_right
                captured['K'] = K
                
                x_in = x_in + atp.dt * (c_right * grad_right + c_left * grad_left)
            return x_in
        
        # 手动跑一次
        with torch.no_grad():
            hook_fn(x)
        
        c = captured['c_right']
        n = captured['norm_right_sq']
        K_val = captured['K'].item()
        
        print(f"\n  K_init={K_init:.1f} (learned K={K_val:.4f}):")
        print(f"    norm_right_sq:  mean={n.mean().item():.4f}  max={n.max().item():.4f}  min={n.min().item():.4f}")
        print(f"    c_right:        mean={c.mean().item():.6f}  min={c.min().item():.8f}  max={c.max().item():.6f}")
        print(f"    c_right < 1e-4: {(c < 1e-4).float().mean().item():.4f} (比例)")
        print(f"    c_right > 0.9:  {(c > 0.9).float().mean().item():.4f} (比例)")
        
        if c.mean().item() < 0.01:
            print(f"    ⚠️  警告: c_right ≈ 0，ATP 扩散几乎不起作用，退化为 avg_pool!")
        elif c.mean().item() > 0.95:
            print(f"    ⚠️  警告: c_right ≈ 1，ATP 过度平滑，边界保护失效!")
        else:
            print(f"    ✅ ATP 扩散正常工作")
    
    # ========== 2. 不同 hidden_dim 的影响 ==========
    print("\n\n--- 测试 2: 不同 hidden_dim 的影响 (K_init=1.0) ---")
    
    for hdim in [64, 128, 256, 512]:
        atp = ATPPool(dim=hdim, pool_size=100, K_init=1.0, diffusion_steps=2, dt=0.1)
        x = torch.randn(1, 1000, hdim)
        for i in range(1, 1000):
            x[0, i] = x[0, i] * 0.3 + x[0, i-1] * 0.7
        
        with torch.no_grad():
            grad_right = torch.zeros_like(x)
            grad_right[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
            norm_right_sq = torch.mean(grad_right ** 2, dim=-1, keepdim=True)
            K = torch.clamp(atp.K, min=1e-3)
            c_right = torch.exp(-norm_right_sq / (K ** 2 + 1e-6))
        
        print(f"  hidden_dim={hdim:>3d}:  norm_sq mean={norm_right_sq.mean().item():.4f}  "
              f"c_right mean={c.mean().item():.6f}  "
              f"ratio<1e-4={(c_right < 1e-4).float().mean().item():.4f}")
    
    # ========== 3. K 与 hidden_dim 的关系 ==========
    print("\n\n--- 测试 3: K 应该随 hidden_dim 缩放 ---")
    print("  理论: norm_sq ≈ hidden_dim * per_dim_var")
    print("  要让 c = exp(-norm_sq/K^2) 有意义, K^2 应 ≈ hidden_dim * typical_per_dim_diff^2")
    
    hdim = 256
    per_dim_diff = 0.1  # 典型的每维差异
    expected_norm_sq = hdim * per_dim_diff ** 2
    suggested_K = (expected_norm_sq ** 0.5) * 2  # 让 c ≈ exp(-4) ≈ 0.018 不会太小
    print(f"\n  hidden_dim={hdim}, per_dim_diff={per_dim_diff}")
    print(f"  预期 norm_sq ≈ {expected_norm_sq:.2f}")
    print(f"  建议 K_init ≈ {suggested_K:.2f}")
    
    # 测试建议的 K
    atp = ATPPool(dim=hdim, pool_size=100, K_init=suggested_K, diffusion_steps=2, dt=0.1)
    x = torch.randn(1, 1000, hdim)
    for i in range(1, 1000):
        x[0, i] = x[0, i] * 0.3 + x[0, i-1] * 0.7
    with torch.no_grad():
        grad_right = torch.zeros_like(x)
        grad_right[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
        norm_right_sq = torch.mean(grad_right ** 2, dim=-1, keepdim=True)
        K = torch.clamp(atp.K, min=1e-3)
        c_right = torch.exp(-norm_right_sq / (K ** 2 + 1e-6))
    print(f"  K_init={suggested_K:.2f}: c_right mean={c_right.mean().item():.6f}  "
          f"ratio<1e-4={(c_right < 1e-4).float().mean().item():.4f}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    diagnose_atp_pool()
