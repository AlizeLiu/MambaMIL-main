import torch
import torch.nn as nn
import torch.nn.functional as F


class ATPPool(nn.Module):
    """
    Anisotropic Topological Pooling on Hilbert Manifold (ATP-Pool)
    基于 Hilbert 流形的各向异性拓扑池化
    """

    def __init__(self, dim, pool_size=100, K_init=0.5, diffusion_steps=2, dt=0.1):
        super(ATPPool, self).__init__()
        self.pool_size = pool_size
        self.diffusion_steps = diffusion_steps

        # K 设为可学习参数 (Learnable Parameter)
        self.K = nn.Parameter(torch.tensor(float(K_init)))

        # 扩散时间步长 (控制每次融合的力度)
        self.dt = dt

        # 诊断模式：记录前 N 步的扩散统计量
        self._diag_enabled = False
        self._diag_step = 0
        self._diag_max_steps = 0
        self._diag_records = []

    def enable_diagnosis(self, max_steps=200):
        """启用诊断模式，记录前 max_steps 次 forward 的扩散统计量"""
        self._diag_enabled = True
        self._diag_step = 0
        self._diag_max_steps = max_steps
        self._diag_records = []

    def disable_diagnosis(self):
        """关闭诊断模式"""
        self._diag_enabled = False

    def get_diagnosis_summary(self):
        """返回诊断汇总"""
        if not self._diag_records:
            return None
        keys = self._diag_records[0].keys()
        summary = {}
        for k in keys:
            vals = [r[k] for r in self._diag_records]
            t = torch.tensor(vals)
            summary[k] = {
                'mean': t.mean().item(),
                'std': t.std().item(),
                'min': t.min().item(),
                'p10': t.kthvalue(max(1, len(vals) // 10)).item(),
                'median': t.median().item(),
                'p90': t.kthvalue(max(1, len(vals) * 9 // 10)).item(),
                'max': t.max().item(),
            }
        return summary

    def _perona_malik_diffusion_1d(self, x):
        """
        核心物理算子：求解 1D Perona-Malik 偏微分方程
        x shape: (Batch, Length, Dim)
        """
        diag_this_step = self._diag_enabled and self._diag_step < self._diag_max_steps

        for step_idx in range(self.diffusion_steps):
            # 向右梯度 (前向差分)
            grad_right = torch.zeros_like(x)
            grad_right[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

            # 向左梯度 (后向差分)
            grad_left = torch.zeros_like(x)
            grad_left[:, 1:, :] = x[:, :-1, :] - x[:, 1:, :]

            # 计算梯度的均方范数 (表示局部异质性剧烈程度)
            # 使用 mean 代替 sum，避免 hidden_dim 缩放导致 c≈0
            norm_right_sq = torch.mean(grad_right ** 2, dim=-1, keepdim=True)
            norm_left_sq = torch.mean(grad_left ** 2, dim=-1, keepdim=True)

            # c = exp(-(|nabla X|^2) / K^2)
            # 防止 K 变成不稳定值
            K = torch.clamp(self.K, min=1e-3)
            c_right = torch.exp(-norm_right_sq / (K ** 2 + 1e-6))
            c_left = torch.exp(-norm_left_sq / (K ** 2 + 1e-6))

            # 诊断：记录统计量 (只在第一个 diffusion step 记录)
            if diag_this_step and step_idx == 0:
                with torch.no_grad():
                    c_flat = c_right.squeeze(-1).flatten()
                    n_flat = norm_right_sq.squeeze(-1).flatten()
                    self._diag_records.append({
                        'K': K.item(),
                        'norm_sq_mean': n_flat.mean().item(),
                        'norm_sq_median': n_flat.median().item(),
                        'norm_sq_p90': n_flat.kthvalue(max(1, int(n_flat.numel() * 0.9))).item(),
                        'c_mean': c_flat.mean().item(),
                        'c_median': c_flat.median().item(),
                        'c_p10': c_flat.kthvalue(max(1, int(n_flat.numel() * 0.1))).item(),
                        'c_p90': c_flat.kthvalue(max(1, int(n_flat.numel() * 0.9))).item(),
                        'ratio_c_lt_01': (c_flat < 0.1).float().mean().item(),
                        'ratio_c_lt_03': (c_flat < 0.3).float().mean().item(),
                        'ratio_c_gt_09': (c_flat > 0.9).float().mean().item(),
                    })

            # 执行流形扩散更新 (Manifold Diffusion Update)
            x = x + self.dt * (c_right * grad_right + c_left * grad_left)

        if diag_this_step:
            self._diag_step += 1

        return x

    def forward(self, x):
        """
        前向传播
        输入: x (B, L, D) - 局部 Mamba 出来的长序列
        输出: pooled_x (B, M, D) - 压缩后的超节点序列
        """
        B, L, D = x.shape

        # 步骤 1：执行各向异性扩散，智能平滑同质区域，保留边界
        x_diffused = self._perona_malik_diffusion_1d(x)

        # 步骤 2：拓扑安全降采样 (Topological Downsampling)
        # 处理长度不能被 pool_size 整除的情况
        pad_len = (self.pool_size - L % self.pool_size) % self.pool_size

        if pad_len > 0:
            x_diffused = x_diffused.transpose(1, 2)
            x_diffused = F.pad(x_diffused, (0, pad_len), mode='replicate')
        else:
            x_diffused = x_diffused.transpose(1, 2)

        # 使用 1D 均值池化进行压缩，池化窗口大小为 pool_size
        pooled_x = F.avg_pool1d(
            x_diffused,
            kernel_size=self.pool_size,
            stride=self.pool_size
        )

        # 转回原始 shape: (B, M, D)
        pooled_x = pooled_x.transpose(1, 2)

        return pooled_x


if __name__ == "__main__":
    dummy_input = torch.randn(1, 10000, 512)

    atp_pool = ATPPool(dim=512, pool_size=100)
    output = atp_pool(dummy_input)

    print(f"输入序列维度: {dummy_input.shape}")
    print(f"输出序列维度: {output.shape}")
    print(f"当前学习到的边界阈值 K: {atp_pool.K.item():.4f}")
