import torch
import torch.nn as nn
import torch.nn.functional as F


class ATPPool(nn.Module):
    """
    Anisotropic Topological Pooling on Hilbert Manifold (ATP-Pool)
    基于 Hilbert 流形的各向异性拓扑池化
    """

    def __init__(self, dim, pool_size=100, K_init=1.0, diffusion_steps=2, dt=0.1):
        super(ATPPool, self).__init__()
        self.pool_size = pool_size
        self.diffusion_steps = diffusion_steps

        #  K 设为可学习参数 (Learnable Parameter)
        self.K = nn.Parameter(torch.tensor(K_init))

        # 扩散时间步长 (控制每次融合的力度)
        self.dt = dt

    def _perona_malik_diffusion_1d(self, x):
        """
        核心物理算子：求解 1D Perona-Malik 偏微分方程
        x shape: (Batch, Length, Dim)
        """
        for _ in range(self.diffusion_steps):
            # 向右梯度 (前向差分)
            grad_right = torch.zeros_like(x)
            grad_right[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

            # 向左梯度 (后向差分)
            grad_left = torch.zeros_like(x)
            grad_left[:, 1:, :] = x[:, :-1, :] - x[:, 1:, :]

            # 2. 计算边界传导系数 (Boundary Conductivity Coefficients)
            # 计算梯度的 L2 范数平方 (表示局部异质性剧烈程度)
            norm_right_sq = torch.sum(grad_right ** 2, dim=-1, keepdim=True)
            norm_left_sq = torch.sum(grad_left ** 2, dim=-1, keepdim=True)

            # c = exp(-(|nabla X|^2) / K^2)
            c_right = torch.exp(-norm_right_sq / (self.K ** 2 + 1e-6))
            c_left = torch.exp(-norm_left_sq / (self.K ** 2 + 1e-6))

            # 3. 执行流形扩散更新 (Manifold Diffusion Update)
            x = x + self.dt * (c_right * grad_right + c_left * grad_left)

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
        # shape 转换为 (B, D, L)
        x_diffused = x_diffused.transpose(1, 2)

        # 使用 1D 均值池化进行暴力压缩，池化窗口大小为 pool_size
        pooled_x = F.avg_pool1d(x_diffused, kernel_size=self.pool_size, stride=self.pool_size)

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