import torch
import torch.nn as nn
import torch.nn.functional as F


class ATPPool(nn.Module):
    """
    Anisotropic Topological Pooling on Hilbert Manifold (ATP-Pool)
    基于 Hilbert 流形的各向异性拓扑池化

    pool_mode:
        "avg"        - 普通 padding + avg_pool1d (无扩散, 无边界残差)
        "diffusion"  - Perona-Malik 扩散 + avg_pool1d (原始模式)
        "residual"   - Boundary Residual Pooling (avg + 边界残差)
    """

    def __init__(
        self,
        dim,
        pool_size=100,
        K_init=0.5,
        diffusion_steps=2,
        dt=0.1,
        norm_type='mean',
        pool_mode='diffusion',
        tau_init=2.0,
        gamma_init=0.0,
    ):
        super(ATPPool, self).__init__()
        self.dim = dim
        self.pool_size = pool_size
        self.diffusion_steps = diffusion_steps
        self.dt = dt
        self.norm_type = norm_type
        self.pool_mode = pool_mode

        # Learnable parameters
        self.K = nn.Parameter(torch.tensor(float(K_init)))
        self.tau_raw = nn.Parameter(torch.tensor(float(tau_init)))
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

        # Diagnosis
        self._diag_enabled = False
        self._diag_step = 0
        self._diag_max_steps = 0
        self._diag_records = []

    # ------------------------------------------------------------------
    # Diagnosis helpers
    # ------------------------------------------------------------------
    def enable_diagnosis(self, max_steps=200):
        self._diag_enabled = True
        self._diag_step = 0
        self._diag_max_steps = max_steps
        self._diag_records = []

    def disable_diagnosis(self):
        self._diag_enabled = False

    def get_diagnosis_summary(self):
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
                'p10': t.kthvalue(max(1, len(vals) // 10)).values.item(),
                'median': t.median().item(),
                'p90': t.kthvalue(max(1, len(vals) * 9 // 10)).values.item(),
                'max': t.max().item(),
            }
        return summary

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _edge_norm(self, x):
        """Compute per-edge squared norm: [B, L-1, 1]"""
        grad = x[:, 1:, :] - x[:, :-1, :]
        if self.norm_type == 'mean':
            norm_sq = torch.mean(grad ** 2, dim=-1, keepdim=True)
        elif self.norm_type == 'sum':
            norm_sq = torch.sum(grad ** 2, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
        return norm_sq

    def _conductance_edges(self, x):
        """Compute conductance on edges: c in [B, L-1, 1]"""
        norm_sq = self._edge_norm(x)
        K = torch.clamp(self.K, min=1e-3)
        c = torch.exp(-norm_sq / (K ** 2 + 1e-6))
        return c, norm_sq

    def _edge_to_token_boundary(self, edge_boundary, L):
        """Convert edge boundary [B, L-1, 1] to token boundary [B, L, 1]"""
        # left[i] = edge_boundary[i-1] (pad front with 0)
        left = F.pad(edge_boundary, (0, 0, 1, 0), value=0.0)
        # right[i] = edge_boundary[i] (pad end with 0)
        right = F.pad(edge_boundary, (0, 0, 0, 1), value=0.0)
        return 0.5 * (left + right)

    def _pad_to_pool_size(self, x, mode='replicate'):
        """Pad x along dim=1 to be divisible by pool_size.
        x: [B, L, D] -> [B, L_pad, D], pad_len
        """
        B, L, D = x.shape
        pad_len = (self.pool_size - L % self.pool_size) % self.pool_size
        if pad_len == 0:
            return x, 0
        # F.pad pads last dim first: (D_left, D_right, L_left, L_right)
        x_pad = F.pad(x.transpose(1, 2), (0, pad_len), mode=mode).transpose(1, 2)
        return x_pad.contiguous(), pad_len

    # ------------------------------------------------------------------
    # Pooling modes
    # ------------------------------------------------------------------
    def _avg_pool(self, x):
        """Plain avg pooling: [B, L, D] -> [B, M, D]"""
        B, L, D = x.shape
        x_pad, _ = self._pad_to_pool_size(x, mode='replicate')
        M = x_pad.shape[1] // self.pool_size
        x_seg = x_pad.reshape(B, M, self.pool_size, D)
        return x_seg.mean(dim=2)

    def _boundary_residual_pool(self, x):
        """Boundary Residual Pooling: avg + gamma * (boundary_weighted - avg)"""
        B, L, D = x.shape

        # Edge conductance and boundary
        c, norm_sq = self._conductance_edges(x)
        edge_b = 1.0 - c  # [B, L-1, 1]

        # Token boundary
        token_b = self._edge_to_token_boundary(edge_b, L)  # [B, L, 1]

        # Pad to pool_size
        x_pad, _ = self._pad_to_pool_size(x, mode='replicate')
        b_pad, _ = self._pad_to_pool_size(token_b, mode='replicate')

        B, Lp, D = x_pad.shape
        M = Lp // self.pool_size

        # Reshape into windows
        x_seg = x_pad.reshape(B, M, self.pool_size, D)
        b_seg = b_pad.reshape(B, M, self.pool_size, 1)

        # Avg pooling (baseline)
        z_avg = x_seg.mean(dim=2)  # [B, M, D]

        # Boundary-weighted pooling
        b_mean = b_seg.mean(dim=2, keepdim=True)
        b_std = b_seg.std(dim=2, keepdim=True, unbiased=False)
        b_norm = (b_seg - b_mean) / (b_std + 1e-6)  # [B, M, pool_size, 1]

        tau = F.softplus(self.tau_raw) + 1e-6
        weights = torch.softmax(tau * b_norm, dim=2)  # [B, M, pool_size, 1]

        z_bd = (weights * x_seg).sum(dim=2)  # [B, M, D]

        # Residual blend
        gamma = torch.tanh(self.gamma_raw)
        z = z_avg + gamma * (z_bd - z_avg)

        # Diagnosis (residual mode only)
        diag_this = self._diag_enabled and self._diag_step < self._diag_max_steps
        if diag_this:
            with torch.no_grad():
                self._diag_records.append({
                    'K': torch.clamp(self.K, min=1e-3).item(),
                    'gamma': gamma.item(),
                    'tau': tau.item(),
                    'boundary_mean': token_b.mean().item(),
                    'boundary_std': token_b.std().item(),
                    'weight_entropy': -(weights * (weights + 1e-8).log()).sum(dim=2).mean().item(),
                    'residual_delta_norm': (z_bd - z_avg).norm(dim=-1).mean().item(),
                })
            self._diag_step += 1

        return z

    def _perona_malik_diffusion_1d(self, x):
        """Perona-Malik diffusion: [B, L, D] -> [B, L, D]"""
        diag_this = self._diag_enabled and self._diag_step < self._diag_max_steps

        for step_idx in range(self.diffusion_steps):
            grad_right = torch.zeros_like(x)
            grad_right[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

            grad_left = torch.zeros_like(x)
            grad_left[:, 1:, :] = x[:, :-1, :] - x[:, 1:, :]

            if self.norm_type == 'mean':
                norm_right_sq = torch.mean(grad_right ** 2, dim=-1, keepdim=True)
                norm_left_sq = torch.mean(grad_left ** 2, dim=-1, keepdim=True)
            else:
                norm_right_sq = torch.sum(grad_right ** 2, dim=-1, keepdim=True)
                norm_left_sq = torch.sum(grad_left ** 2, dim=-1, keepdim=True)

            K = torch.clamp(self.K, min=1e-3)
            c_right = torch.exp(-norm_right_sq / (K ** 2 + 1e-6))
            c_left = torch.exp(-norm_left_sq / (K ** 2 + 1e-6))

            if diag_this and step_idx == 0:
                with torch.no_grad():
                    c_flat = c_right.squeeze(-1).flatten()
                    n_flat = norm_right_sq.squeeze(-1).flatten()
                    self._diag_records.append({
                        'K': K.item(),
                        'norm_sq_mean': n_flat.mean().item(),
                        'norm_sq_median': n_flat.median().item(),
                        'norm_sq_p90': n_flat.kthvalue(max(1, int(n_flat.numel() * 0.9))).values.item(),
                        'c_mean': c_flat.mean().item(),
                        'c_median': c_flat.median().item(),
                        'c_p10': c_flat.kthvalue(max(1, int(n_flat.numel() * 0.1))).values.item(),
                        'c_p90': c_flat.kthvalue(max(1, int(n_flat.numel() * 0.9))).values.item(),
                        'ratio_c_lt_01': (c_flat < 0.1).float().mean().item(),
                        'ratio_c_lt_03': (c_flat < 0.3).float().mean().item(),
                        'ratio_c_gt_09': (c_flat > 0.9).float().mean().item(),
                    })

            x = x + self.dt * (c_right * grad_right + c_left * grad_left)

        if diag_this:
            self._diag_step += 1

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        assert x.shape[-1] == self.dim, f"Expected dim={self.dim}, got {x.shape[-1]}"

        if self.pool_mode == "avg":
            return self._avg_pool(x)

        elif self.pool_mode == "diffusion":
            x_diffused = self._perona_malik_diffusion_1d(x)
            return self._avg_pool(x_diffused)

        elif self.pool_mode == "residual":
            return self._boundary_residual_pool(x)

        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")


if __name__ == "__main__":
    x = torch.randn(1, 10000, 512)
    pool = ATPPool(dim=512, pool_size=100)
    y = pool(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    print(f"K: {pool.K.item():.4f}")
