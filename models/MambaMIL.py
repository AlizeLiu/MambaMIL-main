"""
MambaMIL - IHG-Mamba with parameterized architecture
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F

from .AtpPool import ATPPool

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# ---------------------------------------------------------------------------
# Attention modules for MIL readout
# ---------------------------------------------------------------------------
class SimpleAttention(nn.Module):
    """Standard attention: Linear -> Tanh -> Linear.
    Output shape: [B, M, 1]
    """
    def __init__(self, in_dim, attn_dim=128):
        super(SimpleAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, h):
        """h: [B, M, in_dim] -> A: [B, M, 1]"""
        return self.attn(h)


class GatedAttention(nn.Module):
    """Gated attention: V=Linear->Tanh, U=Linear->Sigmoid, w=Linear.
    Output: w(V(h) * U(h))  shape [B, M, 1]
    """
    def __init__(self, in_dim, attn_dim=128):
        super(GatedAttention, self).__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        """h: [B, M, in_dim] -> A: [B, M, 1]"""
        return self.w(self.V(h) * self.U(h))


class MambaMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        n_classes,
        dropout,
        act,
        survival=False,
        layer=1,
        rate=10,
        type="SRMamba",
        hidden_dim=256,
        local_layers=None,
        global_layers=None,
        pool_size=100,
        use_atp_pool=True,
        diffusion_steps=2,
        K_init=0.5,
        atp_dt=0.1,
        norm_type='mean',
        pool_mode='diffusion',
        tau_init=2.0,
        gamma_init=0.0,
        bp_alpha_init=1.0,
        bp_beta_init=1.0,
        bp_lambda_init=1.0,
        local_segment_mode='none',
        local_segment_size=50,
        attn_type='simple',
        attn_dim=128,
    ):
        super(MambaMIL, self).__init__()

        self.hidden_dim = hidden_dim
        self.survival = survival
        self.rate = rate
        self.type = type
        self.use_atp_pool = use_atp_pool
        self.n_classes = n_classes
        self.local_segment_mode = local_segment_mode
        self.local_segment_size = local_segment_size

        # Validation
        if local_segment_mode == "chunk" and pool_size > local_segment_size:
            raise ValueError(f"pool_size({pool_size}) > local_segment_size({local_segment_size}) is not allowed.")

        # 兼容旧参数：如果没有单独传 local/global，就用 layer
        if local_layers is None:
            local_layers = layer
        if global_layers is None:
            global_layers = layer

        # Build feature extractor
        self._fc1 = [nn.Linear(in_dim, hidden_dim)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)

        # Build Mamba layers
        self.local_layers = self._build_mamba_layers(local_layers, type)
        self.global_layers = self._build_mamba_layers(global_layers, type)

        # Build ATP-Pool or Identity
        if self.use_atp_pool:
            self.atp_pool = ATPPool(
                dim=hidden_dim,
                pool_size=pool_size,
                K_init=K_init,
                diffusion_steps=diffusion_steps,
                dt=atp_dt,
                norm_type=norm_type,
                pool_mode=pool_mode,
                tau_init=tau_init,
                gamma_init=gamma_init,
                bp_alpha_init=bp_alpha_init,
                bp_beta_init=bp_beta_init,
                bp_lambda_init=bp_lambda_init,
            )
        else:
            self.atp_pool = nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim)

        # Attention readout: parameterized by attn_type (simple or gated)
        if attn_type == 'gated':
            self.attention = GatedAttention(hidden_dim, attn_dim=attn_dim)
        elif attn_type == 'simple':
            self.attention = SimpleAttention(hidden_dim, attn_dim=attn_dim)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}. Must be 'simple' or 'gated'.")

        self.classifier = nn.Linear(hidden_dim, self.n_classes)

        self.apply(initialize_weights)

    def _build_mamba_layers(self, layer_num, mamba_type):
        layers = nn.ModuleList()

        for _ in range(layer_num):
            if mamba_type == "SRMamba":
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        SRMamba(
                            d_model=self.hidden_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                            use_fast_path=True
                        )
                    )
                )
            elif mamba_type == "Mamba":
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        Mamba(
                            d_model=self.hidden_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                            use_fast_path=True
                        )
                    )
                )
            elif mamba_type == "BiMamba":
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        BiMamba(
                            d_model=self.hidden_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                            use_fast_path=True
                        )
                    )
                )
            else:
                raise NotImplementedError(f"Mamba [{mamba_type}] is not implemented")

        return layers

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, N, in_dim]

        h = self._fc1(h)  # [B, N, hidden_dim]

        if self.local_segment_mode == 'chunk':
            # ===== Segment-wise Local Mamba =====
            B, L, D = h.shape
            seg_size = self.local_segment_size

            # Pad to segment boundary if needed
            pad_len = (seg_size - L % seg_size) % seg_size
            if pad_len > 0:
                print(f"[WARNING] Segment-wise Local Mamba applied padding: {L} -> {L + pad_len}. "
                      f"Padding tokens are not masked.")
                h = F.pad(h.transpose(1, 2), (0, pad_len), mode='replicate').transpose(1, 2).contiguous()
                B, L, D = h.shape

            S = L // seg_size

            # Reshape: [B, L, D] -> [B*S, seg_size, D]
            h_seg = h.reshape(B, S, seg_size, D)
            h_seg = h_seg.reshape(B * S, seg_size, D)

            # Local Mamba independently per segment
            for layer in self.local_layers:
                h_ = h_seg
                h_seg = layer[0](h_seg)
                h_seg = layer[1](h_seg, rate=self.rate) if self.type == "SRMamba" else layer[1](h_seg)
                h_seg = h_seg + h_

            # ATPPool per segment: [B*S, seg_size, D] -> [B*S, seg_size//pool_size, D]
            pooled_seg = self.atp_pool(h_seg)

            # Reshape back: [B, S * pooled_per_seg, D]
            pooled_per_seg = pooled_seg.shape[1]
            h = pooled_seg.reshape(B, S * pooled_per_seg, D)

        else:
            # ===== Flat Local Mamba (original) =====
            for layer in self.local_layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate) if self.type == "SRMamba" else layer[1](h)
                h = h + h_

            # ATP-Pool: compress sequence length
            h = self.atp_pool(h)  # [B, N // pool_size, hidden_dim] or Identity

        # Phase 2: Global Mamba (macro-context modeling)
        for layer in self.global_layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h, rate=self.rate) if self.type == "SRMamba" else layer[1](h)
            h = h + h_

        h = self.norm(h)

        # Attention pooling
        A = self.attention(h)  # [B, M, 1]
        A = torch.transpose(A, 1, 2)  # [B, 1, M]
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h)  # [B, 1, hidden_dim]
        h = h.squeeze(1)  # [B, hidden_dim]

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = A.detach()  # super-node attention weights
        results_dict = None

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, None

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.local_layers = self.local_layers.to(device)
        self.global_layers = self.global_layers.to(device)
        self.atp_pool = self.atp_pool.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
