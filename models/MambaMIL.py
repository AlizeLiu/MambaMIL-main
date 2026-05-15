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
    ):
        super(MambaMIL, self).__init__()

        self.hidden_dim = hidden_dim
        self.survival = survival
        self.rate = rate
        self.type = type
        self.use_atp_pool = use_atp_pool
        self.n_classes = n_classes

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
            )
        else:
            self.atp_pool = nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim)

        # 注意：这是标准 attention (Linear->Tanh->Linear)，不是 Gated Attention (Linear->sigmoid + Linear->softmax)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

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

        # Phase 1: Local Mamba (micro-environment modeling)
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
