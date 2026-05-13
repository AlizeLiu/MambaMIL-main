"""
IHG-Mamba (Intrinsic Hierarchical Graph-Mamba)
Modified from MambaMIL
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
    def __init__(self, in_dim, n_classes, dropout, act, survival=False, layer=1, rate=10, type="SRMamba", pool_size=100):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 256)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(256)

        self.survival = survival
        self.rate = rate
        self.type = type

        # ===== 2. 构建层级 Mamba 骨架 =====
        # 轻量化：Local=1层, Global=1层，减少参数量防止过拟合
        self.local_layers = self._build_mamba_layers(1, type)
        self.global_layers = self._build_mamba_layers(1, type)

        # ===== 3. 插入各向异性拓扑池化 =====
        self.atp_pool = ATPPool(dim=256, pool_size=pool_size)

        self.n_classes = n_classes
        self.classifier = nn.Linear(256, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.apply(initialize_weights)

    def _build_mamba_layers(self, layer_num, mamba_type):
        """内部方法：用于快速构建指定类型的 Mamba Block 堆叠"""
        layers = nn.ModuleList()
        for _ in range(layer_num):
            if mamba_type == "SRMamba":
                layers.append(nn.Sequential(nn.LayerNorm(256),
                                            SRMamba(d_model=256, d_state=16, d_conv=4, expand=2, use_fast_path=True)))
            elif mamba_type == "Mamba":
                layers.append(nn.Sequential(nn.LayerNorm(256),
                                            Mamba(d_model=256, d_state=16, d_conv=4, expand=2, use_fast_path=True)))
            elif mamba_type == "BiMamba":
                layers.append(nn.Sequential(nn.LayerNorm(256),
                                            BiMamba(d_model=256, d_state=16, d_conv=4, expand=2, use_fast_path=True)))
            else:
                raise NotImplementedError("Mamba [{}] is not implemented".format(mamba_type))
        return layers

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, N, in_dim]

        h = self._fc1(h)  # [B, N, 256] (N 约等于 5000+)

        # ===== 阶段 1: 微观环境建模 (Local Mamba) =====
        for layer in self.local_layers:
            h_ = h
            # 原本的 MambaMIL 中 SRMamba 有额外的 rate 参数
            if self.type == "SRMamba":
                h = layer[1](layer[0](h), rate=self.rate)
            else:

                h = layer[1](layer[0](h))
            h = h + h_

        # ===== 阶段 2: 边界保留降维 (ATP-Pool) =====
        # 将微观特征通过各向异性扩散，压缩成宏观特征
        h = self.atp_pool(h) # 长度瞬间缩小: [B, N // pool_size, 256]

        # ===== 阶段 3: 宏观组织建模 (Global Mamba) =====
        for layer in self.global_layers:
            h_ = h
            if self.type == "SRMamba":
                h = layer[1](layer[0](h), rate=self.rate)
            else:
                h = layer[1](layer[0](h))
            h = h + h_

        # ===== 阶段 4: 双重聚合之 ABMIL 读出头 =====
        h = self.norm(h)
        A = self.attention(h) # [B, M, 1] (M = N // pool_size)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, 1, M]
        h = torch.bmm(A, h) # [B, 1, 256]
        h = h.squeeze(0)    # [B, 256]

        # ===== 阶段 5: 下游任务预测 =====
        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        # 将缩小后的注意力权重返回，后续可视化宏观热力图会用到
        A_raw = A.clone()
        results_dict = None

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, None # 返回 A_raw 替代原本的 None

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        # 将新增加的层挂载到 GPU
        self.local_layers = self.local_layers.to(device)
        self.global_layers = self.global_layers.to(device)
        self.atp_pool = self.atp_pool.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)