# -*- coding: utf-8 -*-
"""
FT-Transformer (lite) 모델 모듈
- Transformer 기반 특성 학습 모델
"""

import torch
import torch.nn as nn


class FTTransformerLite(nn.Module):
    """
    FT-Transformer (lite) 모델
    
    - 연속형 변수: Linear projection + LayerNorm + bias
    - 범주형 변수: Embedding
    - [CLS] 토큰 + Transformer Encoder + Head
    """
    
    def __init__(self, num_dim, cat_dims, d_token=32, n_heads=4, n_layers=2, dropout=0.2):
        """
        Args:
            num_dim: 수치형 변수의 차원
            cat_dims: 각 범주형 변수의 차원 리스트
            d_token: 토큰 차원
            n_heads: Multi-head attention의 헤드 수
            n_layers: Transformer Encoder 레이어 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self._model_type = "ftt"
        self.num_dim = num_dim
        self.cat_dims = cat_dims
        self.d_token = d_token

        # 연속형 피처 임베딩(각 피처별 동일 투영 + 위치 인코딩 대체용 파라미터)
        if num_dim > 0:
            self.num_proj = nn.Linear(1, d_token)
            self.num_ln = nn.LayerNorm(d_token)
            self.num_token_bias = nn.Parameter(torch.zeros(num_dim, d_token))  # feature-wise bias

        # 범주형 임베딩
        self.emb_layers = nn.ModuleList([nn.Embedding(cd, d_token) for cd in cat_dims])

        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 출력 헤드
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat):
        """순전파"""
        B = x_cat.size(0) if x_cat is not None else x_num.size(0)
        tokens = []

        # 연속형 → 토큰
        if self.num_dim > 0:
            # x_num: [B, num_dim] -> [B, num_dim, 1] -> proj -> [B, num_dim, d_token]
            xn = self.num_proj(x_num.unsqueeze(-1))
            xn = self.num_ln(xn + self.num_token_bias)  # feature-wise bias 추가
            tokens.append(xn)

        # 범주형 → 토큰
        if len(self.emb_layers) > 0:
            xcs = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.emb_layers)]  # each [B,1,d]
            xc = torch.cat(xcs, dim=1)  # [B, K, d]
            tokens.append(xc)

        if len(tokens) == 0:
            raise ValueError("No tokens constructed (no numeric and no categorical).")

        # 토큰 결합
        x = torch.cat(tokens, dim=1)  # [B, T, d]
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d]
        x = torch.cat([cls, x], dim=1)          # [B, 1+T, d]

        # Transformer 처리
        z = self.encoder(x)                     # [B, 1+T, d]
        cls_out = z[:, 0, :]                    # [B, d]
        logit = self.head(cls_out).squeeze(1)   # [B]
        
        return logit 