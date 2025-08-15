# -*- coding: utf-8 -*-
"""
MLP 모델 모듈
- BatchNorm과 Dropout이 적용된 다층 퍼셉트론
"""

import torch
import torch.nn as nn


class MLPBNDropout(nn.Module):
    """
    MLP + BatchNorm + Dropout 모델
    
    입력: [수치형 + 원-핫(범주형)] 벡터
    구조: Linear -> BatchNorm -> ReLU -> Dropout -> ... -> Linear
    """
    
    def __init__(self, input_dim, hidden=[128, 64], dropout=0.3):
        """
        Args:
            input_dim: 입력 차원
            hidden: 은닉층 크기 리스트
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        layers = []
        prev = input_dim
        
        # 은닉층 구성
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        
        # 출력층
        layers += [nn.Linear(prev, 1)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """순전파"""
        return self.net(x).squeeze(1)  # logits 