# -*- coding: utf-8 -*-
"""
학습 유틸리티 모듈
- 클래스 가중치 계산, 모델 순전파 등
"""

import numpy as np
import torch
from ..models.mlp import MLPBNDropout
from ..models.wide_deep import WideAndDeep


def compute_class_pos_weight(y_train: np.ndarray):
    """클래스 불균형을 위한 양성 클래스 가중치 계산"""
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0:
        return 1.0
    return max(1.0, float(neg / (pos + 1e-8)))


def model_forward_any(model, xb):
    """모델 타입에 맞게 로그릿 반환"""
    if isinstance(model, MLPBNDropout):
        # num + wide를 하나로 묶어서 사용
        x = xb["num"] if xb.get("wide") is None else torch.cat([xb["num"], xb["wide"]], dim=1)
        return model(x)
    elif isinstance(model, WideAndDeep):
        return model(xb["num"], xb["wide"], xb["cat"])
    elif hasattr(model, "_model_type") and model._model_type == "deepfm":
        return model(xb["num"], xb["wide"], xb["cat"])
    elif hasattr(model, "_model_type") and model._model_type == "ftt":
        # FT-Transformer는 wide 미사용(선택) — 필요시 합칠 수 있음
        return model(xb["num"], xb["cat"])
    else:
        raise ValueError("Unknown model type in forward.") 