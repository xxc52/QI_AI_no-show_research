# -*- coding: utf-8 -*-
"""
신경망 모델 모듈
- MLP, Wide & Deep, DeepFM, FT-Transformer, TabNet 등
"""

from .mlp import MLPBNDropout
from .wide_deep import WideAndDeep
from .deepfm import DeepFM
from .fttransformer import FTTransformerLite

# TabNet은 선택적 (설치 시에만 사용)
try:
    from .tabnet import TabNetWrapper
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

__all__ = [
    'MLPBNDropout',
    'WideAndDeep', 
    'DeepFM',
    'FTTransformerLite',
    'TABNET_AVAILABLE'
]

if TABNET_AVAILABLE:
    __all__.append('TabNetWrapper') 