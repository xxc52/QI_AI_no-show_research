# -*- coding: utf-8 -*-
"""
Neural Network Models for Data Engineering

이 패키지는 Data Engineering을 위한 다양한 신경망 모델들을 제공합니다.
"""

__version__ = "2.0.0"
__author__ = "Data Engineering Team"

# 주요 모듈들
from . import data
from . import models
from . import training
from . import config

__all__ = [
    "data",
    "models", 
    "training",
    "config"
] 