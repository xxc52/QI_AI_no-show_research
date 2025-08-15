# -*- coding: utf-8 -*-
"""
스케일러 모듈
- 수치형 변수의 표준화를 위한 클래스
"""

import numpy as np


class StdScaler:
    """표준화 스케일러 (StandardScaler)"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        """훈련 데이터로부터 평균과 표준편차 계산"""
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8  # 0으로 나누기 방지
        return self

    def transform(self, x: np.ndarray):
        """데이터를 표준화"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform")
        return (x - self.mean_) / self.std_
    
    def fit_transform(self, x: np.ndarray):
        """fit과 transform을 연속으로 실행"""
        return self.fit(x).transform(x) 