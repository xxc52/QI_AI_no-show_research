# -*- coding: utf-8 -*-
"""
PyTorch Dataset 모듈
- 테이블 데이터를 위한 커스텀 Dataset 클래스
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TabDataset(Dataset):
    """테이블 데이터를 위한 PyTorch Dataset"""
    
    def __init__(self, X_num, X_wide, X_cat_idx, y=None):
        """
        Args:
            X_num: 수치형 변수 (numpy array)
            X_wide: 원-핫 인코딩된 범주형 변수 (numpy array)
            X_cat_idx: 범주형 변수의 정수 인덱스 (numpy array)
            y: 타겟 변수 (numpy array, 선택사항)
        """
        self.X_num = X_num.astype(np.float32) if X_num is not None else None
        self.X_wide = X_wide.astype(np.float32) if X_wide is not None else None
        self.X_cat = X_cat_idx.astype(np.int64) if X_cat_idx is not None else None
        self.y = y.astype(np.float32) if y is not None else None

    def __len__(self):
        """데이터셋의 크기 반환"""
        if self.X_cat is not None:
            return len(self.X_cat)
        elif self.X_num is not None:
            return len(self.X_num)
        else:
            return 0

    def __getitem__(self, idx):
        """인덱스에 해당하는 데이터 반환"""
        items = {}
        
        if self.X_num is not None:
            items["num"] = torch.from_numpy(self.X_num[idx])
        if self.X_wide is not None:
            items["wide"] = torch.from_numpy(self.X_wide[idx])
        if self.X_cat is not None:
            items["cat"] = torch.from_numpy(self.X_cat[idx])
        
        if self.y is not None:
            return items, torch.tensor(self.y[idx])
        
        return items 