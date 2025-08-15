# -*- coding: utf-8 -*-
"""
데이터 전처리 모듈
- CSV 로드, 타겟 변수 처리, 범주형/수치형 분리 등
"""

import os
import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """CSV 파일을 로드하고 컬럼명 정리"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_datestr_columns(df: pd.DataFrame, target_col="No-show"):
    """날짜/타임스탬프 문자열 열을 자동 감지하고 드롭"""
    drop_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == object:
            vals = df[c].astype(str).head(20).tolist()
            if any(("T" in v) or ("-" in v and any(ch.isdigit() for ch in v)) or (":" in v) for v in vals):
                drop_cols.append(c)
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols


def prepare_xy(df: pd.DataFrame, target_col="No-show"):
    """데이터프레임을 X, y로 분리하고 전처리"""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # 타겟 변수 처리
    y_raw = df[target_col].copy()
    
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.strip().str.lower().map({
            "no": 0, "n": 0, "0": 0, "false": 0,
            "yes": 1, "y": 1, "1": 1, "true": 1
        })
        if y.isna().any():
            y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = y_raw

    # 결측값이 있는 행 제거
    keep = ~y.isna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int)

    # 날짜 문자열 드롭
    df, dropped_time = normalize_datestr_columns(df, target_col=target_col)

    # 타겟 제거
    X = df.drop(columns=[target_col], errors="ignore")

    # 수치형/범주형 분리 및 결측 대치
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 수치형: 중앙값으로 결측 대치
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    
    # 범주형: "MISSING"으로 결측 대치
    for c in cat_cols:
        X[c] = X[c].fillna("MISSING")

    return X, y, num_cols, cat_cols, dropped_time


def build_category_indexers(df_cat: pd.DataFrame, cat_cols):
    """범주형 변수를 정수 인덱스로 변환하는 매핑 생성"""
    cat_maps, cat_dims = {}, []
    for c in cat_cols:
        vocab = pd.Index(["__UNK__"] + sorted(df_cat[c].astype(str).unique().tolist()))
        stoi = {s: i for i, s in enumerate(vocab)}
        cat_maps[c] = {"stoi": stoi, "itos": vocab}
        cat_dims.append(len(vocab))
    return cat_maps, cat_dims


def encode_categories(df_cat: pd.DataFrame, cat_cols, cat_maps):
    """범주형 변수를 정수 인덱스로 인코딩"""
    cat_arrays = []
    for c in cat_cols:
        stoi = cat_maps[c]["stoi"]
        idx = df_cat[c].astype(str).map(stoi)
        idx = idx.fillna(0).astype(int)  # 미등록 카테고리는 0번(UNK)로
        cat_arrays.append(idx.to_numpy())
    return np.vstack(cat_arrays).T


def build_wide_onehot(cat_indices: np.ndarray, cat_dims):
    """
    범주형 인덱스를 원-핫 벡터로 변환
    
    Args:
        cat_indices: (N, n_cat_cols) 각 항목은 [0..dim-1]
        cat_dims: 각 범주형 변수의 차원 리스트
    
    Returns:
        out: (N, total_dim) 원-핫 벡터
        total_dim: 전체 차원
    """
    N, K = cat_indices.shape
    total_dim = int(np.sum(cat_dims))
    out = np.zeros((N, total_dim), dtype=np.float32)
    offsets = np.cumsum([0] + list(cat_dims[:-1]))
    
    for j in range(K):
        out[np.arange(N), offsets[j] + cat_indices[:, j]] = 1.0
    
    return out, total_dim 