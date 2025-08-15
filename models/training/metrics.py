# -*- coding: utf-8 -*-
"""
평가 메트릭 모듈
- 정확도, 재현율, F1-score, ROC-AUC, AUC-PR 등
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


@torch.no_grad()
def evaluate_threshold(y_true, y_score, thr):
    """주어진 임계값으로 성능 평가"""
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return acc, rec, f1


@torch.no_grad()
def find_threshold_for_target_recall(y_true, y_prob, target_recall=0.7, tie_metric="f1"):
    """목표 재현율을 만족하는 최적 임계값 찾기"""
    ts = np.linspace(0.0, 1.0, 1001)
    cand = []
    
    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = (y_pred[y_pred==1].size and
                (y_true[y_pred==1].sum()/max(1, y_pred[y_pred==1].size))) or 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if rec >= target_recall:
            cand.append((t, {"precision": prec, "f1": f1, "recall": rec}))
    
    if not cand:
        return 0.0  # recall을 만족하는 임계값이 없으면 최소값
    
    key = lambda x: x[1][tie_metric]
    return max(cand, key=key)[0]


def find_best_threshold(y_true, y_prob, metric="f1"):
    """주어진 지표를 최대화하는 최적 임계값 찾기"""
    ts = np.linspace(0.0, 1.0, 1001)
    best_score = -1.0
    best_threshold = 0.5
    
    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold


def metrics_report(y_true, y_prob, thr):
    """종합적인 성능 지표 리포트"""
    # y_prob: sigmoid 확률
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["recall"]   = recall_score(y_true, y_pred, zero_division=0)
    out["f1"]       = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    
    try:
        out["auc_pr"]  = average_precision_score(y_true, y_prob)
    except Exception:
        out["auc_pr"]  = np.nan

    # 결과 출력
    print(f"Accuracy : {out['accuracy']:.4f}")
    print(f"Recall   : {out['recall']:.4f}")
    print(f"F1-score : {out['f1']:.4f}")
    print("ROC-AUC  : {:.4f}".format(out["roc_auc"]) if not np.isnan(out["roc_auc"]) else "ROC-AUC  : N/A")
    print("AUC-PR   : {:.4f}".format(out["auc_pr"]) if not np.isnan(out["auc_pr"]) else "AUC-PR   : N/A")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    
    return out 