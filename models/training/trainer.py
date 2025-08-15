# -*- coding: utf-8 -*-
"""
모델 학습 모듈
- 에포크 학습, 추론, Early Stopping 등
"""

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from .utils import model_forward_any
import numpy as np


def train_epoch(model, loader, device, criterion, optimizer):
    """한 에포크 학습"""
    model.train()
    total_loss = 0.0
    
    for (xb, yb) in loader:
        if isinstance(xb, dict):
            for k in xb: 
                xb[k] = xb[k].to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model_forward_any(model, xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def infer_probs(model, loader, device):
    """모델 추론 및 확률 계산"""
    model.eval()
    probs, ys = [], []
    
    for (xb, yb) in loader:
        if isinstance(xb, dict):
            for k in xb: 
                xb[k] = xb[k].to(device)
        
        logits = model_forward_any(model, xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    
    return np.concatenate(probs).reshape(-1), np.concatenate(ys).astype(int).reshape(-1)


def train_with_early_stopping(model, train_loader, val_loader, device, pos_weight, 
                             lr=1e-3, epochs=50, patience=5):
    """Early Stopping을 적용한 모델 학습"""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = -1.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, train_loader, device, criterion, optimizer)
        val_prob, val_y = infer_probs(model, val_loader, device)
        
        try:
            val_ap = average_precision_score(val_y, val_prob)  # AUC-PR를 조기 종료 지표로
        except Exception:
            val_ap = 0.0

        print(f"[Epoch {ep:03d}] train_loss={tr_loss:.4f}  val_aucpr={val_ap:.4f}")

        if val_ap > best_val:
            best_val = val_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep} (best val AUC-PR={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model 