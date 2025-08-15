# -*- coding: utf-8 -*-
"""
Neural Network Models for Data Engineering - 메인 실행 파일

사용법:
    python main.py --models mlp,wide_deep --epochs 60 --batch_size 2048
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# 로컬 모듈 import
from models.data import (
    load_dataset, prepare_xy, build_category_indexers, 
    encode_categories, build_wide_onehot, StdScaler, TabDataset
)
from models.models import (
    MLPBNDropout, WideAndDeep, DeepFM, FTTransformerLite, TABNET_AVAILABLE
)
from models.training import (
    train_with_early_stopping, infer_probs, 
    find_best_threshold, metrics_report, compute_class_pos_weight
)
from models.config import DEFAULT_CONFIG, MODEL_CONFIGS, PREPROCESSING_CONFIG


def run_tabnet(X_train_num, X_train_cat_idx, X_val_num, X_val_cat_idx, X_test_num, X_test_cat_idx,
               y_train, y_val, y_test, cat_dims, device, epochs=100, batch_size=2048, lr=1e-2):
    """TabNet 모델 실행 (선택적)"""
    if not TABNET_AVAILABLE:
        print("[TabNet] pytorch-tabnet 미설치로 스킵합니다. (pip install pytorch-tabnet)")
        return None

    # TabNet 입력: 연속형 + 범주형을 함께 numpy로 전달
    Xtr = np.hstack([X_train_num, X_train_cat_idx])
    Xva = np.hstack([X_val_num, X_val_cat_idx])
    Xte = np.hstack([X_test_num, X_test_cat_idx])

    num_dim = X_train_num.shape[1]
    cat_idxs = list(range(num_dim, num_dim + X_train_cat_idx.shape[1]))

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_params={"step_size": 20, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=1,
            device_name=device
        )
        
        clf.fit(
            Xtr, y_train,
            eval_set=[(Xva, y_val)],
            eval_name=["val"],
            eval_metric=["auc_pr"],
            max_epochs=epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        # 검증 임계값 튜닝
        val_prob = clf.predict_proba(Xva)[:, 1]
        best_thr = find_best_threshold(y_val, val_prob, metric="f1")

        # 테스트 평가
        test_prob = clf.predict_proba(Xte)[:, 1]
        print("\n==== TabNet (Best threshold on val) ====")
        metrics_report(y_test, test_prob, best_thr)
        
        return {"model": "TabNet", "val_thr": best_thr}
        
    except Exception as e:
        print(f"[TabNet] 실행 중 오류 발생: {e}")
        return None


def main(args):
    """메인 실행 함수"""
    # 데이터 로드 및 전처리
    df = load_dataset(args.csv)
    X, y, num_cols, cat_cols, dropped = prepare_xy(df, target_col=PREPROCESSING_CONFIG["target_col"])
    
    print(f"[INFO] Dropped time-like cols: {dropped}")
    print(f"[INFO] Num cols: {num_cols}")
    print(f"[INFO] Cat cols: {cat_cols}")
    print(f"[INFO] Samples: {len(X)}, Positives: {int(y.sum())}")

    # 데이터 분할: train/val/test (6:2:2)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=PREPROCESSING_CONFIG["test_size"], 
        stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=PREPROCESSING_CONFIG["val_size"], 
        stratify=y_tmp, random_state=args.seed
    )

    # 스케일링(연속형)
    scaler = StdScaler().fit(X_train[num_cols].to_numpy() if len(num_cols) > 0 else np.zeros((len(X_train), 0)))
    Xtr_num = scaler.transform(X_train[num_cols].to_numpy()) if len(num_cols) > 0 else np.zeros((len(X_train), 0))
    Xva_num = scaler.transform(X_val[num_cols].to_numpy()) if len(num_cols) > 0 else np.zeros((len(X_val), 0))
    Xte_num = scaler.transform(X_test[num_cols].to_numpy()) if len(num_cols) > 0 else np.zeros((len(X_test), 0))

    # 범주형 인덱싱 & wide 원-핫
    cat_maps, cat_dims = build_category_indexers(X_train[cat_cols], cat_cols) if len(cat_cols) > 0 else ({}, [])
    Xtr_cat_idx = encode_categories(X_train[cat_cols], cat_cols, cat_maps) if len(cat_cols) > 0 else np.zeros((len(X_train), 0), dtype=int)
    Xva_cat_idx = encode_categories(X_val[cat_cols], cat_cols, cat_maps) if len(cat_cols) > 0 else np.zeros((len(X_val), 0), dtype=int)
    Xte_cat_idx = encode_categories(X_test[cat_cols], cat_cols, cat_maps) if len(cat_cols) > 0 else np.zeros((len(X_test), 0), dtype=int)

    Xtr_wide, wide_dim = build_wide_onehot(Xtr_cat_idx, cat_dims) if len(cat_cols) > 0 else (np.zeros((len(X_train), 0), np.float32), 0)
    Xva_wide, _ = build_wide_onehot(Xva_cat_idx, cat_dims) if len(cat_cols) > 0 else (np.zeros((len(X_val), 0), np.float32), 0)
    Xte_wide, _ = build_wide_onehot(Xte_cat_idx, cat_dims) if len(cat_cols) > 0 else (np.zeros((len(X_test), 0), np.float32), 0)

    # 데이터로더
    train_ds = TabDataset(Xtr_num, Xtr_wide, Xtr_cat_idx, y_train.to_numpy())
    val_ds = TabDataset(Xva_num, Xva_wide, Xva_cat_idx, y_val.to_numpy())
    test_ds = TabDataset(Xte_num, Xte_wide, Xte_cat_idx, y_test.to_numpy())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    pos_weight = compute_class_pos_weight(y_train.to_numpy())
    print(f"[INFO] pos_weight (for BCEWithLogitsLoss): {pos_weight:.3f}")

    results = []

    # ---------------- MLP + BN + Dropout ----------------
    if "mlp" in args.models:
        mlp_input_dim = Xtr_num.shape[1] + Xtr_wide.shape[1]
        mlp = MLPBNDropout(
            input_dim=mlp_input_dim, 
            **MODEL_CONFIGS["mlp"]
        ).to(device)
        
        print("\n[RUN] MLP + BatchNorm + Dropout")
        mlp = train_with_early_stopping(
            mlp, train_loader, val_loader, device, pos_weight,
            lr=args.lr, epochs=args.epochs, patience=args.patience
        )
        
        # 임계값 튜닝(F1 최대화)
        val_prob, val_y = infer_probs(mlp, val_loader, device)
        thr = find_best_threshold(val_y, val_prob, metric="f1")
        test_prob, test_y = infer_probs(mlp, test_loader, device)
        
        print("\n==== MLP+BN+Dropout (Best threshold on val) ====")
        m = metrics_report(test_y, test_prob, thr)
        results.append(("MLP_BN_Dropout", m))

    # ---------------- Wide & Deep ----------------
    if "wide_deep" in args.models:
        wd = WideAndDeep(
            cat_dims=cat_dims, 
            num_dim=Xtr_num.shape[1], 
            wide_dim=wide_dim,
            **MODEL_CONFIGS["wide_deep"]
        ).to(device)
        
        print("\n[RUN] Wide & Deep")
        wd = train_with_early_stopping(
            wd, train_loader, val_loader, device, pos_weight,
            lr=args.lr, epochs=args.epochs, patience=args.patience
        )
        
        val_prob, val_y = infer_probs(wd, val_loader, device)
        thr = find_best_threshold(val_y, val_prob, metric="f1")
        test_prob, test_y = infer_probs(wd, test_loader, device)
        
        print("\n==== Wide & Deep (Best threshold on val) ====")
        m = metrics_report(test_y, test_prob, thr)
        results.append(("Wide_Deep", m))

    # ---------------- DeepFM ----------------
    if "deepfm" in args.models:
        dfm = DeepFM(
            cat_dims=cat_dims, 
            num_dim=Xtr_num.shape[1], 
            wide_dim=wide_dim,
            **MODEL_CONFIGS["deepfm"]
        ).to(device)
        
        print("\n[RUN] DeepFM")
        dfm = train_with_early_stopping(
            dfm, train_loader, val_loader, device, pos_weight,
            lr=args.lr, epochs=args.epochs, patience=args.patience
        )
        
        val_prob, val_y = infer_probs(dfm, val_loader, device)
        thr = find_best_threshold(val_y, val_prob, metric="f1")
        test_prob, test_y = infer_probs(dfm, test_loader, device)
        
        print("\n==== DeepFM (Best threshold on val) ====")
        m = metrics_report(test_y, test_prob, thr)
        results.append(("DeepFM", m))

    # ---------------- FT-Transformer (lite) ----------------
    if "ftt" in args.models:
        ftt = FTTransformerLite(
            num_dim=Xtr_num.shape[1], 
            cat_dims=cat_dims,
            **MODEL_CONFIGS["ftt"]
        ).to(device)
        
        print("\n[RUN] FT-Transformer (lite)")
        ftt = train_with_early_stopping(
            ftt, train_loader, val_loader, device, pos_weight,
            lr=args.lr, epochs=args.epochs, patience=args.patience
        )
        
        val_prob, val_y = infer_probs(ftt, val_loader, device)
        thr = find_best_threshold(val_y, val_prob, metric="f1")
        test_prob, test_y = infer_probs(ftt, test_loader, device)
        
        print("\n==== FT-Transformer (lite) (Best threshold on val) ====")
        m = metrics_report(test_y, test_prob, thr)
        results.append(("FTTransformerLite", m))

    # ---------------- TabNet (optional) ----------------
    if "tabnet" in args.models:
        info = run_tabnet(
            Xtr_num, Xtr_cat_idx, Xva_num, Xva_cat_idx, Xte_num, Xte_cat_idx,
            y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),
            cat_dims=cat_dims, device=device, 
            epochs=max(50, args.epochs), 
            batch_size=args.batch_size, 
            lr=MODEL_CONFIGS["tabnet"]["lr"]
        )
        
        if info is None:
            print("[RUN] TabNet skipped.")
        else:
            results.append(("TabNet", {}))  # 세부 수치는 콘솔에 출력됨

    # 요약
    if results:
        print("\n===== SUMMARY (key metrics) =====")
        for name, m in results:
            if m:
                print(f"{name:12s} | acc={m.get('accuracy'):.4f}  rec={m.get('recall'):.4f}  f1={m.get('f1'):.4f}  roc_auc={m.get('roc_auc', np.nan):.4f}  auc_pr={m.get('auc_pr', np.nan):.4f}")
            else:
                print(f"{name:12s} | (see logs above)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Models for Data Engineering")
    parser.add_argument("--csv", type=str, default="dataV05.csv", help="데이터 파일 경로")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="랜덤 시드")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="배치 크기")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="최대 에포크 수")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"], help="Early stopping 인내심")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="학습률")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_CONFIG["models"]),
                        help="실행할 모델들 (comma-separated)")
    
    args = parser.parse_args()

    # 시드 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 모델 리스트 파싱
    args.models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    
    # 메인 실행
    main(args) 