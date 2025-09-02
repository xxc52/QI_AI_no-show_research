# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드입니다.

## 프로젝트 개요

병원 예약 노쇼(no-show) 예측을 위한 QI AI 연구 프로젝트입니다. 다양한 신경망 모델을 구현하고 비교하여 환자의 노쇼를 예측하는 것이 목표입니다.

## 🚀 Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/xxc52/QI_AI_no-show_research.git
cd QI_AI_no-show_research

# 2. 가상환경 생성 및 활성화 (권장)
python -m venv venv_noshow
# Windows:
venv_noshow\Scripts\activate.bat
# Linux/Mac:
source venv_noshow/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 모델 실행 - 빠른 테스트
python model/main.py --model randomforest --trials 5
python model/main.py --model lightgbm --no-optimize

# 5. 본격적인 하이퍼파라미터 최적화
python model/main.py --model lightgbm --trials 100
```

**GPU 사용자**:
```bash
# LightGBM GPU 활성화
export LIGHTGBM_USE_GPU=1  # Linux/Mac
set LIGHTGBM_USE_GPU=1     # Windows
python model/main.py --model lightgbm --trials 100

# GPU 설정 상세 가이드
cat model/GPU_SETUP_GUIDE.md
```

### 데이터셋 정보

**final_dataset_with_weather_clusters.csv**: 108,296개의 예약 기록, 62,299명의 고유 환자

- **노쇼 비율**: 20.11% (클래스 불균형)
- **데이터 구조**: 환자별로 정렬된 시계열 데이터 (동일 환자의 여러 예약 포함)
- **총 변수**: 54개 (날씨 변수 19개, 지역 클러스터 1개 추가)
- **분석 기간**: 2016년 4월 29일 ~ 6월 8일 (41일간)

**데이터셋 히스토리**:

- `dataV05.csv`: 원본 병원 예약 데이터 (27개 변수)
- `dataV05_with_all_features.csv`: 환자 이력 파생변수 추가 (33개 변수)
- `final_dataset_with_weather_clusters.csv`: 날씨 데이터 + 지역 클러스터 통합 (54개 변수)

### 주요 변수 설명

**기본 변수**:

- `PatientId`: 환자 ID
- `AppointmentID`: 예약 ID
- `Registration_Date`: 예약 등록 일시
- `Appointment_Date`: 진료 예정일
- `Gender_F`: 성별 (여성=1)
- `Age`: 나이
- `Neighbourhood`: 거주 지역
- `Scholarship`: 장학금 수혜 여부
- `Hipertension`: 고혈압 여부
- `Diabetes`: 당뇨 여부
- `Alcoholism`: 알코올 중독 여부
- `Handcap`: 장애 여부
- `SMS_received`: SMS 알림 수신 여부
- `lead_time`: 예약 등록일과 진료일 간의 실제 일수 차이 (평균: 10일)
- `is_same_day`: 등록일과 진료일이 같은 날인지 (당일 예약)
- **`No-show`**: 타겟 변수 (노쇼=1, 출석=0)

**환자 이력 기반 파생 변수** (시계열 특징):

- `patient_previous_noshow_count`: 해당 시점까지 환자의 이전 노쇼 횟수 (0-10회)
- `patient_appointment_count`: 해당 시점까지 환자의 총 예약 횟수 (0-75회)
- `patient_previous_noshow_rate`: 해당 시점까지 환자의 노쇼 비율 (0.0-1.0)
- `days_since_last_appointment`: 마지막 예약으로부터의 경과 일수 (0-40일)
- `time_between_appointments_avg`: 환자의 평균 예약 간격 (0-40일)
- `appointment_regularity`: 예약 간격의 표준편차/규칙성 (0-20)

**지역 클러스터 변수** (차원 축소):

- `neighbourhood_cluster`: 81개 지역을 6개 클러스터로 분류
  - `cluster_0`: 중장년층, 낮은 노쇼율 (18.7%)
  - `cluster_1`: 젊은층, 높은 장학금 비율 (19.6%)
  - `cluster_2`: 높은 노쇼율 지역 (24.7%)
  - `cluster_3`: 중간 노쇼율 지역 (22.4%)
  - `cluster_4`: 중년층, 낮은 노쇼율 (19.4%)
  - `cluster_rare`: 데이터 부족 지역 (23.8%)

**날씨 변수** (외생변수, 19개):

- **기본 기상 데이터**: `temp_avg`, `temp_max`, `temp_min`, `rain_max`, `hum_max`, `hum_min`, `wind_avg`, `rad_max`
- **파생 날씨 변수**:
  - `temp_range`: 일교차
  - `temp_change`: 전날 대비 기온 변화
  - `is_rainy`: 비 온 날 (40.9%)
  - `is_hot`: 고온일 >30°C (78.0%)
  - `is_cold`: 저온일 <20°C (0.0%)
  - `is_humid`: 고습도일 >90% (100.0%)
  - `is_windy`: 강풍일 >3m/s
  - `season`: 계절 (가을 75.9%, 겨울 24.1%)

## Feature Selection 결과

### ML-Ready 데이터셋

**Univariate Statistical Test 결과**:

- **전체 46개 변수** 중 4개 상수 변수 제거 후 42개 변수 분석
- **24개 변수 선택** (57.1% 선택률, p-value < 0.05)
- ANOVA F-test (연속형), Chi-square test (범주형/이진형) 적용

**선택된 주요 변수**:

1. **예약 타이밍 (6개)**: lead_time, is_same_day, Registration_Hour/Month/Day/Shift
2. **환자 이력 (6개)**: 모든 patient\_\* 시계열 변수들
3. **의료 상태 (3개)**: Hipertension, Diabetes, Handcap
4. **환자 정보 (2개)**: Age, Scholarship
5. **시스템 (1개)**: SMS_received
6. **지역 (1개)**: neighbourhood_cluster_encoded
7. **날씨 (4개)**: temp_change, rad_max, temp_range, temp_min
8. **시간 (1개)**: season_encoded

**ML-Ready 데이터셋**:

- `feature_selection_analysis/ml_dataset_all_features.csv`: 108,296 × 44 (전체 43개 feature)
- `feature_selection_analysis/ml_dataset_selected_features.csv`: 108,296 × 27 (선택된 24개 feature)
- 모든 categorical 변수 수치화 완료 (neighbourhood_cluster, season 인코딩)
- **PatientId, AppointmentID 포함** (시계열 분석, data leakage 방지용)

**중요 사항**:

- **Data Leakage 방지**: 환자별 train/test split 필수 (동일 환자가 train/test에 동시 포함 금지)
- **시계열 특성**: 데이터는 PatientId와 시간순으로 정렬됨
- 사용법과 주의사항은 `feature_selection_analysis/feature_lists_and_usage_guide.txt` 참조

## 데이터 분할 (Data Splitting)

### Temporal Split 방식 (현재 방식)

**위치**: `data/` 폴더
**방법**: 시간 기반 8:1:1 순차 분할

```bash
# Temporal split 실행
cd data
python split_dataset_temporal.py
```

**분할 결과**:
- `train.csv`: 86,636개 (80%) - 2016-04-29 ~ 2016-06-02 (34일)
- `val.csv`: 10,829개 (10%) - 2016-06-02 ~ 2016-06-06 (4일) 
- `test.csv`: 10,831개 (10%) - 2016-06-06 ~ 2016-06-08 (2일)

**특징**:
- **현실적**: 과거 데이터로 훈련 → 미래 데이터 예측 (실제 배포 시나리오)
- **단순함**: 복잡한 환자별 규칙 없이 시간순 분할
- **시간적 무결성**: Training < Validation < Test 순서 보장

## 주요 명령어

### Feature Selection 실행

```bash
# Feature selection 분석 (univariate statistical test)
cd feature_selection_analysis
python univariate_feature_selection.py

# ML-ready 데이터셋 생성
python create_ml_ready_dataset.py
```

### 모델 실행 (시간 분할 데이터 사용)

```bash
# Temporal split 데이터로 모델 실행 (권장)
python main.py --csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv --models mlp

# 또는 기존 방식 (전체 데이터셋으로 실행)
python main.py --csv final_dataset_with_weather_clusters.csv --models mlp

# 하이퍼파라미터 조정
python main.py --csv data/train.csv --val_csv data/val.csv --models deepfm --epochs 100 --batch_size 2048 --lr 1e-3 --patience 10
```

**⚠️ 참고**: 실제 모델링 파이프라인에서는 `train.csv`와 `val.csv`를 결합하여 **TimeSeriesSplit**을 사용할 예정입니다. 이는 시계열 데이터의 특성을 더 잘 활용하기 위함입니다.

### 가상환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv_noshow
# Windows:
venv_noshow\Scripts\activate.bat
# Linux/Mac:
source venv_noshow/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 아키텍처 개요

### 🚀 새로운 ML Pipeline (`model/` 폴더) - **현재 사용 중**

완전히 새롭게 구축된 연구용 ML 파이프라인으로, 6개 모델과 포괄적인 평가 시스템을 제공합니다.

#### 핵심 컴포넌트

1. **메인 진입점** (`model/main.py`): 전체 파이프라인 조정 및 모델별 실행
2. **벤치마킹 시스템** (`model/benchmark.py`): 모든 모델 자동 비교 (`--model all`)
3. **데이터 로더** (`model/data/loader.py`): 시간분할 데이터 로딩 및 TimeSeriesSplit
4. **모델 구현** (`model/models/`): 6개 모델 각각 독립 모듈
   - `random_forest.py`: 기준 앙상블 모델
   - `lightgbm_model.py`: 고성능 그래디언트 부스팅
   - `mlp_model.py`: 다층 퍼셉트론 (PyTorch)
   - `deepfm_model.py`: DeepFM (Linear + FM + Deep)
   - `fttransformer_model.py`: Feature Tokenizer + Transformer
   - `tabnet_model.py`: 어텐션 기반 해석 가능 모델
5. **학습 시스템** (`model/training/`):
   - `validator.py`: TimeSeriesSplit 교차 검증
   - `optimizer.py`: Optuna 베이지안 최적화 (PR-AUC 최적화)
6. **평가 시스템** (`model/evaluation/metrics.py`): 15+ 불균형 데이터 특화 지표
7. **설정 관리** (`model/config/config.py`): 모든 모델 하이퍼파라미터 중앙 관리

#### 주요 사용법

```bash
# 단일 모델 실행 (권장)
python model/main.py --model randomforest --trials 50
python model/main.py --model lightgbm --trials 100

# 모든 모델 비교 (벤치마킹) - GPU 권장
python model/main.py --model all --trials 100

# 빠른 테스트 (기본 파라미터로 실행)
python model/main.py --model lightgbm --no-optimize

# GPU 사용 (LightGBM)
export LIGHTGBM_USE_GPU=1  # Linux/Mac
set LIGHTGBM_USE_GPU=1     # Windows
python model/main.py --model lightgbm --trials 100
```

### 📊 지원 모델 및 특징

| 모델 | 타입 | 주요 특징 | 사용 사례 |
|------|------|-----------|-----------|
| **RandomForest** | 앙상블 | 빠른 훈련, 해석 가능 | 기준 모델, 특징 중요도 |
| **LightGBM** | 그래디언트 부스팅 | 고성능, GPU 지원 | 실제 배포, 최고 성능 |
| **MLP** | 딥러닝 | BatchNorm, Dropout, GPU 자동감지 | 딥러닝 기준 |
| **DeepFM** | 하이브리드 | Linear + FM + Deep, GPU 자동감지 | 특징 상호작용 |
| **FT-Transformer** | 트랜스포머 | Self-attention, GPU 자동감지 | 최신 테이블 학습 |
| **TabNet** | 어텐션 | 해석 가능한 어텐션, GPU 자동감지 | 해석 가능한 딥러닝 |

#### 평가 지표 (불균형 데이터 특화)

- **주요 지표**: PR-AUC (Precision-Recall AUC)
- **기본 지표**: Accuracy, Precision, Recall, F1, F2
- **고급 지표**: MCC, Cohen's Kappa, G-Mean, Balanced Accuracy
- **확률 지표**: ROC-AUC, Brier Score, Log Loss
- **성능 지표**: 훈련시간, 예측시간 (ms)

---

### 📚 기존 구현 (`models/` 폴더) - **더 이상 사용 안 함**

이전 버전의 구현으로, 현재는 사용하지 않습니다. 삭제해도 됩니다.

**구성 요소** (참고용):
- `preprocessing.py`: 데이터 로딩, 특징 공학, train/val/test 분할
- `scaler.py`: 수치형 특징을 위한 커스텀 표준 스케일러  
- `dataset.py`: 효율적인 배치 로딩을 위한 PyTorch Dataset 래퍼
- `trainer.py`: validation AUC-PR 기반 조기 종료 포함 학습 루프
- `metrics.py`: 종합 평가 지표 (Accuracy, Recall, F1, ROC-AUC, AUC-PR)
- `utils.py`: 모델 포워딩 및 임계값 최적화 헬퍼 함수

### 주요 설계 패턴

- **통합 데이터 인터페이스**: 모든 모델이 동일한 TabDataset 형식으로 데이터 수신
- **조기 종료**: validation AUC-PR이 개선되지 않으면 자동으로 학습 중단
- **클래스 불균형 처리**: BCEWithLogitsLoss의 pos_weight 자동 계산
- **임계값 최적화**: F1-score 최대화를 위한 검증 세트 활용

### 모델별 특징

- **Wide&Deep, DeepFM**: 선형(wide)과 deep 컴포넌트 결합, 원-핫 인코딩과 임베딩 모두 필요
- **FT-Transformer**: self-attention 메커니즘, 임베딩 차원과 attention head 수에 민감
- **TabNet**: 선택적 의존성 (pytorch-tabnet), 미설치시 우아한 처리
- **MLP**: BatchNorm과 Dropout으로 정규화된 기준 모델

## GPU 지원 및 성능

### 🎮 GPU 자동 지원
- **신경망 모델** (MLP, DeepFM, FT-Transformer, TabNet): PyTorch CUDA 자동 감지
- **LightGBM**: 환경변수 `LIGHTGBM_USE_GPU=1`로 활성화
- **상세 가이드**: `model/GPU_SETUP_GUIDE.md` 참고

### ⚡ 성능 예상치
**CPU 모드** (현재 테스트 환경):
- RandomForest: ~7-10초/trial
- LightGBM: ~10-15초/trial  
- 신경망 모델: 2-5분/trial (trials=1 기준)

**GPU 모드** (CUDA 지원 환경):
- LightGBM: 2-3배 속도 향상
- 신경망 모델: 5-10배 속도 향상

## 중요 사항

- **클래스 불균형 처리**: pos_weight와 scale_pos_weight로 노쇼 클래스(20.11%) 가중치 조정
- **시간 기반 분할**: Temporal split으로 train/val/test (80/10/10) 순차 분할
- **평가 지표**: PR-AUC를 주요 지표로 사용 (불균형 데이터에 적합)
- **결과 저장**: 모든 실행 결과가 `model/results/`에 timestamp와 함께 자동 저장
- **재현 가능성**: 고정 시드(42)와 결정적 프로세스로 일관된 결과 보장

## 연구 기여사항

### 데이터 과학 기여사항
1. **시계열 특징 엔지니어링**: 환자별 이력 기반 6개 파생변수로 예측 성능 향상
2. **지리적 클러스터링**: 81개 지역을 노쇼 패턴 기반 6개 클러스터로 차원 축소
3. **외생변수 통합**: 19개 날씨 변수로 환경적 요인이 노쇼에 미치는 영향 분석
4. **통계적 특징 선택**: Univariate statistical test로 46→24개 변수 선별 (57.1% 선택률)
5. **현실적 데이터 분할**: 시간 기반 순차 분할로 실제 배포 시나리오 반영

### 모델링 및 시스템 기여사항
6. **포괄적 모델 비교**: 6개 최신 모델의 체계적 벤치마킹 시스템
   - 전통적 ML: RandomForest, LightGBM
   - 딥러닝: MLP, DeepFM, FT-Transformer, TabNet
7. **불균형 데이터 특화 평가**: PR-AUC 최적화 및 15+ 평가지표 체계
8. **연구용 하이퍼파라미터 최적화**: Optuna 베이지안 최적화로 효율성 향상
9. **시간 무결성 보장**: TimeSeriesSplit 교차 검증으로 데이터 누출 방지
10. **생산 배포 고려**: 훈련/예측 시간 측정 및 모델 지속성 지원
11. **재현가능한 연구**: 고정 시드, 결정적 프로세스, 포괄적 문서화

## 주요 발견사항

**Feature Selection 결과**:

- **최고 예측력**: lead_time (F=3861.8), is_same_day (χ²=5706.6), SMS_received (χ²=1232.4)
- **환자 이력**: 모든 patient\_\* 변수들이 통계적 유의성 확보 (p<0.001)
- **날씨 변수**: 19개 중 4개만 선택 (temp_change, rad_max, temp_range, temp_min)

**데이터 특성**:

- **날씨 영향**: 비오는 날 노쇼율 0.55%p 감소, 고온일 0.81%p 증가
- **지역 클러스터**: 최대 6.0%p 노쇼율 차이 (cluster_0 vs cluster_2)
- **계절성**: 가을 75.9%, 겨울 24.1% 분포 (브라질 남반구 특성)

## 프로젝트 파일 구조

```
├── final_dataset_with_weather_clusters.csv    # 최종 통합 데이터셋 (54개 변수)
├── data/                                      # 데이터 분할 결과
│   ├── split_dataset_temporal.py             # Temporal split 구현
│   ├── train.csv                             # 훈련 데이터 (80%, 34일간)
│   ├── val.csv                               # 검증 데이터 (10%, 4일간)
│   ├── test.csv                              # 테스트 데이터 (10%, 2일간)
│   ├── temporal_split_analysis.png           # 시각화 결과
│   └── temporal_split_report.md              # 분할 방법론 보고서
├── feature_selection_analysis/                # Feature Selection 분석 결과
│   ├── ml_dataset_all_features.csv           # ML용 전체 데이터셋 (108,296×44)
│   ├── ml_dataset_selected_features.csv      # ML용 선택 데이터셋 (108,296×27)
│   ├── feature_lists_and_usage_guide.txt     # 사용법, Data leakage 방지 가이드
│   ├── univariate_test_results_all_features.csv # 통계분석 상세 결과
│   ├── univariate_feature_selection.py       # Feature selection 분석 스크립트
│   └── create_ml_ready_dataset.py            # ML 데이터셋 생성 스크립트
├── model/                                     # ✅ 새로운 ML 파이프라인 (현재 사용)
│   ├── main.py                               # 메인 실행 파일
│   ├── benchmark.py                          # 포괄적 모델 벤치마킹
│   ├── GPU_SETUP_GUIDE.md                   # GPU 설정 가이드
│   ├── config/config.py                      # 중앙 설정 관리
│   ├── data/loader.py                        # 시간분할 데이터 로더
│   ├── models/                               # 6개 모델 구현
│   │   ├── random_forest.py                 # RandomForest 모델 (✅ 버그 수정)
│   │   ├── lightgbm_model.py                # LightGBM 모델 (✅ GPU 지원)
│   │   ├── mlp_model.py                     # MLP (PyTorch, GPU 자동감지)
│   │   ├── deepfm_model.py                  # DeepFM 모델 (GPU 자동감지)
│   │   ├── fttransformer_model.py           # FT-Transformer (GPU 자동감지)
│   │   └── tabnet_model.py                  # TabNet 모델 (GPU 자동감지)
│   ├── training/                             # 훈련 시스템
│   │   ├── validator.py                     # TimeSeriesSplit 교차 검증
│   │   └── optimizer.py                     # Optuna 베이지안 최적화
│   ├── evaluation/metrics.py                 # 15+ 평가 지표
│   ├── utils/timer.py                        # 성능 측정 도구
│   └── results/                              # 결과 저장소 (폴더 구조만 유지)
│       ├── benchmark/                        # 벤치마크 결과 (.gitkeep)
│       ├── best_params/                      # 최적 하이퍼파라미터 (.gitkeep)
│       ├── models/                           # 훈련된 모델 (.gitkeep)
│       ├── figures/                          # 시각화 결과 (.gitkeep)
│       └── logs/                             # 훈련 로그 (.gitkeep)
├── models/                                    # ❌ 기존 구현 (삭제 가능)
│   ├── config.py                             # 이전 버전 설정
│   ├── data/                                 # 이전 버전 데이터 처리
│   ├── models/                               # 이전 버전 모델들
│   └── training/                             # 이전 버전 훈련 코드
├── neighbourhood_clustering_methodology.md     # 지역 클러스터링 방법론 문서
├── analyse_neighbourhood.py                   # 지역 분석 스크립트
├── validate_weather_data.py                   # 날씨 데이터 검증 스크립트
├── create_final_dataset.py                    # 최종 데이터셋 생성 스크립트
├── neighbourhood_clusters.json                # 지역-클러스터 매핑 파일
└── weather_sum_2015.csv, weather_sum_2016.csv # 브라질 기상청 날씨 데이터
```

### 📂 주요 폴더 설명

- **`model/`**: 🚀 **현재 사용 중인 완전한 ML 파이프라인**
  - 6개 모델, 벤치마킹 시스템, 포괄적 평가 지표
  - 연구용 최적화된 하이퍼파라미터 튜닝
  - 시간분할 데이터, TimeSeriesSplit 교차 검증

- **`models/`**: ❌ **기존 구현 (삭제 가능)**  
  - 이전 버전의 모델 구현
  - 현재는 `model/` 폴더의 새로운 파이프라인 사용
