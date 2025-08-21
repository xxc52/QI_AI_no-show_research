# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드입니다.

## 프로젝트 개요

병원 예약 노쇼(no-show) 예측을 위한 QI AI 연구 프로젝트입니다. 다양한 신경망 모델을 구현하고 비교하여 환자의 노쇼를 예측하는 것이 목표입니다.

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

## 주요 명령어

### 모델 실행

```bash
# 최종 데이터셋으로 모든 모델 실행 (날씨 + 클러스터 포함)
python main.py --csv final_dataset_with_weather_clusters.csv

# 특정 모델만 실행
python main.py --csv final_dataset_with_weather_clusters.csv --models mlp,wide_deep,deepfm,ftt

# 하이퍼파라미터 조정
python main.py --csv final_dataset_with_weather_clusters.csv --models deepfm --epochs 100 --batch_size 2048 --lr 1e-3 --patience 10

# 빠른 테스트 (적은 에포크)
python main.py --csv final_dataset_with_weather_clusters.csv --epochs 20 --batch_size 2048 --models mlp

# 기존 데이터셋 비교 분석용
python main.py --csv dataV05_with_all_features.csv --models mlp  # 날씨/클러스터 제외
```

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

모듈화된 구조로 관심사의 명확한 분리:

### 핵심 컴포넌트

1. **메인 진입점** (`main.py`): 전체 파이프라인 조정 - 데이터 로딩, 전처리, 모델 학습, 평가

2. **데이터 처리 파이프라인** (`models/data/`):
   - `preprocessing.py`: 데이터 로딩, 특징 공학, train/val/test 분할
   - `scaler.py`: 수치형 특징을 위한 커스텀 표준 스케일러
   - `dataset.py`: 효율적인 배치 로딩을 위한 PyTorch Dataset 래퍼

3. **모델 구현** (`models/models/`):
   - MLP, Wide&Deep, DeepFM, FT-Transformer, TabNet 각각 독립 모듈
   - 모든 모델은 `nn.Module` 상속, 일관된 forward() 인터페이스
   - 임베딩을 통한 수치형 및 범주형 특징 처리

4. **학습 프레임워크** (`models/training/`):
   - `trainer.py`: validation AUC-PR 기반 조기 종료 포함 학습 루프
   - `metrics.py`: 종합 평가 지표 (Accuracy, Recall, F1, ROC-AUC, AUC-PR)
   - `utils.py`: 모델 포워딩 및 임계값 최적화 헬퍼 함수

5. **설정** (`models/config.py`): 모든 모델과 학습 설정의 중앙화된 하이퍼파라미터 관리

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

## 중요 사항

- 노쇼 예측의 클래스 불균형을 pos_weight로 처리
- 모든 모델은 동일한 train/val/test 분할 (60/20/20) 및 층화 샘플링 사용  
- 클래스 불균형으로 인해 다양한 지표로 성능 평가 (특히 AUC-PR 중요)
- CPU와 CUDA 실행 모두 지원, 자동 장치 감지

## 연구 기여사항

1. **시계열 특징 엔지니어링**: 환자별 이력 기반 6개 파생변수로 예측 성능 향상
2. **지리적 클러스터링**: 81개 지역을 노쇼 패턴 기반 6개 클러스터로 차원 축소
3. **외생변수 통합**: 19개 날씨 변수로 환경적 요인이 노쇼에 미치는 영향 분석
4. **다중 모델 비교**: MLP, Wide&Deep, DeepFM, FT-Transformer, TabNet 성능 벤치마킹

## 주요 발견사항

- **날씨 영향**: 비오는 날 노쇼율 0.55%p 감소, 고온일 0.81%p 증가
- **지역 클러스터**: 최대 6.0%p 노쇼율 차이 (cluster_0 vs cluster_2)
- **계절성**: 가을 75.9%, 겨울 24.1% 분포 (브라질 남반구 특성)

## 프로젝트 파일 구조

```
├── final_dataset_with_weather_clusters.csv    # 최종 통합 데이터셋 (54개 변수)
├── neighbourhood_clustering_methodology.md     # 지역 클러스터링 방법론 문서
├── analyse_neighbourhood.py                   # 지역 분석 스크립트
├── validate_weather_data.py                   # 날씨 데이터 검증 스크립트  
├── create_final_dataset.py                    # 최종 데이터셋 생성 스크립트
├── neighbourhood_clusters.json                # 지역-클러스터 매핑 파일
└── weather_sum_2015.csv, weather_sum_2016.csv # 브라질 기상청 날씨 데이터
```