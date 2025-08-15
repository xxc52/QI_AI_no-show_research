# 🚀 Neural Network Models for Data Engineering

`models2.py`는 Data Engineering을 위한 다양한 신경망 모델들을 구현하고 성능을 비교하는 Python 스크립트입니다.

##  **파일 구조**

```
.
├── main.py                    # 메인 실행 파일
├── models/                    # 모듈화된 코드
│   ├── __init__.py           # 패키지 초기화
│   ├── config.py             # 설정 파일
│   ├── data/                 # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # 데이터 전처리
│   │   ├── scaler.py        # 스케일러
│   │   └── dataset.py       # PyTorch Dataset
│   ├── models/               # 신경망 모델 모듈
│   │   ├── __init__.py
│   │   ├── mlp.py           # MLP 모델
│   │   ├── wide_deep.py     # Wide & Deep 모델
│   │   ├── deepfm.py        # DeepFM 모델
│   │   ├── fttransformer.py # FT-Transformer 모델
│   │   └── tabnet.py        # TabNet 모델
│   └── training/             # 학습 및 평가 모듈
│       ├── __init__.py
│       ├── trainer.py        # 학습 루틴
│       ├── metrics.py        # 평가 지표
│       └── utils.py          # 학습 유틸리티
├── requirements.txt           # 패키지 의존성
├── README.md                 # 이 파일
├── install.sh                # Linux/Mac 자동 설치 스크립트
├── install.bat               # Windows 자동 설치 스크립트
├── dataV05.csv               # 데이터셋
└── venv_noshow/              # 가상환경
```

##  **사용 방법**

### **기본 실행**

```bash
# 모든 모델 실행 (기본 설정)
python main.py

# 특정 모델만 실행
python main.py --models mlp,wide_deep

# 커스텀 설정으로 실행
python main.py --models deepfm,ftt --epochs 100 --batch_size 1024 --lr 1e-3
```

### **명령행 옵션**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--csv` | 데이터 파일 경로 | `dataV05.csv` |
| `--seed` | 랜덤 시드 | `42` |
| `--batch_size` | 배치 크기 | `1024` |
| `--epochs` | 최대 에포크 수 | `50` |
| `--patience` | Early stopping 인내심 | `6` |
| `--lr` | 학습률 | `1e-3` |
| `--models` | 실행할 모델들 | `mlp,wide_deep,deepfm,ftt,tabnet` |

### **실행 예시**

#### **1. 빠른 테스트 (적은 에포크)**
```bash
python main.py --epochs 20 --batch_size 2048 --models mlp
```

#### **2. 고품질 학습 (많은 에포크)**
```bash
python main.py --epochs 100 --patience 15 --lr 5e-4 --models mlp,wide_deep
```

#### **3. 특정 모델만 실행**
```bash
# DeepFM만
python main.py --models deepfm --epochs 60 --batch_size 2048 --lr 1e-3 --patience 8

# FT-Transformer만
python main.py --models ftt --epochs 60 --batch_size 1024 --lr 7e-4 --patience 8
```

#### **4. 모든 모델 비교**
```bash
python main.py --models mlp,wide_deep,deepfm,ftt --epochs 60 --batch_size 2048 --lr 1e-3 --patience 8
```

##  지원 모델

- **MLP + BatchNorm + Dropout**: 기본적인 다층 퍼셉트론
- **Wide & Deep**: 선형 모델과 신경망의 결합
- **DeepFM**: Factorization Machine과 Deep MLP의 결합
- **FT-Transformer(lite)**: Transformer 기반 특성 학습 모델
- **TabNet**: 선택적 (설치 시에만 사용 가능)

##  설치 방법

###  **방법 1: 자동 설치 스크립트 사용 (권장)**

#### **Linux/Mac 사용자**
```bash
# 스크립트 실행 권한 부여
chmod +x install.sh

# 자동 설치 실행
./install.sh
```

#### **Windows 사용자**
```cmd
# 자동 설치 실행
install.bat
```

**자동 설치 스크립트의 장점:**
-  Python 버전 자동 확인
-  가상환경 자동 생성 및 활성화
-  GPU 자동 감지 및 CUDA 지원 PyTorch 설치
-  모든 필수 패키지 자동 설치
-  설치 상태 자동 확인

---

###  **방법 2: requirements.txt 사용**

#### **1단계: 가상환경 생성 및 활성화**
```bash
# 가상환경 생성
python -m venv venv_noshow

# 가상환경 활성화 (Linux/Mac)
source venv_noshow/bin/activate

# 가상환경 활성화 (Windows)
venv_noshow\Scripts\activate.bat
```

#### **2단계: 패키지 설치**
```bash
# pip 업그레이드
pip install --upgrade pip

# requirements.txt로 모든 패키지 설치
pip install -r requirements.txt

---

###  **방법 3: 개별 패키지 설치**

#### **1단계: 가상환경 생성 및 활성화**
```bash
# 가상환경 생성
python -m venv venv_noshow

# 가상환경 활성화 (Linux/Mac)
source venv_noshow/bin/activate

# 가상환경 활성화 (Windows)
venv_noshow\Scripts\activate.bat
```

#### **2단계: 기본 패키지 설치**
```bash
# 데이터 처리 및 수치 계산
pip install numpy pandas scipy

# 머신러닝
pip install scikit-learn

# 유틸리티
pip install tqdm joblib
```

#### **3단계: 딥러닝 프레임워크 설치**

**PyTorch 설치 (CUDA 지원)**
```bash
# CUDA 12.1 지원 (최신 GPU - RTX 4000 시리즈)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 지원 (구형 GPU - RTX 3000 시리즈)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.7 지원 (구형 GPU - GTX 1000 시리즈)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU 전용 (GPU 없음)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

##  성능 지표

각 모델은 다음 지표들로 평가됩니다:

- **Accuracy**: 정확도
- **Recall**: 재현율 (클래스 불균형 문제 해결에 중요)
- **F1-Score**: 정밀도와 재현율의 조화평균
- **ROC-AUC**: ROC 곡선 아래 면적
- **AUC-PR**: Precision-Recall 곡선 아래 면적 (불균형 데이터에 중요)

##  시스템 요구사항

### 최소 요구사항
- Python 3.8+
- RAM: 8GB+
- 저장공간: 2GB+

### 권장 사양
- Python 3.9+
- RAM: 16GB+
- GPU: NVIDIA GPU (CUDA 지원)
- 저장공간: 5GB+

##  문제 해결

### 일반적인 오류들

#### 1. CUDA 관련 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"
```

#### 2. 메모리 부족 오류
```bash
# 배치 크기 줄이기
python models2.py --batch_size 512

# 모델 수 줄이기
python models2.py --models mlp,wide_deep
```

#### 3. TabNet 설치 오류
```bash
# TabNet 건너뛰기
python models2.py --models mlp,wide_deep,deepfm,ftt
```

##  성능 최적화 팁

### 1. 하이퍼파라미터 튜닝
- **학습률**: 1e-4 ~ 1e-2 범위에서 실험
- **배치 크기**: GPU 메모리에 맞게 조정
- **에포크**: 데이터 크기에 따라 50~200 범위

### 2. 모델 선택 가이드
- **높은 Recall 필요**: MLP 모델 선택
- **균형잡힌 성능**: FT-Transformer 모델 선택
- **불균형 데이터**: DeepFM 모델 선택
- **안정성 중시**: Wide & Deep 모델 선택

### 3. 데이터 전처리
- 범주형 변수가 많은 경우: Wide & Deep, DeepFM 권장
- 수치형 변수가 많은 경우: FT-Transformer 권장
- 클래스 불균형이 심한 경우: pos_weight 자동 조정 활용

- 클래스 불균형이 심한 경우: pos_weight 자동 조정 활용
