#!/bin/bash

# ============================================================================
# models2.py 자동 설치 스크립트
# ============================================================================

echo "🚀 models2.py 설치를 시작합니다..."

# 1. Python 버전 확인
echo "📋 Python 버전 확인 중..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [ $? -ne 0 ]; then
    echo "❌ Python3가 설치되어 있지 않습니다. Python3를 먼저 설치해주세요."
    exit 1
fi

echo "✅ Python 버전: $python_version"

# 2. 가상환경 생성
echo "📦 가상환경 생성 중..."
if [ -d "venv_noshow" ]; then
    echo "⚠️  venv_noshow 디렉토리가 이미 존재합니다. 기존 환경을 사용합니다."
else
    python3 -m venv venv_noshow
    echo "✅ 가상환경 생성 완료"
fi

# 3. 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source venv_noshow/bin/activate

# 4. pip 업그레이드
echo "⬆️  pip 업그레이드 중..."
pip install --upgrade pip

# 5. 기본 패키지 설치
echo "📚 기본 패키지 설치 중..."
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm joblib

# 6. PyTorch 설치 (CUDA 지원)
echo "🔥 PyTorch 설치 중..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU 감지됨. CUDA 지원 PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "💻 CPU 전용 PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 7. TabNet 설치
echo "🧠 TabNet 설치 중..."
pip install pytorch-tabnet

# 8. 설치 확인
echo "✅ 설치 확인 중..."
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ Scikit-learn: {sklearn.__version__}')
print(f'✅ CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA 버전: {torch.version.cuda}')
"

echo ""
echo "🎉 설치가 완료되었습니다!"
echo ""
echo "📋 사용 방법:"
echo "1. 가상환경 활성화: source venv_noshow/bin/activate"
echo "2. 모델 실행: python models2.py --help"
echo ""
echo "🚀 빠른 시작:"
echo "python models2.py --models mlp --epochs 20"
echo ""
echo "📖 자세한 사용법은 README.md를 참조하세요." 