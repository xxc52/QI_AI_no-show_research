"""
Panel Data EDA for Hospital No-show Dataset
이 스크립트는 ML-ready 데이터셋에 대한 종합적인 EDA를 수행합니다.
특히 패널 데이터 구조와 환자별 예약 패턴에 중점을 둡니다.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 스타일 설정
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_and_initial_inspection(file_path):
    """데이터 로드 및 초기 검사"""
    print("="*80)
    print("1. 데이터 로드 및 기본 정보")
    print("="*80)
    
    df = pd.read_csv(file_path)
    print(f"데이터셋 shape: {df.shape}")
    print(f"- 총 예약 수: {df.shape[0]:,}")
    print(f"- 총 변수 수: {df.shape[1]}")
    
    # 고유 환자 수 확인
    n_unique_patients = df['PatientId'].nunique()
    print(f"\n- 고유 환자 수: {n_unique_patients:,}")
    print(f"- 환자당 평균 예약 수: {df.shape[0]/n_unique_patients:.2f}")
    
    # 컬럼 정보
    print("\n컬럼 정보:")
    print(df.info())
    
    # 타겟 변수 분포
    print("\n타겟 변수 (No-show) 분포:")
    noshow_dist = df['No-show'].value_counts()
    noshow_pct = df['No-show'].value_counts(normalize=True) * 100
    print(f"- 출석 (0): {noshow_dist[0]:,} ({noshow_pct[0]:.2f}%)")
    print(f"- 노쇼 (1): {noshow_dist[1]:,} ({noshow_pct[1]:.2f}%)")
    print(f"- 클래스 불균형 비율: 1:{noshow_dist[0]/noshow_dist[1]:.2f}")
    
    return df

def analyze_panel_structure(df):
    """패널 데이터 구조 분석"""
    print("\n" + "="*80)
    print("2. 패널 데이터 구조 분석")
    print("="*80)
    
    # 환자별 예약 횟수 분포
    patient_appointment_counts = df.groupby('PatientId').size()
    
    print("\n환자별 예약 횟수 통계:")
    print(patient_appointment_counts.describe())
    
    # 예약 횟수별 환자 분포
    appointment_distribution = patient_appointment_counts.value_counts().sort_index()
    
    print("\n예약 횟수별 환자 수 분포:")
    print("예약횟수 | 환자수 | 비율(%) | 누적비율(%)")
    print("-" * 50)
    
    cumulative_pct = 0
    for n_appts in range(1, min(11, appointment_distribution.index.max()+1)):
        if n_appts in appointment_distribution.index:
            count = appointment_distribution[n_appts]
            pct = (count / patient_appointment_counts.shape[0]) * 100
            cumulative_pct += pct
            print(f"{n_appts:^10} | {count:^7} | {pct:^7.2f} | {cumulative_pct:^11.2f}")
    
    # 10회 이상 예약 환자
    many_appointments = appointment_distribution[appointment_distribution.index >= 10].sum()
    if many_appointments > 0:
        many_pct = (many_appointments / patient_appointment_counts.shape[0]) * 100
        cumulative_pct += many_pct
        print(f"{'10+':^10} | {many_appointments:^7} | {many_pct:^7.2f} | {cumulative_pct:^11.2f}")
    
    # 중요 통계
    single_appointment_patients = appointment_distribution[1] if 1 in appointment_distribution.index else 0
    single_appointment_pct = (single_appointment_patients / patient_appointment_counts.shape[0]) * 100
    
    print(f"\n핵심 통계:")
    print(f"- 단일 예약 환자: {single_appointment_patients:,}명 ({single_appointment_pct:.2f}%)")
    print(f"- 2회 이상 예약 환자: {patient_appointment_counts.shape[0] - single_appointment_patients:,}명 ({100-single_appointment_pct:.2f}%)")
    print(f"- 5회 이상 예약 환자: {(patient_appointment_counts >= 5).sum():,}명 ({(patient_appointment_counts >= 5).sum()/patient_appointment_counts.shape[0]*100:.2f}%)")
    
    return patient_appointment_counts

def analyze_temporal_features(df):
    """시계열 특징 분석"""
    print("\n" + "="*80)
    print("3. 시계열 특징 분석")
    print("="*80)
    
    temporal_features = [
        'patient_previous_noshow_count',
        'patient_appointment_count', 
        'patient_previous_noshow_rate',
        'days_since_last_appointment',
        'time_between_appointments_avg',
        'appointment_regularity'
    ]
    
    print("\n시계열 특징 변수 통계:")
    print(df[temporal_features].describe())
    
    # 첫 번째 예약과 이후 예약 구분
    first_appointments = df[df['patient_appointment_count'] == 0]
    subsequent_appointments = df[df['patient_appointment_count'] > 0]
    
    print(f"\n첫 번째 예약: {len(first_appointments):,}개 ({len(first_appointments)/len(df)*100:.2f}%)")
    print(f"후속 예약: {len(subsequent_appointments):,}개 ({len(subsequent_appointments)/len(df)*100:.2f}%)")
    
    # 첫 번째 예약과 후속 예약의 노쇼율 비교
    print("\n예약 유형별 노쇼율:")
    print(f"- 첫 번째 예약 노쇼율: {first_appointments['No-show'].mean()*100:.2f}%")
    print(f"- 후속 예약 노쇼율: {subsequent_appointments['No-show'].mean()*100:.2f}%")
    
    # 이전 노쇼 이력과 현재 노쇼의 관계
    print("\n이전 노쇼 이력에 따른 현재 노쇼율:")
    for i in range(0, min(6, int(df['patient_previous_noshow_count'].max()+1))):
        subset = df[df['patient_previous_noshow_count'] == i]
        if len(subset) > 0:
            noshow_rate = subset['No-show'].mean() * 100
            print(f"- 이전 노쇼 {i}회: {noshow_rate:.2f}% (n={len(subset):,})")
    
    # 이전 노쇼율과 현재 노쇼의 관계
    print("\n이전 노쇼율 구간별 현재 노쇼율:")
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    # 이전 예약이 있는 환자만 대상
    subset_with_history = df[df['patient_appointment_count'] > 0].copy()
    if len(subset_with_history) > 0:
        subset_with_history['noshow_rate_bin'] = pd.cut(subset_with_history['patient_previous_noshow_rate'], 
                                                         bins=bins, labels=labels, include_lowest=True)
        for label in labels:
            bin_data = subset_with_history[subset_with_history['noshow_rate_bin'] == label]
            if len(bin_data) > 0:
                current_noshow = bin_data['No-show'].mean() * 100
                print(f"- 이전 노쇼율 {label}: 현재 노쇼율 {current_noshow:.2f}% (n={len(bin_data):,})")

def analyze_selected_features(df):
    """선택된 특징 변수 분석"""
    print("\n" + "="*80)
    print("4. Feature Selection 결과 변수 분석")
    print("="*80)
    
    # 선택된 변수 리스트 (PatientId, AppointmentID, No-show 제외)
    selected_features = [
        # 예약 타이밍
        'lead_time', 'is_same_day', 'Registration_Hour', 
        'Registration_Month', 'Registration_Day', 'Registration_Shift',
        # 환자 이력
        'patient_previous_noshow_count', 'patient_appointment_count',
        'patient_previous_noshow_rate', 'days_since_last_appointment',
        'time_between_appointments_avg', 'appointment_regularity',
        # 의료 상태
        'Hipertension', 'Diabetes', 'Handcap',
        # 환자 정보
        'Age', 'Scholarship',
        # 시스템
        'SMS_received',
        # 지역
        'neighbourhood_cluster_encoded',
        # 날씨
        'temp_change', 'rad_max', 'temp_range', 'temp_min',
        # 시간
        'season_encoded'
    ]
    
    # 실제로 데이터셋에 있는 변수만 필터링
    available_features = [f for f in selected_features if f in df.columns]
    
    print(f"선택된 특징 변수: {len(available_features)}개")
    
    # 변수 타입별 분류
    numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [f for f in available_features if f not in numeric_features]
    
    print(f"\n수치형 변수: {len(numeric_features)}개")
    print(f"범주형 변수: {len(categorical_features)}개")
    
    # 결측치 확인
    print("\n결측치 현황:")
    missing = df[available_features].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("결측치 없음")
    
    return available_features

def analyze_cross_validation_implications(df, patient_appointment_counts):
    """Cross-validation 전략을 위한 분석"""
    print("\n" + "="*80)
    print("5. Cross-validation 전략 분석")
    print("="*80)
    
    # 환자별 노쇼율 계산
    patient_noshow_stats = df.groupby('PatientId').agg({
        'No-show': ['sum', 'mean', 'count']
    })
    patient_noshow_stats.columns = ['total_noshows', 'noshow_rate', 'n_appointments']
    
    # 예약 횟수별 그룹 분석
    print("\n예약 횟수별 그룹 특성:")
    print("예약횟수 | 환자수 | 총예약수 | 평균노쇼율 | 데이터비중")
    print("-" * 60)
    
    for n_appts in [1, 2, 3, 4, 5]:
        patients = patient_noshow_stats[patient_noshow_stats['n_appointments'] == n_appts]
        if len(patients) > 0:
            total_appointments = n_appts * len(patients)
            avg_noshow = patients['noshow_rate'].mean() * 100
            data_proportion = (total_appointments / len(df)) * 100
            print(f"{n_appts:^9} | {len(patients):^7} | {total_appointments:^9} | {avg_noshow:^11.2f} | {data_proportion:^10.2f}%")
    
    # 5회 이상
    many_appts_patients = patient_noshow_stats[patient_noshow_stats['n_appointments'] >= 5]
    if len(many_appts_patients) > 0:
        total_appointments = many_appts_patients['n_appointments'].sum()
        avg_noshow = many_appts_patients['noshow_rate'].mean() * 100
        data_proportion = (total_appointments / len(df)) * 100
        print(f"{'5+':^9} | {len(many_appts_patients):^7} | {total_appointments:^9} | {avg_noshow:^11.2f} | {data_proportion:^10.2f}%")
    
    # 단일 예약 환자 분석
    single_appointment_patients = patient_noshow_stats[patient_noshow_stats['n_appointments'] == 1]
    print(f"\n단일 예약 환자 영향도:")
    print(f"- 환자 수: {len(single_appointment_patients):,}명 ({len(single_appointment_patients)/len(patient_noshow_stats)*100:.2f}%)")
    print(f"- 데이터 비중: {len(single_appointment_patients)/len(df)*100:.2f}%")
    print(f"- 노쇼율: {single_appointment_patients['noshow_rate'].mean()*100:.2f}%")
    
    # 시계열 특징 활용 가능성
    print("\n시계열 특징 활용 가능한 데이터:")
    can_use_temporal = df[df['patient_appointment_count'] > 0]
    print(f"- 시계열 특징 활용 가능 예약: {len(can_use_temporal):,}개 ({len(can_use_temporal)/len(df)*100:.2f}%)")
    print(f"- 시계열 특징 불가능 예약 (첫 번째): {len(df) - len(can_use_temporal):,}개 ({(len(df) - len(can_use_temporal))/len(df)*100:.2f}%)")
    
    return patient_noshow_stats

def generate_visualizations(df, patient_appointment_counts, patient_noshow_stats):
    """주요 시각화 생성"""
    print("\n" + "="*80)
    print("6. 시각화 생성")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 환자별 예약 횟수 분포
    ax1 = axes[0, 0]
    appointment_dist = patient_appointment_counts.value_counts().sort_index()
    ax1.bar(appointment_dist.index[:20], appointment_dist.values[:20])
    ax1.set_xlabel('Number of Appointments per Patient')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Distribution of Appointments per Patient')
    ax1.grid(True, alpha=0.3)
    
    # 2. 노쇼율 by 예약 횟수
    ax2 = axes[0, 1]
    noshow_by_appts = df.groupby('patient_appointment_count')['No-show'].mean()
    ax2.plot(noshow_by_appts.index[:20], noshow_by_appts.values[:20], marker='o')
    ax2.set_xlabel('Previous Appointment Count')
    ax2.set_ylabel('No-show Rate')
    ax2.set_title('No-show Rate by Previous Appointment Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. 이전 노쇼 횟수별 현재 노쇼율
    ax3 = axes[0, 2]
    noshow_by_prev = df.groupby('patient_previous_noshow_count')['No-show'].mean()
    ax3.bar(noshow_by_prev.index[:10], noshow_by_prev.values[:10])
    ax3.set_xlabel('Previous No-show Count')
    ax3.set_ylabel('Current No-show Rate')
    ax3.set_title('Current No-show Rate by Previous No-show Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. Lead time 분포
    ax4 = axes[1, 0]
    ax4.hist(df['lead_time'], bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Lead Time (days)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Lead Time Distribution')
    ax4.axvline(df['lead_time'].mean(), color='red', linestyle='--', label=f'Mean: {df["lead_time"].mean():.1f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 나이 분포 by 노쇼
    ax5 = axes[1, 1]
    df[df['No-show']==0]['Age'].hist(bins=30, alpha=0.5, label='Show', ax=ax5)
    df[df['No-show']==1]['Age'].hist(bins=30, alpha=0.5, label='No-show', ax=ax5)
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Age Distribution by No-show Status')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 환자별 노쇼율 분포
    ax6 = axes[1, 2]
    patient_noshow_rates = patient_noshow_stats[patient_noshow_stats['n_appointments'] > 1]['noshow_rate']
    ax6.hist(patient_noshow_rates, bins=20, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Patient No-show Rate')
    ax6.set_ylabel('Number of Patients')
    ax6.set_title('Distribution of Patient-level No-show Rates\n(Patients with 2+ appointments)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('EDA/panel_data_visualization.png', dpi=300, bbox_inches='tight')
    print("시각화 저장 완료: EDA/panel_data_visualization.png")
    plt.close()

def generate_recommendations(df, patient_appointment_counts, patient_noshow_stats):
    """Cross-validation 전략 권장사항 생성"""
    print("\n" + "="*80)
    print("7. Cross-validation 전략 권장사항")
    print("="*80)
    
    single_appt_patients = (patient_appointment_counts == 1).sum()
    single_appt_pct = single_appt_patients / len(patient_appointment_counts) * 100
    
    print("\n[데이터 특성 요약]")
    print(f"- 총 환자: {len(patient_appointment_counts):,}명")
    print(f"- 단일 예약 환자: {single_appt_patients:,}명 ({single_appt_pct:.1f}%)")
    print(f"- 복수 예약 환자: {len(patient_appointment_counts) - single_appt_patients:,}명 ({100-single_appt_pct:.1f}%)")
    
    print("\n[권장 전략]")
    print("\n1. **Patient-based Stratified K-Fold Cross-validation**")
    print("   - 환자 ID 기준으로 fold 분할 (data leakage 방지)")
    print("   - 각 fold에 노쇼율이 균등하게 분포하도록 stratification")
    print("   - 5-fold 또는 10-fold 권장")
    
    print("\n2. **단일 예약 환자 처리 방안:**")
    print(f"   현재 단일 예약 환자: {single_appt_pct:.1f}%")
    
    if single_appt_pct > 40:
        print("   [권장] 단일 예약 환자 포함")
        print("      - 데이터의 상당 부분을 차지하므로 제외 시 정보 손실 큼")
        print("      - 시계열 특징은 0으로 처리 (첫 예약 표시)")
    else:
        print("   [선택적] 상황에 따라 결정")
        print("      - 포함 시: 전체 데이터 활용, 실제 운영 환경 반영")
        print("      - 제외 시: 시계열 특징 활용 극대화, 예측 성능 향상 가능")
    
    print("\n3. **시계열 특징 처리:**")
    print("   - 첫 번째 예약: 시계열 특징을 0 또는 특별값으로 처리")
    print("   - is_first_appointment 플래그 추가 고려")
    
    print("\n4. **검증 전략:**")
    print("   - Inner CV: 하이퍼파라미터 튜닝용")
    print("   - Outer CV: 최종 성능 평가용")
    print("   - Time-based split도 고려 (시간순 검증)")
    
    print("\n5. **클래스 불균형 처리:**")
    noshow_rate = df['No-show'].mean() * 100
    print(f"   - 현재 노쇼율: {noshow_rate:.2f}%")
    print("   - Stratification 필수")
    print("   - 각 fold의 노쇼율 일관성 확인")
    
    print("\n6. **평가 지표:**")
    print("   - 주요: AUC-PR (클래스 불균형에 robust)")
    print("   - 보조: F1-score, Recall, Precision")
    print("   - Threshold 최적화 필요")

def main():
    """메인 실행 함수"""
    # 데이터 로드
    file_path = 'feature_selection_analysis/ml_dataset_selected_features.csv'
    
    # 1. 데이터 로드 및 기본 정보
    df = load_and_initial_inspection(file_path)
    
    # 2. 패널 구조 분석
    patient_appointment_counts = analyze_panel_structure(df)
    
    # 3. 시계열 특징 분석
    analyze_temporal_features(df)
    
    # 4. 선택된 특징 분석
    selected_features = analyze_selected_features(df)
    
    # 5. Cross-validation 관련 분석
    patient_noshow_stats = analyze_cross_validation_implications(df, patient_appointment_counts)
    
    # 6. 시각화
    generate_visualizations(df, patient_appointment_counts, patient_noshow_stats)
    
    # 7. 권장사항
    generate_recommendations(df, patient_appointment_counts, patient_noshow_stats)
    
    print("\n" + "="*80)
    print("EDA 완료!")
    print("="*80)

if __name__ == "__main__":
    main()