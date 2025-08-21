# -*- coding: utf-8 -*-
"""
Neighbourhood 변수 분석 및 클러스터링 준비
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 데이터 로드
df = pd.read_csv('dataV05_with_all_features.csv')

print('='*60)
print('Neighbourhood Analysis')
print('='*60)
print(f'Total unique neighbourhoods: {df["Neighbourhood"].nunique()}')
print()

# 각 지역별 통계
neigh_stats = df.groupby('Neighbourhood').agg({
    'No-show': ['count', 'mean'],
    'PatientId': 'nunique',
    'Age': 'mean',
    'lead_time': 'mean',
    'SMS_received': 'mean',
    'Scholarship': 'mean',
    'Hipertension': 'mean',
    'Diabetes': 'mean'
}).round(4)

# 컬럼명 정리
neigh_stats.columns = ['appointment_count', 'noshow_rate', 'unique_patients', 
                       'avg_age', 'avg_lead_time', 'sms_rate', 
                       'scholarship_rate', 'hipertension_rate', 'diabetes_rate']

# 노쇼율 기준 정렬
neigh_stats = neigh_stats.sort_values('noshow_rate', ascending=False)

# 상위/하위 10개 지역
print('Top 10 neighbourhoods by no-show rate:')
top10 = neigh_stats[['appointment_count', 'noshow_rate', 'unique_patients']].head(10)
for idx, row in top10.iterrows():
    # 인코딩 문제 회피를 위해 ASCII 문자만 출력
    safe_idx = idx.encode('ascii', 'ignore').decode('ascii') if not idx.isascii() else idx
    print(f"  {safe_idx[:30]:30s} | Count: {row['appointment_count']:5.0f} | Rate: {row['noshow_rate']:.2%} | Patients: {row['unique_patients']:4.0f}")
print()

print('Bottom 10 neighbourhoods by no-show rate:')
bottom10 = neigh_stats[['appointment_count', 'noshow_rate', 'unique_patients']].tail(10)
for idx, row in bottom10.iterrows():
    safe_idx = idx.encode('ascii', 'ignore').decode('ascii') if not idx.isascii() else idx
    print(f"  {safe_idx[:30]:30s} | Count: {row['appointment_count']:5.0f} | Rate: {row['noshow_rate']:.2%} | Patients: {row['unique_patients']:4.0f}")
print()

# 통계 요약
print('='*60)
print('No-show rate distribution:')
print('='*60)
print(f'Min: {neigh_stats["noshow_rate"].min():.2%}')
print(f'Max: {neigh_stats["noshow_rate"].max():.2%}')
print(f'Mean: {neigh_stats["noshow_rate"].mean():.2%}')
print(f'Median: {neigh_stats["noshow_rate"].median():.2%}')
print(f'Std: {neigh_stats["noshow_rate"].std():.2%}')

# 클러스터링 방법 1: No-show rate 기반 (단순)
print()
print('='*60)
print('Method 1: No-show rate based clustering')
print('='*60)

# 노쇼율 기준으로 4개 그룹으로 나누기
neigh_stats['noshow_cluster_simple'] = pd.qcut(neigh_stats['noshow_rate'], 
                                                q=4, 
                                                labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

for cluster in ['Low', 'Medium-Low', 'Medium-High', 'High']:
    cluster_data = neigh_stats[neigh_stats['noshow_cluster_simple'] == cluster]
    print(f"{cluster}: {len(cluster_data)} neighbourhoods, "
          f"avg no-show rate: {cluster_data['noshow_rate'].mean():.2%}, "
          f"total appointments: {cluster_data['appointment_count'].sum()}")

# 클러스터링 방법 2: 다차원 특성 기반 (K-means)
print()
print('='*60)
print('Method 2: Multi-feature K-means clustering')
print('='*60)

# 충분한 데이터가 있는 지역만 선택 (최소 50개 예약)
neigh_stats_filtered = neigh_stats[neigh_stats['appointment_count'] >= 50].copy()
print(f"Neighbourhoods with >= 50 appointments: {len(neigh_stats_filtered)}")

# 클러스터링에 사용할 특성 선택
features_for_clustering = ['noshow_rate', 'avg_age', 'avg_lead_time', 
                          'sms_rate', 'scholarship_rate', 
                          'hipertension_rate', 'diabetes_rate']

X = neigh_stats_filtered[features_for_clustering]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 클러스터링 (5개 그룹)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
neigh_stats_filtered['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# 각 클러스터별 특성
print("\nCluster characteristics:")
for i in range(5):
    cluster_data = neigh_stats_filtered[neigh_stats_filtered['kmeans_cluster'] == i]
    print(f"\nCluster {i}: {len(cluster_data)} neighbourhoods")
    print(f"  No-show rate: {cluster_data['noshow_rate'].mean():.2%}")
    print(f"  Avg age: {cluster_data['avg_age'].mean():.1f}")
    print(f"  Scholarship rate: {cluster_data['scholarship_rate'].mean():.2%}")
    print(f"  SMS rate: {cluster_data['sms_rate'].mean():.2%}")

# 클러스터 매핑 딕셔너리 생성 (전체 지역에 적용)
cluster_mapping = {}

# K-means 클러스터가 있는 지역
for idx, row in neigh_stats_filtered.iterrows():
    cluster_mapping[idx] = f'cluster_{row["kmeans_cluster"]}'

# 데이터가 적은 지역은 별도 그룹으로
for idx in neigh_stats.index:
    if idx not in cluster_mapping:
        if neigh_stats.loc[idx, 'appointment_count'] < 20:
            cluster_mapping[idx] = 'cluster_rare'
        else:
            # 노쇼율 기준으로 가장 가까운 클러스터에 배정
            noshow_rate = neigh_stats.loc[idx, 'noshow_rate']
            # 간단한 규칙 기반 배정
            if noshow_rate < 0.15:
                cluster_mapping[idx] = 'cluster_0'  # Low no-show
            elif noshow_rate < 0.20:
                cluster_mapping[idx] = 'cluster_1'  # Medium-low
            elif noshow_rate < 0.25:
                cluster_mapping[idx] = 'cluster_2'  # Medium
            elif noshow_rate < 0.30:
                cluster_mapping[idx] = 'cluster_3'  # Medium-high
            else:
                cluster_mapping[idx] = 'cluster_4'  # High no-show

# 매핑 저장
import json
with open('neighbourhood_clusters.json', 'w') as f:
    json.dump(cluster_mapping, f, indent=2)

print()
print('='*60)
print('Clustering results saved to neighbourhood_clusters.json')
print('='*60)

# 최종 클러스터 분포
final_clusters = pd.Series(cluster_mapping).value_counts()
print("\nFinal cluster distribution:")
print(final_clusters)

# 원본 데이터에 클러스터 정보 추가
df['neighbourhood_cluster'] = df['Neighbourhood'].map(cluster_mapping)

# 각 클러스터별 노쇼율 확인
print("\nNo-show rate by cluster:")
cluster_noshow = df.groupby('neighbourhood_cluster')['No-show'].agg(['count', 'mean'])
cluster_noshow.columns = ['appointments', 'noshow_rate']
print(cluster_noshow.sort_values('noshow_rate'))

print()
print("Neighbourhood clustering completed!")