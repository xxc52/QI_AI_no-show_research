# -*- coding: utf-8 -*-
"""
최종 데이터셋 생성: 병원 데이터 + 지역 클러스터 + 날씨 데이터 통합
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def create_final_dataset():
    print('='*70)
    print('FINAL DATASET CREATION')
    print('='*70)
    
    # 1. 기존 병원 데이터 로드
    print("1. Loading hospital data...")
    hospital_df = pd.read_csv('dataV05_with_all_features.csv')
    print(f"   Hospital data: {hospital_df.shape}")
    
    # 2. 지역 클러스터 정보 추가
    print("2. Adding neighbourhood clusters...")
    with open('neighbourhood_clusters.json', 'r') as f:
        cluster_mapping = json.load(f)
    
    hospital_df['neighbourhood_cluster'] = hospital_df['Neighbourhood'].map(cluster_mapping)
    print(f"   Cluster assignment success rate: {hospital_df['neighbourhood_cluster'].notna().mean():.1%}")
    
    # 3. 날씨 데이터 준비
    print("3. Processing weather data...")
    
    # 날씨 데이터 로드 및 결합
    weather_2015 = pd.read_csv('weather_sum_2015.csv')
    weather_2016 = pd.read_csv('weather_sum_2016.csv')
    weather_combined = pd.concat([weather_2015, weather_2016], ignore_index=True)
    
    # 날짜 변환
    weather_combined['date'] = pd.to_datetime(weather_combined['DATA (YYYY-MM-DD)'])
    hospital_df['appointment_date'] = pd.to_datetime(hospital_df['Appointment_Date'])
    
    print(f"   Weather data: {weather_combined.shape}")
    print(f"   Weather date range: {weather_combined['date'].min().date()} to {weather_combined['date'].max().date()}")
    
    # 4. Vitoria 지역 날씨 관측소 선택
    # A612가 Vitoria의 주요 관측소로 보임
    vitoria_stations = ['A612', 'A613', 'A614', 'A615', 'A616', 'A617']  # Vitoria area weather stations
    
    # 각 관측소별 데이터 가용성 확인
    print("   Checking Vitoria weather stations:")
    for station in vitoria_stations:
        station_data = weather_combined[weather_combined['ESTACAO'] == station]
        if len(station_data) > 0:
            date_range = f"{station_data['date'].min().date()} to {station_data['date'].max().date()}"
            print(f"     {station}: {len(station_data)} records, {date_range}")
    
    # A612 (Vitoria 공항) 우선 사용, 결측 시 주변 관측소 데이터로 보완
    vitoria_weather = weather_combined[
        weather_combined['ESTACAO'].isin(vitoria_stations) &
        (weather_combined['date'] >= hospital_df['appointment_date'].min()) &
        (weather_combined['date'] <= hospital_df['appointment_date'].max())
    ].copy()
    
    print(f"   Vitoria weather data: {len(vitoria_weather)} records")
    
    # 5. 날씨 데이터 일별 집계 (여러 관측소 평균)
    print("4. Aggregating daily weather data...")
    
    weather_daily = vitoria_weather.groupby('date').agg({
        'temp_avg': 'mean',
        'temp_max': 'max',
        'temp_min': 'min', 
        'rain_max': 'max',
        'hum_max': 'max',
        'hum_min': 'min',
        'wind_avg': 'mean',
        'rad_max': 'max'
    }).round(2)
    
    # 결측치 보간 (선형 보간)
    weather_daily = weather_daily.interpolate(method='linear')
    
    print(f"   Daily weather aggregated: {len(weather_daily)} days")
    print(f"   Missing values after interpolation: {weather_daily.isnull().sum().sum()}")
    
    # 6. 파생 날씨 변수 생성
    print("5. Creating weather-derived features...")
    
    weather_daily['temp_range'] = weather_daily['temp_max'] - weather_daily['temp_min']
    weather_daily['is_rainy'] = (weather_daily['rain_max'] > 0.5).astype(int)
    weather_daily['is_heavy_rain'] = (weather_daily['rain_max'] > 10.0).astype(int)
    weather_daily['is_hot'] = (weather_daily['temp_max'] > 30.0).astype(int)
    weather_daily['is_cold'] = (weather_daily['temp_max'] < 20.0).astype(int)
    weather_daily['is_humid'] = (weather_daily['hum_max'] > 90.0).astype(int)
    weather_daily['is_windy'] = (weather_daily['wind_avg'] > 3.0).astype(int)
    
    # 전날 대비 기온 변화
    weather_daily['temp_change'] = weather_daily['temp_avg'].diff().fillna(0).round(2)
    weather_daily['temp_change_abs'] = weather_daily['temp_change'].abs()
    
    # 계절성 (월별)
    weather_daily['month'] = weather_daily.index.month
    weather_daily['season'] = weather_daily['month'].map({
        12: 'summer', 1: 'summer', 2: 'summer',  # 남반구
        3: 'autumn', 4: 'autumn', 5: 'autumn',
        6: 'winter', 7: 'winter', 8: 'winter',
        9: 'spring', 10: 'spring', 11: 'spring'
    })
    
    # 7. 병원 데이터와 날씨 데이터 조인
    print("6. Merging hospital and weather data...")
    
    # 날짜 기준으로 조인
    hospital_df['merge_date'] = hospital_df['appointment_date'].dt.date.astype(str)
    
    weather_for_merge = weather_daily.reset_index()
    weather_for_merge['merge_date'] = weather_for_merge['date'].dt.date.astype(str)
    
    # 조인 실행
    final_df = hospital_df.merge(
        weather_for_merge, 
        left_on='merge_date', 
        right_on='merge_date', 
        how='left'
    )
    
    print(f"   Merge success rate: {final_df['temp_avg'].notna().mean():.1%}")
    print(f"   Final dataset shape: {final_df.shape}")
    
    # 8. 최종 데이터 정리
    print("7. Final data cleaning...")
    
    # 불필요한 컬럼 제거
    columns_to_drop = ['merge_date', 'date']  # 중복되는 컬럼들
    final_df = final_df.drop(columns=[col for col in columns_to_drop if col in final_df.columns])
    
    # 날씨 변수 결측치를 평균으로 채움 (혹시나 남은 것들)
    weather_cols = [col for col in final_df.columns if col.startswith(('temp_', 'rain_', 'hum_', 'wind_', 'rad_', 'is_'))]
    for col in weather_cols:
        if final_df[col].dtype in ['float64', 'int64']:
            final_df[col] = final_df[col].fillna(final_df[col].mean())
    
    # season 결측치 처리 
    final_df['season'] = final_df['season'].fillna('autumn')  # 4-6월은 주로 가을
    
    print(f"   Remaining missing values: {final_df.isnull().sum().sum()}")
    
    # 9. 최종 검증
    print("8. Final validation...")
    print(f"   Total records: {len(final_df):,}")
    print(f"   Total features: {len(final_df.columns)}")
    print(f"   No-show rate: {final_df['No-show'].mean():.2%}")
    print(f"   Date range: {final_df['appointment_date'].min()} to {final_df['appointment_date'].max()}")
    
    # 새로운 특징들 확인
    new_weather_features = [col for col in final_df.columns if col.startswith(('temp_', 'rain_', 'hum_', 'wind_', 'rad_', 'is_', 'season'))]
    print(f"   Weather features added: {len(new_weather_features)}")
    
    # 10. 저장
    output_filename = 'final_dataset_with_weather_clusters.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\nFinal dataset saved as: {output_filename}")
    
    return final_df, new_weather_features

def analyze_final_dataset(df, weather_features):
    """최종 데이터셋 분석"""
    print('\n' + '='*70)
    print('FINAL DATASET ANALYSIS')
    print('='*70)
    
    # 기본 통계
    print("1. Dataset Overview:")
    print(f"   Records: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Patients: {df['PatientId'].nunique():,}")
    print(f"   Neighbourhoods: {df['Neighbourhood'].nunique()}")
    print(f"   Neighbourhood clusters: {df['neighbourhood_cluster'].nunique()}")
    print(f"   No-show rate: {df['No-show'].mean():.2%}")
    
    # 날씨 특징 분석
    print("\n2. Weather Features Analysis:")
    weather_stats = df[weather_features].describe()
    
    print("   Temperature:")
    print(f"     Average: {weather_stats.loc['mean', 'temp_avg']:.1f}°C")
    print(f"     Range: {weather_stats.loc['min', 'temp_min']:.1f}°C to {weather_stats.loc['max', 'temp_max']:.1f}°C")
    
    print("   Weather conditions:")
    print(f"     Rainy days: {df['is_rainy'].mean():.1%}")
    print(f"     Hot days (>30°C): {df['is_hot'].mean():.1%}")
    print(f"     Cold days (<20°C): {df['is_cold'].mean():.1%}")
    print(f"     High humidity days: {df['is_humid'].mean():.1%}")
    
    print("   Seasonal distribution:")
    season_dist = df['season'].value_counts()
    for season, count in season_dist.items():
        print(f"     {season.title()}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 클러스터별 노쇼율
    print("\n3. No-show Rate by Neighbourhood Cluster:")
    cluster_noshow = df.groupby('neighbourhood_cluster').agg({
        'No-show': ['count', 'mean']
    }).round(3)
    cluster_noshow.columns = ['appointments', 'noshow_rate']
    
    for cluster, row in cluster_noshow.iterrows():
        print(f"   {cluster}: {row['appointments']:,} appointments, {row['noshow_rate']:.1%} no-show rate")
    
    # 날씨와 노쇼의 관계 (간단한 분석)
    print("\n4. Weather Impact on No-show:")
    
    weather_impact = {}
    for feature in ['is_rainy', 'is_hot', 'is_cold', 'is_humid']:
        feature_noshow = df.groupby(feature)['No-show'].mean()
        if len(feature_noshow) == 2:
            diff = feature_noshow[1] - feature_noshow[0]
            weather_impact[feature] = {
                'no_condition': feature_noshow[0],
                'with_condition': feature_noshow[1], 
                'difference': diff
            }
    
    for feature, stats in weather_impact.items():
        condition_name = feature.replace('is_', '').replace('_', ' ')
        print(f"   {condition_name.title()}:")
        print(f"     Normal days: {stats['no_condition']:.2%}")
        print(f"     {condition_name.title()} days: {stats['with_condition']:.2%}")
        print(f"     Difference: {stats['difference']:+.2%}")
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    final_df, weather_features = create_final_dataset()
    analyze_final_dataset(final_df, weather_features)