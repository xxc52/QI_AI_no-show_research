# -*- coding: utf-8 -*-
"""
날씨 데이터 검증 및 분석 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def validate_weather_data():
    print('='*60)
    print('Weather Data Validation')
    print('='*60)
    
    # 2015, 2016 날씨 데이터 로드
    weather_2015 = pd.read_csv('weather_sum_2015.csv')
    weather_2016 = pd.read_csv('weather_sum_2016.csv')
    
    print(f"Weather 2015 shape: {weather_2015.shape}")
    print(f"Weather 2016 shape: {weather_2016.shape}")
    print()
    
    # 컬럼 확인
    print("Columns:")
    print(weather_2015.columns.tolist())
    print()
    
    # 날짜 범위 확인
    weather_2015['date'] = pd.to_datetime(weather_2015['DATA (YYYY-MM-DD)'])
    weather_2016['date'] = pd.to_datetime(weather_2016['DATA (YYYY-MM-DD)'])
    
    print("Date ranges:")
    print(f"2015: {weather_2015['date'].min()} to {weather_2015['date'].max()}")
    print(f"2016: {weather_2016['date'].min()} to {weather_2016['date'].max()}")
    print()
    
    # 기상관측소 확인
    print("Weather stations:")
    print(f"2015 stations: {weather_2015['ESTACAO'].unique()}")
    print(f"2016 stations: {weather_2016['ESTACAO'].unique()}")
    print()
    
    # 결측치 확인
    print("Missing values check:")
    for year, df in [('2015', weather_2015), ('2016', weather_2016)]:
        print(f"\n{year}:")
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            print("  No missing values found")
        else:
            print(missing_counts[missing_counts > 0])
    
    # 기본 통계 요약
    print("\n" + "="*60)
    print("Weather Statistics Summary")
    print("="*60)
    
    weather_combined = pd.concat([weather_2015, weather_2016], ignore_index=True)
    
    stats_cols = ['temp_avg', 'temp_max', 'temp_min', 'rain_max', 'hum_max', 'hum_min', 'wind_avg']
    stats = weather_combined[stats_cols].describe()
    
    print("Temperature (Celsius):")
    print(f"  Average: {stats.loc['mean', 'temp_avg']:.1f}°C")
    print(f"  Max: {stats.loc['max', 'temp_max']:.1f}°C")
    print(f"  Min: {stats.loc['min', 'temp_min']:.1f}°C")
    
    print(f"\nRainfall (mm):")
    print(f"  Average daily max: {stats.loc['mean', 'rain_max']:.1f}mm")
    print(f"  Max daily: {stats.loc['max', 'rain_max']:.1f}mm")
    
    print(f"\nHumidity (%):")
    print(f"  Average max: {stats.loc['mean', 'hum_max']:.1f}%")
    print(f"  Average min: {stats.loc['mean', 'hum_min']:.1f}%")
    
    # 이상치 검사
    print("\n" + "="*60)
    print("Outlier Detection")
    print("="*60)
    
    outliers = {}
    
    # 온도 이상치 (브라질 기준으로 합리적인 범위)
    temp_outliers = (weather_combined['temp_avg'] < 10) | (weather_combined['temp_avg'] > 40)
    if temp_outliers.sum() > 0:
        outliers['temperature'] = temp_outliers.sum()
    
    # 습도 이상치
    humidity_outliers = (weather_combined['hum_max'] < 30) | (weather_combined['hum_max'] > 100)
    if humidity_outliers.sum() > 0:
        outliers['humidity'] = humidity_outliers.sum()
    
    # 강수량 이상치 (일일 200mm 이상은 매우 많음)
    rain_outliers = weather_combined['rain_max'] > 200
    if rain_outliers.sum() > 0:
        outliers['rainfall'] = rain_outliers.sum()
    
    if outliers:
        print("Potential outliers found:")
        for var, count in outliers.items():
            print(f"  {var}: {count} records")
    else:
        print("No obvious outliers detected")
    
    # 병원 데이터와 날짜 매칭 확인
    print("\n" + "="*60)
    print("Date Coverage for Hospital Data")
    print("="*60)
    
    hospital_data = pd.read_csv('dataV05_with_all_features.csv')
    hospital_data['appointment_date'] = pd.to_datetime(hospital_data['Appointment_Date'])
    
    hospital_date_range = (hospital_data['appointment_date'].min(), 
                          hospital_data['appointment_date'].max())
    
    print(f"Hospital appointments: {hospital_date_range[0].date()} to {hospital_date_range[1].date()}")
    
    weather_date_range = (weather_combined['date'].min(), 
                         weather_combined['date'].max())
    
    print(f"Weather data: {weather_date_range[0].date()} to {weather_date_range[1].date()}")
    
    # 겹치는 기간 확인
    overlap_start = max(hospital_date_range[0], weather_date_range[0])
    overlap_end = min(hospital_date_range[1], weather_date_range[1])
    
    if overlap_start <= overlap_end:
        print(f"Overlap period: {overlap_start.date()} to {overlap_end.date()}")
        
        # 겹치는 기간의 병원 예약 건수
        overlap_appointments = hospital_data[
            (hospital_data['appointment_date'] >= overlap_start) & 
            (hospital_data['appointment_date'] <= overlap_end)
        ]
        
        print(f"Appointments in overlap period: {len(overlap_appointments):,}")
        print(f"Percentage of total appointments: {len(overlap_appointments)/len(hospital_data)*100:.1f}%")
    else:
        print("WARNING: No date overlap between hospital and weather data!")
    
    # 날씨 변수별 분포 확인
    print("\n" + "="*60)
    print("Weather Variable Distributions")
    print("="*60)
    
    # 계절성 확인 (월별)
    weather_combined['month'] = weather_combined['date'].dt.month
    monthly_stats = weather_combined.groupby('month').agg({
        'temp_avg': 'mean',
        'rain_max': 'mean',
        'hum_max': 'mean'
    }).round(1)
    
    print("Monthly averages:")
    print("Month | Temp(°C) | Rain(mm) | Humidity(%)")
    print("-" * 40)
    for month, row in monthly_stats.iterrows():
        print(f"{month:5d} | {row['temp_avg']:8.1f} | {row['rain_max']:8.1f} | {row['hum_max']:11.1f}")
    
    return weather_combined, hospital_data

if __name__ == "__main__":
    weather_data, hospital_data = validate_weather_data()
    print("\nWeather data validation completed successfully!")