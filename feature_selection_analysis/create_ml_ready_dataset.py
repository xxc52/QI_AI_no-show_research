"""
Create ML-ready dataset with all categorical variables encoded
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    """Load the original dataset"""
    df = pd.read_csv('../final_dataset_with_weather_clusters.csv')
    print(f"Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def encode_all_categorical_variables(df):
    """Encode all categorical variables to numeric"""
    df_encoded = df.copy()
    
    # Dictionary to store encoding mappings
    encoding_info = {}
    
    # 1. Encode neighbourhood_cluster (string → numeric)
    if 'neighbourhood_cluster' in df_encoded.columns:
        le_cluster = LabelEncoder()
        df_encoded['neighbourhood_cluster_encoded'] = le_cluster.fit_transform(df_encoded['neighbourhood_cluster'])
        encoding_info['neighbourhood_cluster'] = dict(zip(le_cluster.classes_, le_cluster.transform(le_cluster.classes_)))
        print(f"neighbourhood_cluster encoded: {encoding_info['neighbourhood_cluster']}")
    
    # 2. Encode season (string → numeric)
    if 'season' in df_encoded.columns:
        season_map = {'autumn': 0, 'winter': 1}
        df_encoded['season_encoded'] = df_encoded['season'].map(season_map)
        encoding_info['season'] = season_map
        print(f"season encoded: {season_map}")
    
    return df_encoded, encoding_info

def create_ml_dataset_all_features(df):
    """Create ML dataset with ALL features (before selection)"""
    
    # Columns to exclude from ML dataset (only original string versions)
    exclude_cols = [
        'Registration_Date', 'Appointment_Date',  # Original date strings
        'Registration_Date_no_time', 'appointment_date',  # Processed dates
        'Neighbourhood',  # Original neighbourhood (use encoded version)
        'neighbourhood_cluster',  # Original cluster strings (use encoded version)
        'season'  # Original season strings (use encoded version)
    ]
    
    # IMPORTANT: Keep PatientId and AppointmentID for:
    # - Time series analysis by patient
    # - Preventing data leakage (patient-based train/test split)
    # - Data tracking and result analysis
    
    # Target variable
    target = 'No-show'
    
    # All feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target]]
    
    # Remove constant features identified earlier
    constant_features = ['Registration_Year', 'Appointment_Year', 'is_heavy_rain', 'is_cold', 'is_humid']
    feature_cols = [col for col in feature_cols if col not in constant_features]
    
    # Create final dataset: features + target
    final_cols = feature_cols + [target]
    ml_dataset = df[final_cols].copy()
    
    print(f"\nML dataset (all features): {ml_dataset.shape[0]:,} rows × {ml_dataset.shape[1]} columns")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target}")
    
    return ml_dataset, feature_cols

def create_ml_dataset_selected_features(df):
    """Create ML dataset with only SELECTED features from univariate analysis"""
    
    # Load selected features from univariate analysis
    selected_df = pd.read_csv('selected_features_univariate.csv')
    selected_features = selected_df['feature'].tolist()
    
    print(f"\nSelected features from univariate analysis: {len(selected_features)}")
    print("Selected features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2}. {feat}")
    
    # Target variable and essential IDs
    target = 'No-show'
    essential_cols = ['PatientId', 'AppointmentID']  # Keep for time series and data tracking
    
    # Create final dataset: IDs + selected features + target
    final_cols = essential_cols + selected_features + [target]
    ml_dataset_selected = df[final_cols].copy()
    
    print(f"\nML dataset (selected features): {ml_dataset_selected.shape[0]:,} rows × {ml_dataset_selected.shape[1]} columns")
    print(f"Includes: PatientId, AppointmentID + {len(selected_features)} features + target")
    
    return ml_dataset_selected

def validate_dataset(df, dataset_name):
    """Validate the ML dataset"""
    print(f"\n{'='*60}")
    print(f"VALIDATION: {dataset_name}")
    print(f"{'='*60}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Check target distribution
    if 'No-show' in df.columns:
        target_dist = df['No-show'].value_counts()
        noshow_rate = df['No-show'].mean()
        print(f"\nTarget distribution:")
        print(f"  Show (0): {target_dist.get(0, 0):,} ({1-noshow_rate:.1%})")
        print(f"  No-show (1): {target_dist.get(1, 0):,} ({noshow_rate:.1%})")
    
    # Check for any remaining object columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"\n[WARNING] Object columns found: {object_cols}")
        for col in object_cols:
            print(f"  - {col}: {df[col].nunique()} unique values")
    else:
        print(f"\n[OK] All columns are numeric")
    
    # Value ranges for key features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric feature ranges (first 10):")
        for col in numeric_cols[:10]:
            if col != 'No-show':
                print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

def main():
    """Main pipeline to create ML-ready datasets"""
    
    print("="*80)
    print("CREATING ML-READY DATASETS")
    print("="*80)
    
    # Load original data
    df = load_original_data()
    
    # Encode categorical variables
    df_encoded, encoding_info = encode_all_categorical_variables(df)
    
    # Create ML dataset with ALL features
    ml_all_features, all_feature_cols = create_ml_dataset_all_features(df_encoded)
    
    # Validate all features dataset
    validate_dataset(ml_all_features, "ALL FEATURES DATASET")
    
    # Save all features dataset
    ml_all_features.to_csv('ml_dataset_all_features.csv', index=False)
    print(f"\n[SAVED] ml_dataset_all_features.csv")
    
    # Create ML dataset with SELECTED features only
    ml_selected_features = create_ml_dataset_selected_features(df_encoded)
    
    # Validate selected features dataset  
    validate_dataset(ml_selected_features, "SELECTED FEATURES DATASET")
    
    # Save selected features dataset
    ml_selected_features.to_csv('ml_dataset_selected_features.csv', index=False)
    print(f"\n[SAVED] ml_dataset_selected_features.csv")
    
    # Sort datasets by PatientId and then by some time indicator for time series analysis
    # This ensures patient data is grouped together and in chronological order
    print("\nSorting datasets for time series analysis...")
    
    # Sort by PatientId first, then by Registration_Date-related features
    sort_cols = ['PatientId', 'Registration_Month', 'Registration_Day', 'Registration_Hour']
    available_sort_cols = [col for col in sort_cols if col in ml_all_features.columns]
    
    ml_all_features = ml_all_features.sort_values(available_sort_cols).reset_index(drop=True)
    ml_selected_features = ml_selected_features.sort_values(available_sort_cols).reset_index(drop=True)
    
    print(f"Sorted by: {available_sort_cols}")
    
    # Save feature lists and documentation
    with open('feature_lists_and_usage_guide.txt', 'w') as f:
        f.write("ML-READY DATASETS - FEATURE LISTS AND USAGE GUIDE\n")
        f.write("="*60 + "\n\n")
        
        f.write("IMPORTANT: DATA LEAKAGE PREVENTION\n")
        f.write("-"*40 + "\n")
        f.write("1. Use PatientId for train/validation/test split\n")
        f.write("2. NEVER split randomly - same patient should not appear in both train and test\n")
        f.write("3. Recommended split: 70% patients for train, 15% for val, 15% for test\n")
        f.write("4. Data is already sorted by PatientId and time for time series analysis\n\n")
        
        f.write("TIME SERIES CONSIDERATIONS\n")
        f.write("-"*40 + "\n")
        f.write("- Each patient has multiple appointments over time\n")
        f.write("- Patient history features already incorporate temporal information\n")
        f.write("- For time series prediction, use patient_appointment_count as sequence position\n\n")
        
        f.write(f"ALL FEATURES DATASET ({len(all_feature_cols)} features + IDs + target):\n")
        f.write("-"*40 + "\n")
        f.write("Essential columns: PatientId, AppointmentID, No-show\n")
        for i, feat in enumerate(all_feature_cols, 1):
            f.write(f"{i:3}. {feat}\n")
        
        f.write(f"\nSELECTED FEATURES DATASET ({len(ml_selected_features.columns)-3} features + IDs + target):\n")
        f.write("-"*40 + "\n")
        f.write("Essential columns: PatientId, AppointmentID, No-show\n")
        selected_cols = [col for col in ml_selected_features.columns if col not in ['PatientId', 'AppointmentID', 'No-show']]
        for i, feat in enumerate(selected_cols, 1):
            f.write(f"{i:3}. {feat}\n")
        
        f.write("\nENCODING INFORMATION:\n")
        f.write("-"*40 + "\n")
        for var, mapping in encoding_info.items():
            f.write(f"{var}: {mapping}\n")
        
        f.write("\nUSAGE EXAMPLES:\n")
        f.write("-"*40 + "\n")
        f.write("# Patient-based train/test split\n")
        f.write("unique_patients = df['PatientId'].unique()\n")
        f.write("train_patients = unique_patients[:int(0.7*len(unique_patients))]\n")
        f.write("test_patients = unique_patients[int(0.85*len(unique_patients)):]\n")
        f.write("train_df = df[df['PatientId'].isin(train_patients)]\n")
        f.write("test_df = df[df['PatientId'].isin(test_patients)]\n\n")
        
        f.write("# For model training, exclude IDs from features\n")
        f.write("feature_cols = [col for col in df.columns if col not in ['PatientId', 'AppointmentID', 'No-show']]\n")
        f.write("X = df[feature_cols]\n")
        f.write("y = df['No-show']\n")
    
    print(f"\n[SAVED] feature_lists_and_usage_guide.txt")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"[OK] ml_dataset_all_features.csv: {ml_all_features.shape[0]:,} x {ml_all_features.shape[1]} (PatientId + AppointmentID + {len(all_feature_cols)} features + target)")
    print(f"[OK] ml_dataset_selected_features.csv: {ml_selected_features.shape[0]:,} x {ml_selected_features.shape[1]} (PatientId + AppointmentID + {len(ml_selected_features.columns)-3} selected features + target)")
    print(f"[OK] feature_lists_and_usage_guide.txt: Complete documentation with data leakage prevention guide")
    print(f"\nBoth datasets are ready for ML/DL models!")
    
    return ml_all_features, ml_selected_features

if __name__ == "__main__":
    all_features_df, selected_features_df = main()