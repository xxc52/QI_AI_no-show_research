"""
Univariate Feature Selection Analysis
Complete and accurate analysis of ALL 46 features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data and prepare for analysis"""
    df = pd.read_csv('../final_dataset_with_weather_clusters.csv')
    print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def encode_categorical_variables(df):
    """Encode string categorical variables to numeric"""
    df_encoded = df.copy()
    
    # Encode neighbourhood_cluster (object type)
    if 'neighbourhood_cluster' in df_encoded.columns:
        le = LabelEncoder()
        df_encoded['neighbourhood_cluster_encoded'] = le.fit_transform(df_encoded['neighbourhood_cluster'])
        print(f"Encoded neighbourhood_cluster: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Encode season (object type)
    if 'season' in df_encoded.columns:
        season_map = {'autumn': 0, 'winter': 1}
        df_encoded['season_encoded'] = df_encoded['season'].map(season_map)
        print(f"Encoded season: {season_map}")
    
    return df_encoded

def classify_features(df):
    """Classify all 46 features by their statistical type"""
    
    # Features to exclude from analysis
    exclude_cols = [
        'PatientId', 'AppointmentID',
        'Registration_Date', 'Appointment_Date',
        'Registration_Date_no_time', 'appointment_date',
        'Neighbourhood',  # Excluded - using neighbourhood_cluster instead
        'No-show',  # Target variable
        'neighbourhood_cluster',  # Original string version
        'season'  # Original string version
    ]
    
    # Get all feature columns (should be 46)
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    # Initialize lists
    continuous_features = []
    binary_features = []
    categorical_features = []
    
    for col in all_features:
        unique_vals = df[col].nunique()
        
        # Classify based on number of unique values and data type
        if unique_vals == 2:
            binary_features.append(col)
        elif unique_vals <= 10:  # Small number of categories
            categorical_features.append(col)
        else:  # Continuous or high-cardinality
            continuous_features.append(col)
    
    return continuous_features, binary_features, categorical_features

def detect_constant_features(df, features):
    """Identify features with zero or near-zero variance"""
    constant_features = []
    
    for col in features:
        if df[col].var() == 0:
            constant_features.append(col)
            print(f"  - {col}: Zero variance (constant value = {df[col].iloc[0]})")
        elif df[col].nunique() == 1:
            constant_features.append(col)
            print(f"  - {col}: Single unique value")
    
    return constant_features

def perform_univariate_tests(df, continuous_features, binary_features, categorical_features, 
                            target='No-show', p_threshold=0.05):
    """
    Perform univariate statistical tests
    - ANOVA F-test for continuous features
    - Chi-square test for categorical/binary features
    """
    
    y = df[target]
    all_results = []
    
    # 1. ANOVA F-test for continuous features
    if continuous_features:
        print(f"\nANOVA F-test for {len(continuous_features)} continuous features...")
        X_cont = df[continuous_features]
        f_scores, p_values = f_classif(X_cont, y)
        
        for i, col in enumerate(continuous_features):
            all_results.append({
                'feature': col,
                'type': 'continuous',
                'test': 'ANOVA F-test',
                'statistic': f_scores[i],
                'p_value': p_values[i],
                'significant': p_values[i] < p_threshold,
                'unique_values': df[col].nunique()
            })
    
    # 2. Chi-square test for categorical and binary features
    cat_and_binary = binary_features + categorical_features
    if cat_and_binary:
        print(f"Chi-square test for {len(cat_and_binary)} categorical/binary features...")
        
        # Ensure non-negative values for chi2
        X_cat = df[cat_and_binary].copy()
        for col in cat_and_binary:
            if X_cat[col].min() < 0:
                X_cat[col] = X_cat[col] - X_cat[col].min()
        
        chi2_scores, p_values = chi2(X_cat, y)
        
        for i, col in enumerate(cat_and_binary):
            feature_type = 'binary' if col in binary_features else 'categorical'
            all_results.append({
                'feature': col,
                'type': feature_type,
                'test': 'Chi-square',
                'statistic': chi2_scores[i],
                'p_value': p_values[i],
                'significant': p_values[i] < p_threshold,
                'unique_values': df[col].nunique()
            })
    
    return pd.DataFrame(all_results)

def main():
    """Main analysis pipeline"""
    
    print("="*80)
    print("UNIVARIATE FEATURE SELECTION ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Classify features
    continuous_features, binary_features, categorical_features = classify_features(df)
    
    print(f"\nFeature Classification:")
    print(f"  Continuous: {len(continuous_features)} features")
    print(f"  Binary: {len(binary_features)} features") 
    print(f"  Categorical: {len(categorical_features)} features")
    print(f"  Total: {len(continuous_features) + len(binary_features) + len(categorical_features)} features")
    
    # Detect constant features
    print("\nChecking for constant features...")
    all_features = continuous_features + binary_features + categorical_features
    constant_features = detect_constant_features(df, all_features)
    
    if constant_features:
        print(f"Found {len(constant_features)} constant features to remove")
        # Remove constant features from lists
        continuous_features = [f for f in continuous_features if f not in constant_features]
        binary_features = [f for f in binary_features if f not in constant_features]
        categorical_features = [f for f in categorical_features if f not in constant_features]
        
        print(f"\nAfter removing constant features:")
        print(f"  Continuous: {len(continuous_features)}")
        print(f"  Binary: {len(binary_features)}")
        print(f"  Categorical: {len(categorical_features)}")
        print(f"  Total: {len(continuous_features) + len(binary_features) + len(categorical_features)}")
    
    # Perform univariate tests
    print("\nPerforming statistical tests...")
    results_df = perform_univariate_tests(df, continuous_features, binary_features, categorical_features)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    # Summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    total = len(results_df)
    selected = results_df['significant'].sum()
    print(f"Total features tested: {total}")
    print(f"Significant features (p < 0.05): {selected} ({selected/total:.1%})")
    
    # By type
    for ftype in ['continuous', 'binary', 'categorical']:
        type_df = results_df[results_df['type'] == ftype]
        if len(type_df) > 0:
            selected_type = type_df['significant'].sum()
            total_type = len(type_df)
            print(f"  {ftype.capitalize()}: {selected_type}/{total_type} ({selected_type/total_type:.1%})")
    
    # Save results
    results_df.to_csv('univariate_test_results_all_features.csv', index=False)
    print(f"\nResults saved to 'univariate_test_results_all_features.csv'")
    
    # Print top 15 features
    print("\nTop 15 Most Significant Features:")
    print("-"*80)
    print(f"{'Feature':<35} {'Type':<12} {'Test':<15} {'Statistic':>10} {'P-value':<12} {'Significant'}")
    print("-"*80)
    
    for idx, row in results_df.head(15).iterrows():
        sig = "YES" if row['significant'] else "NO"
        p_str = f"{row['p_value']:.2e}" if row['p_value'] < 0.001 else f"{row['p_value']:.6f}"
        print(f"{row['feature']:<35} {row['type']:<12} {row['test']:<15} {row['statistic']:10.2f} {p_str:<12} {sig}")
    
    # Save selected features only
    selected_features = results_df[results_df['significant']].copy()
    selected_features.to_csv('selected_features_univariate.csv', index=False)
    print(f"\nSelected features saved to 'selected_features_univariate.csv'")
    
    # Create summary for paper
    create_summary_for_paper(results_df)
    
    return results_df

def create_summary_for_paper(results_df):
    """Create a clean summary for academic paper"""
    
    selected = results_df[results_df['significant']].copy()
    
    # Group by category
    categories = {
        'Appointment Timing': ['lead_time', 'is_same_day', 'Registration_Day', 'Registration_Month', 
                              'Registration_Hour', 'Registration_Shift', 'Registration_Weekday',
                              'Appointment_Day', 'Appointment_Month', 'Appointment_Weekday'],
        'Patient Demographics': ['Age', 'Gender_F', 'Scholarship'],
        'Medical Conditions': ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap'],
        'Patient History': ['patient_previous_noshow_count', 'patient_previous_noshow_rate',
                           'patient_appointment_count', 'days_since_last_appointment',
                           'time_between_appointments_avg', 'appointment_regularity'],
        'System Features': ['SMS_received'],
        'Location': ['neighbourhood_cluster_encoded'],
        'Weather': ['temp_avg', 'temp_min', 'temp_max', 'temp_range', 'temp_change', 'temp_change_abs',
                   'rain_max', 'hum_max', 'hum_min', 'wind_avg', 'rad_max',
                   'is_rainy', 'is_hot', 'is_windy'],
        'Temporal': ['month', 'season_encoded']
    }
    
    with open('feature_selection_summary.txt', 'w') as f:
        f.write("UNIVARIATE FEATURE SELECTION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        for category, features in categories.items():
            cat_features = selected[selected['feature'].isin(features)]
            if len(cat_features) > 0:
                f.write(f"\n{category} ({len(cat_features)} features selected):\n")
                f.write("-"*40 + "\n")
                for _, row in cat_features.iterrows():
                    p_str = "p < 0.001" if row['p_value'] < 0.001 else f"p = {row['p_value']:.3f}"
                    f.write(f"  - {row['feature']}: {row['test']}, {p_str}\n")
        
        f.write(f"\n\nTotal: {len(selected)}/{len(results_df)} features selected\n")
    
    print("\nSummary for paper saved to 'feature_selection_summary.txt'")

if __name__ == "__main__":
    results = main()