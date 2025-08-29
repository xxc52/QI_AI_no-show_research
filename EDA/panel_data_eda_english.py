"""
Panel Data EDA for Hospital No-show Dataset (English Version)
Analysis focused on last-appointment-based train/validation/test split strategy
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

# Style settings
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_and_initial_inspection(file_path):
    """Load data and initial inspection"""
    print("="*80)
    print("1. DATA LOADING AND BASIC INFORMATION")
    print("="*80)
    
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"- Total appointments: {df.shape[0]:,}")
    print(f"- Total features: {df.shape[1]}")
    
    # Unique patients
    n_unique_patients = df['PatientId'].nunique()
    print(f"\n- Unique patients: {n_unique_patients:,}")
    print(f"- Average appointments per patient: {df.shape[0]/n_unique_patients:.2f}")
    
    # Target variable distribution
    print("\nTarget variable (No-show) distribution:")
    noshow_dist = df['No-show'].value_counts()
    noshow_pct = df['No-show'].value_counts(normalize=True) * 100
    print(f"- Show (0): {noshow_dist[0]:,} ({noshow_pct[0]:.2f}%)")
    print(f"- No-show (1): {noshow_dist[1]:,} ({noshow_pct[1]:.2f}%)")
    print(f"- Class imbalance ratio: 1:{noshow_dist[0]/noshow_dist[1]:.2f}")
    
    return df

def analyze_panel_structure(df):
    """Analyze panel data structure"""
    print("\n" + "="*80)
    print("2. PANEL DATA STRUCTURE ANALYSIS")
    print("="*80)
    
    # Appointments per patient distribution
    patient_appointment_counts = df.groupby('PatientId').size()
    
    print("\nAppointments per patient statistics:")
    print(patient_appointment_counts.describe())
    
    # Distribution by appointment count
    appointment_distribution = patient_appointment_counts.value_counts().sort_index()
    
    print("\nPatient distribution by number of appointments:")
    print("Appointments | Patients | Percentage | Cumulative %")
    print("-" * 55)
    
    cumulative_pct = 0
    for n_appts in range(1, min(11, appointment_distribution.index.max()+1)):
        if n_appts in appointment_distribution.index:
            count = appointment_distribution[n_appts]
            pct = (count / patient_appointment_counts.shape[0]) * 100
            cumulative_pct += pct
            print(f"{n_appts:^12} | {count:^8} | {pct:^10.2f} | {cumulative_pct:^12.2f}")
    
    # 10+ appointments
    many_appointments = appointment_distribution[appointment_distribution.index >= 10].sum()
    if many_appointments > 0:
        many_pct = (many_appointments / patient_appointment_counts.shape[0]) * 100
        cumulative_pct += many_pct
        print(f"{'10+':^12} | {many_appointments:^8} | {many_pct:^10.2f} | {cumulative_pct:^12.2f}")
    
    # Key statistics
    single_appointment_patients = appointment_distribution[1] if 1 in appointment_distribution.index else 0
    single_appointment_pct = (single_appointment_patients / patient_appointment_counts.shape[0]) * 100
    
    # Calculate data proportion for single vs multiple appointments
    single_appt_data = single_appointment_patients  # 1 appointment each
    total_appointments = df.shape[0]
    single_appt_data_pct = (single_appt_data / total_appointments) * 100
    
    print(f"\nKey Statistics:")
    print(f"- Single appointment patients: {single_appointment_patients:,} ({single_appointment_pct:.2f}% of patients)")
    print(f"- Single appointment data points: {single_appt_data:,} ({single_appt_data_pct:.2f}% of all data)")
    print(f"- Multiple appointment patients: {patient_appointment_counts.shape[0] - single_appointment_patients:,} ({100-single_appointment_pct:.2f}% of patients)")
    print(f"- Multiple appointment data points: {total_appointments - single_appt_data:,} ({100-single_appt_data_pct:.2f}% of all data)")
    
    return patient_appointment_counts

def analyze_last_appointment_split(df, patient_appointment_counts):
    """Analyze implications of using last appointment for validation/test"""
    print("\n" + "="*80)
    print("3. LAST-APPOINTMENT-BASED SPLIT ANALYSIS")
    print("="*80)
    
    # Patients with multiple appointments who can contribute to train AND val/test
    multi_appt_patients = (patient_appointment_counts > 1).sum()
    multi_appt_pct = (multi_appt_patients / len(patient_appointment_counts)) * 100
    
    print(f"\nPatients with multiple appointments (can use last for val/test):")
    print(f"- Count: {multi_appt_patients:,} patients ({multi_appt_pct:.2f}%)")
    
    # Calculate how many appointments would be in val/test
    # For patients with 1 appointment: all go to train (can't split)
    # For patients with 2+ appointments: last one goes to val/test
    potential_val_test = multi_appt_patients  # Each multi-appt patient contributes 1 appointment
    potential_val_test_pct = (potential_val_test / df.shape[0]) * 100
    
    print(f"\nPotential validation/test set size:")
    print(f"- Max possible: {potential_val_test:,} appointments ({potential_val_test_pct:.2f}% of data)")
    print(f"- From {multi_appt_patients:,} patients with 2+ appointments")
    
    # Analyze temporal features availability
    print("\nTemporal features in potential splits:")
    
    # For training: all except last appointments of multi-appt patients
    train_with_temporal = df[df['patient_appointment_count'] > 0].shape[0]
    
    # For val/test: only multi-appt patients' last appointments have temporal features
    print(f"- Training set temporal features: {train_with_temporal:,} appointments")
    print(f"- Val/Test temporal features: {potential_val_test:,} appointments (all have history)")
    
    # Single appointment patients handling
    single_appt_patients = (patient_appointment_counts == 1).sum()
    print(f"\nSingle appointment patients (cannot split):")
    print(f"- Count: {single_appt_patients:,} patients")
    print(f"- These must stay entirely in training set")
    print(f"- They have NO temporal features (first appointments)")
    
    # Analyze no-show rates by appointment position
    print("\nNo-show rates by appointment position:")
    
    # First appointments
    first_appts = df[df['patient_appointment_count'] == 0]
    print(f"- First appointments: {first_appts['No-show'].mean()*100:.2f}% no-show rate")
    
    # Last appointments (for patients with 2+ appointments)
    multi_appt_patient_ids = patient_appointment_counts[patient_appointment_counts > 1].index
    df_multi = df[df['PatientId'].isin(multi_appt_patient_ids)]
    last_appts = df_multi.groupby('PatientId').tail(1)
    print(f"- Last appointments (multi-appt patients): {last_appts['No-show'].mean()*100:.2f}% no-show rate")
    
    # Middle appointments
    middle_appts = df_multi[~df_multi.index.isin(last_appts.index)]
    middle_appts = middle_appts[middle_appts['patient_appointment_count'] > 0]
    if len(middle_appts) > 0:
        print(f"- Middle appointments: {middle_appts['No-show'].mean()*100:.2f}% no-show rate")
    
    return multi_appt_patients, single_appt_patients, last_appts

def analyze_temporal_features(df):
    """Analyze temporal features"""
    print("\n" + "="*80)
    print("4. TEMPORAL FEATURES ANALYSIS")
    print("="*80)
    
    temporal_features = [
        'patient_previous_noshow_count',
        'patient_appointment_count', 
        'patient_previous_noshow_rate',
        'days_since_last_appointment',
        'time_between_appointments_avg',
        'appointment_regularity'
    ]
    
    print("\nTemporal features statistics:")
    print(df[temporal_features].describe().round(2))
    
    # Relationship between previous no-shows and current no-show
    print("\nNo-show rate by previous no-show count:")
    for i in range(0, min(6, int(df['patient_previous_noshow_count'].max()+1))):
        subset = df[df['patient_previous_noshow_count'] == i]
        if len(subset) > 0:
            noshow_rate = subset['No-show'].mean() * 100
            print(f"- Previous no-shows = {i}: {noshow_rate:.2f}% current no-show rate (n={len(subset):,})")
    
    # Correlation with target
    print("\nCorrelation with No-show target:")
    for feat in temporal_features:
        corr = df[feat].corr(df['No-show'])
        print(f"- {feat}: {corr:.4f}")

def analyze_split_strategy(df, patient_appointment_counts):
    """Analyze different split strategies"""
    print("\n" + "="*80)
    print("5. SPLIT STRATEGY COMPARISON")
    print("="*80)
    
    print("\nOption 1: Last Appointment for Val/Test (Time-based)")
    print("-" * 55)
    
    # Calculate exact splits
    multi_appt_patients = patient_appointment_counts[patient_appointment_counts > 1]
    single_appt_patients = patient_appointment_counts[patient_appointment_counts == 1]
    
    # For multi-appointment patients
    multi_patient_appointments = df[df['PatientId'].isin(multi_appt_patients.index)]
    last_appointments = multi_patient_appointments.groupby('PatientId').tail(1)
    train_appointments = multi_patient_appointments[~multi_patient_appointments.index.isin(last_appointments.index)]
    
    # Add single appointment patients to training
    single_patient_appointments = df[df['PatientId'].isin(single_appt_patients.index)]
    train_with_single = pd.concat([train_appointments, single_patient_appointments])
    
    # Split last appointments into val and test (50/50 or 60/40)
    np.random.seed(42)
    val_size = int(len(last_appointments) * 0.5)
    val_indices = np.random.choice(last_appointments.index, size=val_size, replace=False)
    val_set = last_appointments.loc[val_indices]
    test_set = last_appointments[~last_appointments.index.isin(val_indices)]
    
    print(f"Training set: {len(train_with_single):,} appointments ({len(train_with_single)/len(df)*100:.2f}%)")
    print(f"- From multiple-appt patients: {len(train_appointments):,}")
    print(f"- From single-appt patients: {len(single_patient_appointments):,}")
    print(f"Validation set: {len(val_set):,} appointments ({len(val_set)/len(df)*100:.2f}%)")
    print(f"Test set: {len(test_set):,} appointments ({len(test_set)/len(df)*100:.2f}%)")
    
    print(f"\nNo-show rates:")
    print(f"- Training: {train_with_single['No-show'].mean()*100:.2f}%")
    print(f"- Validation: {val_set['No-show'].mean()*100:.2f}%")
    print(f"- Test: {test_set['No-show'].mean()*100:.2f}%")
    
    print("\nOption 2: Random Patient-based Split (Traditional)")
    print("-" * 55)
    
    # Traditional 60/20/20 split by patients
    n_patients = len(patient_appointment_counts)
    n_train = int(n_patients * 0.6)
    n_val = int(n_patients * 0.2)
    
    print(f"Training: ~{n_train:,} patients (~{n_train/n_patients*100:.0f}% of appointments)")
    print(f"Validation: ~{n_val:,} patients (~{n_val/n_patients*100:.0f}% of appointments)")
    print(f"Test: ~{n_patients-n_train-n_val:,} patients (~{(n_patients-n_train-n_val)/n_patients*100:.0f}% of appointments)")
    
    print("\nComparison:")
    print("- Last-appointment split: Realistic for time-series prediction")
    print("- Random patient split: More data for val/test but less realistic")
    
    return train_with_single, val_set, test_set

def generate_visualizations(df, patient_appointment_counts):
    """Generate key visualizations"""
    print("\n" + "="*80)
    print("6. GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of appointments per patient
    ax1 = axes[0, 0]
    appointment_dist = patient_appointment_counts.value_counts().sort_index()
    ax1.bar(appointment_dist.index[:20], appointment_dist.values[:20], color='steelblue')
    ax1.set_xlabel('Number of Appointments per Patient')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Distribution of Appointments per Patient')
    ax1.grid(True, alpha=0.3)
    
    # 2. No-show rate by appointment count
    ax2 = axes[0, 1]
    noshow_by_appts = df.groupby('patient_appointment_count')['No-show'].mean()
    ax2.plot(noshow_by_appts.index[:20], noshow_by_appts.values[:20], marker='o', color='coral')
    ax2.set_xlabel('Previous Appointment Count')
    ax2.set_ylabel('No-show Rate')
    ax2.set_title('No-show Rate by Previous Appointment Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. No-show rate by previous no-shows
    ax3 = axes[0, 2]
    noshow_by_prev = df.groupby('patient_previous_noshow_count')['No-show'].mean()
    ax3.bar(noshow_by_prev.index[:10], noshow_by_prev.values[:10], color='salmon')
    ax3.set_xlabel('Previous No-show Count')
    ax3.set_ylabel('Current No-show Rate')
    ax3.set_title('Current No-show Rate by Previous No-show Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. Lead time distribution
    ax4 = axes[1, 0]
    ax4.hist(df['lead_time'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax4.set_xlabel('Lead Time (days)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Lead Time Distribution')
    ax4.axvline(df['lead_time'].mean(), color='red', linestyle='--', label=f'Mean: {df["lead_time"].mean():.1f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Age distribution by no-show
    ax5 = axes[1, 1]
    df[df['No-show']==0]['Age'].hist(bins=30, alpha=0.5, label='Show', ax=ax5, color='blue')
    df[df['No-show']==1]['Age'].hist(bins=30, alpha=0.5, label='No-show', ax=ax5, color='red')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Age Distribution by No-show Status')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Data split visualization
    ax6 = axes[1, 2]
    patient_groups = patient_appointment_counts.value_counts().sort_index()
    single_count = patient_groups[1] if 1 in patient_groups.index else 0
    multi_count = sum(patient_groups[patient_groups.index > 1])
    
    labels = ['Single Appointment\nPatients', 'Multiple Appointment\nPatients']
    sizes = [single_count, multi_count]
    colors = ['lightcoral', 'lightblue']
    explode = (0.1, 0)
    
    ax6.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax6.set_title('Patient Distribution for Split Strategy')
    
    plt.tight_layout()
    plt.savefig('EDA/panel_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved: EDA/panel_data_analysis.png")
    plt.close()

def main():
    """Main execution function"""
    # Load data
    file_path = 'feature_selection_analysis/ml_dataset_selected_features.csv'
    
    # 1. Load and inspect
    df = load_and_initial_inspection(file_path)
    
    # 2. Panel structure analysis
    patient_appointment_counts = analyze_panel_structure(df)
    
    # 3. Last appointment split analysis
    multi_appt_patients, single_appt_patients, last_appts = analyze_last_appointment_split(
        df, patient_appointment_counts
    )
    
    # 4. Temporal features analysis
    analyze_temporal_features(df)
    
    # 5. Split strategy comparison
    train_set, val_set, test_set = analyze_split_strategy(df, patient_appointment_counts)
    
    # 6. Generate visualizations
    generate_visualizations(df, patient_appointment_counts)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    
    # Return key statistics for report
    return {
        'total_appointments': len(df),
        'unique_patients': patient_appointment_counts.shape[0],
        'single_appt_patients': single_appt_patients,
        'multi_appt_patients': multi_appt_patients,
        'noshow_rate': df['No-show'].mean() * 100,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set)
    }

if __name__ == "__main__":
    stats = main()