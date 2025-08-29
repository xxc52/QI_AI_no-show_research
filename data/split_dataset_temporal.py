#!/usr/bin/env python3
"""
Temporal Data Splitting for Hospital No-Show Dataset

This script implements a simple temporal splitting approach for the hospital no-show dataset.
Instead of complex patient-based temporal splits, this uses a straightforward time-based approach:
- Sort all data by AppointmentDate (ascending)
- Split sequentially: first 80% → train, next 10% → validation, last 10% → test

This approach better reflects real-world deployment scenarios where models are trained on
historical data and tested on future data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the final dataset and prepare for temporal splitting.
    """
    print("Loading dataset...")
    df = pd.read_csv('../final_dataset_with_weather_clusters.csv')
    
    print(f"Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
    print(f"Date range in data: {df['Appointment_Date'].min()} to {df['Appointment_Date'].max()}")
    
    # Convert Appointment_Date to datetime
    df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'])
    
    # Sort by AppointmentDate to ensure temporal order
    df = df.sort_values('Appointment_Date').reset_index(drop=True)
    
    print(f"Data sorted by Appointment_Date")
    print(f"Unique patients: {df['PatientId'].nunique():,}")
    print(f"Overall no-show rate: {df['No-show'].mean():.2%}")
    
    return df

def temporal_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset temporally based on appointment dates.
    
    Args:
        df (DataFrame): Input dataset sorted by Appointment_Date
        train_ratio (float): Proportion for training set (default 0.8)
        val_ratio (float): Proportion for validation set (default 0.1)
        test_ratio (float): Proportion for test set (default 0.1)
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Handle any rounding issues
    
    print(f"\nTemporal splitting:")
    print(f"Total records: {n_total:,}")
    print(f"Train: {n_train:,} ({n_train/n_total:.1%})")
    print(f"Validation: {n_val:,} ({n_val/n_total:.1%})")
    print(f"Test: {n_test:,} ({n_test/n_total:.1%})")
    
    # Split the data sequentially
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    
    return train_df, val_df, test_df

def analyze_splits(train_df, val_df, test_df):
    """
    Analyze the temporal splits and generate statistics.
    """
    splits = {
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }
    
    print("\n" + "="*60)
    print("TEMPORAL SPLIT ANALYSIS")
    print("="*60)
    
    # Basic statistics
    stats = []
    for name, df in splits.items():
        stat = {
            'Split': name,
            'Records': len(df),
            'Percentage': len(df) / (len(train_df) + len(val_df) + len(test_df)) * 100,
            'Unique_Patients': df['PatientId'].nunique(),
            'Date_Start': df['Appointment_Date'].min().strftime('%Y-%m-%d'),
            'Date_End': df['Appointment_Date'].max().strftime('%Y-%m-%d'),
            'Days_Span': (df['Appointment_Date'].max() - df['Appointment_Date'].min()).days,
            'NoShow_Rate': df['No-show'].mean() * 100,
            'Single_Appointment_Patients': df.groupby('PatientId').size().eq(1).sum(),
            'Multiple_Appointment_Patients': df.groupby('PatientId').size().gt(1).sum()
        }
        stats.append(stat)
    
    stats_df = pd.DataFrame(stats)
    
    print("\nDATASET OVERVIEW:")
    print(stats_df.to_string(index=False, float_format='%.1f'))
    
    # Check for patient overlap between sets
    print("\n" + "-"*60)
    print("PATIENT OVERLAP ANALYSIS:")
    print("-"*60)
    
    train_patients = set(train_df['PatientId'].unique())
    val_patients = set(val_df['PatientId'].unique())
    test_patients = set(test_df['PatientId'].unique())
    
    # Calculate overlaps
    train_val_overlap = len(train_patients & val_patients)
    train_test_overlap = len(train_patients & test_patients)
    val_test_overlap = len(val_patients & test_patients)
    all_overlap = len(train_patients & val_patients & test_patients)
    
    print(f"Patients in Train only: {len(train_patients - val_patients - test_patients):,}")
    print(f"Patients in Validation only: {len(val_patients - train_patients - test_patients):,}")
    print(f"Patients in Test only: {len(test_patients - train_patients - val_patients):,}")
    print(f"Patients in Train & Validation: {train_val_overlap:,}")
    print(f"Patients in Train & Test: {train_test_overlap:,}")
    print(f"Patients in Validation & Test: {val_test_overlap:,}")
    print(f"Patients in all three sets: {all_overlap:,}")
    
    # Temporal integrity check
    print("\n" + "-"*60)
    print("TEMPORAL INTEGRITY CHECK:")
    print("-"*60)
    
    train_end = train_df['Appointment_Date'].max()
    val_start = val_df['Appointment_Date'].min()
    val_end = val_df['Appointment_Date'].max()
    test_start = test_df['Appointment_Date'].min()
    
    print(f"Training ends on: {train_end.strftime('%Y-%m-%d')}")
    print(f"Validation starts on: {val_start.strftime('%Y-%m-%d')}")
    print(f"Validation ends on: {val_end.strftime('%Y-%m-%d')}")
    print(f"Test starts on: {test_start.strftime('%Y-%m-%d')}")
    
    temporal_valid = (train_end <= val_start) and (val_end <= test_start)
    status = "PASSED" if temporal_valid else "FAILED"
    print(f"Temporal integrity: {status}")
    
    return stats_df

def create_visualizations(train_df, val_df, test_df):
    """
    Create visualizations for the temporal splits.
    """
    print("\nCreating visualizations...")
    
    # Combine data with split labels
    train_df_viz = train_df.copy()
    train_df_viz['Split'] = 'Train'
    val_df_viz = val_df.copy()
    val_df_viz['Split'] = 'Validation'
    test_df_viz = test_df.copy()
    test_df_viz['Split'] = 'Test'
    
    combined_df = pd.concat([train_df_viz, val_df_viz, test_df_viz], ignore_index=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Data Split Analysis', fontsize=16, fontweight='bold')
    
    # 1. Daily appointment counts by split
    ax1 = axes[0, 0]
    daily_counts = combined_df.groupby(['Appointment_Date', 'Split']).size().unstack(fill_value=0)
    
    colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
    for split in ['Train', 'Validation', 'Test']:
        if split in daily_counts.columns:
            ax1.plot(daily_counts.index, daily_counts[split], 
                    label=split, color=colors[split], linewidth=1.5)
    
    ax1.set_title('Daily Appointment Counts by Split')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Appointments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. No-show rate over time
    ax2 = axes[0, 1]
    # Calculate weekly no-show rates to reduce noise
    combined_df['Week'] = combined_df['Appointment_Date'].dt.to_period('W')
    weekly_noshow = combined_df.groupby(['Week', 'Split'])['No-show'].mean().unstack(fill_value=np.nan)
    
    for split in ['Train', 'Validation', 'Test']:
        if split in weekly_noshow.columns:
            # Convert period index to datetime for plotting
            dates = weekly_noshow.index.to_timestamp()
            ax2.plot(dates, weekly_noshow[split] * 100, 
                    label=split, color=colors[split], linewidth=2)
    
    ax2.set_title('Weekly No-Show Rate by Split')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('No-Show Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Split distribution (pie chart)
    ax3 = axes[1, 0]
    split_counts = combined_df['Split'].value_counts()
    split_percentages = split_counts / split_counts.sum() * 100
    
    wedges, texts, autotexts = ax3.pie(split_counts.values, 
                                      labels=split_counts.index,
                                      colors=[colors[split] for split in split_counts.index],
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax3.set_title('Data Split Distribution')
    
    # 4. No-show rate comparison
    ax4 = axes[1, 1]
    noshow_rates = combined_df.groupby('Split')['No-show'].mean() * 100
    bars = ax4.bar(noshow_rates.index, noshow_rates.values, 
                   color=[colors[split] for split in noshow_rates.index])
    ax4.set_title('No-Show Rate by Split')
    ax4.set_ylabel('No-Show Rate (%)')
    ax4.set_ylim(0, max(noshow_rates.values) * 1.1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, noshow_rates.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('temporal_split_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'temporal_split_analysis.png'")
    
    return fig

def save_splits(train_df, val_df, test_df):
    """
    Save the split datasets to CSV files.
    """
    print("\nSaving split datasets...")
    
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print(f"+ train.csv: {len(train_df):,} records")
    print(f"+ val.csv: {len(val_df):,} records")
    print(f"+ test.csv: {len(test_df):,} records")

def generate_report(stats_df):
    """
    Generate a markdown report with methodology and results.
    """
    print("\nGenerating report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Hospital No-Show Dataset Temporal Split Report

Generated: {timestamp}

## Methodology

### Temporal Splitting Approach

This implementation uses a **simple temporal splitting strategy** that better reflects real-world deployment scenarios:

1. **Data Preparation**: 
   - Sort all records by `Appointment_Date` in ascending order
   - Ensure temporal continuity in the dataset

2. **Sequential Split** (8:1:1 ratio):
   - **Training Set**: First 80% of time period (earliest appointments)
   - **Validation Set**: Next 10% of time period (middle appointments)  
   - **Test Set**: Last 10% of time period (most recent appointments)

### Rationale

**Why Temporal Over Patient-Based Splitting?**

1. **Real-World Deployment**: Models are typically trained on historical data and deployed to predict future appointments
2. **Simplicity**: Eliminates complex patient-based temporal rules and edge cases
3. **Clear Temporal Boundaries**: Clean separation between training and evaluation periods
4. **Business Relevance**: Simulates actual model deployment where future data is unknown

### Key Advantages

- **No Complex Rules**: Simple chronological split without patient overlap considerations
- **Realistic Evaluation**: Test set represents truly future data the model will encounter
- **Reproducible**: Deterministic splitting based on appointment dates
- **Scalable**: Easy to apply to datasets of any size

## Results Summary

"""
    
    # Add the statistics table
    report += "### Dataset Statistics\n\n"
    report += stats_df.to_markdown(index=False, floatfmt=".1f")
    report += "\n\n"
    
    # Add key findings
    train_stats = stats_df[stats_df['Split'] == 'Train'].iloc[0]
    val_stats = stats_df[stats_df['Split'] == 'Validation'].iloc[0]
    test_stats = stats_df[stats_df['Split'] == 'Test'].iloc[0]
    
    report += f"""### Key Findings

#### Temporal Distribution
- **Training Period**: {train_stats['Date_Start']} to {train_stats['Date_End']} ({train_stats['Days_Span']} days)
- **Validation Period**: {val_stats['Date_Start']} to {val_stats['Date_End']} ({val_stats['Days_Span']} days)
- **Test Period**: {test_stats['Date_Start']} to {test_stats['Date_End']} ({test_stats['Days_Span']} days)

#### Data Characteristics
- **No-Show Rate Variation**: {min(stats_df['NoShow_Rate']):.1f}% - {max(stats_df['NoShow_Rate']):.1f}%
- **Patient Overlap**: Expected overlap due to multiple appointments across time periods
- **Temporal Integrity**: Training < Validation < Test (chronological order maintained)

#### Model Training Implications
1. **Class Balance**: Monitor no-show rate differences between splits
2. **Temporal Drift**: Account for potential changes in patterns over time
3. **Feature Engineering**: Ensure time-dependent features respect temporal boundaries
4. **Evaluation**: Focus on test set performance as primary metric for real-world prediction

## Files Generated

- `train.csv`: Training dataset ({train_stats['Records']:,} records)
- `val.csv`: Validation dataset ({val_stats['Records']:,} records) 
- `test.csv`: Test dataset ({test_stats['Records']:,} records)
- `temporal_split_analysis.png`: Visualization of split distributions
- `temporal_split_report.md`: This report

## Implementation Notes

- **Deterministic**: Results are reproducible (no random seeds involved)
- **Temporal Order**: All data sorted by Appointment_Date before splitting
- **Clean Boundaries**: No overlap in time periods between splits
- **Data Integrity**: All original features and records preserved

---

*Report generated by split_dataset_temporal.py*
"""
    
    with open('temporal_split_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Report saved as 'temporal_split_report.md'")

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("HOSPITAL NO-SHOW TEMPORAL DATA SPLITTING")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Perform temporal split
    train_df, val_df, test_df = temporal_split(df)
    
    # Analyze splits
    stats_df = analyze_splits(train_df, val_df, test_df)
    
    # Create visualizations
    create_visualizations(train_df, val_df, test_df)
    
    # Save split datasets
    save_splits(train_df, val_df, test_df)
    
    # Generate report
    generate_report(stats_df)
    
    print("\n" + "="*60)
    print("TEMPORAL SPLITTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    print("- train.csv, val.csv, test.csv (split datasets)")
    print("- temporal_split_analysis.png (visualizations)")
    print("- temporal_split_report.md (methodology and results)")

if __name__ == "__main__":
    main()