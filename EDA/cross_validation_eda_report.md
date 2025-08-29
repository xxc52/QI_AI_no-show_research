# Cross-Validation Strategy for Hospital No-Show Panel Data
## EDA Report and Recommendations

**Date**: 2025-08-28  
**Dataset**: `ml_dataset_selected_features.csv`  
**Total Records**: 108,296 appointments  
**Unique Patients**: 62,299 patients  

---

## Executive Summary

This report analyzes the panel structure of the hospital no-show dataset to determine the optimal train/validation/test split strategy. Given the time-series nature of the data where patients have sequential appointments, we recommend a **last-appointment-based split strategy** that mirrors real-world prediction scenarios.

---

## 1. Dataset Overview

### Basic Statistics
- **Total appointments**: 108,296
- **Unique patients**: 62,299
- **Average appointments per patient**: 1.74
- **No-show rate**: 20.11% (class imbalance ratio 1:3.97)

### Panel Data Structure

| Appointments per Patient | Number of Patients | % of Patients | % of Total Data |
|-------------------------|--------------------|---------------|-----------------|
| 1 | 38,423 | 61.68% | 35.48% |
| 2 | 13,806 | 22.16% | 25.50% |
| 3 | 5,405 | 8.68% | 14.97% |
| 4+ | 4,665 | 7.49% | 24.05% |
| **Total** | **62,299** | **100%** | **100%** |

### Key Findings
- **61.68% of patients have only a single appointment** (38,423 patients)
- These single appointments represent **35.48% of all data points**
- **38.32% of patients have multiple appointments** (23,876 patients)  
- Multiple appointments represent **64.52% of all data points**

---

## 2. Temporal Features Analysis

### Feature Importance
The dataset includes 6 temporal features derived from patient history:
- `patient_previous_noshow_count`
- `patient_appointment_count`
- `patient_previous_noshow_rate`
- `days_since_last_appointment`
- `time_between_appointments_avg`
- `appointment_regularity`

### Temporal Feature Availability
- **First appointments (57.53% of data)**: No temporal features available (all zeros)
- **Subsequent appointments (42.47% of data)**: Full temporal features available

### No-show Patterns by History
| Previous No-shows | Current No-show Rate | Sample Size |
|------------------|---------------------|-------------|
| 0 | 18.95% | 93,202 |
| 1 | 25.59% | 11,661 |
| 2 | 29.86% | 2,358 |
| 3 | 35.24% | 664 |
| 4 | 40.00% | 240 |
| 5+ | 50.00% | 171 |

**Clear trend**: Higher previous no-shows strongly predict future no-shows

---

## 3. Recommended Split Strategy: Last-Appointment Based

### Why Last-Appointment Split?

For time-series panel data, using each patient's **last appointment** for validation/test is the most realistic approach because:

1. **Mimics real-world prediction**: We predict future appointments based on past history
2. **Preserves temporal ordering**: Training data always precedes validation/test data
3. **No data leakage**: Patient's future appointments never inform predictions about their past
4. **Validates temporal features**: Tests if historical patterns genuinely predict future behavior

### Proposed Data Split

| Split | Appointments | Percentage | Composition |
|-------|-------------|------------|-------------|
| **Training** | 84,420 | 77.95% | • All single-appointment patients (38,423)<br>• All but last appointments of multi-appointment patients (45,997) |
| **Validation** | 11,938 | 11.02% | • 50% of last appointments from multi-appointment patients |
| **Test** | 11,938 | 11.02% | • 50% of last appointments from multi-appointment patients |

### Split Characteristics

#### Training Set (77.95%)
- Contains **ALL single-appointment patients** (cannot be split)
- Includes all intermediate appointments from multi-appointment patients
- Mix of appointments with and without temporal features
- No-show rate: 19.79%

#### Validation Set (11.02%)
- Only from patients with 2+ appointments
- **ALL have temporal features** (guaranteed history)
- Used for hyperparameter tuning and threshold optimization
- No-show rate: 20.85%

#### Test Set (11.02%)
- Only from patients with 2+ appointments
- **ALL have temporal features** (guaranteed history)
- Final model evaluation
- No-show rate: 21.59%

---

## 4. Implementation Considerations

### Single-Appointment Patients
- **Must remain in training set** (no history to split)
- Represent 35.48% of total data
- Have NO temporal features (all zeros)
- Consider adding `is_first_appointment` flag

### Multi-Appointment Patients
- Provide both training and validation/test data
- Last appointments have full temporal history
- Enable realistic evaluation of temporal features

### No-show Rate Distribution
- Training: 19.79%
- Validation: 20.85%
- Test: 21.59%
- Slight increase in later appointments (realistic pattern)

---

## 5. Alternative Strategies Comparison

| Strategy | Pros | Cons |
|----------|------|------|
| **Last-Appointment Split** (Recommended) | • Realistic time-series evaluation<br>• No data leakage<br>• Tests temporal features properly | • Smaller val/test sets (22% combined)<br>• Cannot split single-appointment patients |
| **Random Patient Split** | • Larger val/test sets (40% combined)<br>• All patients can be split | • Unrealistic for time-series<br>• Potential data leakage<br>• Doesn't test temporal prediction |
| **Random Appointment Split** | • Easy to implement<br>• Large val/test sets | • Severe data leakage<br>• Invalid for panel data<br>• Meaningless results |

---

## 6. Recommendations

### Primary Recommendation
Implement **last-appointment-based split** with the following steps:

1. **Sort data by PatientId and appointment date** (ensure temporal ordering)
2. **Identify patients with multiple appointments** (23,876 patients)
3. **Extract last appointment** for each multi-appointment patient
4. **Randomly split** these last appointments 50/50 for validation and test
5. **Combine remaining data** with single-appointment patients for training

### Code Structure
```python
# Pseudo-code for implementation
def create_temporal_split(df):
    # Group by patient
    patient_groups = df.groupby('PatientId')
    
    # Separate single vs multiple appointment patients
    single_appt_data = []
    train_data = []
    last_appts = []
    
    for patient_id, group in patient_groups:
        if len(group) == 1:
            single_appt_data.append(group)
        else:
            train_data.append(group.iloc[:-1])  # All but last
            last_appts.append(group.iloc[-1:])  # Last appointment
    
    # Combine training data
    train = pd.concat(single_appt_data + train_data)
    
    # Split last appointments for val/test
    last_appts_df = pd.concat(last_appts)
    val, test = train_test_split(last_appts_df, test_size=0.5, 
                                  random_state=42, stratify=last_appts_df['No-show'])
    
    return train, val, test
```

### Additional Considerations

1. **Stratification**: Ensure balanced no-show rates across splits
2. **Random seed**: Fix for reproducibility
3. **Feature engineering**: Add `is_first_appointment` flag
4. **Evaluation metrics**: Focus on AUC-PR due to class imbalance
5. **Cross-validation**: For hyperparameter tuning, use time-based splits within training data

---

## 7. Expected Model Performance Implications

### Advantages of This Approach
- **Realistic performance estimates**: Results reflect actual deployment scenarios
- **Valid temporal feature evaluation**: Properly tests if history predicts future
- **No overfitting risk**: Clean separation between train and test

### Challenges
- **Smaller validation/test sets**: 11% each (but high quality)
- **Different distributions**: Val/test have only experienced patients
- **Feature coverage**: Val/test always have temporal features, training mixed

### Mitigation Strategies
1. Use **stratified sampling** when splitting last appointments
2. Consider **weighted loss functions** to handle class imbalance
3. Monitor performance separately for first vs. subsequent appointments
4. Implement **early stopping** based on validation performance

---

## Conclusion

The **last-appointment-based split strategy** is the most appropriate approach for this panel dataset. While it results in smaller validation and test sets (11% each), it provides the most realistic evaluation of model performance in a production environment where we predict future appointments based on patient history.

This approach ensures:
- No data leakage
- Proper temporal ordering
- Realistic performance estimates
- Valid evaluation of temporal features

The trade-off of smaller validation/test sets is acceptable given the significant improvement in evaluation validity and real-world applicability.