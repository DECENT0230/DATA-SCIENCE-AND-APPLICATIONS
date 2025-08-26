# DATA-SCIENCE-AND-APPLICATIONS
assigment
# Predictive Maintenance Analysis

## Project Overview
This project implements machine learning models to predict equipment failure days based on sensor data from manufacturing equipment.

## Problem Statement
Predict equipment failure days using sensor data (Temperature, Vibration, Pressure, Runtime) to enable proactive maintenance scheduling.

##Dataset
- **File**: `Question1.csv`
- **Features**: Temperature (°C), Vibration (mm/s), Pressure (psi), Runtime (hours)
- **Target**: Days to Failure
- **Samples**: 200 equipment records

##Implementation

### Models Used:
1. **Random Forest Regressor** (Primary model)
2. **Linear Regression** (Baseline comparison)
3. **Hybrid Approach** (Random Forest + K-Means clustering)

### Evaluation Metrics:
- Root Mean Squared Error (RMSE)
- R-squared (R²) Score
- 5-Fold Cross-Validation

##Results

###Performance Summary:
- **Random Forest RMSE**: 155.78
- **Linear Regression RMSE**: 151.46  
- **Hybrid Approach RMSE**: 157.86
- **Cross-Validation RMSE**: 133.77 ± 15.32

###Key Findings:
1. Sensor data shows weak correlation with failure timing
2. Ensemble methods didn't significantly outperform linear regression
3. Runtime and Temperature are the most important features
4. Data limitations identified as primary constraint

##Installation

```bash
# Clone repository
git clone <your-repo-link>
cd Predictive-Maintenance-Analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py
