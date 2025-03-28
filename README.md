# Payment Fraud Detection Project

## Overview
This project focuses on detecting payment fraud using machine learning techniques, specifically Logistic Regression. The goal is to build a predictive model that can identify potentially fraudulent payment transactions.

## Project Structure
- Data analysis and preprocessing
- Feature scaling
- Model training using Logistic Regression
- Model evaluation and performance metrics

## Dependencies
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Data Preprocessing
- Loaded payment fraud dataset
- Checked for null values
- Performed label encoding on payment methods
- Scaled features using StandardScaler

## Exploratory Data Analysis
### Payment Method Distribution
- Visualized distribution of payment methods using bar plot
- Analyzed class distribution in the target variable

## Model Training
- Used Logistic Regression classifier
- Split data into training (75%) and testing (25%) sets
- Performed feature scaling before model training

## Model Evaluation
### Metrics
- Accuracy Score
- Classification Report
- Confusion Matrix

## Key Findings
- Model performance details are available in the classification report
- Confusion matrix provides insights into model's prediction accuracy

## How to Run
```bash
python payment_fraud_detection.py
```

## Future Improvements
- Experiment with other machine learning algorithms
- Implement more advanced feature engineering
- Try ensemble methods for potentially better performance
