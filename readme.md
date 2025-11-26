# Ensemble Learning System for Financial Fraud Detection
## Project Overview
This project constructs an ensemble learning-based financial fraud detection system. By integrating multiple advanced gradient boosting models, it achieves high-precision financial fraud detection. The system adopts a complete data preprocessing workflow and various ensemble learning strategies, which significantly improves the accuracy and reliability of detection.

## Core Features
- Complete data preprocessing workflow (integration, cleaning, transformation, reduction)
- Hybrid sampling technology to address class imbalance issues
- Gradient boosting models based on XGBoost, LightGBM, and CatBoost
- Optuna Bayesian hyperparameter optimization
- Weighted soft voting and stacking ensemble methods
- Comprehensive model evaluation and visualization analysis

## Directory Structure
```
├── Dataset/              # Raw dataset directory
│   ├── Debt-Paying Capacity/    # Debt-paying capacity indicator data
│   ├── Development Capacity/      # Development capacity indicator data
│   ├── Disclosed Financial Indicators/  # Disclosed financial indicator data
│   ├── Per-Share Indicators/    # Per-share indicator data
│   ├── Profitability/     # Profitability indicator data
│   ├── Operational Capacity/      # Operational capacity indicator data
│   ├── Dividend Distribution/     # Dividend distribution data
│   ├── Fraud Information Summary/ # Fraud information data
│   └── Risk Level/        # Risk level indicator data
├── data_integration.py    # Data integration script
├── data_cleaning.py       # Data cleaning script
├── data_transformation.py # Data transformation script
├── data_reduction.py      # Data reduction script
├── financial_fraud_ensemble_final.ipynb  # Main ensemble learning notebook
├── integrated_data.csv    # Integrated data
├── cleaned_data.csv       # Cleaned data
├── transformed_data.csv   # Transformed data
├── reduced_data.csv       # Reduced data
├── integrated_model_performance_report.csv  # Model performance report
└── model_performance_comparison.csv  # Model performance comparison
```

## System Architecture
### Data Processing Workflow
1. **Data Integration**: Integrate financial indicator data scattered across multiple Excel files with fraud information
2. **Data Cleaning**: Handle missing values and outliers to improve data quality
3. **Data Transformation**: Create derived features through feature engineering, and perform standardization and encoding
4. **Data Reduction**: Reduce data dimensionality through feature selection and PCA dimensionality reduction

### Model Architecture
#### Base Model Layer
- XGBoost: A high-performance gradient boosting framework with strong regularization capabilities
- LightGBM: An efficient histogram-based gradient boosting framework
- CatBoost: A gradient boosting framework optimized for categorical features

#### Ensemble Strategy Layer
- Weighted Soft Voting: Allocate weights according to model performance
- Stacking Ensemble: Use logistic regression or ExtraTrees as the meta-model

## Tech Stack
- **Data Processing**: Python, Pandas, NumPy, Scikit-learn
- **Data Sampling**: imbalanced-learn (SMOTE, RandomUnderSampler)
- **Machine Learning Models**: XGBoost, LightGBM, CatBoost
- **Hyperparameter Optimization**: Optuna
- **Evaluation & Visualization**: Matplotlib, Seaborn, Scikit-learn metrics
- **Development Environment**: Jupyter Notebook

## Core Implementation
### 1. Data Preprocessing
#### Data Integration
- Integrate multi-dimensional financial indicators with fraud information
- Construct fraud labels (0/1)
- Generate `integrated_data.csv` containing 56 features and 119,060 records

#### Data Cleaning & Transformation
- Handle missing values and outliers
- Extract time features and calculate financial ratios
- Perform feature standardization and encoding
- Generate 24 principal components through PCA dimensionality reduction, retaining 96.22% of the variance

### 2. Imbalanced Data Handling
A hybrid sampling strategy combining SMOTE oversampling and RandomUnderSampler undersampling is adopted to effectively balance the dataset and improve the recognition capability for minority classes.

### 3. Model Training & Optimization
Each base model undergoes Bayesian hyperparameter optimization via Optuna, with cross-validation AUC as the optimization target:
- **XGBoost Optimization Parameters**: n_estimators, max_depth, learning_rate, gamma, etc.
- **LightGBM Optimization Parameters**: n_estimators, max_depth, learning_rate, num_leaves, etc.
- **CatBoost Optimization Parameters**: n_estimators, max_depth, learning_rate, l2_leaf_reg, etc.

### 4. Ensemble Learning Implementation
#### Weighted Soft Voting Ensemble
- Automatically allocate weights based on model AUC performance
- Support weight adjustment based on expert experience
- Implemented using sklearn's VotingClassifier

#### Stacking Ensemble
- Custom `StackingEnsemble` class to implement stratified K-fold cross-validation stacking
- **First Layer**: XGBoost, LightGBM, CatBoost base models
- **Second Layer**: Logistic regression or ExtraTrees meta-model

## Performance Results
AUC performance of each model on the test set:

| Model | AUC |
|--------------------------------|--------|
| Weighted Ensemble | 0.8039 |
| VotingClassifier | 0.8039 |
| Stacking Ensemble (Logistic Regression) | 0.8023 |
| LightGBM | 0.7917 |
| XGBoost | 0.7892 |
| CatBoost | 0.7881 |
| Stacking Ensemble (ExtraTrees) | 0.7791 |

The ensemble method achieves an approximately 1.54% improvement compared to the best single model (LightGBM).

## User Guide
### Environment Configuration
```
pip install -r requirements.txt
```

### Workflow
1. **Data Preparation**: Place raw data in the Dataset directory
2. **Data Preprocessing**: Run the data preprocessing scripts in sequence
   ```
   python data_integration.py
   python data_cleaning.py
   python data_transformation.py
   python data_reduction.py
   ```
3. **Model Training & Evaluation**: Run the Jupyter Notebook
   ```
   jupyter notebook financial_fraud_ensemble_final.ipynb
   ```

## Practical Application Suggestions
1. **Real-Time Monitoring & Update**: Establish a regular model update mechanism to track changes in model performance
2. **Human-Machine Collaborative Decision-Making**: The model provides risk scores, and experts conduct final reviews
3. **Interpretability Enhancement**: Use tools such as SHAP values to explain model decisions
4. **Compliance Considerations**: Ensure the model complies with financial regulatory requirements and maintain decision records

## Future Improvement Directions
1. **Dynamic Feature Engineering**: Develop more complex time-series features
2. **Deep Learning Integration**: Incorporate deep learning models into the ensemble framework
3. **Active Learning**: Reduce labeling costs
4. **Federated Learning**: Conduct model training while protecting data privacy
5. **Causal Inference**: Explore the causal relationship between financial indicators and fraudulent behaviors

## Summary
This project constructs a high-performance financial fraud detection system through a complete data preprocessing workflow and advanced ensemble learning technologies. The weighted soft voting ensemble method performs best in tests, with significant performance improvements over single models. The system's modular design makes it easy to extend and maintain, and it can be adjusted and optimized according to actual business needs.
