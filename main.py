import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Load and prepare data
data = pd.read_csv('housing.csv')
housing = pd.DataFrame(data=data)
# Create income categories for stratification
housing['income_cat'] = pd.cut(housing['median_income'], 
                              bins=[0, 1.5, 3.0, 4.5, 6., np.inf], 
                              labels=[1, 2, 3, 4, 5])

# Stratified sampling to maintain income distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove the income_cat attribute
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# CORRECTED: Extract features and labels from TRAINING set
housing_features = strat_train_set.drop("median_house_value", axis=1)
housing_features.dropna()
housing_label = strat_train_set['median_house_value'].copy()
test_features = strat_test_set.drop("median_house_value", axis=1)
test_labels = strat_test_set['median_house_value'].copy()
# CORRECTED: Feature engineering function
def add_extra_features(data, add_bedrooms_per_room=True):
    """Add engineering features that often improve model performance"""
    """ Adding the engineering features that work both with Dataframe and numpy arrays"""
    if isinstance(data,np.ndarray):
        columns_names=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income']
        housing_data=pd.DataFrame(data=data,columns=columns_names)
    else:
        housing_data = data.copy()
    
    # Fix column names and logic
    housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
    housing_data['population_per_household'] = housing_data['population'] / housing_data['households']
    
    # Add the missing bedrooms_per_room feature
    if add_bedrooms_per_room:
        housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
    
    return housing_data

# Custom transformer for feature engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

# this fit and tranform are used for the ease of implementing tools in the transformer in scikit-learn 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_extra_features(X, self.add_bedrooms_per_room)

# Identify numerical and categorical columns
numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                 'total_bedrooms', 'population', 'households', 'median_income']
categorical_cols = ['ocean_proximity']

# CORRECTED: Create preprocessing pipelines with proper instantiation
# Restructured pipeline with correct order
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_eng', FeatureEngineering()),  # After imputation
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Clean ColumnTransformer without duplicate column processing
preprocessing_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Apply preprocessing to training data
housing_prepared = preprocessing_pipeline.fit_transform(housing_features)

print(f'Original features: {len(housing_features.columns)}')
print(f'After processing: {housing_prepared.shape[1]} features')
print("Preprocessing complete!")

def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    """Get feature names after preprocessing"""
    num_features = numerical_cols.copy()
    cat_features = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)) # ask this line later 
    
    # Engineering features (now correctly implemented)
    eng_features = ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
    
    return num_features + cat_features + eng_features

feature_names = get_feature_names(preprocessing_pipeline, numerical_cols, categorical_cols)
print("Final feature names:", feature_names)

# time for model training 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# pipeline has already been imported before

linear_pipeline=Pipeline([('preprocessing',preprocessing_pipeline),# existing preprocessing
('model',LinearRegression()) ])

forest_pipeline=Pipeline([('preprocessing',preprocessing_pipeline),('model',RandomForestRegressor(n_estimators=100,random_state=42))])

tree_pipeline=Pipeline([('preprocessing',preprocessing_pipeline),('model',DecisionTreeRegressor(random_state=42))])

# dictionary to store the models 

models={'Linear Regression':linear_pipeline,
        'Decision tree':tree_pipeline,
        'Random forest':forest_pipeline}

model_result={}

for model_name,pipeline in models.items():
    print(f"Training {model_name}")
    # training the pipeline
    pipeline.fit(housing_features,housing_label)
    # sotre the trained pipeline
    model_result[model_name]=pipeline
    print(f'{model_name} training complete!')

# model evaluation time 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,validation_curve
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import PredictionErrorDisplay
import warnings
warnings.filterwarnings('ignore')

# comprehensive model evaluation on training data 

print('Evaluating all the trained models on training and test data')

# Initialize the evaluation results dictionary
evaluation_results = {}

# Store your trained models (assuming you have them in model_results)
for model_name, pipeline in model_result.items():
    print(f"-----Evaluating----{model_name}")
    
    # Make predictions on both training and test data
    train_predictions = pipeline.predict(housing_features)
    test_predictions = pipeline.predict(test_features)
    
    # Calculate training metrics
    train_rmse = np.sqrt(mean_squared_error(housing_label, train_predictions))
    train_mae = mean_absolute_error(housing_label, train_predictions)
    train_r2 = r2_score(housing_label, train_predictions)
    
    # Calculate test metrics
    test_rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
    test_mae = mean_absolute_error(test_labels, test_predictions)
    test_r2 = r2_score(test_labels, test_predictions)
    
    # CORRECTED: Store results in proper dictionary structure
    evaluation_results[model_name] = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions  
    }
    
    # Print results
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test R²: {test_r2:.4f}")

# cross-validation for model performance increment 

print('Cross validation Analysis')
print("performing 5-fold cross validation for robust performance estimates....")

cv_results={}

for model_name,pipeline in model_result.items():
    print(f'Cross validating {model_name}')

    # perform 5-fold cv

    cv_score=cross_val_score(pipeline,housing_features,housing_label,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)

    cv_rmse_scores=np.sqrt(-cv_score)
    cv_mean=cv_rmse_scores.mean()
    cv_std=cv_rmse_scores.std()

    cv_results[model_name]={
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_rmse_scores
    }
    print(f"CV RMSE: {cv_mean:.2f} (+/- {cv_std * 2:.2f})")
    print(f"Individual fold scores: {cv_rmse_scores}")

# model comaparison data 

print("------------Model comparison Table----------")

comparison_data=[]

for model_name in evaluation_results.keys():
    comparison_data.append({
        'model':model_name,
        'Train RMSE': f"{evaluation_results[model_name]['train_rmse']:.3f}",
        'Test RMSE': f"{evaluation_results[model_name]['test_rmse']:.3f}",
        'CV RMSE (Mean)': f"{cv_results[model_name]['cv_mean']:.3f}",
        'CV RMSE (Std)': f"{cv_results[model_name]['cv_std']:.3f}",
        'Test R²': f"{evaluation_results[model_name]['test_r2']:.4f}",
        'Overfitting': f"{evaluation_results[model_name]['train_rmse'] - evaluation_results[model_name]['test_rmse']:.3f}"
    })

comaprison_df=pd.DataFrame(comparison_data)
print(comaprison_df.to_string(index=False))

print("--------Creating Prediction Visualisations----------")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted House Values - Test Set Performance', fontsize=16)

for idx, (model_name, results) in enumerate(evaluation_results.items()):
    ax = axes[idx]
    
    # Create scatter plot
    ax.scatter(test_labels, results['test_predictions'], 
              alpha=0.6, color=plt.cm.Set1(idx))
    
    # Add perfect prediction line
    min_val = min(test_labels.min(), results['test_predictions'].min())
    max_val = max(test_labels.max(), results['test_predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Formatting
    ax.set_xlabel('Actual House Values')
    ax.set_ylabel('Predicted House Values')
    ax.set_title(f'{model_name}\nRMSE: {results["test_rmse"]:.3f}, R²: {results["test_r2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== RESIDUAL ANALYSIS ===")

# Create residual plots for the best performing model
best_model_name = min(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x]['test_rmse'])
best_results = evaluation_results[best_model_name]

print(f"Analyzing residuals for best model: {best_model_name}")

# Calculate residuals
residuals = test_labels - best_results['test_predictions']

# Create residual plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Residual Analysis - {best_model_name}', fontsize=16)

# Residuals vs Predicted
axes[0, 0].scatter(best_results['test_predictions'], residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of residuals
axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].axvline(x=0, color='r', linestyle='--')

# Q-Q plot for normality check
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')

# Residuals vs Order (to check for patterns)
axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Observation Order')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Order')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# this is for random forest 

if 'Random Forest' in best_model_name:
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get feature importance from the best model
    rf_model = model_result[best_model_name].named_steps['model']
    feature_importance = rf_model.feature_importances_
    
    # Get feature names (you'll need to create this based on your preprocessing)
    feature_names = get_feature_names(preprocessing_pipeline, numerical_cols, categorical_cols)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# final model selection and summary

print(f"\n=== FINAL MODEL SELECTION ===")
print(f"Best performing model: {best_model_name}")
print(f"Final Test RMSE: ${best_results['test_rmse']:.2f} (hundreds of thousands)")
print(f"Final Test MAE: ${best_results['test_mae']:.2f}")
print(f"Final Test R²: {best_results['test_r2']:.4f}")
print(f"This means the model explains {best_results['test_r2']*100:.1f}% of house price variance")

rmse_dollars=best_results['test_rmse']*100000
mae_dollars=best_results['test_mae']*100000

print(f"\nPractical Interpretation:")
print(f"- Average prediction error: ${rmse_dollars:,.0f}")
print(f"- Typical prediction error: ${mae_dollars:,.0f}")
print(f"- Model explains {best_results['test_r2']*100:.1f}% of price variation")

if best_results['train_rmse'] - best_results['test_rmse'] > 0.05:
    print(f"⚠️  Warning: Potential overfitting detected (train-test RMSE diff: {best_results['train_rmse'] - best_results['test_rmse']:.3f})")
else:
    print(f"✅ Good generalization: minimal overfitting detected")