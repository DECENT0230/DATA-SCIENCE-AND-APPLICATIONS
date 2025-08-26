

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== Predictive Maintenance Analysis ===")
print("Loading and preparing data...")

# Load and prepare data
df = pd.read_csv('Question1.csv')
df = df.drop('Index', axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target: {df.columns[-1]}")

# Prepare features and target
X = df[['Temperature', 'Vibration', 'Pressure', 'Runtime']]
y = df['Days to Failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R² Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                           cv=5, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores.mean()

print(f"\n5-Fold Cross-Validation:")
print(f"Average CV RMSE: {cv_rmse:.2f}")
print(f"Individual Fold RMSE: {-cv_scores}")

# Compare with Linear Regression
print("\n=== Baseline Comparison ===")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print(f"Linear Regression RMSE: {rmse_linear:.2f}")
print(f"Random Forest vs Linear: {rmse_linear - rmse:.2f} RMSE difference")

# Hybrid approach with clustering
print("\n=== Hybrid Approach with Clustering ===")
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

X_train_hybrid = np.column_stack((X_train_scaled, train_clusters))
X_test_hybrid = np.column_stack((X_test_scaled, test_clusters))

rf_hybrid = RandomForestRegressor(random_state=42)
rf_hybrid.fit(X_train_hybrid, y_train)
y_pred_hybrid = rf_hybrid.predict(X_test_hybrid)
rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))

print(f"Hybrid Approach RMSE: {rmse_hybrid:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Visualization
print("\nGenerating visualizations...")
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Failure Prediction')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')

# Save results
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression', 'Hybrid Approach'],
    'RMSE': [rmse, rmse_linear, rmse_hybrid],
    'R2_Score': [r2, r2_score(y_test, y_pred_linear), r2_score(y_test, y_pred_hybrid)]
})

results_df.to_csv('results/model_results.csv', index=False)
feature_importance.to_csv('results/feature_importance.csv', index=False)

print("\n=== Analysis Complete ===")
print("Results saved to:")
print("- results/model_results.csv")
print("- results/feature_importance.csv")
print("- images/feature_importance.png")

print("\n=== Key Findings ===")
print("1. Sensor data shows limited predictive power for failure timing")
print("2. Random Forest performed similarly to Linear Regression")
print("3. Feature importance: Runtime and Temperature are most influential")
print("4. Recommendation: Focus on condition-based maintenance rather than prediction")