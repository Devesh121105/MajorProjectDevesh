import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("WASTE MANAGEMENT SYSTEM - MODEL TRAINING")
print("="*60)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Generate synthetic dataset with area information
print("\n[1/6] Generating dataset...")

np.random.seed(42)
n_samples = 5000

# Area/Zones
areas = ['Downtown', 'Industrial', 'Residential', 'Commercial', 'Suburban', 'Riverside', 'Airport', 'Harbor']
area_weights = {
    'Downtown': 1.5, 'Industrial': 2.0, 'Residential': 1.2,
    'Commercial': 1.8, 'Suburban': 0.8, 'Riverside': 1.0,
    'Airport': 1.3, 'Harbor': 1.4
}

# Features
area_list = np.random.choice(areas, n_samples, p=[area_weights[a]/sum(area_weights.values()) for a in areas])
population = np.random.randint(500, 50000, n_samples)
area_sqkm = np.random.uniform(0.5, 25, n_samples)
num_industries = np.random.randint(0, 50, n_samples)
num_hospitals = np.random.randint(0, 10, n_samples)
num_schools = np.random.randint(0, 20, n_samples)
avg_income = np.random.randint(20000, 150000, n_samples)
waste_per_capita = np.random.uniform(0.3, 2.5, n_samples)
recycling_rate = np.random.uniform(0, 60, n_samples)
temperature = np.random.uniform(15, 40, n_samples)
humidity = np.random.uniform(30, 90, n_samples)
day_of_week = np.random.randint(0, 6, n_samples)
is_weekend = (day_of_week >= 5).astype(int)

# Calculate total waste generated
total_waste = (population * waste_per_capita * 
               (1 + num_industries/200) * 
               (1 + (temperature - 25)/50) +
               np.random.normal(0, 200, n_samples))
total_waste = np.maximum(total_waste, 100)

# Waste classification (target for classifier)
def classify_waste(total_w, pop, industries, recycling):
    ratio = total_w / pop
    if industries > 20:
        return 'Industrial Hazardous'
    elif ratio > 2.0:
        return 'High Volume Organic'
    elif recycling < 15:
        return 'Mixed Unsegregated'
    elif ratio < 0.8:
        return 'Low Volume Dry'
    else:
        return 'Standard Mixed'

waste_type = [classify_waste(tw, p, ind, rec) 
              for tw, p, ind, rec in zip(total_waste, population, num_industries, recycling_rate)]

# Create AQI data based on waste and industries
aqi = (50 + (total_waste / 1000) * 10 + 
       num_industries * 2 + 
       np.random.normal(0, 10, n_samples))
aqi = np.clip(aqi, 30, 500)

# Create DataFrame
df = pd.DataFrame({
    'area': area_list,
    'population': population,
    'area_sqkm': area_sqkm,
    'num_industries': num_industries,
    'num_hospitals': num_hospitals,
    'num_schools': num_schools,
    'avg_income': avg_income,
    'waste_per_capita': waste_per_capita,
    'recycling_rate': recycling_rate,
    'temperature': temperature,
    'humidity': humidity,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'total_waste': total_waste,
    'aqi': aqi,
    'waste_type': waste_type
})

# Save dataset
df.to_csv('data/waste_data.csv', index=False)
print(f"   Dataset saved: {len(df)} records, 8 areas")

print("\n[2/6] Data preprocessing...")

# Encode categorical variables
label_encoders = {}
le_area = LabelEncoder()
df['area_encoded'] = le_area.fit_transform(df['area'])
label_encoders['area'] = le_area

le_waste = LabelEncoder()
df['waste_type_encoded'] = le_waste.fit_transform(df['waste_type'])
label_encoders['waste_type'] = le_waste

# Prepare features for classification
class_features = ['population', 'area_sqkm', 'num_industries', 'num_hospitals',
                  'num_schools', 'avg_income', 'recycling_rate', 'temperature',
                  'humidity', 'day_of_week', 'is_weekend', 'area_encoded']

X_class = df[class_features]
y_class = df['waste_type_encoded']

# Scale features
scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"   Training set: {len(X_train_c)} samples")
print(f"   Test set: {len(X_test_c)} samples")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

# ==================== RANDOM FOREST CLASSIFIER ====================
print("\n[3/6] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                   min_samples_split=5, 
                                   min_samples_leaf=2,
                                   random_state=42,
                                   n_jobs=-1)
rf_model.fit(X_train_c, y_train_c)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_c, y_train_c, cv=5)
print(f"   Cross-validation scores: {cv_scores}")
print(f"   Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Evaluate
y_pred_rf = rf_model.predict(X_test_c)
rf_accuracy = accuracy_score(y_test_c, y_pred_rf)
print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"   Classification Report:\n{classification_report(y_test_c, y_pred_rf, target_names=le_waste.classes_)}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test_c, y_pred_rf)
print(f"   Confusion Matrix:\n{cm_rf}")

joblib.dump(rf_model, 'models/rf_classifier.pkl')

# ==================== XGBOOST CLASSIFIER ====================
print("\n[4/6] Training XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=8, 
                               learning_rate=0.1, 
                               subsample=0.8,
                               colsample_bytree=0.8,
                               random_state=42,
                               use_label_encoder=False,
                               eval_metric='mlogloss')
xgb_model.fit(X_train_c, y_train_c)

y_pred_xgb = xgb_model.predict(X_test_c)
xgb_accuracy = accuracy_score(y_test_c, y_pred_xgb)
print(f"   XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"   Confusion Matrix:\n{confusion_matrix(y_test_c, y_pred_xgb)}")

joblib.dump(xgb_model, 'models/xgb_classifier.pkl')

# ==================== GRADIENT BOOSTING CLASSIFIER ====================
print("\n[5/6] Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=6,
                                       learning_rate=0.1,
                                       subsample=0.8,
                                       random_state=42)
gb_model.fit(X_train_c, y_train_c)

y_pred_gb = gb_model.predict(X_test_c)
gb_accuracy = accuracy_score(y_test_c, y_pred_gb)
print(f"   Gradient Boosting Accuracy: {gb_accuracy:.4f}")

joblib.dump(gb_model, 'models/gb_classifier.pkl')

# ==================== NEURAL NETWORK (MLP) ====================
print("\n   Training Neural Network (MLP Classifier)...")
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                          activation='relu',
                          solver='adam',
                          max_iter=300,
                          random_state=42,
                          early_stopping=True,
                          validation_fraction=0.1)
mlp_model.fit(X_train_c, y_train_c)

y_pred_mlp = mlp_model.predict(X_test_c)
mlp_accuracy = accuracy_score(y_test_c, y_pred_mlp)
print(f"   MLP Neural Network Accuracy: {mlp_accuracy:.4f}")

joblib.dump(mlp_model, 'models/mlp_classifier.pkl')

# ==================== REGRESSION MODELS for Waste Prediction ====================
print("\n[6/6] Training Regression Models for Waste Prediction...")

# Prepare regression features
reg_features = ['population', 'num_industries', 'num_hospitals', 'num_schools',
                'avg_income', 'recycling_rate', 'temperature', 'humidity',
                'day_of_week', 'area_encoded']
X_reg = df[reg_features]
y_reg = df['total_waste']

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# Linear Regression
print("\n   Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
y_pred_lr = lr_model.predict(X_test_r)
lr_r2 = r2_score(y_test_r, y_pred_lr)
lr_mse = mean_squared_error(y_test_r, y_pred_lr)
lr_mae = mean_absolute_error(y_test_r, y_pred_lr)
print(f"   Linear Regression - R²: {lr_r2:.4f}, MSE: {lr_mse:.2f}, MAE: {lr_mae:.2f}")

# Ridge Regression
print("\n   Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_r, y_train_r)
y_pred_ridge = ridge_model.predict(X_test_r)
ridge_r2 = r2_score(y_test_r, y_pred_ridge)
print(f"   Ridge Regression - R²: {ridge_r2:.4f}")

# Random Forest Regressor
print("\n   Training Random Forest Regressor...")
rf_regressor = RandomForestRegressor(n_estimators=150, max_depth=12,
                                      min_samples_split=5, random_state=42,
                                      n_jobs=-1)
rf_regressor.fit(X_train_r, y_train_r)
y_pred_rf_reg = rf_regressor.predict(X_test_r)
rf_reg_r2 = r2_score(y_test_r, y_pred_rf_reg)
rf_reg_mse = mean_squared_error(y_test_r, y_pred_rf_reg)
print(f"   Random Forest Regressor - R²: {rf_reg_r2:.4f}, MSE: {rf_reg_mse:.2f}")

# XGBoost Regressor
print("\n   Training XGBoost Regressor...")
xgb_regressor = xgb.XGBRegressor(n_estimators=150, max_depth=8,
                                  learning_rate=0.1, random_state=42)
xgb_regressor.fit(X_train_r, y_train_r)
y_pred_xgb_reg = xgb_regressor.predict(X_test_r)
xgb_reg_r2 = r2_score(y_test_r, y_pred_xgb_reg)
print(f"   XGBoost Regressor - R²: {xgb_reg_r2:.4f}")

# Save best regression model (Random Forest performed best)
joblib.dump(rf_regressor, 'models/best_regressor.pkl')
joblib.dump(lr_model, 'models/linear_regression.pkl')
joblib.dump(ridge_model, 'models/ridge_regression.pkl')
joblib.dump(xgb_regressor, 'models/xgb_regressor.pkl')
joblib.dump(scaler_reg, 'models/scaler_reg.pkl')

# ==================== Time Series Simulation using Moving Average ====================
print("\n   Creating Time Series Forecasting Model (Moving Average + Trend)...")

def create_timeseries_model():
    """Create a simple but effective time series model using historical patterns"""
    
    # Generate time series data for each area
    ts_models = {}
    
    for area in areas:
        area_data = df[df['area'] == area]['total_waste'].values
        
        if len(area_data) >= 30:
            # Calculate seasonal patterns
            daily_pattern = np.zeros(7)
            for i in range(min(len(area_data), 100)):
                daily_pattern[i % 7] += area_data[i]
            daily_pattern = daily_pattern / min(len(area_data), 100)
            
            # Calculate trend
            trend = np.polyfit(range(len(area_data[:100])), area_data[:100], 1)[0]
            
            ts_models[area] = {
                'base_level': np.mean(area_data[:50]),
                'daily_pattern': daily_pattern.tolist(),
                'trend': float(trend),
                'volatility': float(np.std(area_data[:50]))
            }
    
    return ts_models

ts_models = create_timeseries_model()
joblib.dump(ts_models, 'models/timeseries_models.pkl')

print("\n   Time series models created for all areas")

# Model Comparison Summary
print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print("\nCLASSIFICATION MODEL COMPARISON:")
print(f"   {'Model':<25} {'Accuracy':<12} {'CV Score':<12}")
print(f"   {'-'*50}")
print(f"   {'Random Forest':<25} {rf_accuracy:.4f}{'':<8} {cv_scores.mean():.4f}")
print(f"   {'XGBoost':<25} {xgb_accuracy:.4f}{'':<8} {'N/A':<12}")
print(f"   {'Gradient Boosting':<25} {gb_accuracy:.4f}{'':<8} {'N/A':<12}")
print(f"   {'Neural Network (MLP)':<25} {mlp_accuracy:.4f}{'':<8} {'N/A':<12}")

print("\nREGRESSION MODEL COMPARISON:")
print(f"   {'Model':<25} {'R² Score':<12} {'MSE':<12}")
print(f"   {'-'*50}")
print(f"   {'Linear Regression':<25} {lr_r2:.4f}{'':<8} {lr_mse:.2f}")
print(f"   {'Ridge Regression':<25} {ridge_r2:.4f}{'':<8} {'N/A':<12}")
print(f"   {'Random Forest':<25} {rf_reg_r2:.4f}{'':<8} {rf_reg_mse:.2f}")
print(f"   {'XGBoost':<25} {xgb_reg_r2:.4f}{'':<8} {'N/A':<12}")

print("\n" + "="*60)
print("All models saved successfully in 'models/' directory!")
print("="*60)