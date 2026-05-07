import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("WASTE MANAGEMENT SYSTEM - MODEL TRAINING")
print("="*60)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ==================== LOAD CSV FROM DATA FOLDER ====================
print("\n[1/6] Loading CSV file from data folder...")

# Look for CSV file
csv_path = None
possible_names = ['sample_data.csv', 'waste_data.csv', 'data.csv', 'waste.csv']

for name in possible_names:
    test_path = os.path.join('data', name)
    if os.path.exists(test_path):
        csv_path = test_path
        break

if csv_path is None:
    print(f"\n❌ ERROR: No CSV file found in 'data' folder!")
    exit(1)

print(f"   ✓ Found CSV file: {csv_path}")

# Read CSV - try different delimiters
df = None
for delim in [',', '\t', ';']:
    try:
        df = pd.read_csv(csv_path, sep=delim, encoding='utf-8')
        if len(df.columns) > 1:
            print(f"   ✓ Successfully read with delimiter: '{delim}'")
            break
    except:
        continue

if df is None:
    print("   ❌ Could not read CSV file")
    exit(1)

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()
print(f"   ✓ Columns: {list(df.columns)}")
print(f"   ✓ Loaded {len(df)} records")

# ==================== DATA PREPROCESSING ====================
print("\n[2/6] Data preprocessing...")

# City is MAIN AREA, Area_Name is SUB-AREA
df['Main_City'] = df['City_Name']  # Main area
df['Sub_Area'] = df['Area_Name']    # Sub area
df['Full_Location'] = df['Main_City'] + " - " + df['Sub_Area']

# Get unique cities and their sub-areas
cities = df['Main_City'].unique().tolist()
city_subareas = {}
for city in cities:
    city_subareas[city] = df[df['Main_City'] == city]['Sub_Area'].unique().tolist()

print(f"   ✓ Main Cities: {cities}")
print(f"   ✓ Total Locations: {len(df['Full_Location'].unique())}")
for city, subareas in city_subareas.items():
    print(f"      - {city}: {len(subareas)} sub-areas")

# Encode all categorical variables
le_city = LabelEncoder()
df['city_encoded'] = le_city.fit_transform(df['Main_City'])

le_subarea = LabelEncoder()
df['subarea_encoded'] = le_subarea.fit_transform(df['Sub_Area'])

le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['Full_Location'])

le_waste = LabelEncoder()
df['waste_type_encoded'] = le_waste.fit_transform(df['Waste_Category'])

label_encoders = {
    'city': le_city,
    'subarea': le_subarea,
    'location': le_location,
    'waste_type': le_waste
}

print(f"   ✓ Waste categories: {le_waste.classes_.tolist()}")

# Prepare features for classification
feature_columns = ['Population', 'Industrial_Units', 'Commercial_Units', 
                    'Organic_Waste_kg', 'Plastic_Waste_kg', 'Metal_Waste_kg',
                    'Hazardous_Waste_kg', 'E_Waste_kg', 'Recycling_Rate_percent',
                    'Collection_Frequency_days', 'Air_Quality_Index']

# Add encoded columns
class_features = feature_columns + ['city_encoded', 'subarea_encoded']

print(f"   ✓ Features: {class_features}")

X_class = df[class_features].copy()
y_class = df['waste_type_encoded']

# Handle missing values
X_class = X_class.fillna(X_class.mean())

# Scale features
scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\n   Training set: {len(X_train_c)} samples")
print(f"   Test set: {len(X_test_c)} samples")

# Save preprocessing objects
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(class_features, 'models/class_features.pkl')
joblib.dump(cities, 'models/cities.pkl')
joblib.dump(city_subareas, 'models/city_subareas.pkl')

# ==================== RANDOM FOREST ====================
print("\n[3/6] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                   min_samples_split=5, 
                                   min_samples_leaf=2,
                                   random_state=42,
                                   n_jobs=-1)
rf_model.fit(X_train_c, y_train_c)

y_pred_rf = rf_model.predict(X_test_c)
rf_accuracy = accuracy_score(y_test_c, y_pred_rf)
print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")

joblib.dump(rf_model, 'models/rf_classifier.pkl')

# ==================== XGBOOST ====================
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

joblib.dump(xgb_model, 'models/xgb_classifier.pkl')

# ==================== GRADIENT BOOSTING ====================
print("\n[5/6] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=6,
                                       learning_rate=0.1,
                                       subsample=0.8,
                                       random_state=42)
gb_model.fit(X_train_c, y_train_c)

y_pred_gb = gb_model.predict(X_test_c)
gb_accuracy = accuracy_score(y_test_c, y_pred_gb)
print(f"   Gradient Boosting Accuracy: {gb_accuracy:.4f}")

joblib.dump(gb_model, 'models/gb_classifier.pkl')

# ==================== NEURAL NETWORK ====================
print("\n   Training Neural Network...")
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                          activation='relu',
                          solver='adam',
                          max_iter=500,
                          random_state=42,
                          early_stopping=True,
                          validation_fraction=0.1)
mlp_model.fit(X_train_c, y_train_c)

y_pred_mlp = mlp_model.predict(X_test_c)
mlp_accuracy = accuracy_score(y_test_c, y_pred_mlp)
print(f"   Neural Network Accuracy: {mlp_accuracy:.4f}")

joblib.dump(mlp_model, 'models/mlp_classifier.pkl')

# ==================== CONFUSION MATRIX ====================
print("\n   Generating Confusion Matrix...")

# Get confusion matrix with all classes
all_classes = list(range(len(le_waste.classes_)))
cm = confusion_matrix(y_test_c, y_pred_xgb, labels=all_classes)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_waste.classes_,
            yticklabels=le_waste.classes_,
            annot_kws={'size': 14})
plt.title('Confusion Matrix - Waste Classification (XGBoost)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Confusion matrix saved to static/confusion_matrix.png")

# ==================== REGRESSION ====================
print("\n[6/6] Training Regression Model...")

X_reg = df[class_features].copy()
y_reg = df['Total_Waste_kg']

X_reg = X_reg.fillna(X_reg.mean())

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

rf_regressor = RandomForestRegressor(n_estimators=150, max_depth=12,
                                      min_samples_split=5, random_state=42,
                                      n_jobs=-1)
rf_regressor.fit(X_train_r, y_train_r)
y_pred_rf_reg = rf_regressor.predict(X_test_r)
rf_reg_r2 = r2_score(y_test_r, y_pred_rf_reg)
print(f"   Random Forest Regressor - R²: {rf_reg_r2:.4f}")

joblib.dump(rf_regressor, 'models/best_regressor.pkl')
joblib.dump(scaler_reg, 'models/scaler_reg.pkl')

# ==================== AREA STATISTICS ====================
print("\n   Creating area statistics...")
area_stats = {}

for idx, row in df.iterrows():
    location = row['Full_Location']
    area_stats[location] = {
        'city': row['Main_City'],
        'subarea': row['Sub_Area'],
        'area_type': row['Area_Type'],
        'population': int(row['Population']),
        'avg_total_waste': float(row['Total_Waste_kg']),
        'avg_recycling_rate': float(row['Recycling_Rate_percent']),
        'avg_aqi': float(row['Air_Quality_Index']),
        'industrial_units': int(row['Industrial_Units']),
        'commercial_units': int(row['Commercial_Units']),
        'organic_waste': float(row['Organic_Waste_kg']),
        'plastic_waste': float(row['Plastic_Waste_kg']),
        'metal_waste': float(row['Metal_Waste_kg']),
        'hazardous_waste': float(row['Hazardous_Waste_kg']),
        'ewaste': float(row['E_Waste_kg']),
        'waste_type': row['Waste_Category'],
        'collection_frequency': int(row['Collection_Frequency_days']),
        'waste_source': row['Waste_Source']
    }

joblib.dump(area_stats, 'models/area_stats.pkl')

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\n📊 Summary:")
print(f"   Total records: {len(df)}")
print(f"   Main Cities: {len(cities)}")
print(f"   Total Locations: {len(df['Full_Location'].unique())}")
print(f"   Waste categories: {le_waste.classes_.tolist()}")
print(f"\n🤖 Model Performance:")
print(f"   Random Forest: {rf_accuracy:.4f}")
print(f"   XGBoost: {xgb_accuracy:.4f}")
print(f"   Gradient Boosting: {gb_accuracy:.4f}")
print(f"   Neural Network: {mlp_accuracy:.4f}")
print(f"   Regression R²: {rf_reg_r2:.4f}")
print(f"\n📁 Files saved in 'models/' folder")
print(f"🖼️ Confusion matrix: static/confusion_matrix.png")
print("="*60)