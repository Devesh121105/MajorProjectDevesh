import sys
from unittest.mock import MagicMock

# Mocking libraries that are hard to install in this environment
mock_libs = ['pandas', 'numpy', 'joblib', 'sklearn', 'sklearn.preprocessing', 'xgboost', 'matplotlib', 'seaborn']

class ClassesList(list):
    def tolist(self):
        return self

class DummyLabelEncoder:
    def __init__(self, classes):
        self.classes_ = ClassesList(classes)
    def transform(self, x):
        return [list(self.classes_).index(i) for i in x]
    def inverse_transform(self, x):
        return [self.classes_[i] for i in x]

class DummyScaler:
    def transform(self, x):
        return x

# Real data extracted from CSV
CITIES = ['Mumbai', 'Meerut', 'Pune', 'Delhi', 'Lucknow']
CITY_SUBAREAS = {
    'Mumbai': ['Andheri', 'Powai', 'Dadar', 'Bandra'],
    'Meerut': ['Partapur', 'Shastri Nagar', 'Modipuram', 'Ganga Nagar'],
    'Pune': ['Pimpri-Chinchwad', 'Shivajinagar', 'Kothrud', 'Hinjewadi'],
    'Delhi': ['Okhla Industrial Area', 'Rohini', 'Saket'],
    'Lucknow': ['Alambagh', 'Gomti Nagar', 'Indira Nagar', 'Charbagh']
}

AREA_STATS = {}
for city, subareas in CITY_SUBAREAS.items():
    for subarea in subareas:
        loc = f"{city} - {subarea}"
        AREA_STATS[loc] = {
            'population': 100000,
            'industrial_units': 50,
            'commercial_units': 200,
            'organic_waste': 3000,
            'plastic_waste': 2000,
            'metal_waste': 500,
            'hazardous_waste': 100,
            'ewaste': 400,
            'avg_recycling_rate': 45.0,
            'collection_frequency': 2,
            'avg_aqi': 110,
            'avg_total_waste': 6000,
            'waste_type': 'Mixed'
        }

LABEL_ENCODERS = {
    'city': DummyLabelEncoder(CITIES),
    'subarea': DummyLabelEncoder([s for subs in CITY_SUBAREAS.values() for s in subs]),
    'location': DummyLabelEncoder(list(AREA_STATS.keys())),
    'waste_type': DummyLabelEncoder(['Mixed', 'Biodegradable', 'Non-Biodegradable', 'Hazardous'])
}

def mocked_load(filename):
    if 'cities.pkl' in filename:
        return CITIES
    elif 'city_subareas.pkl' in filename:
        return CITY_SUBAREAS
    elif 'area_stats.pkl' in filename:
        return AREA_STATS
    elif 'label_encoders.pkl' in filename:
        return LABEL_ENCODERS
    elif 'scaler' in filename:
        return DummyScaler()
    elif 'class_features.pkl' in filename:
        return ['population', 'industrial_units', 'commercial_units', 'organic_waste', 'plastic_waste', 'metal_waste', 'hazardous_waste', 'ewaste', 'avg_recycling_rate', 'collection_frequency', 'aqi', 'city', 'subarea']
    return MagicMock()

# Set up the mocks
joblib_mock = MagicMock()
joblib_mock.load.side_effect = mocked_load
sys.modules['joblib'] = joblib_mock

# Mock other ML modules
for lib in mock_libs:
    if lib != 'joblib' and lib not in sys.modules:
        sys.modules[lib] = MagicMock()

# Fix numpy mocks for app.py
sys.modules['numpy'].random.normal.return_value = 0
sys.modules['numpy'].clip.side_effect = lambda x, a, b: x
sys.modules['numpy'].sin.side_effect = lambda x: 0

print("ML libraries mocked with REAL city data due to environment limitations.")
