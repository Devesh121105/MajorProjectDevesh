import sys
import os

# Mock pandas to see if it's strictly needed
class MockPandas:
    def __init__(self):
        pass

sys.modules['pandas'] = MockPandas()

try:
    import numpy as np
    import joblib
    print("Loading models...")
    rf_model = joblib.load('models/rf_classifier.pkl')
    area_stats = joblib.load('models/area_stats.pkl')
    print("Successfully loaded models without real pandas!")
except Exception as e:
    print(f"Failed to load models: {e}")
    import traceback
    traceback.print_exc()
