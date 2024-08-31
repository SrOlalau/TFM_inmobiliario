import joblib
import os

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(script_dir, 'modelo_test_lr.pkl'))