from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore
from utility.preprocess import preprocess1, preprocess2, tokenizer
import pickle

def test_model(text_list: list):
    with open("models/weights/stack_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
        
    
    predictions = pipeline.predict(text_list)
    return predictions