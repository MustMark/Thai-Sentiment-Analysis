from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore
from models.preprocess import preprocess1, preprocess2
import pickle

def test_model(text: str):
    with open("models/SVM/sentiment_model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)
    
    preprocessed_text = preprocess2(text)
    
    vec = vectorizer.transform(preprocessed_text)
    prediction = model.predict(vec)

    return prediction[0]