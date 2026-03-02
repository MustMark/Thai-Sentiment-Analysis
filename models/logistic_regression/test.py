from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore
from models.preprocess import preprocess1, preprocess2

def test_model(text: str):
    vectorizer, model = joblib.load("models/logistic_regression/model.pkl")
    # vectorizer = joblib.load("models/logistic_regression/vectorizer.pkl")
    
    preprocessed_text = preprocess2(text)
    
    new_vector = vectorizer.transform(preprocessed_text)
    prediction = model.predict(new_vector)

    return prediction[0]