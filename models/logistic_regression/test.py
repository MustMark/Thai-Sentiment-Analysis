from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore

def test_model(text: str):
    model = joblib.load("models/logistic_regression/model.pkl")
    vectorizer = joblib.load("models/logistic_regression/vectorizer.pkl")
    
    new_vector = vectorizer.transform([text])

    prediction = model.predict(new_vector)

    return prediction[0]