import pickle

def test_model(text_list: list):
    with open("models/weights/stack_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
        
    
    predictions = pipeline.predict(text_list)
    return predictions