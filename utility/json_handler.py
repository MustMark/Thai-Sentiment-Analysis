import json
import pandas as pd
from models.test import test_model

def handle_json_file(json_file):
    data = json.load(json_file)
    df = pd.DataFrame(data)
    text = df['text'].tolist()
    predictions = test_model(text)
    df['sentiment'] = predictions
    result_json = df.to_dict()
    return result_json