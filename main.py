from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

app = FastAPI()


# Load the model
model = pickle.load(open('model.pkl','rb'))

# Define the class
class Data(BaseModel):
    battery_power: float
    fc: float
    int_memory: float
    mobile_wt: float
    px_height: float
    px_width: float
    ram: float
    sc_w: float
    talk_time: float

# Define the prediction function
def predict(data):
    data = data.dict()
    battery_power = data['battery_power']
    fc = data['fc']
    int_memory = data['int_memory']
    mobile_wt = data['mobile_wt']
    px_height = data['px_height']
    px_width = data['px_width']
    ram = data['ram']
    sc_w = data['sc_w']
    talk_time = data['talk_time']
    prediction = model.predict([[battery_power, fc, int_memory, mobile_wt, px_height, px_width, ram, sc_w, talk_time]])
    if prediction == 0:
        prediction = 'low'
    elif prediction == 1:
        prediction = 'medium'
    elif prediction == 2:
        prediction = 'high'
    else:
        prediction = 'very high'
    return prediction



# Define the home page
@app.get('/')
def index():
    return "HELLO!"

# Define the post method
@app.post('/predict')
def get_prediction(data: Data):
    prediction = predict(data)
    return {
        'prediction': prediction
    }

# Define the main method
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# Path: model.pkl

