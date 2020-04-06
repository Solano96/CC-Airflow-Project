from flask import Flask, request, jsonify, Response
import os
import pickle
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Print beautiful JSON
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Get humidity arima model
with open('./arima_humidity.pkl', 'rb') as pkl:
    model_humidity = pickle.load(pkl)

# Get temperature arima model
with open('./arima_temperature.pkl', 'rb') as pkl:
    model_temperature = pickle.load(pkl)

@app.route('/servicio/v1/prediccion/<int:n_periods>horas/', methods=['GET'])
def index(n_periods):
    predictions = []
    # Get predictions
    fc_temp, confint = model_temperature.predict(n_periods=n_periods, return_conf_int=True)
    fc_hum, confint = model_humidity.predict(n_periods=n_periods, return_conf_int=True)

    # Initialize the prediction time
    pred_time = datetime.now()

    # Get a list with time and predictions
    for i in range(0,n_periods):
        predictions.append({'date': pred_time.strftime('%d-%m-%Y'), 'hour': pred_time.strftime("%H:00"), 'temp': fc_temp[i], 'hum': fc_hum[i]})
        pred_time = pred_time + timedelta(hours=1)

    # Return in json format
    return jsonify(list(predictions)), 200
