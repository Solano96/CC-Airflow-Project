from flask import Flask, request, jsonify, Response
import os
import pickle
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Print beautiful JSON
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Get humidity arima model
with open('/tmp/airflow_data/v2/exp_smoothing_humidity.pkl', 'rb') as pkl:
    model_humidity = pickle.load(pkl)

# Get temperature arima model
with open('/tmp/airflow_data/v2/exp_smoothing_temperature.pkl', 'rb') as pkl:
    model_temperature = pickle.load(pkl)

@app.route('/servicio/v2/prediccion/<int:n_periods>horas/', methods=['GET'])
def index(n_periods):
    predictions = []
    # Get predictions
    fc_temp = model_temperature.predict(24, 23+n_periods)
    fc_hum = model_humidity.predict(24, 23+n_periods)

    # Initialize the prediction time
    pred_time = datetime.now()

    # Get a list with time and predictions
    for i in range(0,n_periods):
        predictions.append({'date': pred_time.strftime('%d-%m-%Y'), 'hour': pred_time.strftime("%H:00"), 'temp': fc_temp[24+i], 'hum': fc_hum[24+i]})
        pred_time = pred_time + timedelta(hours=1)

    # Return in json format
    return jsonify(list(predictions)), 200
