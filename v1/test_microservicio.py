import pytest
import sys
import json
import os
from flask import Flask, request, jsonify, Response
import os
import pickle
from datetime import datetime, timedelta
import json
from microservicio import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()

    yield client


def test_len_24horas(client):
    now_time = datetime.now()

    test_data_1 = {
        "date": now_time.strftime('%d-%m-%Y'),
        "hour": now_time.strftime("%H:00"),
        "hum": 29.720270734703718,
        "temp": 301.82215386331615
    }

    response = client.get('/servicio/v1/prediccion/24horas/')
    assert json.loads(response.data)[0] == test_data_1
    assert len(json.loads(response.data)) == 24

def test_len_48horas(client):
    now_time = datetime.now()

    test_data_1 = {
        "date": now_time.strftime('%d-%m-%Y'),
        "hour": now_time.strftime("%H:00"),
        "hum": 29.720270734703718,
        "temp": 301.82215386331615
    }

    response = client.get('/servicio/v1/prediccion/48horas/')
    assert json.loads(response.data)[0] == test_data_1
    assert len(json.loads(response.data)) == 48

def test_len_72horas(client):
    now_time = datetime.now()

    test_data_1 = {
        "date": now_time.strftime('%d-%m-%Y'),
        "hour": now_time.strftime("%H:00"),
        "hum": 29.720270734703718,
        "temp": 301.82215386331615
    }

    response = client.get('/servicio/v1/prediccion/72horas/')
    assert json.loads(response.data)[0] == test_data_1
    assert len(json.loads(response.data)) == 72
