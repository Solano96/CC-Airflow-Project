from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import pandas as pd
import json
import pymongo

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pmdarima as pm
import pickle

def merge_humidity_temperature(**kwargs):
    # Get arguments
    directory_name = kwargs['directory_name']
    humidity_file = kwargs['humidity_file']
    temperature_file = kwargs['temperature_file']
    output_file = kwargs['output_file']
    join_by = kwargs['join_by']
    selected_column = kwargs['selected_column']

    # Read data
    df_humidity = pd.read_csv(f'{directory_name}/{humidity_file}')
    df_temperature = pd.read_csv(f'{directory_name}/{temperature_file}')

    # Select 'San Francisco' column
    df_humidity = df_humidity[[join_by, selected_column]]
    df_temperature = df_temperature[[join_by, selected_column]]

    # Rename columns
    df_humidity = df_humidity.rename(columns={selected_column: "humidity"}, errors="raise")
    df_temperature = df_temperature.rename(columns={selected_column: "temperature"}, errors="raise")

    # Merge data
    merge_data = pd.merge(df_humidity, df_temperature, on=join_by, how='inner')
    # Remove nan values
    merge_data = merge_data.dropna()
    # Save merge data in csv format
    merge_data.to_csv(f'{directory_name}/{output_file}', index = False, header = True)


def get_mongo_collection(host, port, db_name, collection_name):
    client = pymongo.MongoClient(host=host, port=port)
    db = client[db_name]
    return db[collection_name]


def insert_data_in_collection(**kwargs):
    # Get arguments
    host = kwargs['host']
    port = kwargs['port']
    db_name = kwargs['db_name']
    collection_name = kwargs['collection_name']
    path_file = kwargs['path_file']

    # Connect mongo client and get collection
    collection = get_mongo_collection(host, port, db_name, collection_name)

    # Read csv
    df = pd.read_csv('/tmp/airflow_data/merge.csv')

    # Insert data in collection
    collection.insert_many(df.to_dict('records'))


def train_model(**kwargs):
    # Get arguments
    host = kwargs['host']
    port = kwargs['port']
    db_name = kwargs['db_name']
    collection_name = kwargs['collection_name']
    column_name = kwargs['column_name']
    model_name = kwargs['model_name']

    # Connect mongo client and get collection
    collection = get_mongo_collection(host, port, db_name, collection_name)

    df = pd.DataFrame(list(collection.find({})))
    df = df.dropna()
    model = None

    if model_name == 'SimpleExpSmoothing':
        model = SimpleExpSmoothing(df[column_name]).fit()

        with open(f'/tmp/airflow_data/v2/exp_smoothing_{column_name}.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)

    elif model_name == 'arima':
        model = pm.auto_arima(df[column_name], start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=3, max_q=3, # maximum p and q
                              m=1,              # frequency of series
                              d=None,           # let model determine 'd'
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        with open(f'/tmp/airflow_data/v1/arima_{column_name}.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)

    return model

temp_dir = '/tmp/airflow_data'
humidity_file = 'humidity.csv.zip'
temperature_file = 'temperature.csv.zip'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

#Inicialización del grafo DAG de tareas para el flujo de trabajo
dag = DAG(
    'practica2',
    default_args=default_args,
    description='Un grafo simple de tareas',
    schedule_interval=timedelta(days=1),
)

# ------------------------ Operators or Tasks ------------------------ #

# Create temp dir
create_temp_dir = BashOperator(
    task_id='create_temp_dir',
    bash_command=f'mkdir -p {temp_dir}; mkdir -p {temp_dir}/v1; mkdir -p {temp_dir}/v2;',
    dag=dag,
)

# Download humidity.csv.zip
download_humidity = BashOperator(
    task_id='download_humidity',
    bash_command=f'curl -o {temp_dir}/{humidity_file} https://raw.githubusercontent.com/manuparra/MaterialCC2020/master/{humidity_file}',
    dag=dag,
)

# Download temperature.csv
download_temperature = BashOperator(
    task_id='download_temperature',
    bash_command=f'curl -o {temp_dir}/{temperature_file} https://raw.githubusercontent.com/manuparra/MaterialCC2020/master/{temperature_file}',
    dag=dag,
)

# Unzip humidity.csv.zip
unzip_humidity = BashOperator(
    task_id='unzip_humidity',
    bash_command=f'unzip -od {temp_dir} {temp_dir}/{humidity_file}',
    dag=dag,
)

# Unzip temperature.csv.zip
unzip_temperature = BashOperator(
    task_id='unzip_temperature',
    bash_command=f'unzip -od {temp_dir} {temp_dir}/{temperature_file}',
    dag=dag,
)

# Merge merge_data
merge_data = PythonOperator(
    task_id='merge_data',
    provide_context=True,
    python_callable=merge_humidity_temperature,
    op_kwargs={
        'directory_name': temp_dir,
        'humidity_file': 'humidity.csv',
        'temperature_file': 'temperature.csv',
        'output_file': 'merge.csv',
        'join_by': 'datetime',
        'selected_column': 'San Francisco',
    },
    dag=dag,
)

# Deploy Docker container
deploy_docker = BashOperator(
    task_id='deploy_docker',
    bash_command=f'docker run -d -p 27018:27017 --name mongodb mongo:latest',
    dag=dag,
)

# Insert data
insert_data = PythonOperator(
    task_id='insert_data',
    provide_context=True,
    python_callable=insert_data_in_collection,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'path_file': '/tmp/airflow_data/merge.csv',
    },
    dag=dag,
)

# Train arima model with humidity data
train_arima_humidity = PythonOperator(
    task_id='train_arima_humidity',
    provide_context=True,
    python_callable=train_model,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'humidity',
        'model_name': 'arima'
    },
    dag=dag,
)

# Train arima model with temperature data
train_arima_temperature = PythonOperator(
    task_id='train_arima_temperature',
    provide_context=True,
    python_callable=train_model,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'temperature',
        'model_name': 'arima'
    },
    dag=dag,
)

# Train SimpleExpSmoothing model with humidity data
train_exp_smoothing_humidity = PythonOperator(
    task_id='train_exp_smoothing_humidity',
    provide_context=True,
    python_callable=train_model,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'humidity',
        'model_name': 'SimpleExpSmoothing'
    },
    dag=dag,
)

# Train SimpleExpSmoothing model with temperature data
train_exp_smoothing_temperature = PythonOperator(
    task_id='train_exp_smoothing_temperature',
    provide_context=True,
    python_callable=train_model,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'temperature',
        'model_name': 'SimpleExpSmoothing'
    },
    dag=dag,
)

download_v1 = BashOperator(
    task_id='download_v1',
    bash_command=f'''curl -o {temp_dir}/v1/Dockerfile https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/Dockerfile;
                    curl -o {temp_dir}/v1/requirements.txt https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/requirements.txt;
                    curl -o {temp_dir}/v1/microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/microservicio.py;
                    curl -o {temp_dir}/v1/test_microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/test_microservicio.py;''',
    dag=dag,
)

download_v2 = BashOperator(
    task_id='download_v2',
    bash_command=f'''curl -o {temp_dir}/v2/Dockerfile https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v2/Dockerfile;
                    curl -o {temp_dir}/v2/requirements.txt https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v2/requirements.txt;
                    curl -o {temp_dir}/v2/microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v2/microservicio.py;
                    curl -o {temp_dir}/v2/test_microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v2/test_microservicio.py;''',
    dag=dag,
)

run_test_v1 = BashOperator(
    task_id='run_test_v1',
    bash_command='python3 -m pytest /tmp/airflow_data/v1',
    dag=dag,
)

run_test_v2 = BashOperator(
    task_id='run_test_v2',
    bash_command='python3 -m pytest /tmp/airflow_data/v2',
    dag=dag,
)

create_image_v1 = BashOperator(
    task_id='create_image_v1',
    bash_command='docker build /tmp/airflow_data/v1 -t microservicio_v1',
    dag=dag,
)

create_image_v2 = BashOperator(
    task_id='create_image_v2',
    bash_command='docker build /tmp/airflow_data/v2 -t microservicio_v2',
    dag=dag,
)

deploy_microservice_v1 = BashOperator(
    task_id='deploy_microservice_v1',
    bash_command='docker run --detach -p 80:5000 microservicio_v1',
    dag=dag,
)

deploy_microservice_v2 = BashOperator(
    task_id='deploy_microservice_v2',
    bash_command='docker run --detach -p 81:5000 microservicio_v2',
    dag=dag,
)

create_temp_dir >> [download_humidity, download_temperature]

download_humidity >> unzip_humidity
download_temperature >> unzip_temperature

[unzip_humidity, unzip_temperature] >> merge_data >> deploy_docker >> insert_data

insert_data >> [train_arima_humidity, train_arima_temperature] >> run_test_v1
insert_data >> [train_exp_smoothing_humidity, train_exp_smoothing_temperature] >> run_test_v2

create_temp_dir >> [download_v1, download_v2]
download_v1 >> run_test_v1
download_v2 >> run_test_v2

run_test_v1 >> create_image_v1 >> deploy_microservice_v1
run_test_v2 >> create_image_v2 >> deploy_microservice_v2
