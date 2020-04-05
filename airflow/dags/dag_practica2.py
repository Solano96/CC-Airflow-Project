from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import utils

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
    'practica2_v1',
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
    python_callable=utils.merge_humidity_temperature,
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
    python_callable=utils.insert_data_in_collection,
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
    python_callable=utils.train_arima,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'humidity'
    },
    dag=dag,
)

# Train arima model with temperature data
train_arima_temperature = PythonOperator(
    task_id='train_arima_temperature',
    provide_context=True,
    python_callable=utils.train_arima,
    op_kwargs={
        'host': 'localhost',
        'port': 27018,
        'db_name': 'airflow_db',
        'collection_name': 'SanFrancisco',
        'column_name': 'temperature'
    },
    dag=dag,
)

download_v1 = BashOperator(
    task_id='download_v1',
    bash_command='''curl -o {temp_dir}/v1/Dockerfile https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/Dockerfile;
                    curl -o {temp_dir}/v1/requirements.txt https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/requirements.txt;
                    curl -o {temp_dir}/v1/microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/microservicio.py;
                    curl -o {temp_dir}/v1/test_microservicio.py https://raw.githubusercontent.com/Solano96/CC-Airflow-Project/master/v1/test_microservicio.py''',
    dag=dag,
)

run_test_v1 = BashOperator(
    task_id='run_test_v1',
    bash_command='python3 -m pytest /tmp/airflow_data/v1',
    dag=dag,
)

create_image_v1 = BashOperator(
    task_id='create_image_v1',
    bash_command='docker build /tmp/airflow_data/v1 -t microservicio_v1',
    dag=dag,
)


deploy_microservice_v1 = BashOperator(
    task_id='deploy_microservice_v1',
    bash_command='docker run --detach -p 80:5000 microservicio_v1',
    dag=dag,
)

create_temp_dir >> [download_humidity, download_temperature]
download_humidity >> unzip_humidity
download_temperature >> unzip_temperature
[unzip_humidity, unzip_temperature] >> merge_data >> deploy_docker >> insert_data
insert_data >> [train_arima_humidity, train_arima_temperature] >> run_test_v1

create_temp_dir >> download_v1 >> run_test_v1

run_test_v1 >> create_image_v1 >> deploy_microservice_v1
