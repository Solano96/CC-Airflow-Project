import pandas as pd
import json
import pymongo

from statsmodels.tsa.arima_model import ARIMA
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


def train_arima(**kwargs):
    # Get arguments
    host = kwargs['host']
    port = kwargs['port']
    db_name = kwargs['db_name']
    collection_name = kwargs['collection_name']
    column_name = kwargs['column_name']

    # Connect mongo client and get collection
    collection = get_mongo_collection(host, port, db_name, collection_name)

    df = pd.DataFrame(list(collection.find({})))
    df = df.dropna()

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
