# ===============================================================
# Airflow DAG: Heart Clustering Pipeline
# Author: Lochan Enugula
# ===============================================================

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
import sys, os

# 👇 Add this block so Airflow can find your local src package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import functions from src/lab2.py
from src.lab2 import load_data, data_preprocessing, build_save_model, load_model_elbow

# Enable XCom pickling to allow byte data transfer
conf.set('core', 'enable_xcom_pickling', 'True')

# Default DAG arguments
default_args = {
    'owner': 'lochan_enugula',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 5),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='heart_clustering_dag',
    default_args=default_args,
    description='Heart Dataset KMeans Clustering DAG',
    schedule_interval=None,   # manual trigger only
    catchup=False,
    tags=['mlops', 'heart', 'clustering'],
) as dag:

    # --- Tasks ---
    load_heart_data = PythonOperator(
        task_id='load_heart_data',
        python_callable=load_data
    )

    preprocess_heart_data = PythonOperator(
        task_id='preprocess_heart_data',
        python_callable=data_preprocessing,
        op_args=[load_heart_data.output]
    )

    build_heart_model = PythonOperator(
        task_id='build_heart_model',
        python_callable=build_save_model,
        op_args=[preprocess_heart_data.output, "heart_model.sav"]
    )

    load_elbow_and_predict = PythonOperator(
        task_id='load_elbow_and_predict',
        python_callable=load_model_elbow,
        op_args=["heart_model.sav", build_heart_model.output]
    )

    # Define task dependencies
    load_heart_data >> preprocess_heart_data >> build_heart_model >> load_elbow_and_predict


if __name__ == "__main__":
    dag.cli()
