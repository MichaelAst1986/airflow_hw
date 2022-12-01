import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from modules.pipeline import pipeline
from modules.predict import predict

path = os.path.expanduser('/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)


args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='First_dag',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    make_pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag,
    )
    make_predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag,
    )

    make_pipeline >> make_predict

