from airflow import DAG
from datetime import datetime, timedelta
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow import configuration as conf
from airflow.utils.dates import days_ago
from airflow.models import Variable

default_args = {
    'owner': 'lorenz',
    'start_date': days_ago(5),
    'email': ['lorenz.wackenhut@netcheck.de'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
}

default_conf_spark = {
    'spark.kubernetes.authenticate.driver.serviceAccountName': 'spark',
    'spark.executor.cores': 3,
    'spark.executor.instances': 4,
    'spark.executor.memory': '8g',
    'spark.driver.memory': '8g',
    'spark.driver.cores': 3,
    'spark.kubernetes.container.image': 'REDACTED.azurecr.io/spark-python:ivy2',
    'spark.hadoop.fs.azure.account.auth.type.REDACTED.dfs.core.windows.net': 'SharedKey',
    'spark.hadoop.fs.azure.account.key.REDACTED.dfs.core.windows.net': 'REDACTED'
}

spark_home = Variable.get("SPARK_HOME")

dag = DAG('prediction_pipeline_k8',
          schedule_interval='@once',
          default_args=default_args)

with dag:
    ingestion_prediction = SparkSubmitOperator(
        task_id='ingestion_prediction',
        name='ingestion_prediction',
        conn_id='spark_k8s',
        proxy_user='root',
        application='abfss://py-files@REDACTED.dfs.core.windows.net/main_kubernetes.py',
        files='abfss://py-files@REDACTED.dfs.core.windows.net/configuration/config_extraction_prediction.yml',
        py_files='abfss://py-files@REDACTED.dfs.core.windows.net/jobs.zip',
        conf=default_conf_spark,
        execution_timeout=timedelta(minutes=120),
        dag=dag)

    processing_prediction = SparkSubmitOperator(
        task_id='processing_prediction',
        name='processing_prediction',
        conn_id='spark_k8s',
        proxy_user='root',
        application='abfss://py-files@REDACTED.dfs.core.windows.net/main_kubernetes.py',
        files='abfss://py-files@REDACTED.dfs.core.windows.net/configuration/config_processing_prediction.yml',
        py_files='abfss://py-files@REDACTED.dfs.core.windows.net/jobs.zip',
        conf=default_conf_spark,
        execution_timeout=timedelta(minutes=120),
        dag=dag)

    prediction = SparkSubmitOperator(
        task_id='prediction',
        name='prediction',
        conn_id='spark_k8s',
        proxy_user='root',
        application='abfss://py-files@REDACTED.dfs.core.windows.net/main_kubernetes.py',
        files='abfss://py-files@REDACTED.dfs.core.windows.net/configuration/config_prediction.yml',
        py_files='abfss://py-files@REDACTED.dfs.core.windows.net/jobs.zip',
        conf=default_conf_spark,
        execution_timeout=timedelta(minutes=120),
        dag=dag)

ingestion_prediction >> processing_prediction >> prediction
