from jobs.evaluation.evaluator import Evaluator
from jobs.training.trainer_gradient_boosted_trees_hyper import TrainerGradientBoostedTreesHyper
from jobs.training.trainer_random_forest_hyper import TrainerRandomForestHyper
from jobs.transformation.transformer_training import TransformerTraining
from jobs.transformation.transformer_prediction import TransformerPrediction
from jobs.prediction.predictor import Predictor
from jobs.extraction.extractor_prediction import ExtractorPrediction
from jobs.extraction.extractor_training import ExtractorTraining
from jobs.loading.loader import Loader
import pyspark
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
import sys
import glob
import yaml


def main():
    """
    Main method for cluster deployment with Airflow
    """
    def loadConfig(path_config):
        """
        Loads a YAML configuration file from a specified path

        Parameters
        -----
        path_config: str, required
            path to the file
        """
        with open(path_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    # initialize spark session
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    # initialize spark logger
    sc.setLogLevel("WARN")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.warn("pyspark logger initialized")

    # load yaml config file
    config_path = glob.glob(f"{pyspark.SparkFiles.get('')}/*.yml")[0]
    config = loadConfig(config_path)
    LOGGER.warn('yaml configuration loaded')

    # read adls account credentials from config
    adls_account_name = config['adls_account_name']
    adls_account_key = config['adls_account_key']

    # set credentials in order to access adls
    spark.conf.set("fs.azure.account.auth.type." +
                   adls_account_name + ".dfs.core.windows.net", "SharedKey")
    spark.conf.set("fs.azure.account.key." + adls_account_name +
                   ".dfs.core.windows.net", adls_account_key)

    # create loader object
    loader = Loader(spark, adls_account_name)
    LOGGER.warn('loader initialized')

    # iterate through pipeline stages
    stages = []
    for task_config in config['processing_tasks']:
        klass = globals()[task_config['classname']]
        task = klass(loader, **task_config['initialization_parameters'])
        stages.append(task)
        LOGGER.warn(
            f'Added task {task_config["classname"]} to pipeline stages')

    for i, task in enumerate(stages):
        LOGGER.warn(
            f'Running task {config["processing_tasks"][i]["classname"]}')
        task.runTransformation(
            **config['processing_tasks'][i]['transformation_parameters'])

    sc.stop()


if __name__ == "__main__":
    main()
