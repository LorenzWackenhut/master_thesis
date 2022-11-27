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
    Main method for consecutiv run without Airflow
    For testing purposes
    """
    def loadConfig(path_config):
        with open(path_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    config_path = glob.glob(f"{pyspark.SparkFiles.get('')}/*.yml")[0]
    config = loadConfig(config_path)

    adls_account_name = config['adls_account_name']
    adls_account_key = config['adls_account_key']

    print(adls_account_name)
    print(adls_account_key)

    spark.conf.set("fs.azure.account.auth.type." +
                   adls_account_name + ".dfs.core.windows.net", "SharedKey")
    spark.conf.set("fs.azure.account.key." + adls_account_name +
                   ".dfs.core.windows.net", adls_account_key)

    loader = Loader(spark, adls_account_name)

    extractor = ExtractorPrediction(
        loader, container_name='data', path_raw_tables='/masterdata/', schema_path='/schema/schema_prediction.pickle')
    df_extract = extractor.extract()

    transformer = TransformerPrediction(loader=loader,
                                        mode='pass', df=df_extract)
    df_transform = transformer.transform(mode='return')

    models_dictionairy = {
        'GBTClassificationModel': '/model/final/GBTClassificationModel'}
    predictor = Predictor(loader=loader, mode_model='load', container_name_model='data',
                          model_path_dictionary=models_dictionairy, mode_df='pass', df=df_transform)
    predictor.predict(mode='save', path_df_predict='/final/')

    sc.stop()


if __name__ == "__main__":
    main()
