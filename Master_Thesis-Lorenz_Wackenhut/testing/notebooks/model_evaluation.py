import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import pyspark.ml.feature as ML
import pyspark.sql.functions as F
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import *


class Loader():
    """
    Includes all methods to read or write from the ADLS storage
    """

    def __init__(self, spark_session, adls_account_name='REDACTED'):
        self.spark_session = spark_session
        self.spark_context = spark_session.sparkContext
        self.adls_account_name = adls_account_name

    def readDataframeFromAdls(self, table_path, container_name='data'):
        """
        Reads a dataframe in parquet format from ADLS

        Paramters
        -----
        table_path: str, required
            absolute path to table in ADLS container
        container_name: str, optional
            name of the container in ADLS

        Returns
        -----
        Spark dataframe
        """
        return (
            self
            .spark_session
            .read
            .parquet(
                f'abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{table_path}')
        )

    def readModelFromAdls(self, model_path, model_class, container_name='data'):
        """
        Reads a RandomForestClassificationModel from ADLS

        Paramters
        -----
        model_class; str, required
            class name of the model e.g. RandomForestClassificationModel
        model_path: str, required
            absolute path to model in ADLS container
        container_name: str, optional
            name of the container in ADLS

        Returns
        -----
        PipelineModel
        """
        model = globals()[model_class]
        return (
            model
            .load(
                f"abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{model_path}"
            )
        )


class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter,
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2,
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


def main():
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    account_name = "REDACTED"
    loader = Loader(spark, account_name)

    df_transform = loader.readDataframeFromAdls(
        '/transformation/df_transform_training')
    _, df_predict = df_transform.randomSplit([0.8, 0.2], seed=42)
    gbt_hyper = loader.readModelFromAdls(
        '/model/gbt_hyper', 'CrossValidatorModel')
    rfc_hyper = loader.readModelFromAdls(
        '/model/rfc_hyper', 'CrossValidatorModel')

    df_predicted_gbt = gbt_hyper.transform(df_predict)
    df_predicted_rfc = rfc_hyper.transform(df_predict)

    def getPerformanceMetrics(df):
        metrics_dictionary = {}
        evaluator_multi = MulticlassClassificationEvaluator(
            labelCol='Y', predictionCol="prediction")
        evaluator_binary = BinaryClassificationEvaluator(
            labelCol='Y', rawPredictionCol="prediction", metricName='areaUnderROC')

        # Get metrics
        metrics_dictionary['acc'] = evaluator_multi.evaluate(
            df, {evaluator_multi.metricName: "accuracy"})
        metrics_dictionary['f1'] = evaluator_multi.evaluate(
            df, {evaluator_multi.metricName: "f1"})
        metrics_dictionary['weighted_precision'] = evaluator_multi.evaluate(
            df, {evaluator_multi.metricName: "weightedPrecision"})
        metrics_dictionary['weighted_recall'] = evaluator_multi.evaluate(
            df, {evaluator_multi.metricName: "weightedRecall"})
        metrics_dictionary['auc'] = evaluator_binary.evaluate(df)

        print(f"Accuracy: {metrics_dictionary['acc']}")
        print(f"F1: {metrics_dictionary['f1']}")
        print(
            f"Weighted Precision: {metrics_dictionary['weighted_precision']}")
        print(f"Weighted Recall: {metrics_dictionary['weighted_recall']}")
        print(f"AUC: {metrics_dictionary['auc']}")

        return metrics_dictionary

    def printConfusionMatrix(df):
        y_true = df.select(['Y']).collect()
        y_pred = df.select(['prediction']).collect()

        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(
            f'True negatives:{tn}, false positives:{fp}, false negatives:{fn}, true positives:{tp}')

    def plotROC(df1, df2, name1, name2):

        df1_dict = {name1: [df1]}
        df2_dict = {name2: [df2]}
        df_dict_list = [df1_dict, df2_dict]

        def getValues(df):
            preds = df.select('Y', 'probability').rdd.map(
                lambda row: (float(row['probability'][1]), float(row['Y'])))
            points = CurveMetrics(preds).get_curve('roc')
            y_true = df.select(['Y']).collect()
            y_pred = df.select(['prediction']).collect()
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            return points, roc_auc

        plt.figure()
        for df_dict in df_dict_list:
            df_dict[list(df_dict)[0]].extend(
                getValues(df_dict[list(df_dict)[0]][0]))
            x_val = [x[0] for x in df_dict[list(df_dict)[0]][1]]
            y_val = [x[1] for x in df_dict[list(df_dict)[0]][1]]
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.plot(x_val, y_val, label=f'{list(df_dict)[0]} (area = %0.3f)' % df_dict[list(
                df_dict)[0]][2], alpha=0.7)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(
            '/home/lorenz.wackenhut/master_thesis/spark/notebooks/roc_curve.png', dpi=300)

    def ExtractFeatureImp(model, df, featuresCol='X'):
        actual_model = model.bestModel
        featureImp = actual_model.featureImportances
        list_extract = []
        for i in df.schema[featuresCol].metadata["ml_attr"]["attrs"]:
            list_extract = list_extract + \
                df.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
        varlist = pd.DataFrame(list_extract)
        varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
        return(varlist.sort_values('score', ascending=False))

    def getHyperparameters(model):
        print(model.getEstimatorParamMaps()[np.argmax(model.avgMetrics)])

    plotROC(df_predicted_rfc, df_predicted_gbt,
            'RandomForestClassifier', 'GradientBoostedTreesClassifier')

    print('Random Forest Classification')
    getPerformanceMetrics(df_predicted_rfc)
    printConfusionMatrix(df_predicted_rfc)
    print(ExtractFeatureImp(rfc_hyper, df_predicted_rfc).head(10))
    getHyperparameters(rfc_hyper)

    print('Gradient Boosted Trees Classification')
    getPerformanceMetrics(df_predicted_gbt)
    printConfusionMatrix(df_predicted_gbt)
    print(ExtractFeatureImp(gbt_hyper, df_predicted_gbt).head(10))
    getHyperparameters(gbt_hyper)

    sc.stop()


if __name__ == "__main__":
    main()
