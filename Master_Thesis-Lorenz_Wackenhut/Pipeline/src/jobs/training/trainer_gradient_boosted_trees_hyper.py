from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np

from .trainer import Trainer


class TrainerGradientBoostedTreesHyper(Trainer):
    def __init__(self, loader, mode_df=None, df=None, container_name_df=None, table_path=None):
        """
        Initiates the training process for a gradient boosted trees classifier

        Parameters
        -----
        loader: loader object, required
        mode_df: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        df: dataframe, optional
        container_name_df: str, optional
        table_path: str, optional
            format like '/directory/'

        Returns
        -----
        Trainer object
        """
        super(TrainerGradientBoostedTreesHyper, self).__init__(
            loader,
            mode_df,
            df,
            container_name_df,
            table_path
        )

    def runTransformation(self, **kwargs):
        """
        Executes the transformations 

        Parameters
        -----
        kwargs: dict, required
            dictionary loaded from yaml config 
        """
        mode_model = kwargs['mode_model']
        path_model = kwargs['path_model']
        hyper_tuning = kwargs['hyper_tuning']
        self.train(mode_model=mode_model, path_model=path_model,
                   hyper_tuning=hyper_tuning)

    def train(self, label_col='Y', feature_col='X', mode_model='return', path_model=None, hyper_tuning=False):
        """
        Trains the specified ML model and performs hyperparameter tuning

        Parameters
        -----
        label_col: str, required
            name of the label column
        feature_col: str, required
            name of the feature column
        mode_model: str, optional
            Determines if the tables should be saved to ADLS or returned
            Possible values: ['return', 'save']
        path_model: str, optional
            format like '/directory/'

        Returns
        -----
        ML model
        """
        super(TrainerGradientBoostedTreesHyper, self).train(
            label_col, feature_col, mode_model, path_model, hyper_tuning)

        model = (
            GBTClassifier(
                labelCol=label_col,
                featuresCol=feature_col
            )
        )

        # perform hyperparameter tuning if stated in yaml config
        if hyper_tuning:
            model_tuning = self.hyperparamterTuning(model, self.df_train)
        else:
            model_tuning = model.fit(self.df_train)

        if mode_model == 'save':
            self.loader.writeModelToAdls(model_tuning, path_model)
        else:
            return model_tuning

    def hyperparamterTuning(self, model, df_train):
        # define grid for parameter search
        param_grid = (
            ParamGridBuilder()
            .addGrid(model.maxDepth, [1, 3, 5])
            .addGrid(model.stepSize, [0.05, 0.1, 0.2])
            .addGrid(model.minInstancesPerNode, [1, 2, 3])
            .build()
        )

        evaluator = BinaryClassificationEvaluator(
            labelCol='Y',
            rawPredictionCol="prediction",
            metricName='areaUnderROC'
        )

        cross_val = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=10
        )

        return cross_val.fit(df_train)
