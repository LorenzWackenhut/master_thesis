from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from .trainer import Trainer


class TrainerGradientBoostedTrees(Trainer):
    def __init__(self, loader, mode_df=None, df=None, container_name_df=None, table_path=None):
        """
        Initiates the training process for a gradient boosted trees

        Parameters
        -----
        loader: loader object, required
        mode_df: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        df: dataframe, optional
        container_name_df: str, required
        table_path: str, required
            format like '/directory/'

        Returns
        -----
        Trainer object

        """
        super(TrainerGradientBoostedTrees, self).__init__(
            loader,
            mode_df,
            df,
            container_name_df,
            table_path
        )

    def runTransformation(self, **kwargs):
        mode_model = kwargs['mode_model']
        path_model = kwargs['path_model']
        #label_col = kwargs['label_col']
        #feature_col = kwargs['feature_col']
        #label_col=label_col, feature_col=feature_col,
        self.train(mode_model=mode_model, path_model=path_model)

    def train(self, label_col='Y', feature_col='X', mode_model='return', path_model=None):
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
        Spark ML model
        """
        super(TrainerGradientBoostedTrees, self).train(
            label_col, feature_col, mode_model, path_model)

        model = (
            GBTClassifier(
                labelCol=label_col,
                featuresCol=feature_col
            )
        )

        model_tuning = self.hyperparamterTuning(model, self.df_train)

        if mode_model == 'save':
            self.loader.writeModelToAdls(model_tuning, path_model)
        else:
            return model_tuning

    def hyperparamterTuning(self, model, df_train):
        """
        Performs hyperparameter tuning with model and dataframe

        Parameters
        -----
        model: Spark ML object, required
        df_train: Spark dataframe, required
        """
        return model.fit(df_train)
