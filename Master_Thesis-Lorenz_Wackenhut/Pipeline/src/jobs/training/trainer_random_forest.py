from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from .trainer import Trainer


class TrainerRandomForest(Trainer):
    def __init__(self, loader, mode_df=None, df=None, container_name_df=None, table_path=None):
        """
        Initiates a trainer object

        Parameters
        -----
        loader: Loader object, required
        mode_df: str, required
            possible inputs: ['pass', 'load']
        df: Spark dataframe, required
        container_name_df: str, required
        table_path: str, required
            format like '/directory/'

        Returns
        -----
        Trainer object

        """
        super(TrainerRandomForest, self).__init__(
            loader,
            mode_df,
            df,
            container_name_df,
            table_path
        )

    def runTransformation(self, **kwargs):
        '''
        Calls the transformation method with a dictionairy of parameters
        '''
        mode_model = kwargs['mode_model']
        path_model = kwargs['path_model']
        self.train(mode_model=mode_model, path_model=path_model)

    def train(self, label_col='Y', feature_col='X', mode_model='return', path_model=None):
        """
        Trains an ML model according to the feature and label columns passed

        Parameters
        -----
        label_col: str, optional
            Name of the label column
        feature_col: str, optional
            Name of the feature vector column
        mode_model: str, optional
            Determines if the model should be saved to ADLS or returned
            Possible values: ['return', 'save']
        path_model: str, required
            format like '/directory/'
        hyper_tuning: boolean, required
            determines if hyperparameter tuning should be performed
        """
        super(TrainerRandomForest, self).train(
            label_col, feature_col, mode_model, path_model)

        model = (
            RandomForestClassifier(
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
