from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from ..prediction.predictor import Predictor


class Evaluator():
    def __init__(
            self,
            loader,
            mode_df=None,
            mode_models=None,
            models_dictionary=None,
            df=None,
            container_name=None,
            table_path=None,
            models_path_dictionary=None):
        """
        CHANGE!
        Initiates an evaluator object

        Parameters
        -----
        loader: Loader object, required
        mode_df: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        mode_models: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        models_dictionary: dict, optional
            dict with name and models
        df: dataframe, optional    
        container_name: str, required
        table_path: str, required
            format like '/directory/'
        models_path_dictionary: dict, optional
            dict with name and paths to models

        Returns
        -----
        Evaluator object

        """
        if mode_df not in ['pass', 'load'] or mode_models not in ['pass', 'load']:
            raise ValueError(
                "mode has to be 'pass' or 'load'"
            )
        if mode_df == 'pass':
            if df is None:
                raise ValueError(
                    "df cannot be none if mode='pass'"
                )
        if mode_models == 'pass':
            if models_dictionary is None:
                raise ValueError(
                    "models_dictionary cannot be none if mode='pass'"
                )
        elif mode_df == 'load':
            if (loader is None) or (container_name is None) or (table_path is None):
                raise ValueError(
                    "loader, container_name, table_path cannot be none if mode='load'"
                )
        elif mode_models == 'load':
            if (loader is None) or (container_name is None) or (models_path_dictionary is None):
                raise ValueError(
                    "loader, container_name, models_path_dictionary cannot be none if mode='load'"
                )

        if models_dictionary is None:
            self.models_dictionary = {}
        else:
            self.models_dictionary = models_dictionary
        self.loader = loader
        self.df = df
        self.container_name = container_name
        self.table_path = table_path
        self.models_path_dictionary = models_path_dictionary
        self.mode_df = mode_df
        self.mode_models = mode_models

    def runTransformation(self, **kwargs):
        """
        Executes the transformations 

        Parameters
        -----
        kwargs: dict, required
            dictionary loaded from yaml config 
        """
        mode = kwargs['mode']
        label_col = kwargs['label_col']
        prediction_col = kwargs['prediction_col']
        self.evaluate(label_col=label_col, prediction_col=prediction_col,
                      mode=mode)

    def evaluate(self, label_col='Y', prediction_col='X', mode='return', path_final_model="/model/final/"):
        """
        Evaluates the models and deployes the best one

        Parameters
        -----
        label_col: str, optional
            Name of the label column
        prediction_col: str, optional
            Name of the predicted column
        mode: str, optional
            Determines if the model should be saved to ADLS or returned
            Possible values: ['return', 'save']
        path_final_model: str, required
            format like '/directory/'
        """
        if mode not in ['save', 'return']:
            raise ValueError(
                "mode has to be 'save' or 'return'"
            )

        if self.mode_df == 'load':
            self.df = self.loader.readDataframeFromAdls(self.table_path)
        _, self.df_test = self.df.randomSplit([0.8, 0.2], seed=42)

        if self.mode_models == 'load':
            for name_model, path_model in self.models_path_dictionary.items():
                self.models_dictionary[name_model] = self.loader.readModelFromAdls(
                    model_path=path_model, model_class=name_model)

        models_metrics_dictionary = {}
        for name, model in self.models_dictionary.items():
            predictor = Predictor(
                self.loader, mode_model='pass', mode_df='pass', model=model, df=self.df_test)
            df_prediction = predictor.predict(mode='return')
            models_metrics_dictionary[name] = self._calculate_metrics(
                df_prediction)

        return(self._save_best_model(models_metrics_dictionary, path_final_model=path_final_model))

    def _calculate_metrics(self, df_prediction):
        """
        Calculates a set of metrics for the models

        Parameters
        -----
        df_prediction: Spark dataframe, required
        """
        metrics_dictionary = {}
        evaluator_multi = MulticlassClassificationEvaluator(
            labelCol='Y', predictionCol="prediction")
        evaluator_binary = BinaryClassificationEvaluator(
            labelCol='Y', rawPredictionCol="prediction", metricName='areaUnderROC')

        # Get metrics
        metrics_dictionary['acc'] = evaluator_multi.evaluate(
            df_prediction, {evaluator_multi.metricName: "accuracy"})
        metrics_dictionary['f1'] = evaluator_multi.evaluate(
            df_prediction, {evaluator_multi.metricName: "f1"})
        metrics_dictionary['weighted_precision'] = evaluator_multi.evaluate(
            df_prediction, {evaluator_multi.metricName: "weightedPrecision"})
        metrics_dictionary['weighted_recall'] = evaluator_multi.evaluate(
            df_prediction, {evaluator_multi.metricName: "weightedRecall"})
        metrics_dictionary['auc'] = evaluator_binary.evaluate(df_prediction)

        print(f"Accuracy: {metrics_dictionary['acc']}")
        print(f"F1: {metrics_dictionary['f1']}")
        print(
            f"Weighted Precision: {metrics_dictionary['weighted_precision']}")
        print(f"Weighted Recall: {metrics_dictionary['weighted_recall']}")
        print(f"AUC: {metrics_dictionary['auc']}")

        return metrics_dictionary

    def _save_best_model(self, models_metrics_dictionary, path_final_model, metric='auc'):
        """
        Deploys the best model

        Parameters
        -----
        models_metrics_dictionary: dict, required
            dict with name and metrics of models
        path_model: str, required
            format like '/directory/'
        metric: str, optional
            metric by which the best model will be decided
        """
        if metric not in ['acc', 'f1', 'weighted_precision', 'auc']:
            raise ValueError(
                "metric has to be in ['acc', 'f1', 'weighted_precision', 'auc']"
            )

        def sort_metric_dictionary(models_metrics_dictionary, metric):
            """
            Sorts the models_metrics_dictionary

            Parameters
            -----
            models_metrics_dictionary: dict, required
                dict with name and metrics of models
            metric: str, optional
                metric by which the best model will be decided
            """
            models_metrics_dictionary_sorted = (
                {k: v for k, v in
                 sorted(
                     [(name, metric_dict[metric])
                      for name, metric_dict in models_metrics_dictionary.items()],
                     key=lambda item: item[1],
                     reverse=True
                 )
                 }
            )
            return models_metrics_dictionary_sorted

        # get first key of the dictionairy as this is the model with the highest metric
        best_model_name = next(iter(sort_metric_dictionary(
            models_metrics_dictionary, metric)))
        best_model = self.models_dictionary[best_model_name]

        self.loader.writeModelToAdls(
            model=best_model, model_path=path_final_model+best_model_name)

        return best_model
