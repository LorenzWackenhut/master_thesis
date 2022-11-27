from pyspark.ml.classification import RandomForestClassifier


class Predictor():
    def __init__(
            self,
            loader,
            mode_model=None,
            mode_df=None,
            model=None,
            df=None,
            container_name_model=None,
            container_name_df=None,
            model_path_dictionary=None,
            table_path=None):
        """
        Initiates a predictor object which sets the label column on an unseen dataframe

        Parameters
        -----
        loader: loader object, required
        mode_model: str, required
            possible inputs: ['pass', 'load']
            the model either gets passed as an object or gets loaded from the adls
        mode_df: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        model: Spark ML model, optional
            model can be passed
        df: dataframe, optional
            dataframe can be passed
        container_name_model: str, optional
        container_name_df: str, optional
        model_path_dictionary: dict, optional
            multiple paths to models can be statet in a dict
       table_path: str, optional
            format like '/directory/'       

        Returns
        -----
        Predictor object
        """
        if (mode_df not in ['pass', 'load']) or (mode_model not in ['pass', 'load']):
            raise ValueError(
                "mode has to be 'pass' or 'load'"
            )
        if (mode_df == 'pass'):
            if (df is None):
                raise ValueError(
                    "df cannot be none if mode='pass'"
                )
        elif (mode_df == 'load'):
            if (loader is None) or (container_name_df is None) or (table_path is None):
                raise ValueError(
                    "loader, container_name, path cannot be none if mode_df='load'"
                )
        if (mode_model == 'pass'):
            if (model is None):
                raise ValueError(
                    "model cannot be none if mode='pass'"
                )

        elif (mode_model == 'load'):
            if (loader is None) or (container_name_model is None) or (model_path_dictionary is None):
                raise ValueError(
                    "loader, container_name, path cannot be none if mode_model='load'"
                )

        self.model = model
        self.loader = loader
        self.df = df
        self.container_name_df = container_name_df
        self.container_name_model = container_name_model
        self.table_path = table_path
        self.model_path_dictionary = model_path_dictionary
        self.mode_df = mode_df
        self.mode_model = mode_model

    def runTransformation(self, **kwargs):
        """
        Executes the transformations 

        Parameters
        -----
        kwargs: dict, required
            dictionary loaded from yaml config 
        """
        mode = kwargs['mode']
        path_df_predict = kwargs['path_df_predict']
        self.predict(mode=mode, path_df_predict=path_df_predict)

    def predict(self, mode='return', path_df_predict=None):
        """
        Predicts the indoor/outdoor label on the dataframe

        Parameters
        -----
        mode: str, optional
            Determines if the dataframe should be saved to ADLS or returned
            Possible values: ['return', 'save']
        path_df_predict: str, optional
            format like '/directory/'

        Returns
        -----
        Spark dataframe
        """
        if mode not in ['save', 'return']:
            raise ValueError(
                "mode has to be 'save' or 'return'"
            )
        elif mode == "save":
            if path_df_predict is None:
                raise ValueError(
                    "path_df_predict cannot be none if mode='save'"
                )
        if self.mode_df == 'load':
            self.df = (
                self.loader
                .readDataframeFromAdls(
                    self.table_path,
                    self.container_name_df)
            )
        if self.mode_model == 'load':
            self.model = (
                self.loader
                .readModelFromAdls(
                    model_path=next(iter(self.model_path_dictionary.values())),
                    model_class=next(iter(self.model_path_dictionary.keys()))
                )
            )

        df_predict = self.model.transform(self.df)
        if mode == "save":
            print("Saving result to ADLS")
            self.loader.writeDataframeToAdls(
                df_predict,
                path_df_predict,
                "df_predict",
                self.container_name_model
            )
        return df_predict
