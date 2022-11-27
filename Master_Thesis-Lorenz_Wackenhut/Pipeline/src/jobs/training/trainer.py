from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
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
        if mode_df not in ['pass', 'load']:
            raise ValueError(
                "mode has to be 'pass' or 'load'"
            )
        if mode_df == 'pass':
            if df is None:
                raise ValueError(
                    "df cannot be none if mode='pass'"
                )
        elif mode_df == 'load':
            if (loader is None) or (container_name_df is None) or (table_path is None):
                raise ValueError(
                    "loader, container_name_df, table_path cannot be none if mode='load'"
                )

        self.loader = loader
        self.df = df
        self.container_name_df = container_name_df
        self.table_path = table_path
        self.mode_df = mode_df

    @abstractmethod
    def runTransformation(self, **kwargs):
        '''
        Calls the transformation method with a dictionairy of parameters
        '''

    @abstractmethod
    def train(self, label_col='Y', feature_col='X', mode_model='return', path_model=None, hyper_tuning=None):
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
        if mode_model not in ['save', 'return']:
            raise ValueError(
                "mode has to be 'save' or 'return'"
            )
        elif mode_model == "save":
            if path_model is None:
                raise ValueError(
                    "path_model cannot be none if mode='save'"
                )
        if self.mode_df == 'load':
            self.df = (
                self.loader
                .readDataframeFromAdls(
                    self.table_path,
                    self.container_name_df)
            )

        self.df_train, self.df_test = self.df.randomSplit([0.8, 0.2], seed=42)

    @abstractmethod
    def hyperparamterTuning(self, model, df_train):
        """
        Performs hyperparameter tuning with model and dataframe

        Parameters
        -----
        model: Spark ML object, required
        df_train: Spark dataframe, required
        """
