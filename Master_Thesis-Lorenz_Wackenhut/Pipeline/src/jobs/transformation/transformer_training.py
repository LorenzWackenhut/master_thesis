from pyspark.ml.pipeline import Pipeline
from jobs.transformation.categorical_filter import CategoricalFilter
from .transformer import TransformerAbstract
from .weekend_extractor import WeekendExtractor
from .time_extractor import TimeExtractor
from .level_calculator import LevelCalculator
from .categorical_filter import CategoricalFilter
from .numerical_imputer import NumericalImputer
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.ml.feature as ML


class TransformerTraining(TransformerAbstract):
    def __init__(self, loader, mode=None, df=None, container_name=None, table_path=None):
        """
        Inherits from Transformer
        """
        super(TransformerTraining, self).__init__(
            mode,
            df,
            loader,
            container_name,
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
        mode = kwargs['mode']
        path_df_transform = kwargs['path_df_transform']
        self.transform(mode=mode, path_df_transform=path_df_transform)

    def transform(self, mode='return', path_df_transform=None):
        """
        Transforms the dataframe by running the pipeline and either returns or saves to adls

        Parameters
        -----
        mode: str, optional
            Determines if the tables should be saved to ADLS or returned
            Possible values: ['return', 'save']
        path_df_transform: str, optional
            Path to the ADLS blob storage where the table should be saved to
        """
        super(TransformerTraining, self).transform(mode, path_df_transform)
        self._buildStages()
        transformation_pipeline = Pipeline(stages=self.stages)
        self.df_transform = (
            transformation_pipeline
            .fit(self.df)
            .transform(self.df)
        )

        if mode == "save":
            print('Saving df_transform_training')
            self.loader.writeDataframeToAdls(
                self.df_transform,
                path_df_transform,
                "df_transform_training",
                self.container_name
            )
        return self.df_transform

    def _buildStages(self):
        """
        Initiates the transformation classes and arranges them in a pipeline
        """
        weekend_extractor = WeekendExtractor(input_col='STIME_SEC')
        time_extractor = TimeExtractor(input_col='STIME_SEC')
        level_calculator = LevelCalculator(input_col='LEVEL')
        categorical_filter_activity = (
            CategoricalFilter(
                'ACTIVITY_STATE',
                self.activity_state_categorical
            )
        )
        categorical_filter_activity_lag = (
            CategoricalFilter(
                'LAG_ACTIVITY',
                self.activity_state_categorical
            )
        )
        categorical_filter_activity_lead = (
            CategoricalFilter(
                'LEAD_ACTIVITY',
                self.activity_state_categorical
            )
        )
        categorical_filter_batt_charge = (
            CategoricalFilter(
                'BATT_CHARGE',
                self.batt_charge_categorical
            )
        )
        categorical_filter_stype = (
            CategoricalFilter(
                'STYPE',
                self.stype_categorical
            )
        )
        categorical_filter_source = (
            CategoricalFilter(
                'SOURCE',
                self.source_categorical
            )
        )
        categorical_filter_type = (
            CategoricalFilter(
                'TYPE',
                self.type_categorical
            )
        )
        categorical_filter_time_day = (
            CategoricalFilter(
                'TIME_DAY',
                self.time_day_categorical
            )
        )
        median_imputer = (
            ML.Imputer(
                inputCols=self.median_imputer_input,
                outputCols=self.median_imputer_output
            )
            .setStrategy("median")
        )
        numerical_imputer = NumericalImputer(-1)
        string_indexer = (
            ML.StringIndexer(
                inputCols=self.string_indexer_input,
                outputCols=self.string_indexer_output
            )
        )
        one_hot_encoder = (
            ML.OneHotEncoder(
                inputCols=self.one_hot_encoder_input,
                outputCols=self.one_hot_encoder_output
            )
        )
        vector_assembler = (
            ML.VectorAssembler(
                inputCols=self.x_columns_training,
                outputCol='X'
            )
        )

        self.stages = [
            weekend_extractor,
            time_extractor,
            level_calculator,
            categorical_filter_activity,
            categorical_filter_activity_lag,
            categorical_filter_activity_lead,
            categorical_filter_batt_charge,
            categorical_filter_stype,
            categorical_filter_source,
            categorical_filter_type,
            categorical_filter_time_day,
            median_imputer,
            numerical_imputer,
            string_indexer,
            one_hot_encoder,
            vector_assembler
        ]
