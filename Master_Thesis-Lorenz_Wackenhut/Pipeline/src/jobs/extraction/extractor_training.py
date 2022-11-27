from .extractor import Extractor
from pyspark.sql.types import *
import pyspark.sql.functions as F


class ExtractorTraining(Extractor):
    def __init__(self, loader, container_name, path_raw_tables, schema_path):
        """
        Inherits from Extractor
        """
        super(ExtractorTraining, self).__init__(
            loader,
            container_name,
            path_raw_tables,
            schema_path
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
        path_df_extract = kwargs['path_df_extract']
        self.extract(mode=mode, path_df_extract=path_df_extract)

    def extract(self, mode='return', path_df_extract=None):
        """
        Extracts the tables and combines them in one dataframe

        Parameters
        -----
        mode: str, optional
            Determines if the tables should be saved to ADLS or returned
            Possible values: return, save
        path_df_extract: str, optional
            Path to the ADLS blob storage where the table should be saved to
        """
        super(ExtractorTraining, self).extract(mode, path_df_extract)
        self._getTables()
        self._combineTables()
        # increase dataset to same size as inference set
        # self._increaseDataset(k=69)
        # already set the label column to 0 or 1 as it would be difficult later on
        self.df_extract = (
            self.df_extract
            .withColumn(
                'Y',
                F.when(
                    F.col('ENV_LABEL').isin(['METRO', 'INDOOR', 'i']), 0
                )
                .otherwise(1)
            )
            .withColumn(
                'WIFI_CONNECTED_INT',
                F.col('WIFI_CONNECTED').cast('int')
            )
        )
        self._validateSchema()
        if mode == "save":
            self.loader.writeDataframeToAdls(
                self.df_extract,
                path_df_extract,
                "df_extract_training",
                self.container_name
            )
        return self.df_extract

    def _getTables(self):
        """
        Loads the tables from ADLS so the labels are balanced between indoor and outdoor
        Compare with legacy pipeline for further information
        """
        self.df_env_indoor = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_env_indoor')
            .sample(0.633)
        )

        self.df_env_outdoor = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_env_outdoor')
        )

        self.df_geo_labeled_io = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_geo_labeled')
            .filter(F.col('ENV_LABEL').isin(['i', 'o']))
        )

        self.df_geo_labeled_v = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_geo_labeled')
            .filter(F.col('ENV_LABEL') == 'v')
            .sample(0.009)
        )

        self.df_man_labeled_i = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_man_labeled')
            .filter(F.col('ENV_LABEL').isin(['METRO', 'INDOOR']))
            .sample(0.093)
        )

        self.df_man_labeled_o = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_man_labeled')
            .filter(~F.col('ENV_LABEL').isin(['METRO', 'INDOOR']))
        )

    def _combineTables(self):
        """
        Creates a union between the previously loaded tables 
        """
        self.df_extract = (
            self.df_env_indoor
            .union(self.df_env_outdoor)
            .union(self.df_geo_labeled_io)
            .union(self.df_geo_labeled_v)
            .union(self.df_man_labeled_i)
            .union(self.df_man_labeled_o)
            .withColumn('STIME', F.to_timestamp(F.col('STIME')))
            .withColumn('STIME_SEC', F.col('STIME').cast('long'))
        )

    def _validateSchema(self):
        """
        Validates the schema of the unified dataframe
        """
        #self.loader.writeSchemaToAdls(self.df_extract, self.schema_path)
        return super()._validateSchema()

    def _increaseDataset(self, k=10):
        """
        Unions a dataframe with itself k times in order to increase the dataset for testing purposes
        Parameters
        -----
        k: int, optional
            Factor by which to increase the dataset
        """
        df_union = self.df_extract
        for _ in range(k):
            self.df_extract = self.df_extract.union(df_union)
