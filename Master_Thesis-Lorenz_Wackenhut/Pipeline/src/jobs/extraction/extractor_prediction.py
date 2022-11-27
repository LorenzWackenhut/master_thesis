from .extractor import Extractor
from pyspark.ml.pipeline import Transformer
from pyspark.sql.types import *
import pyspark.sql.functions as F


class ExtractorPrediction(Extractor):
    def __init__(self, loader, container_name, path_raw_tables, schema_path):
        """
        Inherits from Extractor
        """
        super(ExtractorPrediction, self).__init__(
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
        super(ExtractorPrediction, self).extract(mode, path_df_extract)
        self._getTables()
        self._setJoinConditions()
        self._combineTables()
        self.df_extract = (
            self.df_extract
            .withColumn(
                'WIFI_CONNECTED_INT',
                F.when(
                    F.col('WIFI_CONNECTED').isNotNull(), 1
                )
                .otherwise(0)
            )
        )
        self._validateSchema()
        if mode == "save":
            self.loader.writeDataframeToAdls(
                self.df_extract,
                path_df_extract,
                "df_extract_prediction",
                self.container_name
            )
        return self.df_extract

    def _getTables(self):
        """
        Loads the tables from ADLS
        Performs some light preprocessing as it would be difficult when the tables are joined
        """
        self.df_qb_file = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qb_file')
            .select(['ID_FILE'])
        )

        self.df_qd_state = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state')
            .select(['ID', 'ID_FILE', 'DAY', 'MONTH', 'YEAR', 'STIME', 'STYPE', 'ACTIVITY_STATE', 'ACTIVITY_CONFID'])
            .withColumn('STIME_SEC', F.round((F.col('STIME') / 1000)).cast(IntegerType()))
            .withColumnRenamed('ID', 'ID_STATE')
        )

        self.df_qd_state_gps = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state_gps')
            .select(['ID_STATE', 'ID_FILE', 'LTIME', 'SOURCE', 'SPEED', 'ACCURACY', 'LON', 'LAT'])
            .withColumn('LTIME_SEC', F.round((F.col('LTIME') / 1000)).cast(IntegerType()))
        )

        self.df_qd_state_cell = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state_cell')
            .select(['ID_STATE', 'ID_FILE', 'TYPE', 'LEVEL', 'QUAL', 'SLOT'])
        )

        self.df_qd_state_wifi = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state_wifi')
            .select(['ID_STATE', 'ID_FILE'])
            .withColumn('WIFI_CONNECTED', F.col('ID_STATE'))
        )

        self.df_qd_state_sens = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state_sens')
            .select(['ID_STATE', 'ID_FILE', 'LIGHT', 'MAGNET_X', 'MAGNET_Y', 'MAGNET_Z', 'PROXIMITY'])
            .withColumn(
                'MAGNET_X',
                F.when(
                    (F.abs(F.col('MAGNET_X')) < 100000), F.col('MAGNET_X')
                )
                .otherwise(F.lit(None)))
            .withColumn(
                'MAGNET_Y',
                F.when(
                    (F.abs(F.col('MAGNET_X')) < 100000), F.col('MAGNET_X')
                )
                .otherwise(F.lit(None)))
            .withColumn(
                'MAGNET_Z',
                F.when(
                    (F.abs(F.col('MAGNET_X')) < 100000), F.col('MAGNET_X')
                )
                .otherwise(F.lit(None)))
        )

        self.df_qd_state_batt = (
            self.loader
            .readDataframeFromAdls(table_path=f'{self.path_raw_tables}df_qd_state_batt')
            .select(['ID_STATE', 'ID_FILE', 'BATT_CHARGE'])
        )

    def _setJoinConditions(self):
        """
        Sets the join conditions in order to join the tables
        """
        self.condition_qbFile = ['ID_FILE']
        self.condition_qdStateGps = [
            (self.df_qd_state.ID_STATE == self.df_qd_state_gps.ID_STATE) &
            (self.df_qd_state.ID_FILE == self.df_qd_state_gps.ID_FILE) &
            (self.df_qd_state_gps.LON.between(-180, 180)) &
            (self.df_qd_state_gps.LAT.between(-90, 90))
        ]
        self.condition_qdStateCell = [
            (self.df_qd_state.ID_STATE == self.df_qd_state_cell.ID_STATE) &
            (self.df_qd_state.ID_FILE == self.df_qd_state_cell.ID_FILE) &
            (self.df_qd_state_cell.SLOT == 0)
        ]
        self.condition_qdStateWifi = ['ID_FILE', 'ID_STATE']
        self.condition_qdStateSens = ['ID_FILE', 'ID_STATE']
        self.condition_qdStateBatt = ['ID_FILE', 'ID_STATE']

    def _combineTables(self):
        """
        Joins the previously loaded tables on the specified conditions 
        """
        self.df_extract = (
            self.df_qd_state
            .join(self.df_qb_file, on=self.condition_qbFile, how='inner')
            .join(self.df_qd_state_gps, on=self.condition_qdStateGps, how='inner')
            .drop(self.df_qd_state_gps.ID_FILE)
            .drop(self.df_qd_state_gps.ID_STATE)
            .join(self.df_qd_state_cell, on=self.condition_qdStateCell, how='left')
            .drop(self.df_qd_state_cell.ID_FILE)
            .drop(self.df_qd_state_cell.ID_STATE)
            .join(self.df_qd_state_wifi, on=self.condition_qdStateWifi, how='left')
            .join(self.df_qd_state_sens, on=self.condition_qdStateSens, how='left')
            .join(self.df_qd_state_batt, on=self.condition_qdStateBatt, how='left')
        )

    def _validateSchema(self):
        """
        Validates the schema of the unified dataframe
        """
        # self.loader.writeSchemaToAdls(self.df_extract, self.schema_path)
        return super()._validateSchema()
