from pyspark.ml.pipeline import Transformer
from pyspark.sql.window import Window
import pyspark.sql.functions as F


class LagCalculator(Transformer):
    # LagCalculator herits of property of Transformer
    def __init__(self, input_col):
        """
        Initiates a LagCalculator transformer

        Parameters
        -----
        input_col: str, required
            possible inputs: ['LTIME', 'SPEED', 'ACTIVITY', 'LAT', 'LON']

        Returns
        -----
        Transformer object

        """
        if input_col not in ['LTIME', 'SPEED', 'ACTIVITY', 'LAT', 'LON']:
            raise ValueError(
                "input_col has to be in ['LTIME', 'SPEED', 'ACTIVITY', 'LAT', 'LON']"
            )
        self.input_col = input_col
        self.window_spec = (
            Window
            .partitionBy('ID_FILE')
            .orderBy(['ID_FILE', 'ID_STATE'])
        )
        self.output_col = f'LAG_{input_col}'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("LagCalculator"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        windowCondition = self._getWindowConditionLagGps()
        if self.input_col == 'LTIME':
            windowResult = self._getWindowResultLagGpsTime()
        elif self.input_col == 'SPEED':
            windowResult = self._getWindowResultLagGpsSpeed()
        elif self.input_col == 'ACTIVITY':
            windowCondition = self._getWindowConditionLagGpsActivity()
            windowResult = self._getWindowResultLagGpsActivity()
        elif self.input_col == 'LAT':
            windowResult = self._getWindowResultLagGpsLat()
        elif self.input_col == 'LON':
            windowResult = self._getWindowResultLagGpsLon()

        return (
            df.withColumn(
                self.output_col,
                F.when(
                    windowCondition, windowResult
                )
                .otherwise(F.lit(None))
            )
        )

    def _getWindowConditionLagGps(self):
        return (
            (F.col('STIME_SEC') - F.lag(F.col('STIME_SEC')).over(self.window_spec)).between(0, 900) &
            (F.col('STIME_SEC') - F.col('LTIME_SEC')).between(0, 180) &
            ((F.lag(F.col('STIME_SEC'))).over(self.window_spec) -
             F.lag(F.col('LTIME_SEC')).over(self.window_spec))
            .between(0, 180)
        )

    def _getWindowConditionLagGpsActivity(self):
        return (
            (F.col('STIME_SEC') -
             F.lag(F.col('STIME_SEC'))
             .over(self.window_spec)) < 900
        )

    def _getWindowResultLagGpsSpeed(self):
        return (
            F.lag(F.col('SPEED')).over(self.window_spec)
        )

    def _getWindowResultLagGpsActivity(self):
        return (
            F.lag(F.col('ACTIVITY_STATE')).over(self.window_spec)
        )

    def _getWindowResultLagGpsTime(self):
        return (
            F.col('LTIME_SEC') - F.lag(F.col('LTIME_SEC')).over(self.window_spec)
        )

    def _getWindowResultLagGpsLat(self):
        return (
            F.lag(F.col('LAT')).over(self.window_spec)
        )

    def _getWindowResultLagGpsLon(self):
        return (
            F.lag(F.col('LON')).over(self.window_spec)
        )
