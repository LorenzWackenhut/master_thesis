from pyspark.ml.pipeline import Transformer
from pyspark.sql.window import Window
import pyspark.sql.functions as F


class LeadCalculator(Transformer):
    # LeadCalculator herits of property of Transformer
    def __init__(self, input_col):
        """
        Initiates a LeadCalculator transformer

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
        self.output_col = f'LEAD_{input_col}'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("LeadCalculator"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        windowCondition = self._getWindowConditionLeadGps()
        if self.input_col == 'LTIME':
            windowResult = self._getWindowResultLeadGpsTime()
        elif self.input_col == 'SPEED':
            windowResult = self._getWindowResultLeadGpsSpeed()
        elif self.input_col == 'ACTIVITY':
            windowCondition = self._getWindowConditionLeadGpsActivity()
            windowResult = self._getWindowResultLeadGpsActivity()

        return (
            df.withColumn(
                self.output_col,
                F.when(
                    windowCondition, windowResult
                )
                .otherwise(F.lit(None))
            )
        )

    def _getWindowConditionLeadGps(self):
        return (
            (F.lead(F.col('STIME_SEC')).over(self.window_spec) - F.col('STIME_SEC')).between(0, 960) &
            (F.col('STIME_SEC') - F.col('LTIME_SEC')).between(0, 180) &
            ((F.lead(F.col('STIME_SEC'))).over(self.window_spec) -
                F.lead(F.col('LTIME_SEC')).over(self.window_spec))
            .between(0, 180)
        )

    def _getWindowConditionLeadGpsActivity(self):
        return (
            (F.lead(F.col('STIME_SEC')).over(
                self.window_spec) - F.col('STIME_SEC')) < 900
        )

    def _getWindowResultLeadGpsSpeed(self):
        return (
            F.lead(F.col('SPEED')).over(self.window_spec)
        )

    def _getWindowResultLeadGpsActivity(self):
        return (
            F.lead(F.col('ACTIVITY_STATE')).over(self.window_spec)
        )

    def _getWindowResultLeadGpsTime(self):
        return (
            F.lead(F.col('LTIME_SEC')).over(
                self.window_spec) - F.col('LTIME_SEC')
        )
