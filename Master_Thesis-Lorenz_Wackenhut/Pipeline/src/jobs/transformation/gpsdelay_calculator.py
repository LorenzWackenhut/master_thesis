from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *
from math import radians, cos, sin, asin, sqrt


class GpsDelayCalculator(Transformer):
    # GpsDelayCalculator herits of property of Transformer
    def __init__(self, stime, ltime):
        """
        Initiates a GpsDelayCalculator transformer

        Parameters
        -----
        stime: str, required

        ltime: str, required

        Returns
        -----
        Transformer object

        """
        self.stime = stime
        self.ltime = ltime
        self.output_col = 'GPS_DELAY'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("GpsDelayCalculator"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        return (
            df.withColumn(
                'GPS_DELAY',
                F.when(
                    F.col('STIME_SEC') - F.col('LTIME_SEC') < -32000, -32000
                )
                .otherwise(
                    F.when(
                        F.col('STIME_SEC') - F.col('LTIME_SEC') > 32000, -32000
                    )
                    .otherwise(
                        F.col('STIME_SEC') - F.col('LTIME_SEC')
                    )
                )
            )
        )
