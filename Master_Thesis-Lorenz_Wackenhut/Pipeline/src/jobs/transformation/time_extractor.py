from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import pytz


class TimeExtractor(Transformer):
    # TimeExtractor herits of property of Transformer
    def __init__(self, input_col):
        """
        Initiates a TimeExtractor transformer

        Parameters
        -----
        input_col: str, required
            column name of timestamp as epoch in seconds

        Returns
        -----
        Transformer object

        """
        self.input_col = input_col
        self.output_col = 'TIME_DAY'
        if self._isSummerTime():
            self.hours_added = 2
        else:
            self.hours_added = 1

    def this():
        # define an unique ID
        this(Identifiable.randomUID("TimeExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        @F.udf(returnType=IntegerType())
        def epochToHour(epoch):
            return int(datetime.fromtimestamp(epoch).strftime('%-H'))

        return (
            df.withColumn(
                'HOUR',
                epochToHour(F.col('STIME_SEC')))
            .withColumn(
                'TIME_DAY',
                F.when(
                    (F.col('HOUR') + F.lit(self.hours_added)
                     ).between(6, 11), 'MORNING'
                )
                .otherwise(
                    F.when(
                        (F.col('HOUR') + F.lit(self.hours_added)
                         ).between(12, 17), 'AFTERNOON'
                    )
                    .otherwise(
                        F.when(
                            (F.col('HOUR') + F.lit(self.hours_added)
                             ).between(18, 24), 'NIGHT'
                        )
                        .otherwise('NIGHT')
                    )
                )
            )
        )

    def _isSummerTime(self):
        dt = datetime.now()
        timezone = "Europe/Berlin"
        timezone = pytz.timezone(timezone)
        timezone_aware_date = timezone.localize(dt, is_dst=None)
        return timezone_aware_date.tzinfo._dst.seconds != 0
