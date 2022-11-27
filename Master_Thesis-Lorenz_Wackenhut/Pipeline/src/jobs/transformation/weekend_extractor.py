from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime


class WeekendExtractor(Transformer):
    # WeekendExtractor herits of property of Transformer
    def __init__(self, input_col):
        """
        Initiates a WeekendExtractor transformer

        Parameters
        -----
        input_col: str, required
            column name of timestamp as epoch in seconds

        Returns
        -----
        Transformer object

        """
        self.input_col = input_col
        self.output_col = 'WEEKEND'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("WeekendExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        @F.udf(returnType=IntegerType())
        def epochToWeekday(epoch):
            return int(datetime.fromtimestamp(epoch).strftime('%w'))

        return (
            df.withColumn(
                'WEEKDAY',
                epochToWeekday(F.col(self.input_col))
            )
            .withColumn(
                self.output_col,
                F.when(
                    F.col('WEEKDAY').isin([6, 0]), 1
                )
                .otherwise(0)
            )
        )
