from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *


class LevelCalculator(Transformer):
    # LevelCalculator herits of property of Transformer
    def __init__(self, input_col):
        """
        Initiates a LevelCalculator transformer

        Parameters
        -----
        input_col: str, required
            column name of level in dbm

        Returns
        -----
        Transformer object

        """
        self.input_col = input_col
        self.output_col = 'LEVEL_MW'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("LevelCalculator"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        @F.udf(returnType=DoubleType())
        def dbmToMw(level):
            if level is None:
                return None
            else:
                return 10**((level)/10.)

        return (
            df.withColumn(
                self.output_col,
                dbmToMw(F.col(self.input_col)))
        )
