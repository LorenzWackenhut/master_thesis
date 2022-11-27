from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *


class CategoricalFilter(Transformer):
    # CategoricalFilter herits of property of Transformer
    def __init__(self, input_col, categorical_values):
        """
        Initiates a CategoricalFilter transformer

        Parameters
        -----
        input_col: str, required
            column name of level in dbm
        categorical_values: list, required
            list of attributes that should be filtered

        Returns
        -----
        Transformer object

        """
        self.input_col = input_col
        self.output_col = input_col + '_CAT'
        self.categorical_values = categorical_values

    def this():
        # define an unique ID
        this(Identifiable.randomUID("CategoricalFilter"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        return (
            df.withColumn(
                self.output_col,
                F.when(
                    F.col(self.input_col).isin(self.categorical_values), F.col(self.input_col))
                .otherwise(F.lit('UNKNOWN'))
            )
        )
