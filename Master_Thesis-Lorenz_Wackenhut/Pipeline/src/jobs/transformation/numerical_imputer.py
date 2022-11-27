from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *


class NumericalImputer(Transformer):
    # NumericalImputer herits of property of Transformer
    def __init__(self, impute_value, input_cols=None):
        """
        Initiates a NumericalImputer transformer

        Parameters
        -----
        input_col: str, required
            name of column to impute
        impute_value: int, required
            integer that should be imputed

        Returns
        -----
        Transformer object
        """
        self.input_cols = input_cols
        self.impute_value = impute_value

    def this():
        # define an unique ID
        this(Identifiable.randomUID("NumericalImputer"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        if self.input_cols is None:
            return df.fillna(self.impute_value)
        else:
            return df.fillna(self.impute_value, subset=self.input_cols)
