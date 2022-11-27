from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import *


class Loader():
    """
    Includes all methods to read or write from the ADLS storage
    """

    def __init__(self, spark_session, adls_account_name='REDACTED'):
        self.spark_session = spark_session
        self.spark_context = spark_session.sparkContext
        self.adls_account_name = adls_account_name

    def readDataframeFromAdls(self, table_path, container_name='data'):
        """
        Reads a dataframe in parquet format from ADLS

        Paramters
        -----
        table_path: str, required
            absolute path to table in ADLS container
        container_name: str, optional
            name of the container in ADLS

        Returns
        -----
        Spark dataframe
        """
        return (
            self
            .spark_session
            .read
            .parquet(
                f'abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{table_path}')
        )

    def readModelFromAdls(self, model_path, model_class, container_name='data'):
        """
        Reads a RandomForestClassificationModel from ADLS

        Paramters
        -----
        model_class; str, required
            class name of the model e.g. RandomForestClassificationModel
        model_path: str, required
            absolute path to model in ADLS container
        container_name: str, optional
            name of the container in ADLS

        Returns
        -----
        PipelineModel
        """
        model = globals()[model_class]
        return (
            model
            .load(
                f"abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{model_path}"
            )
        )

    def writeDataframeToAdls(self, df, table_path, table_name, container_name='data', mode='overwrite'):
        """
        Writes a datframe to ADLS in parquet format

        Paramters
        -----
        df: Spark dataframe, required
        table_path: str, required
            absolute path to table in ADLS container
        table_name: str, required
            name for the table
        container_name: str, optional
            name of the container in ADLS
        mode: str, optional
            Spark write mode
        """
        (
            df
            .write
            .mode(mode)
            .format(table_name)
            .parquet(
                f'abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{table_path}/{table_name}'
            )
        )

    def writeModelToAdls(self, model, model_path, container_name='data'):
        """
        Writes a RandomForestClassificationModel to ADLS

        Paramters
        -----
        model: RandomForestClassificationModel, required
        model_path: str, required
            absolute path to model in ADLS container
        container_name: str, optional
            name of the container in ADLS
        mode: str, optional
            Spark write mode
        """
        (
            model
            .write()
            .overwrite()
            .save(
                f'abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{model_path}'
            )
        )

    def writeSchemaToAdls(self, df, schema_path, container_name='data'):
        """
        Writes a Dataframe schema to ADLS

        Paramters
        -----
        df: Dataframe, required
        schema_path: str, required
            absolute path to schema in ADLS container
        container_name: str, optional
            name of the container in ADLS
        mode: str, optional
            Spark write mode
        """
        temp_rdd = self.spark_context.parallelize(df.schema)
        temp_rdd.coalesce(1).saveAsPickleFile(
            f"abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{schema_path}")

    def readSchemaFromAdls(self, schema_path, container_name='data'):
        """
        Reads a Dataframe schema from ADLS

        Paramters
        -----
        schema_path: str, required
            absolute path to schema in ADLS container
        container_name: str, optional
            name of the container in ADLS
        mode: str, optional
            Spark write mode
        """
        schema_rdd = self.spark_context.pickleFile(
            f"abfss://{container_name}@{self.adls_account_name}.dfs.core.windows.net{schema_path}")
        return StructType(schema_rdd.collect())
