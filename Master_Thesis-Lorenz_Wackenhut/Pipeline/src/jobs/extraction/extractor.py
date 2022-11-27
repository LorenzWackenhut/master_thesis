from abc import ABC, abstractmethod


class Extractor(ABC):
    """
    Abstract base class for the extraction and combining of raw tables
    """
    @abstractmethod
    def __init__(self, loader, container_name, path_raw_tables, schema_path):
        """
        Initiates an extractor object

        Parameters
        -----
        loader: Loader object, required
        container_name: str, required
        path_raw_tables: str, required
            format like '/directory/'

        Returns
        -----
        Extractor object

        """
        self.loader = loader
        self.container_name = container_name
        self.path_raw_tables = path_raw_tables
        self.schema_path = schema_path

    @abstractmethod
    def runTransformation(self, **kwargs):
        """
        Executes the transformations 

        Parameters
        -----
        kwargs: dict, required
            dictionary loaded from yaml config 
        """

    @abstractmethod
    def extract(self, mode='return', path_df_extract=None):
        """
        Extracts the tables and combines them in one dataframe

        Parameters
        -----
        mode: str, optional
            Determines if the tables should be saved to ADLS or returned
            Possible values: return, save
        """
        if mode not in ['save', 'return']:
            raise ValueError(
                "mode has to be 'save' or 'return'"
            )
        if mode == "save":
            if path_df_extract is None:
                raise ValueError(
                    "path_df_extract cannot be none if mode='save'"
                )

    @abstractmethod
    def _getTables(self):
        """
        Loads tables from ADLS
        """

    @abstractmethod
    def _combineTables(self):
        """
        Combines the loaded tables to one dataframe
        """

    @abstractmethod
    def _validateSchema(self):
        """
        Validates the schema of the unified dataframe
        """
        def assert_schema(schema1, schema2):
            schema2 = set(schema2)
            diff = [item for item in schema1 if item not in schema2]
            assert not diff

        reading_schema = self.loader.readSchemaFromAdls(
            schema_path=self.schema_path)
        fields_compare = list(reading_schema.fields)
        fields_df_extract = list(self.df_extract.schema.fields)
        assert_schema(fields_compare, fields_df_extract)
