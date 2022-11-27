from abc import ABC, abstractmethod


class TransformerAbstract(ABC):
    """
    Abstract base class for the transformation and feature engineering of extracted tables
    """
    @abstractmethod
    def __init__(self, mode, df=None, loader=None, container_name=None, table_path=None):
        """
        Initiates a transformer object

        Parameters
        -----
        mode: str, required
            possible inputs: ['pass', 'load']
            the dataframe either gets passed as an object or gets loaded from the adls
        loader: Loader object, required
        container_name: str, required
        table_path: str, required
            format like '/directory/'

        Returns
        -----
        Transformer object

        """
        if mode not in ['pass', 'load']:
            raise ValueError(
                "mode has to be 'pass' or 'load'"
            )
        if mode == 'pass':
            if df is None:
                raise ValueError(
                    "df cannot be none if mode='pass'"
                )
        elif mode == 'load':
            if (loader is None) or (container_name is None) or (table_path is None):
                raise ValueError(
                    "loader, container_name, table_path cannot be none if mode='load'"
                )

        self.loader = loader
        self.df = df
        self.loader = loader
        self.container_name = container_name
        self.table_path = table_path
        self.mode = mode
        self._setCategoricalFeatures()
        self.median_imputer_input = [
            'LEVEL_MW',
            'QUAL',
            'GPS_DELAY',
            'SPEED',
            'DISTANCE',
            'LIGHT',
            'MAGNET_X',
            'MAGNET_Y',
            'MAGNET_Z',
            'PROXIMITY'
        ]
        self.median_imputer_output = [
            f'{col}_IMPUTE' for col in self.median_imputer_input
        ]
        self.string_indexer_input = [
            'STYPE_CAT',
            'TYPE_CAT',
            'SOURCE_CAT',
            'BATT_CHARGE_CAT',
            'ACTIVITY_STATE_CAT',
            'LAG_ACTIVITY_CAT',
            'LEAD_ACTIVITY_CAT',
            'TIME_DAY_CAT'
        ]
        self.string_indexer_output = [
            f'{col}_INDEX' for col in self.string_indexer_input
        ]
        self.one_hot_encoder_input = self.string_indexer_output
        self.one_hot_encoder_output = [
            f'{col}_HOT' for col in self.one_hot_encoder_input
        ]
        self.x_columns_training = [
            'ACCURACY',
            'ACTIVITY_CONFID',
            'ACTIVITY_STATE_CAT_INDEX_HOT',
            'BATT_CHARGE_CAT_INDEX_HOT',
            'DISTANCE_IMPUTE',
            'GPS_DELAY_IMPUTE',
            'LAG_ACTIVITY_CAT_INDEX_HOT',
            'LAG_LTIME',
            'LAG_SPEED',
            'LEAD_ACTIVITY_CAT_INDEX_HOT',
            'LEAD_LTIME',
            'LEAD_SPEED',
            'LEVEL_MW_IMPUTE',
            'LIGHT_IMPUTE',
            'MAGNET_X_IMPUTE',
            'MAGNET_Y_IMPUTE',
            'MAGNET_Z_IMPUTE',
            'PROXIMITY_IMPUTE',
            'QUAL_IMPUTE',
            'SOURCE_CAT_INDEX_HOT',
            'SPEED_IMPUTE',
            'STYPE_CAT_INDEX_HOT',
            'TIME_DAY_CAT_INDEX_HOT',
            'TYPE_CAT_INDEX_HOT',
            'WEEKEND',
            'WIFI_CONNECTED_INT'
        ]
        self.x_columns_prediction = [
            'ACCURACY',
            'ACTIVITY_CONFID',
            'ACTIVITY_STATE_CAT_INDEX_HOT',
            'BATT_CHARGE_CAT_INDEX_HOT',
            'DISTANCE',
            'GPS_DELAY',
            'LAG_ACTIVITY_CAT_INDEX_HOT',
            'LAG_LTIME',
            'LAG_SPEED',
            'LEAD_ACTIVITY_CAT_INDEX_HOT',
            'LEAD_LTIME',
            'LEAD_SPEED',
            'LEVEL_MW',
            'LIGHT',
            'MAGNET_X',
            'MAGNET_Y',
            'MAGNET_Z',
            'PROXIMITY',
            'QUAL',
            'SOURCE_CAT_INDEX_HOT',
            'SPEED',
            'STYPE_CAT_INDEX_HOT',
            'TIME_DAY_CAT_INDEX_HOT',
            'TYPE_CAT_INDEX_HOT',
            'WEEKEND',
            'WIFI_CONNECTED_INT'
        ]

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
    def transform(self, mode='return', path_df_transform=None):
        """
        Transforms the dataframe and either returns of saves to adls

        Parameters
        -----
        mode: str, optional
            Determines if the tables should be saved to ADLS or returned
            Possible values: ['return', 'save']
        """
        if mode not in ['save', 'return']:
            raise ValueError(
                "mode has to be 'save' or 'return'"
            )
        if mode == "save":
            if path_df_transform is None:
                raise ValueError(
                    "path_df_transform cannot be none if mode='save'"
                )
        if self.mode == 'load':
            self.df = (
                self.loader
                .readDataframeFromAdls(
                    self.table_path,
                    self.container_name)
            )

    @abstractmethod
    def _buildStages(self):
        """
        Initiates transformer objects

        Returns
        -----
        list
        """

    def _setCategoricalFeatures(self):
        """
        Sets the values for categorical features
        """
        self.type_categorical = [
            'GSM',
            'LTE',
            'UNKNOWN',
            'WCDMA'
        ]
        self.source_categorical = [
            'FUSED',
            'GPS',
            'NET'
        ]
        self.stype_categorical = [
            'LOCATION',
            'NF_BOOST_TECHNOLOGY',
            'NF_SERVICE',
            'NF_BOOST_COVERAGE',
            'TIMER',
            'WORKER',
            'UNKNOWN'
        ]
        self.batt_charge_categorical = [
            'CHARGING',
            'FULL',
            'NOT_CHARGING',
            'UNCHARGING',
            'UNKNOWN'
        ]
        self.activity_state_categorical = [
            'IN_VEHICLE',
            'ON_BICYCLE',
            'ON_FOOT',
            'STILL',
            'TILTING',
            'WALKING',
            'UNKNOWN'
        ]
        self.time_day_categorical = [
            'MORNING',
            'AFTERNOON',
            'NIGHT'
        ]
