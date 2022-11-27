from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import *
from math import radians, cos, sin, asin, sqrt


class DistanceCalculator(Transformer):
    # DistanceCalculator herits of property of Transformer
    def __init__(self, lat_a, lon_a, lat_b, lon_b):
        """
        Initiates a DistanceCalculator transformer

        Parameters
        -----
        lat_a: str, required
            latitude coordinat of point a

        lon_a: str, required
            longitude coordinat of point a

        lat_b: str, required
            latitude coordinat of point b

        lon_b: str, required
            longitude coordinat of point b

        Returns
        -----
        Transformer object

        """
        self.lat_a = lat_a
        self.lon_a = lon_a
        self.lat_b = lat_b
        self.lon_b = lon_b
        self.output_col = 'DISTANCE'

    def this():
        # define an unique ID
        this(Identifiable.randomUID("DistanceCalculator"))

    def copy(extra):
        defaultCopy(extra)

    def _transform(self, df):
        @F.udf(returnType=IntegerType())
        def _calcDistance(lon_a, lat_a, lon_b, lat_b):
            """
            Calculates the distance between two points in meter
            """
            geo_list = [lon_a, lat_a, lon_b, lat_b]
            if any(x is None for x in geo_list):
                return None
            else:
                # Transform to radians
                lon_a, lat_a, lon_b, lat_b = map(radians, geo_list)
                dist_lon = lon_b - lon_a
                dist_lat = lat_b - lat_a
                # Calculate area
                area = sin(dist_lat/2)**2 + cos(lat_a) * \
                    cos(lat_b) * sin(dist_lon/2)**2
                # Calculate the central angle
                central_angle = 2 * asin(sqrt(area))
                radius = 6371
                # Calculate distance
                distance = central_angle * radius
                return int(abs(round(distance, 3)) * 1000)

        return (
            df.withColumn(
                self.output_col,
                _calcDistance(
                    F.col(self.lon_a),
                    F.col(self.lat_a),
                    F.col(self.lon_b),
                    F.col(self.lat_b))
            )
        )
