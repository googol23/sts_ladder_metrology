
from dataclasses import dataclass
from functools import cached_property

from sts_ladder_metrology.sensor_shape import SensorShape


@dataclass(frozen=True)
class SensorFit:
    """
    It hold the result of fitting a set of sensor measurements to a SensorShape, including:
        - resulting SensorShape
        - residuals
        - alignment (residuals respect to nominal)
    """

    shape: SensorShape
    param: tuple[float,...]
    error: tuple[float,...]
    
    @cached_property
    def n_par(self)->int:
        return len(self.param)