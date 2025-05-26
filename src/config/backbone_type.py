from enum import Enum, auto


class BackboneType(Enum):
    RESNET_50 = auto()
    RESNET_100 = auto()
    DEFORMED_RESNET_50 = auto()
    DEFORMED_RESNET_100 = auto()
