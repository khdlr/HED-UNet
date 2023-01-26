# flake8: noqa
from .hed_unet import HEDUNet
from .multitask_hed_unet import MultitaskHEDUNet

def get_model(model_name):
    try:
        return globals()[model_name]
    except KeyError:
        raise ValueError(f'Can\'t provide Model called "{model_name}"')
