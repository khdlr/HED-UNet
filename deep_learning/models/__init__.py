# flake8: noqa
from .hed_unet import HEDUNet

def get_model(model_name):
    try:
        return globals()[model_name]
    except KeyError:
        raise ValueError(f'Can\'t provide Model called "{model_name}"')
