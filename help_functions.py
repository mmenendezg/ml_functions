import os
from datetime import datetime


def get_logdir(date_type="date", path_folder=None):
    """This function creates the name of a folder for Tensorboard
    logs using the current date or datetime

    Args:
        date_type (str, optional): Format of the second part of the folder name. Defaults to "date".
        path_folder (str, optional): String of the path to add before the folder name. Defaults to None.

    Returns:
        str: Name of the folder or path of the folder.
    """
    if date_type == "datetime":
        log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"run_{log_dir}"
    elif date_type == "date":
        log_dir = datetime.now().strftime("%Y%m%d")
        log_dir = f"run_{log_dir}"

    if path_folder:
        log_dir = os.path.join(path_folder, log_dir)

    return log_dir
