import importlib
import os

from .decorators import plugin_decorator, plugins_dict


def load_plugins():
    """
    Digs through plugins folder for possible plugins to be imported
    and consequently added to the plugins_dict if properly decorated

    :return: None
    """
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    for folder in os.listdir(plugin_path):
        option_path = os.path.join(plugin_path, folder)
        if os.path.isdir(option_path):
            if folder == "__pycache__":
                continue
            for filename in os.listdir(option_path):
                if filename is None or not filename.endswith(".py"):
                    continue
                spec = importlib.util.spec_from_file_location(
                    filename, os.path.join(option_path, filename)
                )
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)


def get_plugins(typ):
    """
    Fetches a dictionary of plugins of a certain type

    :param typ: Broader classification/type of a plugin
    :return: dict
    """
    # Direct dictionary access is marginally faster than .get when key presence is guaranteed.
    # However, to preserve exact behavior (since .get returns None if key absent),
    # check membership and return None if not found.
    if typ in plugins_dict:
        return plugins_dict[typ]
    else:
        return None
