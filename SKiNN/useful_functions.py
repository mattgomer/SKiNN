import os
import numpy as np
import joblib
import urllib.request
from pathlib import Path

# Dataverse URL
_WEIGHTS_URL = "https://dataverse.uliege.be/api/access/datafile/46845"
_WEIGHTS_FILENAME = "weights.ckpt"


def get_module_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))


def get_scalers_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "scalers"))


def _default_cache_dir() -> Path:
    """
    Choose a writable cache directory.
    - If XDG_CACHE_HOME is set, use it (Linux convention)
    - Else use ~/.cache/SKiNN
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "SKiNN"
    return Path.home() / ".cache" / "SKiNN"


def get_weights_path(download_if_missing: bool = True) -> str:
    """
    Returns the path to weights.ckpt.

    Priority:
    1) If SKINN_WEIGHTS_PATH env var is set, use that path.
    2) Else use a per-user cache dir (~/.cache/SKiNN/weights.ckpt).
    3) (Optional) if download_if_missing=True and the file is missing, download it.
    """
    # 1) user override
    override = os.environ.get("SKINN_WEIGHTS_PATH")
    if override:
        weights_path = Path(override).expanduser()
    else:
        # 2) writable cache location
        weights_path = _default_cache_dir() / _WEIGHTS_FILENAME

    if download_if_missing and not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SKiNN weights to {weights_path} ...")
        urllib.request.urlretrieve(_WEIGHTS_URL, str(weights_path))

    return str(weights_path.resolve())


def get_scaling_y():
    return np.load(os.path.join(get_scalers_path(), "scaler_y.npy"))


def get_scaling_x():
    return joblib.load(os.path.join(get_scalers_path(), "scaler_x"))
