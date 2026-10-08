"""Image validation and preprocessing (resize; normalisation happens inside the model's Rescaling layer)."""
import numpy as np
from PIL import Image, UnidentifiedImageError
from .config import ALLOWED_EXT, IMG_SIZE


def load_and_validate(uploaded_file) -> Image.Image:
    """Return an RGB PIL image or raise ValueError with a user-friendly message."""
    ext = "." + uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported format '{ext}'. Please upload JPG or PNG.")
    try:
        Image.open(uploaded_file).verify()          # detects truncated/corrupted files
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("The image appears to be corrupted or unreadable.")


def to_model_input(img: Image.Image) -> np.ndarray:
    """Resize to model size -> float32 array of shape (1, H, W, 3) with values 0-255."""
    arr = np.asarray(img.resize(IMG_SIZE), dtype="float32")
    return np.expand_dims(arr, 0)
