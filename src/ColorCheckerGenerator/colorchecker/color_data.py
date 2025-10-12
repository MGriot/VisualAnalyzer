"""
color_data.py

Color presets and chart definitions.
If 'colour' library is present, it pulls official ColorChecker data; otherwise uses fallback sRGB values.
"""

from typing import List, Tuple
import numpy as np

try:
    import colour
    _HAS_COLOUR = True
except Exception:
    _HAS_COLOUR = False

def get_colorchecker_classic() -> np.ndarray:
    """
    Return a (4,6,3) uint8 array with sRGB values for ColorChecker Classic.
    If 'colour' library available, use its ColorChecker values (approx).
    Otherwise use reasonable fallback sRGB triplets.
    """
    if _HAS_COLOUR:
        try:
            data = colour.CCS_COLOURCHECKERS.get('Color checker (24 patches)') or colour.CCS_COLOURCHECKERS.get('ColorChecker Classic') or next(iter(colour.CCS_COLOURCHECKERS.values()))
            rgb = []
            for name, val in data.items():
                # val may be a list/tuple where sRGB is available
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    s = val[1]
                else:
                    s = val
                try:
                    svals = [int(x*255) for x in s]
                except Exception:
                    # fallback: try flattening
                    svals = [int(float(x)*255) for x in s]
                rgb.append(np.clip(svals, 0, 255).astype(np.uint8))
            arr = np.array(rgb, dtype=np.uint8).reshape((4, 6, 3))
            return arr
        except Exception:
            pass

    # Fallback approximate sRGB values for the 24 patches (rows x cols = 4 x 6)
    colors = [
        (115,  82,  68), (194, 150, 130), ( 98, 122, 157), ( 87, 108,  67), (133, 128, 177), (103, 189, 170),
        (214, 126,  44), ( 80,  91, 166), (193,  90,  99), ( 94,  60, 108), (157, 188,  64), (224, 163,  46),
        ( 56,  61, 150), ( 70, 148,  73), (175,  54,  60), (231, 199,  31), (187,  86, 149), (  8, 133, 161),
        (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), ( 85,  85,  85), ( 52,  52,  52)
    ]
    return np.array(colors, dtype=np.uint8).reshape((4, 6, 3))


def get_calibration_colors() -> List[Tuple[str, str]]:
    """
    Return a list of (name, hex) pairs like the example card in user's image.
    """
    return [
        ("Blue",    "#0000FF"),
        ("Cyan",    "#00FFFF"),
        ("Green",   "#00FF00"),
        ("Yellow",  "#FFFF00"),
        ("Red",     "#FF0000"),
        ("Magenta", "#FF00FF"),
        ("White",   "#FFFFFF"),
        ("Grey",    "#A0A0A0"),
        ("Black",   "#000000"),
    ]
