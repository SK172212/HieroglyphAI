from pathlib import Path
from typing import Any, Literal

import streamlit.components.v1 as components


_COMPONENT_FRONTEND = (
    Path(__file__).resolve().parent
    / "component_frontend"
)

_hieroglyph_camera = components.declare_component(
    "hieroglyph_camera",
    path=str(_COMPONENT_FRONTEND),
)


def hieroglyph_camera(
    *,
    mode: Literal["camera", "upload"] = "camera",
    key: str | None = None,
) -> Any:
    """
    Render the custom image capture and crop component.

    mode="camera":
        Open the live device camera, take a photo, and crop it.

    mode="upload":
        Choose an existing image from Photos or Files, then crop it.
    """
    if mode not in {"camera", "upload"}:
        raise ValueError(
            "mode must be either 'camera' or 'upload'"
        )

    return _hieroglyph_camera(
        mode=mode,
        key=key,
        default=None,
    )