import os
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class MCOnetSession(BaseSession):
    """
    This class represents a MCOnet session, which is a subclass of BaseSession.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the ONNX runtime session here instead of downloading the model
        model_path = self.download_models(*args, **kwargs)  # Now returns the path to the local model
        self.inner_session = ort.InferenceSession(model_path)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the output masks for the input image using the inner session.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: The list of output masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Instead of downloading, return the path to the local ONNX model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the local model file.
        """
        # Replace the following path with the path to your local .onnx model
        local_model_path = "rembg/sessions/MCOnet.onnx"
        return local_model_path

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the MCOnet session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "MCOnet"  # Update this to reflect your model's name if needed
