from __future__ import annotations
import numpy as np
import torch


class FeatureExtractor:
    def __init__(self, model_name: str = 'osnet_x1_0', model_path: str = '', device: str = 'cpu'):
        try:
            from torchreid.reid.utils import FeatureExtractor as TorchExtractor
        except ModuleNotFoundError:
            from torchreid.reid.utils import FeatureExtractor as TorchExtractor
        self.extractor = TorchExtractor(
            model_name=model_name,
            model_path=model_path if model_path else None,
            device=device,
            verbose=False
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError('Empty crop')
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image
        with torch.no_grad():
            feature = self.extractor(rgb)
        feature = feature.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 1e-12:
            feature /= norm
        return feature