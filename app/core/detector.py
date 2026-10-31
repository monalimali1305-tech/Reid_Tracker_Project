from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None = None


class PersonDetector:
    def __init__(self, model_name: str, confidence: float = 0.35, device: str = ''):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence, self.device = confidence, device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(
            frame,
            classes=[0],
            conf=self.confidence,
            iou=0.7,
            imgsz=320,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        masks = None
        if result.masks is not None:
            masks_np = result.masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            if masks_np.shape[1:] != (h, w):
                masks_resized = []
                for m in masks_np:
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks_resized.append(m_resized)
                masks = np.array(masks_resized)
            else:
                masks = masks_np

        detections = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            mask = masks[i] if masks is not None else None
            detections.append(Detection(tuple(box), float(score), mask))
        return detections