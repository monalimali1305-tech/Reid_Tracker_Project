from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple
import cv2
from app.core.gallery import Gallery


# ---- Kalman Filter ----
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.ndim = 8                     # was 7
        self.f = np.eye(self.ndim)
        self.h = np.eye(4, self.ndim)     # now 4×8
        for i in range(4):
            self.f[i, i+4] = self.dt
        self.q = np.eye(self.ndim) * 0.01 # 8×8
        self.r = np.eye(4) * 0.1          # 4×4
        self.x = None
        self.p = None

    def initiate(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / h if h > 0 else 1.0
        self.x = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)  # added one extra zero
        self.p = np.eye(self.ndim) * 10.0

    def predict(self):
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.get_bbox()

    def update(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / h if h > 0 else 1.0
        z = np.array([cx, cy, s, r], dtype=np.float32)
        y = z - self.h @ self.x
        s_matrix = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s_matrix)
        self.x = self.x + k @ y
        self.p = (np.eye(self.ndim) - k @ self.h) @ self.p

    def get_bbox(self) -> np.ndarray:
        cx, cy, s, r = self.x[:4]
        # Clamp to prevent negative or zero values
        s = max(s, 1e-6)
        r = max(r, 0.1)   # prevent division by zero or negative aspect
        area = s * r
        if area <= 0:
            area = s * 0.5   # fallback
        w = np.sqrt(area)
        h = s / w if w > 0 else 0.1
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

# ---- Track state ----
class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


@dataclass
class Track:
    tracker_id: int
    bbox: np.ndarray
    body_embedding: np.ndarray
    face_embedding: np.ndarray | None = None
    state: int = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    name: str = 'Unknown'
    score: float = 0.0

    def __post_init__(self):
        self.kalman = KalmanFilter()
        self.kalman.initiate(self.bbox)

    def predict(self) -> np.ndarray:
        self.bbox = self.kalman.predict()
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray, body_emb: np.ndarray, face_emb: np.ndarray | None):
        self.kalman.update(bbox)
        self.bbox = bbox
        self.body_embedding = body_emb
        self.face_embedding = face_emb
        self.time_since_update = 0
        self.age += 1
        self.hits += 1

    def mark_lost(self):
        self.state = TrackState.LOST


# ---- Main ReID Tracker ----
class ReIDTracker:
    def __init__(self, gallery: Gallery, extractor, face_processor=None,
                 threshold: float = 0.75, margin: float = 0.15,
                 max_lost_frames: int = 30, min_hits_to_confirm: int = 3,
                 iou_threshold: float = 0.3, appearance_weight: float = 0.5,
                 w_body: float = 0.5, w_face: float = 0.5):
        self.gallery = gallery
        self.extractor = extractor
        self.face_processor = face_processor
        self.threshold = threshold
        self.margin = margin
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.w_body = w_body
        self.w_face = w_face
        self.tracks: List[Track] = []
        self.next_id: int = 0
        self.frame_count: int = 0

    def _hybrid_similarity(self, body1, face1, body2, face2) -> float:
        sim, wsum = 0.0, 0.0
        if body1 is not None and body2 is not None:
            sim += self.w_body * np.dot(body1, body2)
            wsum += self.w_body
        if (face1 is not None and face2 is not None
                and np.isfinite(face1).all() and np.isfinite(face2).all()):
            sim += self.w_face * np.dot(face1, face2)
            wsum += self.w_face
        if wsum == 0:
            return -1.0
        return sim / wsum

    def _extract_embeddings(self, image, bbox, mask):
        x1, y1, x2, y2 = bbox.astype(int)
        # Body extraction (unchanged)
        if mask is not None:
            mask_bool = mask > 0.5
            masked = image.copy()
            masked[~mask_bool] = 0
            crop = masked[y1:y2, x1:x2]
        else:
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            body_emb = np.zeros((256,), dtype=np.float32)
        else:
            crop_resized = cv2.resize(crop, (128, 256))
            body_emb = self.extractor.extract(crop_resized)
            if body_emb is None or body_emb.size == 0:
                body_emb = np.zeros((256,), dtype=np.float32)

        # ---------- Face extraction with safety ----------
        face_emb = None
        if self.face_processor is not None:
            roi = image[y1:y2, x1:x2]
            # Only try if ROI is large enough and not empty
            if roi.size > 0 and roi.shape[0] >= 30 and roi.shape[1] >= 30:
                try:
                    # Resize ROI to speed up (optional)
                    # roi_small = cv2.resize(roi, (160, 160))
                    _, emb = self.face_processor.extract(roi)  # or roi_small
                    if emb is not None and emb.size > 0:
                        face_emb = emb
                except Exception:
                    pass   # ignore any errors
            # else: face_emb stays None
        return body_emb, face_emb

    def _compute_iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (a[2]-a[0]) * (a[3]-a[1])
        area2 = (b[2]-b[0]) * (b[3]-b[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _associate_detections(self, tracks, det_bboxes, det_bodies, det_faces):
        if not tracks or not det_bboxes:
            return [], list(range(len(tracks))), list(range(len(det_bboxes)))
        track_bboxes = [t.bbox for t in tracks]
        track_bodies = [t.body_embedding for t in tracks]
        track_faces = [t.face_embedding for t in tracks]
        cost = np.zeros((len(tracks), len(det_bboxes)))
        for i, (tb, tbody, tface) in enumerate(zip(track_bboxes, track_bodies, track_faces)):
            for j, (db, dbody, dface) in enumerate(zip(det_bboxes, det_bodies, det_faces)):
                iou = self._compute_iou(tb, db)
                iou_cost = 1.0 - iou
                app_sim = self._hybrid_similarity(tbody, tface, dbody, dface)
                app_cost = 1.0 - max(0.0, app_sim)
                cost[i, j] = (1 - self.appearance_weight) * iou_cost + self.appearance_weight * app_cost
        row_idx, col_idx = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(det_bboxes)))
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 1.0 - self.iou_threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)
        return matches, unmatched_tracks, unmatched_detections

    def _frame_level_identity_assignment(self):
        active = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        if not active or not self.gallery.has_body():
            return
        N, M = len(active), len(self.gallery.names)
        sim = np.zeros((N, M))
        for i, t in enumerate(active):
            for j, name in enumerate(self.gallery.names):
                g_body = self.gallery.body_means[j]
                g_face = self.gallery.face_means[j] if self.gallery.has_face() else None
                sim[i, j] = self._hybrid_similarity(t.body_embedding, t.face_embedding, g_body, g_face)
        row_idx, col_idx = linear_sum_assignment(-sim)
        assigned_names = set()
        assignments = {}
        for r, c in zip(row_idx, col_idx):
            score = sim[r, c]
            name = self.gallery.names[c]
            sorted_scores = np.sort(sim[r])[::-1]
            best = sorted_scores[0]
            second = sorted_scores[1] if len(sorted_scores) > 1 else -1.0
            margin_ok = (best - second) >= self.margin
            if score >= self.threshold and margin_ok and name not in assigned_names:
                assignments[active[r].tracker_id] = (name, score)
                assigned_names.add(name)
            else:
                assignments[active[r].tracker_id] = ('Unknown', 0.0)
        for track in active:
            if track.tracker_id in assignments:
                track.name, track.score = assignments[track.tracker_id]
            else:
                track.name, track.score = 'Unknown', 0.0

    def update(self, image: np.ndarray, detections: list) -> List[Track]:
        self.frame_count += 1
        # Advance every track on every frame, including frames with no detections.
        for track in self.tracks:
            track.predict()
        det_bboxes, det_bodies, det_faces = [], [], []
        for det in detections:
            bbox = np.array(det.box, dtype=np.float32)
            body, face = self._extract_embeddings(image, bbox, det.mask)
            det_bboxes.append(bbox)
            det_bodies.append(body)
            det_faces.append(face)
        matches, unmatched_tracks, unmatched_dets = self._associate_detections(
            self.tracks, det_bboxes, det_bodies, det_faces)
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_bboxes[d_idx], det_bodies[d_idx], det_faces[d_idx])
            if self.tracks[t_idx].hits >= self.min_hits_to_confirm:
                self.tracks[t_idx].state = TrackState.CONFIRMED
        for d_idx in unmatched_dets:
            track = Track(
                tracker_id=self.next_id,
                bbox=det_bboxes[d_idx],
                body_embedding=det_bodies[d_idx],
                face_embedding=det_faces[d_idx],
                state=(TrackState.CONFIRMED if self.min_hits_to_confirm <= 1 else TrackState.TENTATIVE),
                hits=1,
            )
            self.tracks.append(track)
            self.next_id += 1
        for t_idx in unmatched_tracks:
            if self.tracks[t_idx].time_since_update > self.max_lost_frames:
                self.tracks[t_idx].mark_lost()
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]
        self._frame_level_identity_assignment()
        return self.tracks
