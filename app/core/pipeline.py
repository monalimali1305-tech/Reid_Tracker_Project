from __future__ import annotations
import cv2
from pathlib import Path
from datetime import datetime
from app.config import Settings, database_timestamp
from app.core.detector import PersonDetector
from app.core.reid import FeatureExtractor
from app.core.face import FaceProcessor
from app.core.gallery import Gallery
from app.core.tracker import ReIDTracker, TrackState
from app.core.presence import PresenceTracker
from app.storage.mysql_store import MySQLPresenceStore
from app.utils.utils import draw_label, draw_presence_panel


class ReIDPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.extractor = FeatureExtractor(
            model_name=settings.reid_model,
            model_path=settings.reid_weights,
            device=settings.device,
        )
        self.face_processor = FaceProcessor(
            model_name=settings.face_model,
            det_size=settings.face_det_size,
        )
        self.gallery = Gallery.load(
            settings.gallery_dir,
            body_extractor=self.extractor,
            face_processor=self.face_processor
        )
        self.detector = PersonDetector(
            model_name=str(settings.detector_model),
            confidence=settings.confidence,
            device=settings.device,
        )
        self.tracker = ReIDTracker(
            gallery=self.gallery,
            extractor=self.extractor,
            face_processor=self.face_processor,
            threshold=settings.match_threshold,
            w_body=settings.hybrid_weight_body,
            w_face=settings.hybrid_weight_face,
        )
        self.presence = PresenceTracker()
        self.store = None
        if settings.mysql_host:
            self.store = MySQLPresenceStore(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
            )

    def run_on_video(self, source: str | int, save_to_db: bool = True,
                     output_video: Path | None = None, max_frames: int = -1) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open source: {source}')
        started_at = database_timestamp()
        prev_time = datetime.now()
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = None
        try:
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video), fourcc, fps,
                                      (int(cap.get(3)), int(cap.get(4))))
                if not out.isOpened():
                    raise RuntimeError(f'Cannot create output video: {output_video}')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if max_frames > 0 and frame_count > max_frames:
                    break
                current_time = datetime.now()
                elapsed = (current_time - prev_time).total_seconds()
                prev_time = current_time
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(frame, detections)
                visible_names = {t.name for t in tracks if t.state == 1 and t.name != 'Unknown'}
                self.presence.update(visible_names, elapsed, current_time)
                for track in tracks:
                    if track.time_since_update != 0:
                        continue
                    known = track.state == TrackState.CONFIRMED and track.name != 'Unknown'
                    label = track.name if known else f'Person {track.tracker_id}'
                    draw_label(frame, tuple(track.bbox.astype(int)), label, known)
                draw_presence_panel(frame, self.presence.records, self.presence.active_names)
                if out is not None:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            completed_at = database_timestamp()
            if self.store:
                try:
                    if save_to_db:
                        self.store.save_run(
                            source=str(source),
                            started_at=started_at,
                            completed_at=completed_at,
                            records=self.presence.records,
                            events=self.presence.events,
                        )
                        print('Run saved to database.')
                finally:
                    self.store.close()
            print('Processing complete.')
