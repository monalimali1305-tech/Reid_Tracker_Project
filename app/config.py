"""Runtime configuration for the person re-identification pipeline."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)


def database_timestamp() -> datetime:
    timezone_name = os.getenv('REID_TIMEZONE', 'Asia/Kolkata')
    return datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    detector_model: str = r'models\yolo11s-seg.pt'
    reid_model: str = 'osnet_x1_0'
    reid_weights: str = r'models\osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    device: str = 'cpu'
    confidence: float = 0.1
    match_threshold: float = 0.55
    known_retention_threshold: float = float(os.getenv('REID_KNOWN_RETENTION_THRESHOLD', '0.45'))
    gallery_dir: Path | None = None
    output_dir: Path | None = None

    hybrid_weight_body: float = 0.3
    hybrid_weight_face: float = 0.7

    face_model: str = 'buffalo_l'
    face_det_size: tuple[int, int] = (640, 640)
    face_det_threshold: float = 0.5

    mysql_host: str = os.getenv('REID_MYSQL_HOST', 'localhost')
    mysql_port: int = int(os.getenv('REID_MYSQL_PORT', '3306'))
    mysql_user: str = os.getenv('REID_MYSQL_USER', 'root')
    mysql_password: str = os.getenv('REID_MYSQL_PASSWORD', 'S108108w')
    mysql_database: str = os.getenv('REID_MYSQL_DATABASE', 'person_reid')

    def __post_init__(self) -> None:
        self.gallery_dir = self.gallery_dir or self.project_root / 'known_people'
        self.output_dir = self.output_dir or self.project_root / 'output'
        detector_path = Path(self.detector_model)
        weights_path = Path(self.reid_weights)
        if not detector_path.is_absolute():
            self.detector_model = str(self.project_root / detector_path)
        if not weights_path.is_absolute():
            self.reid_weights = str(self.project_root / weights_path)
