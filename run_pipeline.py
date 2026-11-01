#!/usr/bin/env python
import sys
from pathlib import Path
from app.config import Settings
from app.core.pipeline import ReIDPipeline

if __name__ == '__main__':
    settings = Settings()
    pipeline = ReIDPipeline(settings)

    source_arg = sys.argv[1] if len(sys.argv) > 1 else '0'
    source = int(source_arg) if source_arg.isdecimal() else source_arg
    output = settings.output_dir / 'output_video.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f'Processing source: {source}')
    pipeline.run_on_video(source, save_to_db=True, output_video=output)
