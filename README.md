# Person ReID Pipeline with Hybrid Face+Body Matching

A complete person re-identification and presence tracking system with:
- YOLO segmentation for clean body cropping
- OSNet body embeddings
- InsightFace face embeddings
- Hybrid matching (weighted body+face similarity)
- Advanced Kalman + Hungarian tracker
- Presence accounting (IN/OUT, dwell time)
- MySQL storage with audit trail
- Streamlit dashboard

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your credentials.

3. Enroll people:
   ```
   python -m app.scripts.enroll
   ```

4. Build gallery embeddings:
   ```
   python -m app.scripts.build_gallery
   ```

5. Run pipeline on a video or camera:
   ```
   python run_pipeline.py [video_path]
   ```

6. Launch dashboard:
   ```
   streamlit run streamlit_app.py
   ```