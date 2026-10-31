from __future__ import annotations
import os
from pathlib import Path
import pymysql
import streamlit as st
from dotenv import dotenv_values
from app.config import Settings


def apply_theme():
    st.markdown(
        '''
        <style>
        .stApp { background: radial-gradient(circle at 8% 0%, #17355d 0%, #0b1220 38%, #070b13 100%); color: #e8eef9; }
        #MainMenu, footer { visibility: hidden; } .block-container { max-width: 1360px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 2rem 2.2rem; border: 1px solid rgba(151,193,255,.24); border-radius: 22px; background: linear-gradient(125deg,rgba(24,58,99,.94),rgba(13,25,44,.9)); box-shadow: 0 18px 46px rgba(0,0,0,.24); }
        .eyebrow { color:#75b8ff; font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
        .hero h1 { margin:.3rem 0; font-size:2.2rem; letter-spacing:-.04em; color:#fff; } .hero p { margin:0; color:#b8c8df; }
        .status { float:right; margin-top:-2.3rem; color:#a8f3c8; background:rgba(40,180,104,.14); padding:.42rem .75rem; border-radius:999px; font-size:.8rem; font-weight:650; }
        .metric-card { min-height:120px; padding:1.1rem 1.2rem; border-radius:16px; background:rgba(18,31,52,.78); border:1px solid rgba(167,197,232,.16); }
        .metric-name { color:#aebed4; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; } .metric-value { color:#f7fbff; font-size:1.8rem; font-weight:740; margin-top:.5rem; } .metric-detail { color:#72d6a1; font-size:.84rem; margin-top:.3rem; }
        .section-title { color:#f2f7ff; font-size:1.18rem; font-weight:680; margin:1.8rem 0 .65rem; }
        .stButton > button { border-radius:9px; border:1px solid rgba(117,184,255,.46); background:#247fd2; color:white; font-weight:650; } .stButton > button:hover { background:#3592e9; }
        div[data-testid='stDataFrame'] { border:1px solid rgba(167,197,232,.14); border-radius:12px; overflow:hidden; }
        [data-testid='stSidebar'] { background:#0b1422; min-width:280px !important; max-width:280px !important; }
        </style>
        ''',
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.markdown('### 👥 Person ReID')
        st.caption('Control Center')
        st.page_link('streamlit_app.py', label='Home', icon='⌂')
        st.page_link('/analytics', label='Analytics', icon='📊')
        st.page_link('/admin', label='Admin', icon='⚙️')
        st.divider()
        st.caption('Presence intelligence')


def connection():
    settings = Settings()
    return pymysql.connect(host=settings.mysql_host, port=settings.mysql_port, user=settings.mysql_user,
                           password=settings.mysql_password, database=settings.mysql_database,
                           cursorclass=pymysql.cursors.DictCursor, connect_timeout=5, read_timeout=10, autocommit=True)


@st.cache_data(ttl=3)
def load_runs() -> list[dict]:
    db = connection()
    try:
        with db.cursor() as cursor:
            cursor.execute('SELECT id, source, started_at, completed_at FROM reid_runs WHERE deleted_at IS NULL ORDER BY id DESC LIMIT 100')
            return list(cursor.fetchall())
    finally:
        db.close()


@st.cache_data(ttl=3)
def load_run_details(run_id: int) -> tuple[list[dict], list[dict]]:
    db = connection()
    try:
        with db.cursor() as cursor:
            cursor.execute('SELECT person_name, total_seconds, entries_count, exits_count FROM person_presence WHERE run_id=%s AND deleted_at IS NULL ORDER BY person_name', (run_id,))
            summary = list(cursor.fetchall())
            cursor.execute('SELECT person_name, event_type, occurred_at FROM person_presence_events WHERE run_id=%s AND deleted_at IS NULL ORDER BY occurred_at', (run_id,))
            return summary, list(cursor.fetchall())
    finally:
        db.close()


def delete_run(run_id: int) -> None:
    db = connection()
    try:
        with db.cursor() as cursor:
            cursor.execute('UPDATE person_presence_events SET deleted_at=NOW() WHERE run_id=%s AND deleted_at IS NULL', (run_id,))
            cursor.execute('UPDATE person_presence SET deleted_at=NOW() WHERE run_id=%s AND deleted_at IS NULL', (run_id,))
            cursor.execute('UPDATE reid_runs SET deleted_at=NOW() WHERE id=%s AND deleted_at IS NULL', (run_id,))
    finally:
        db.close()


def admin_authenticated() -> bool:
    password = dotenv_values(Path(__file__).resolve().parents[2] / '.env').get('REID_ADMIN_PASSWORD') or os.getenv('REID_ADMIN_PASSWORD', '')
    if not password:
        st.warning('Set `REID_ADMIN_PASSWORD` in `.env` to enable administration.')
        return False
    if st.session_state.get('admin_authenticated'):
        return True
    entered = st.text_input('Admin password', type='password')
    if st.button('Sign in'):
        if entered == password:
            st.session_state.admin_authenticated = True
            st.rerun()
        st.error('Incorrect admin password.')
    return False