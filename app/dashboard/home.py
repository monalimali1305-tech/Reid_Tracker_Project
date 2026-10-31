import streamlit as st
from app.dashboard.shared import apply_theme, load_runs

apply_theme()
st.markdown(
    '''
    <div class='hero'>
        <div class='eyebrow'>Presence intelligence</div>
        <div class='status'>● DATABASE CONNECTED</div>
        <h1>Person ReID Control Center</h1>
        <p>Monitor recognized visitors, review activity, and manage your enrolled identities.</p>
    </div>
    ''',
    unsafe_allow_html=True
)
runs = load_runs()
columns = st.columns(3)
for column, label, value, detail in zip(
    columns,
    ['Completed runs', 'Data source', 'Navigation'],
    [str(len(runs)), 'MySQL', '3 pages'],
    ['Available for analysis', 'Connected presence store', 'Home · Analytics · Admin'],
):
    with column:
        st.markdown(
            f'''<div class='metric-card'><div class='metric-name'>{label}</div><div class='metric-value'>{value}</div><div class='metric-detail'>{detail}</div></div>''',
            unsafe_allow_html=True
        )
st.markdown("<div class='section-title'>Get started</div>", unsafe_allow_html=True)
st.write('Use **Analytics** in the sidebar to inspect camera runs and detailed IN/OUT history. Use **Admin** to enroll people or archive stored runs.')
