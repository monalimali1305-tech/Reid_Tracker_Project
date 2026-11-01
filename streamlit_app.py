# Person ReID Streamlit application entry point.
import streamlit as st

st.set_page_config(
    page_title='Person ReID Control Center',
    page_icon='👥',
    layout='wide',
    initial_sidebar_state='expanded',
)

navigation = st.navigation(
    [
        st.Page('app/dashboard/home.py', title='Home', icon='🏠', default=True),
        st.Page('app/dashboard/analytics.py', title='Analytics', icon='📊', url_path='analytics'),
        st.Page('app/dashboard/admin.py', title='Admin', icon='⚙️', url_path='admin'),
    ],
    position='sidebar',
)
navigation.run()