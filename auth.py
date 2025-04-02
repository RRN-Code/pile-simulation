import streamlit as st

def login_required():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("🔐 Access denied. Please log in from the main page.")
        st.stop()
