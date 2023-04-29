import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.info("Select a page above.")

st.markdown(
    """
    Select a page above to get started!
    **👈 
    ### Pages
    - ...
"""
)
