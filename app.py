import streamlit as st

st.set_page_config(layout="wide", page_icon="https://clarifai.com/favicon.svg")

st.sidebar.info("Select a page above.")

st.markdown(
    """
    **Helper module that allows you to upload and explore large amount of texts** 👈 
    ### Pages
    - Investigate: Use this page to explore the texts you uploaded to a Clarifai application. Find similar texts and continue your investigation.
    - Upload: Upload and chunk a PDF to a Clarifai application.
"""
)
