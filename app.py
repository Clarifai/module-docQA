import streamlit as st
import os

st.set_page_config(layout="wide", page_icon="https://clarifai.com/favicon.svg")

st.sidebar.info("Select a page above.")

# Check if API key is in environment variables
if "OPENAI_API_KEY" not in os.environ:
    OPENAI_API_KEY = st.sidebar.text_input("Enter OpenAI API key here", type="password")
    st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY

st.markdown(
    """
    **Helper module that allows you to upload and explore large amount of texts**
    ## 📄 Pages
    #### 🌐 Geo Search
    Use this page to find explore texts that are related to a location.
    #### 🔍 Investigate
    Use this page to explore the texts you uploaded to a Clarifai application. Find similar texts and continue your investigation.
    #### 📟 Upload
    Upload and chunk a PDF to a Clarifai application.
    #### 📟 🌐 Upload with geo
    Upload and chunk with geo data a PDF to a Clarifai application.
"""
)
