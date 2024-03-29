import streamlit as st
from clarifai.modules.css import ClarifaiStreamlitCSS

st.set_page_config(layout="wide", page_icon="https://clarifai.com/favicon.svg")

ClarifaiStreamlitCSS.insert_default_css(st)

st.markdown("""
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
""")
