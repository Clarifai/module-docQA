import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import ConversationChain, AnalyzeDocumentChain
from utils.prompts import NER_LOC_RADIUS_PROMPT
from utils.geo_search_utils import search_with_geopoints, process_post_searches_response, llm_output_to_json, get_location_data, display_location_info, get_summarization_output
import plotly.express as px


# os.environ["OPENAI_API_KEY"] = "API_KEY"

# Set Streamlit page configuration
st.set_page_config(
    page_title="GEOINT NER Investigation", page_icon="https://clarifai.com/favicon.svg", layout="wide"
)

# Authenticate user and get stub for Clarifai API
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

# Get user's task query
task_query = st.text_area("Enter your task here")

# If task_query is not empty
if task_query:
    # Create OpenAI language model
    llm_chatgpt = OpenAI(temperature=0, max_tokens=1500, model_name="gpt-3.5-turbo")
    # Create prompt template that retrieves the location and radius from the task query
    prompt = PromptTemplate(template=NER_LOC_RADIUS_PROMPT, input_variables=["page_content"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)
    # Run language model chain to get location object from task query
    chain_output = llm_chain(task_query)
    chain_output_json = llm_output_to_json(chain_output["text"])
    location_obj = get_location_data(chain_output_json["LOC"])
    
    print(chain_output_json)


    # If location object is found, display address, latitude, longitude, and radius
    if location_obj is not None:
        display_location_info(location_obj, chain_output_json['RADIUS'])
    # If location object is not found, display error message
    else:
        st.error(f"Coordinates not found for this location: {chain_output_json['LOC']}")


    # Search posts with geopoints using Clarifai API
    post_searches_response = search_with_geopoints(
        stub, userDataObject, location_obj.longitude, location_obj.latitude, float(chain_output_json['RADIUS'])
    )
    # Process post search response into a dictionary list
    input_dict_list = process_post_searches_response(post_searches_response)
    # Convert dictionary list to pandas DataFrame
    input_df = pd.DataFrame(input_dict_list)
    
    # If DataFrame is empty, display warning message
    if input_df.empty:
        st.warning("No searches found for this query")
    # If DataFrame is not empty, proceed with displaying and summarizing searches
    else:

        # Create a column with random floats between 0 and 0.5
        input_df["random"] = np.random.rand(len(input_df))  # / 2

        # add random column to latitude and longitude and remove random column
        input_df["lat"] = input_df["lat"] + input_df["random"]
        input_df["lon"] = input_df["lon"] + input_df["random"]
        input_df = input_df.drop(columns=["random"])

        st.dataframe(input_df)

         # Create a map scatter map plot of the search results
        fig = px.scatter_mapbox(
            input_df,
            lat="lat",
            lon="lon",
            zoom=3,
            hover_name="source",
            hover_data=["input_id", "page_number", "page_chunk_number"],
            color_discrete_sequence=["red"],
            height=800,
            width=800,
            template="plotly_white",
        )

        # Update the map style and layout
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        # Display the map plot in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Summarize Searches"):
            texts = input_df["text"].to_list()
            text_summary = get_summarization_output(texts)
            st.write(text_summary)
