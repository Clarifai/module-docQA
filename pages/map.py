import json
import requests
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import os
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationChain, AnalyzeDocumentChain
from langchain.docstore.document import Document
from geopy.geocoders import Nominatim
from utils.prompts import NER_LOC_RADIUS_PROMPT
import plotly.express as px

## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
from clarifai_grpc.grpc.api.service_pb2 import GetInputCountRequest, StreamInputsRequest

geolocator = Nominatim(user_agent="test")
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(
    page_title="GEOINT NER Investigation", page_icon="https://clarifai.com/favicon.svg", layout="wide"
)

# init session state
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


# Create function that searches with a given longitude and latitude
@st.cache_resource
def search_with_geopoints(_stub, userDataObject, longitude, latitude):
    post_searches_response = stub.PostSearches(
        service_pb2.PostSearchesRequest(
            user_app_id=userDataObject,
            query=resources_pb2.Query(
                ands=[
                    resources_pb2.And(
                        input=resources_pb2.Input(
                            data=resources_pb2.Data(
                                geo=resources_pb2.Geo(
                                    geo_point=resources_pb2.GeoPoint(
                                        longitude=longitude,
                                        latitude=latitude,
                                    ),
                                    geo_limit=resources_pb2.GeoLimit(type="withinKilometers", value=500.0),
                                )
                            )
                        )
                    )
                ]
            ),
        ),
    )

    if post_searches_response.status.code != status_code_pb2.SUCCESS:
        print(post_searches_response)
        raise Exception("Post searches failed, status: " + post_searches_response.status.description)

    print("Found inputs:")
    print(len(post_searches_response.hits))
    return post_searches_response


@st.cache_resource
def url_to_text(url):
    try:
        response = requests.get(url)
        response.encoding = response.apparent_encoding
    except Exception as e:
        print(f"Error: {e}")
        response = None
    return response.text if response else ""


@st.cache_resource
def process_post_searches_response(post_searches_response):
    input_success_status = {
        status_code_pb2.INPUT_DOWNLOAD_SUCCESS,
        status_code_pb2.INPUT_DOWNLOAD_PENDING,
        status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
    }

    input_dict_list = []
    for idx, hit in enumerate(post_searches_response.hits):
        input = hit.input
        if input.status.code not in input_success_status:
            continue

        # Initializations
        input_dict = {}
        input_dict["input_id"] = input.id
        input_dict["text"] = url_to_text(input.data.text.url)
        input_dict["source"] = input.data.metadata["source"]
        input_dict["text_length"] = input.data.metadata["text_length"]
        input_dict["page_number"] = input.data.metadata["page_number"]
        input_dict["page_chunk_number"] = input.data.metadata["page_chunk_number"]
        input_dict["lat"] = input.data.geo.geo_point.latitude
        input_dict["lon"] = input.data.geo.geo_point.longitude
        input_dict_list.append(input_dict)

    return input_dict_list


task_query = st.text_area("Enter your task here")


@st.cache_resource
def llm_output_to_json(llm_output):
    if isinstance(llm_output, dict) and len(llm_output) == 2:
        return llm_output
    elif isinstance(llm_output, str):
        entity_dict = {}
        llm_output = llm_output.strip()
        if "output" in llm_output[:20].lower():
            llm_output = llm_output.split("Output:")[1].strip()
        try:
            entity_dict = json.loads(llm_output)
        except Exception as e:
            st.error(f"output: {llm_output}")
            st.error(f"error: {e}")
        return entity_dict


# Function to get latitudes and longitudes from the location
def get_location_data(location_str):
    try:
        location_obj = geolocator.geocode(location_str)
        return location_obj
    except Exception as e:
        print(f"Error: {e}")
        return None


if task_query:
    llm_chatgpt = OpenAI(temperature=0, max_tokens=1500, model_name="gpt-3.5-turbo")
    prompt = PromptTemplate(template=NER_LOC_RADIUS_PROMPT, input_variables=["page_content"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)
    chain_output = llm_chain(task_query)
    chain_output_json = llm_output_to_json(chain_output["text"])
    location_obj = get_location_data(chain_output_json["LOC"])

    if location_obj is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"{location_obj.address}")
        with col2:
            st.info(f"LAT: {location_obj.latitude} - LON: {location_obj.longitude}")
        with col3:
            st.info(f"{chain_output_json['RADIUS']:0.2f} KM Radius")
    else:
        st.error(f"Coordinates not found for this location: {chain_output_json['LOC']}")

    post_searches_response = search_with_geopoints(
        stub, userDataObject, location_obj.longitude, location_obj.latitude
    )
    input_dict_list = process_post_searches_response(post_searches_response)
    input_df = pd.DataFrame(input_dict_list)
    if input_df.empty:
        st.warning("No searches found for this query")
    else:

        # Create a column with random floats between 0 and 0.5
        input_df["random"] = np.random.rand(len(input_df))  # / 2

        # add random column to latitude and longitude and remove random column
        input_df["lat"] = input_df["lat"] + input_df["random"]
        input_df["lon"] = input_df["lon"] + input_df["random"]
        input_df = input_df.drop(columns=["random"])

        st.dataframe(input_df)

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

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Summarize Searches"):
            # Function that gets summarization output using LLM chain
            @st.cache_resource
            def get_summarization_output(texts):

                from langchain.docstore.document import Document

                docs = [Document(page_content=t) for t in texts[:3]]

                llm = OpenAI(temperature=0, max_tokens=1024)
                summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                # summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
                text_summary = summary_chain.run(docs)
                return text_summary

            texts = input_df["text"].to_list()
            text_summary = get_summarization_output(texts)
            st.write(text_summary)
            # for i, row in input_df.iterrows():
            #     st.markdown(f"### {row['source']}")
            #     texts = row["text"]
            #     text_summary = get_summarization_output(text)
            #     st.write(text_summary)

    # lat_lon = [
    #     [18.971187, -72.285215],
    #     [42.546245, 1.601554],
    #     [7.425554, 150.550812],
    #     [12.262776, 61.604171],
    #     [39.074208, 21.824312],
    # ]
    # df = pd.DataFrame(lat_lon, columns=["lat", "lon"])
    # print(df)
    # df["Well Name"] = [str({"HOHIHO": "HEYU"})] * 5
    # df["Well Name_2"] = [str({"HOHIHO": "HEYU"})] * 5
    # df["size"] = [1, 1, 1, 1, 1]
    # st.map(df)

    # m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=1, control_scale=True)

    # # Loop through each row in the dataframe
    # for i, row in df.iterrows():
    #     # Setup the content of the popup
    #     iframe = folium.IFrame("Well Name: " + row["Well Name"])

    #     # Initialise the popup using the iframe
    #     popup = folium.Popup(iframe, min_width=300, max_width=300)

    #     # Add each row to the map
    #     folium.Marker(location=[row["lat"], row["lon"]], popup=popup, c=row["Well Name"]).add_to(m)

    # st_data = st_folium(m, width=700)

    # import folium
    # import streamlit as st
    # from folium.plugins import Draw

    # from streamlit_folium import st_folium

    # m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
    # Draw(export=True).add_to(m)

    # c1, c2 = st.columns(2)
    # with c1:
    #     output = st_folium(m, width=700, height=500)

    # with c2:
    #     st.write(output)
