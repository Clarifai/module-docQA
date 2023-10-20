"""Python file to serve as the frontend"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.auth import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import service_pb2
from clarifai.client.user import User
from streamlit_chat import message

from utils.investigate_utils import (
    combine_dicts, create_retrieval_qa_chat_chain, get_clarifai_docsearch, get_full_text,
    get_search_query_text, get_summarization_output, load_custom_llm_chain, parallel_process_input)
from utils.prompts import NER_PROMPT

st.set_page_config(
    page_title="GEOINT NER Investigation",
    page_icon="https://clarifai.com/favicon.svg",
    layout="wide",
)

ClarifaiStreamlitCSS.insert_default_css(st)

# init session state
if "full_text" not in st.session_state:
  st.session_state.full_text = ""

if "search_input_df" not in st.session_state:
  st.session_state.search_input_df = pd.DataFrame()

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

#############
# We need a cache ID so that if a user changes their app we
#############
app = stub.GetApp(service_pb2.GetAppRequest(user_app_id=userDataObject))

cache_id = "clarifai_app_cache_id"
if app.app.extra_info and  app.app.extra_info.search_revision_marker and app.app.extra_info.search_revision_marker != cache_id:
  cache_id = app.app.extra_info.search_revision_marker

st.markdown(
    "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This will collect a shortlist of the docs and then try to summarize the shortlist into one cohesive paragraph. So it's succesptible to combining lots of unrelated information that is retrieved. "
)

user_input = get_search_query_text()
number_of_docs = st.selectbox(
    "Select number of documents to return", options=[2, 4, 6, 8, 10, 12], index=2)

if user_input:
  docs = get_clarifai_docsearch(user_input, number_of_docs, cache_id)

  with st.expander(f"{len(docs)} Docs answering from:"):
    for idx, doc in enumerate(docs):
      st.markdown(f"### Search Result: {idx+1}")
      st.write(f"**{doc.page_content}**")
      st.write(doc.metadata)
      st.text("")

  if docs != []:
    input_list = [{
        "page_content": doc.page_content,
        "source": doc.metadata["source"]
    } for doc in docs]

    llm_chain = load_custom_llm_chain(prompt_template=NER_PROMPT)

    llm_chain.save("blah.json")

    # Measure execution time
    entities_list = parallel_process_input(llm_chain, input_list)

    combined_entities = combine_dicts(entities_list)
    entities_help = (
        "- PER (person): Refers to individuals, including their names and titles.\n"
        " - ORG (organization): Refers to institutions, companies, government bodies, and other groups.\n"
        " - LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.\n"
        " - TIME (date or year): Refers to dates, years, and other time-related expressions.\n"
        " - MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.\n"
        " - Sources (list of sources of the text)")

    # Create columns for each entity type
    st.markdown("### Entities", help=entities_help)
    columns = st.columns(len(combined_entities))
    for idx, (entity_type, entity_list) in enumerate(combined_entities.items()):
      columns[idx].info(f"**{entity_type}**")
      columns[idx].json(entity_list, expanded=False)

    doc_selection = st.selectbox(
        "Select search result to investigate",
        [idx if idx != 0 else "-" for idx in range(len(docs) + 1)],
        index=0,
    )

    highlighted_text_list = []
    if doc_selection and doc_selection != "-":
      doc_selection = int(doc_selection) - 1
      st.markdown(f"### {docs[doc_selection].metadata['source']}")

      if st.button("Summarize"):
        full_text, search_input_df = get_full_text(docs, doc_selection, cache_id)
        st.session_state.full_text = full_text
        st.session_state.search_input_df = search_input_df
        st.write(get_summarization_output(full_text, cache_id))

      if not st.session_state.search_input_df.empty:
        st.dataframe(st.session_state.search_input_df)

      if st.session_state.full_text != "":
        button1 = st.button("Chat with Document")
        if st.session_state.get("button") != True:
          st.session_state["button"] = button1

        if st.session_state["button"] == True:
          with st.expander("Full Text"):
            st.write(st.session_state.full_text)

          if st.session_state.full_text != "":
            split_texts = st.session_state.search_input_df.text.to_list()
            retrieval_qa_chat_chain = create_retrieval_qa_chat_chain(split_texts, cache_id)

            if "generatedcl2" not in st.session_state:
              st.session_state["generatedcl2"] = []

            if "pastcl2" not in st.session_state:
              st.session_state["pastcl2"] = []

            def get_question_text():
              input_text = st.text_input(
                  "You: ",
                  placeholder=f"Hello, what questions do you have about content in this app?",
                  key="input_clqa2",
              )
              return input_text

            user_input = get_question_text()

            if user_input:
              output = retrieval_qa_chat_chain.run(user_input)
              st.session_state.pastcl2.append(user_input)
              st.session_state.generatedcl2.append(output)

            if st.session_state["generatedcl2"]:
              for i in range(len(st.session_state["generatedcl2"]) - 1, -1, -1):
                message(
                    st.session_state["generatedcl2"][i],
                    key=str(i) + "cl2",
                )
                message(
                    st.session_state["pastcl2"][i],
                    is_user=True,
                    key=str(i) + "_usercl2",
                )

  else:
    st.warning("Found no documents related to your query.")
