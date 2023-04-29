from __future__ import annotations

import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from langchain import OpenAI
from langchain.agents.agent_toolkits import (VectorStoreInfo, VectorStoreToolkit,
                                             create_vectorstore_agent)
from langchain.llms import OpenAI

from pages.vectorstore import Clarifai

# docsearch = Clarifai.from_documents(texts, embeddings, collection_name="state-of-union")

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

docsearch = Clarifai(user_id="zeiler", app_id="arxiv-understanding", pat=auth._pat)

input_text = st.text_input(
    "You: ",
    placeholder=f"Hello, what questions do you have about content in this app?",
    key="input_clqa")

vectorstore_info = VectorStoreInfo(
    name="research_articles",
    description=
    "useful for when you need to answer questions about articles. Input should be a fully formed question.",
    vectorstore=docsearch)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
llm = OpenAI(temperature=0)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# tools = load_tools(["searx-search"], searx_host="https://search.bus-hit.me/", llm=llm)

# agent_executor = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

output = agent_executor.run(input_text)

st.write("Answer:")
st.write(output)
