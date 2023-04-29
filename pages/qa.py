"""Python file to serve as the frontend"""
from __future__ import annotations

import os

import PyPDF2
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.chains import ConversationChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_chat import message

from pages.vectorstore import Clarifai

# FIXME(zeiler): don't hardcode.
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


def load_chat_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = ConversationChain(llm=llm)
    return chain


def load_sum_chain():
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True)
    return chain


def load_qa_with_sources():
    # Note that the chatGPT model doesn't work.
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0), chain_type="refine", return_intermediate_steps=True
    )
    return chain


def load_qa_agent_with_sources(docstore):
    vectorstore_info = VectorStoreInfo(
        name="database",
        description="useful data source of information for when you need to answer questions. Input should be a fully formed question.",
        vectorstore=docsearch,
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent_executor


clarifaiAgent, clarifaiQA, openai_directly, chattab, sumtab, pdftab, pdfQA = st.tabs(
    ["Clarifai Agent", "Clarifai Q&A", "OpenAI LLM", "Chat", "Summarization Text", "Summarize PDF", "Q&A PDF"]
)

with chattab:

    st.markdown("Allows you to chat directly with the LLM")

    chain = load_chat_chain()

    # From here down is all the StreamLit UI.
    st.header("LangChain Demo")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    def get_text():
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        output = chain.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

with sumtab:

    st.markdown(
        "Take in the snippet of text and summarize it. Has to fit into the context of a LLM and summarizes it directly in the LLM."
    )
    chain = load_sum_chain()
    text_splitter = CharacterTextSplitter()

    text_input = st.text_input("Input the text to summarize")

    if text_input:

        texts = text_splitter.split_text(text_input)

        docs = [Document(page_content=t) for t in texts[:3]]

        # chain.run(docs)
        response = chain({"input_documents": docs}, return_only_outputs=True)

        st.title("Summary:")
        st.write(response)

with pdftab:

    st.markdown("Take in the PDF and summarizes it, but all at once.")

    chain = load_sum_chain()
    text_splitter = CharacterTextSplitter()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:

        # bytes_data = uploaded_file.getvalue()a

        reader = PyPDF2.PdfReader(uploaded_file)

        texts = []
        for p in reader.pages:
            texts.append(p.extract_text())

        with st.expander("Read text"):
            st.write(texts)

        # texts = text_splitter.split_text(text)

        docs = [Document(page_content=t) for t in texts[:3]]

        # chain.run(docs)
        response = chain({"input_documents": docs}, return_only_outputs=True)

        st.title("Summary:")
        st.write(response)

with pdfQA:
    st.markdown(
        "This will let you ask questions about the uploaded PDF by embedding chunks of it using OpenAI embeddings and then summarizing what it finds. This does not call our API at all."
    )

    chain = load_qa_with_sources()
    text_splitter = CharacterTextSplitter()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="qapdf")

    if uploaded_file:

        # bytes_data = uploaded_file.getvalue()a

        reader = PyPDF2.PdfReader(uploaded_file)

        texts = []
        for p in reader.pages:
            texts.append(p.extract_text())

        texts = [t for t in texts if t]

        # texts = texts[:10]

        with st.expander("Read text"):
            st.write(texts)
        st.write("Number of texts: %d" % len(texts))

        @st.cache_resource
        def get_docsearch(texts):
            embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_texts(
                texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]
            )
            return docsearch

        docsearch = get_docsearch(texts)

        if "generatedqa" not in st.session_state:
            st.session_state["generatedqa"] = []

        if "pastqa" not in st.session_state:
            st.session_state["pastqa"] = []

        def get_text():
            input_text = st.text_input(
                "You: ",
                placeholder=f"Hello, what questions do you have about {uploaded_file.name}?",
                key="input_qapdf",
            )
            return input_text

        user_input = get_text()

        if user_input:
            docs = docsearch.similarity_search(user_input)

            st.write("Found %d documents" % len(docs))

            output = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
            output = output["output_text"]

            st.session_state.pastqa.append(user_input)
            st.session_state.generatedqa.append(output)

        if st.session_state["generatedqa"]:

            for i in range(len(st.session_state["generatedqa"]) - 1, -1, -1):
                message(st.session_state["generatedqa"][i], key=str(i) + "qa")
                message(st.session_state["pastqa"][i], is_user=True, key=str(i) + "_userqa")

        # st.title("Summary:")
        # st.write(response)

with clarifaiQA:

    st.markdown(
        "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This will collect a shortlist of the docs and then try to summarize the shortlist into one cohesive paragraph. So it's succesptible to combining lots of unrelated information that is retrieved. "
    )

    chain = load_qa_with_sources()
    text_splitter = CharacterTextSplitter()

    if "generatedcl" not in st.session_state:
        st.session_state["generatedcl"] = []

    if "pastcl" not in st.session_state:
        st.session_state["pastcl"] = []

    def get_text():
        input_text = st.text_input(
            "You: ",
            placeholder=f"Hello, what questions do you have about content in this app?",
            key="input_clqa",
        )
        return input_text

    user_input = get_text()

    if user_input:

        ########################################
        # Use Clarifai text that is embedded as the docsearch
        auth = ClarifaiAuthHelper.from_streamlit(st)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        docsearch = Clarifai(user_id=userDataObject.user_id, app_id=userDataObject.app_id, pat=auth._pat)

        docs = docsearch.similarity_search(user_input)
        with st.expander("Docs answering from:"):
            st.write(docs)

        st.write("Found %d documents" % len(docs))
        st.write("Now going to use the LLM to understand and summarize the information...")

        ner_prompt = """
        Using the context, do entity recognition of these texts using PER (person), ORG (organization), LOC (place name or location), TIME (actually date or year), and MISC (formal agreements and projects).
        """
        ner_output = chain({"input_documents": docs, "question": ner_prompt}, return_only_outputs=True)
        output_1 = ner_output["output_text"]

        connection_prompt = """
        Using the context and the following entities {output_1}, do relationship extraction to find the relationships between the entities and source document.
        """
        connection_output = chain(
            {"input_documents": docs, "question": connection_prompt}, return_only_outputs=True
        )
        output_2 = connection_output["output_text"]

        st.session_state.pastcl.append(user_input)
        st.session_state.generatedcl.append(output_1)

        st.session_state.pastcl.append(user_input)
        st.session_state.generatedcl.append(output_2)

    if st.session_state["generatedcl"]:

        for i in range(len(st.session_state["generatedcl"]) - 1, -1, -1):
            message(st.session_state["generatedcl"][i], key=str(i) + "cl")
            message(st.session_state["pastcl"][i], is_user=True, key=str(i) + "_usercl")

with clarifaiAgent:
    st.markdown(
        "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This seems better than the Q&A concatenation approach as it can iterate on coming to a good answer, can chain thoughts together and doesn't seem to concatenate useless information."
    )
    ########################################
    # Use Clarifai text that is embedded as the docsearch
    docsearch = Clarifai(user_id=userDataObject.user_id, app_id=userDataObject.app_id, pat=auth._pat)

    chain = load_qa_agent_with_sources(docsearch)
    if "generatedcl2" not in st.session_state:
        st.session_state["generatedcl2"] = []

    if "pastcl2" not in st.session_state:
        st.session_state["pastcl2"] = []

    def get_text():
        input_text = st.text_input(
            "You: ",
            placeholder=f"Hello, what questions do you have about content in this app?",
            key="input_clqa2",
        )
        return input_text

    user_input = get_text()

    if user_input:
        output = chain.run(user_input)
        st.markdown(output)

        st.session_state.pastcl2.append(user_input)
        st.session_state.generatedcl2.append(output)

    if st.session_state["generatedcl2"]:

        for i in range(len(st.session_state["generatedcl2"]) - 1, -1, -1):
            message(st.session_state["generatedcl2"][i], key=str(i) + "cl2")
            message(st.session_state["pastcl2"][i], is_user=True, key=str(i) + "_usercl2")

with openai_directly:
    input_text = st.text_input(
        "Question for openAI LLM: ",
        placeholder=f"Hello, what questions do you have for the LLM?",
        key="input_openai",
    )

    template = """Question: {question}

  Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = llm_chain.run(input_text)

    st.write("Answer:")
    st.write(output)
