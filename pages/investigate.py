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
import pandas as pd
from vector.vectorstore import Clarifai

# FIXME(zeiler): don't hardcode.
os.environ["OPENAI_API_KEY"] = "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"

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


def load_custom_qa_with_sources():
    refine_template = (
        "The original question is as follows: {question}\n"
        "We have provided an existing answer, including sources: {existing_answer}\n"
        "We have the opportunity to update the list of named entities"
        "(only if needed) with some more context below.\n"
        "Make sure any entities extracted are part of the context below. If not, do not add them to the list."
        "If you see any entities that are not extracted, add them."
        "Use only the context below.\n"
        "------------\n"
        "{context_str}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "create a more accurate list of named entities."
        "If you do update it, please update the sources as well. "
        "If the context isn't useful, return the original answer."
    )
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_template,
    )

    question_template = (
        "Context information is below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {question}\n"
    )
    question_prompt = PromptTemplate(input_variables=["context_str", "question"], template=question_template)

    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0),
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
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


clarifai_qa, clarifai_agent, summarize_tab, pdf_tab, pdf_qa = st.tabs(
    ["Clarifai Q&A", "Clarifai Agent", "Summarization Text", "Summarize PDF", "Q&A PDF"]
)


with summarize_tab:

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

with pdf_tab:

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

with pdf_qa:
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

with clarifai_qa:

    st.markdown(
        "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This will collect a shortlist of the docs and then try to summarize the shortlist into one cohesive paragraph. So it's succesptible to combining lots of unrelated information that is retrieved. "
    )

    chain = load_qa_with_sources()
    custom_chain = load_custom_qa_with_sources()
    text_splitter = CharacterTextSplitter()

    if "generatedcl" not in st.session_state:
        st.session_state["generatedcl"] = []

    if "pastcl" not in st.session_state:
        st.session_state["pastcl"] = []

    def get_text():
        input_text = st.text_input(
            "You: ",
            placeholder=f"Type words or sentences to search for in your app",
            key="input_clqa",
        )
        return input_text

    user_input = get_text()

    if user_input:
        # Use Clarifai text that is embedded as the docsearch
        auth = ClarifaiAuthHelper.from_streamlit(st)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        docsearch = Clarifai(user_id=userDataObject.user_id, app_id=userDataObject.app_id, pat=auth._pat)

        print("Searching for: %s" % user_input)
        docs = docsearch.similarity_search(user_input)

        with st.expander("Docs answering from:"):
            for idx, doc in enumerate(docs):
                st.subheader(f"Search Result: {idx+1}")
                st.markdown(f"**{doc.page_content}**")
                st.write(doc.metadata)
                st.text("")

        if docs != []:
            st.write(f"Found {len(docs)} documents")
            st.write("Now going to use the LLM to understand and summarize the information...")

            ner_prompt = """Using the context, do entity recognition of these texts using PER (person), ORG (organization),
            LOC (place name or location), TIME (actually date or year), and MISC (formal agreements and projects) and the Sources (the name of the document where the text is extracted from).
            
            The format is:
            - PER: {list of people}
            - ORG: {list of organizations}
            - LOC: {list of locations}
            - TIME: {list of times}
            - MISC: {list of formal agreements and projects}
            - Sources: {list of sources}

            Here are the definitions with a few examples:
            PER (person): Refers to individuals, including their names and titles.
            Example:
            - Barack Obama, former President of the United States
            - J.K. Rowling, author of the Harry Potter series
            - Elon Musk, CEO of SpaceX and Tesla

            ORG (organization): Refers to institutions, companies, government bodies, and other groups.
            Example:
            - Microsoft Corporation, a multinational technology company
            - United Nations, an intergovernmental organization
            - International Red Cross, a humanitarian organization

            LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.
            Example:
            - London, capital of England
            - Eiffel Tower, a landmark in Paris, France
            - Great Barrier Reef, a coral reef system in Australia

            TIME (date or year): Refers to dates, years, and other time-related expressions.
            Example:
            - January 1st, 2023, the start of a new year
            - 1995, the year Toy Story was released

            MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.
            Example:
            - Kyoto Protocol, an international agreement to address climate change
            - Apollo program, a series of manned spaceflight missions undertaken by NASA
            Obamacare, a healthcare reform law in the United States.
            
            Sources (list of sources of the text).
            Example:
            - Tom Clancy
            - The New York Times
            - Harry Potter and the Sorcerer's Stone
            ----------------

            Output:
            """

            # ner_output = chain({"input_documents": docs, "question": ner_prompt}, return_only_outputs=True)
            ner_output = custom_chain(
                {"input_documents": docs, "question": ner_prompt}, return_only_outputs=True
            )
            output_1 = ner_output["output_text"]
            st.markdown(output_1)
            print("output_1: ", output_1)

            # connection_prompt = """Using the named entity recognition (NER) annotations for the set of texts, identify any connections or commonalities between the texts. Consider how the entities mentioned in each text relate to each other, and whether any patterns emerge across the set. In particular, look for similarities or differences in the types of entities mentioned, and consider how these may be relevant to the themes or topics covered in the texts.

            # Here are the annotations:
            # {output_1}
            # """
            # connection_output = chain(
            #     {"input_documents": docs, "question": connection_prompt}, return_only_outputs=True
            # )
            # output_2 = connection_output["output_text"]

            st.session_state.pastcl.append(user_input)
            st.session_state.generatedcl.append(output_1)

            # st.session_state.pastcl.append(user_input)
            # st.session_state.generatedcl.append(output_2)
            if st.session_state["generatedcl"]:
                for i in range(len(st.session_state["generatedcl"]) - 1, -1, -1):
                    message(st.session_state["generatedcl"][i], key=str(i) + "cl")
                    message(st.session_state["pastcl"][i], is_user=True, key=str(i) + "_usercl")

        else:
            st.warning("Found no documents related to your query.")


with clarifai_agent:
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
