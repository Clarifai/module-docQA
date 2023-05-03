"""Python file to serve as the frontend"""
from __future__ import annotations

import os

import PyPDF2
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.chains import ConversationChain, AnalyzeDocumentChain
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
from utils.prompts import NER_PROMPT, NER_QUESTION_TEMPLATE, NER_REFINE_TEMPLATE
from utils.investigate_utils import search_with_metadata, process_post_searches_response
import json
import concurrent.futures
import time
import spacy
from spacy_streamlit import visualize_ner

# FIXME(zeiler): don't hardcode.
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(
    page_title="GEOINT NER Investigation", page_icon="https://clarifai.com/favicon.svg", layout="wide"
)

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
        "We have provided an existing answer: {existing_answer}\n"
        "Do not remove any entities or sources from the existing answer.\n"
        "We have the opportunity to update the list of named entities and sources\n"
        "(only if needed) using the context below delimited by triple backticks.\n"
        "Make sure any entities extracted are part of the context below. If not, do not add them to the list.\n"
        "If you see any entities that are not extracted, add them.\n"
        "The new source can be extracted at the end of the context after 'Source: '.\n"
        "Use only the context below.\n"
        "------------\n"
        "{context_str}\n"
        "------------\n"
        "Given the new context, update the original answer to extract additional entities and sources.\n"
        "Create a more accurate list of named entities and sources.\n"
        "If you do update it, please update the sources as well while keeping the existing sources.\n"
        "If the context isn't useful, return the existing answer in JSON format unchanged.\n"
        "Output in JSON format:"
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
        "Output in JSON format:"
    )
    question_prompt = PromptTemplate(input_variables=["context_str", "question"], template=question_template)

    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0, max_tokens=-1),  # model_name="gpt-3.5-turbo",  max_tokens=2000
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


clarifai_qa, clarifai_agent = st.tabs(["Clarifai Q&A", "Clarifai Agent"])


with clarifai_qa:

    st.markdown(
        "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This will collect a shortlist of the docs and then try to summarize the shortlist into one cohesive paragraph. So it's succesptible to combining lots of unrelated information that is retrieved. "
    )

    chain = load_qa_with_sources()
    custom_chain = load_custom_qa_with_sources()
    text_splitter = CharacterTextSplitter()

    def get_text():
        input_text = st.text_input(
            "Search Query: ",
            placeholder=f"Type words or sentences to search for in your app",
            key="input_clqa",
            help="This will search for similar text in your Clarifai app.",
        )
        return input_text

    user_input = get_text()

    # Function that gets relevant doc using Clariifai's vector search
    @st.cache_resource
    def get_clarifai_docsearch(user_input):
        auth = ClarifaiAuthHelper.from_streamlit(st)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        docsearch = Clarifai(user_id=userDataObject.user_id, app_id=userDataObject.app_id, pat=auth._pat)

        print("Searching for: %s" % user_input)
        docs = docsearch.similarity_search(user_input)
        return docs

    if user_input:
        docs = get_clarifai_docsearch(user_input)

        with st.expander(f"{len(docs)} Docs answering from:"):
            for idx, doc in enumerate(docs):
                st.markdown(f"### Search Result: {idx+1}")
                st.write(f"**{doc.page_content}**")
                st.write(doc.metadata)
                st.text("")

        if docs != []:
            # Function that gets ner output using LLM
            @st.cache_resource
            def get_ner_output(_custom_chain, _docs, ner_prompt, user_input):
                ner_output = _custom_chain(
                    {"input_documents": _docs, "question": ner_prompt}, return_only_outputs=True
                )
                return ner_output

            # llm_ada = OpenAI(temperature=0, max_tokens=-1)

            llm_chatgpt = OpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")

            input_list = [
                {"page_content": doc.page_content, "source": doc.metadata["source"]} for doc in docs
            ]

            prompt = PromptTemplate(template=NER_PROMPT, input_variables=["page_content"])
            llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)

            @st.cache_resource
            def get_openai_output(_llm_chain, input):
                chat_output = llm_chain(input)
                return chat_output

            @st.cache_resource
            def extract_entities(llm_output):
                # print("output_str: ", llm_output)
                # print(type(llm_output))
                if isinstance(llm_output, dict) and len(llm_output) == 6:
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

            # # Measure execution time
            # start = time.time()
            # entities_list = []
            # for input in input_list:
            #     chat_output = llm_chatgpt(NER_PROMPT.replace("{page_content}", input["page_content"]))
            #     entities = extract_entities(chat_output)
            #     entities["SOURCES"] = [input["source"]]
            #     entities_list.append(entities)
            # end = time.time()
            # print(f"Time taken: {end - start}")

            def process_input(input):
                chat_output = get_openai_output(llm_chatgpt, input["page_content"])
                entities = extract_entities(chat_output["text"])
                entities["SOURCES"] = [input["source"]]
                return entities

            @st.cache_resource
            def parallel_process_input(input_list):
                entities_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    future_entities = [executor.submit(process_input, input) for input in input_list]
                    for future in concurrent.futures.as_completed(future_entities):
                        entities = future.result()
                        entities_list.append(entities)
                return entities_list

            # Measure execution time
            start = time.time()
            entities_list = parallel_process_input(input_list)
            end = time.time()
            print(f"Time taken: {end - start}")

            def combine_dicts(list_of_dicts):
                result = {}
                for dict_ in list_of_dicts:
                    for key, value in dict_.items():
                        result.setdefault(key, set()).update(value)
                return {k: sorted(list(v)) for k, v in result.items()}

            combined_entities = combine_dicts(entities_list)
            print(combined_entities)

            entities_help = "- PER (person): Refers to individuals, including their names and titles.\n - ORG (organization): Refers to institutions, companies, government bodies, and other groups.\n - LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.\n - TIME (date or year): Refers to dates, years, and other time-related expressions.\n - MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.\n - Sources (list of sources of the text)"

            # Create columns for each entity type
            st.markdown("### Entities", help=entities_help)
            columns = st.columns(len(combined_entities))
            for idx, (entity_type, entity_list) in enumerate(combined_entities.items()):
                columns[idx].info(f"**{entity_type}**")
                columns[idx].json(entity_list)

            import spacy
            from spacy_streamlit import visualize_ner

            doc_selection = st.selectbox(
                "Select search result to visualize",
                [idx if idx != 0 else "-" for idx in range(len(docs) + 1)],
                index=0,
            )

            if doc_selection and doc_selection != "-":
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(docs[doc_selection].page_content)
                st.markdown(f"### {docs[doc_selection].metadata['source']}")
                visualize_ner(doc, labels=combined_entities.keys(), show_table=False, title="Entities")

                # Function that gets summarization output using LLM chain
                @st.cache_resource
                def get_summarization_output(doc_selection):
                    auth = ClarifaiAuthHelper.from_streamlit(st)
                    stub = create_stub(auth)
                    userDataObject = auth.get_user_app_id_proto()

                    post_searches_response = search_with_metadata(
                        stub,
                        userDataObject,
                        search_metadata_key="source",
                        search_metadata_value=docs[doc_selection].metadata["source"],
                    )
                    search_input_df = pd.DataFrame(process_post_searches_response(post_searches_response))
                    search_input_df = search_input_df.sort_values(["page_number", "page_chunk_number"])
                    search_input_df.reset_index(drop=True, inplace=True)
                    st.dataframe(search_input_df)
                    full_text = "\n".join(search_input_df.text.to_list())

                    llm = OpenAI(temperature=0, max_tokens=1024)
                    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
                    text_summary = summarize_document_chain.run(full_text)
                    return text_summary

                if st.button("Summarize"):
                    st.write(get_summarization_output(doc_selection))

            # connection_prompt = """Using the named entity recognition (NER) annotations for the set of texts, identify any connections or commonalities between the texts. Consider how the entities mentioned in each text relate to each other, and whether any patterns emerge across the set. In particular, look for similarities or differences in the types of entities mentioned, and consider how these may be relevant to the themes or topics covered in the texts.

            # Here are the annotations:
            # {output_1}
            # """
            # connection_output = chain(
            #     {"input_documents": docs, "question": connection_prompt}, return_only_outputs=True
            # )
            # output_2 = connection_output["output_text"]

            # st.session_state.pastcl.append(user_input)
            # st.session_state.generatedcl.append(output_1)

            # # st.session_state.pastcl.append(user_input)
            # # st.session_state.generatedcl.append(output_2)
            # if st.session_state["generatedcl"]:
            #     for i in range(len(st.session_state["generatedcl"]) - 1, -1, -1):
            #         message(st.session_state["generatedcl"][i], key=str(i) + "cl")
            #         message(st.session_state["pastcl"][i], is_user=True, key=str(i) + "_usercl")

        else:
            st.warning("Found no documents related to your query.")


with clarifai_agent:
    st.markdown(
        "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This seems better than the Q&A concatenation approach as it can iterate on coming to a good answer, can chain thoughts together and doesn't seem to concatenate useless information."
    )

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
