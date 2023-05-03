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

            page_content = f"{docs[2].page_content}\nSource: {docs[2].metadata['source']}"

            NER_PROMPT_1 = """Using the context, do entity recognition of these texts using PER (person), ORG (organization),
            LOC (place name or location), TIME (actually date or year), and MISC (formal agreements and projects) and the Sources (the name of the document where the text is extracted from).
            The source can be extracted after 'Source: '.
            Make sure the source is prefixed with 'Source: ' and is on a new line. Do not include any sources that are part of the context.


            FORMAT:
            Provide them in JSON format with the following 6 keys:
            - PER: (list of people)
            - ORG: (list of organizations)
            - LOC: (list of locations)
            - TIME: (list of times)
            - MISC: (list of formal agreements and projects)
            - SOURCES: (list of sources)


            EXAMPLES:
            Here are the definitions with a few examples, do not use these examples to answer the question:
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
            - Tom Clancy's Jack Ryan
            - The New York Times
            - Harry Potter and the Sorcerer's Stone
            ----------------

            Before you generate the output, make sure that the named entities are correct and part of the context. 
            If the named entities are not part of the context, do not include them in the output. 
            Please add a comma after each value in the list except for the last one. 
            
            Context: {page_content}
            Source: {source}
            
            Output in JSON format: 
            """

            prompt = PromptTemplate(template=NER_PROMPT_1, input_variables=["page_content", "source"])
            llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)

            st.write(NER_PROMPT_1)

            @st.cache_resource
            def get_openai_output(_llm_chain, page_content, source):
                chat_output = llm_chain.run({"page_content": page_content, "source": source})
                return chat_output

            chat_output = get_openai_output(llm_chatgpt, docs[2].page_content, docs[2].metadata["source"])
            print(chat_output)

            # ner_output = get_ner_output(custom_chain, docs, NER_PROMPT, user_input=user_input)
            # output_text = ner_output["output_text"]
            # print("output_1: ", output_text)

            def extract_entities(output_str):
                entity_dict = {}
                output_str = output_str.strip()
                if "output" in output_str[:20].lower():
                    output_str = output_str.split("Output:")[1].strip()
                try:
                    entity_dict = json.loads(output_str)
                except Exception as e:
                    st.error(f"output: {output_str}")
                    st.error(f"error: {e}")
                return entity_dict

            # Extract the entities from the output
            entities = extract_entities(chat_output)
            print("entities: ", entities)

            # # Extract the entities from the output
            # entities = extract_entities(output_text)
            # print("entities: ", entities)

            entities_help = "- PER (person): Refers to individuals, including their names and titles.\n - ORG (organization): Refers to institutions, companies, government bodies, and other groups.\n - LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.\n - TIME (date or year): Refers to dates, years, and other time-related expressions.\n - MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.\n - Sources (list of sources of the text)"
            # entities_help = """- PER (person): Refers to individuals, including their names and titles.
            # - ORG (organization): Refers to institutions, companies, government bodies, and other groups.
            # - LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.
            # - TIME (date or year): Refers to dates, years, and other time-related expressions.
            # - MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.
            # - Sources (list of sources of the text)"""

            # Create columns for each entity type
            st.markdown("### Entities", help=entities_help)
            columns = st.columns(len(entities))
            for idx, (entity_type, entity_list) in enumerate(entities.items()):
                columns[idx].info(f"**{entity_type}**")
                columns[idx].json(entity_list)

            # import spacy
            # from spacy_streamlit import visualize_ner

            # doc_selection = st.selectbox(
            #     "Select search result to visualize",
            #     [idx if idx != 0 else "-" for idx in range(len(docs) + 1)],
            #     index=0,
            # )

            # if doc_selection and doc_selection != "-":
            #     nlp = spacy.load("en_core_web_sm")
            #     doc = nlp(docs[doc_selection].page_content)
            #     visualize_ner(doc, labels=entities.keys(), show_table=False, title="Entities")

            #     # Function that gets summarization output using LLM chain
            #     @st.cache_resource
            #     def get_summarization_output(doc_selection):
            #         auth = ClarifaiAuthHelper.from_streamlit(st)
            #         stub = create_stub(auth)
            #         userDataObject = auth.get_user_app_id_proto()

            #         post_searches_response = search_with_metadata(
            #             stub,
            #             userDataObject,
            #             search_metadata_key="source",
            #             search_metadata_value=docs[doc_selection].metadata["source"],
            #         )
            #         search_input_df = pd.DataFrame(process_post_searches_response(post_searches_response))
            #         search_input_df = search_input_df.sort_values(["page_number", "page_chunk_number"])
            #         search_input_df.reset_index(drop=True, inplace=True)
            #         full_text = "\n".join(search_input_df.text.to_list())

            #         llm = OpenAI(temperature=0, max_tokens=512)
            #         summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            #         summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
            #         text_summary = summarize_document_chain.run(full_text)
            #         return text_summary

            #     if st.button("Summarize"):
            #         st.write(get_summarization_output(doc_selection))

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
