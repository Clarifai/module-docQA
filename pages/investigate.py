"""Python file to serve as the frontend"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents.agent_toolkits import (VectorStoreInfo,
                                             VectorStoreToolkit,
                                             create_vectorstore_agent)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from streamlit_chat import message
from utils.investigate_utils import (combine_dicts,
                                     create_retrieval_qa_chat_chain,
                                     get_clarifai_docsearch,
                                     get_search_query_text,
                                     get_summarization_output,
                                     load_custom_llm_chain,
                                     parallel_process_input)
from utils.prompts import NER_PROMPT
from vector.vectorstore import Clarifai

# from annotated_text import annotated_text


# os.environ["OPENAI_API_KEY"] = "API_KEY"

st.set_page_config(
    page_title="GEOINT NER Investigation",
    page_icon="https://clarifai.com/favicon.svg",
    layout="wide",
)

# init session state
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "search_input_df" not in st.session_state:
    st.session_state.search_input_df = pd.DataFrame()

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


# def load_custom_qa_with_sources():
#     refine_template = (
#         "The original question is as follows: {question}\n"
#         "We have provided an existing answer: {existing_answer}\n"
#         "Do not remove any entities or sources from the existing answer.\n"
#         "We have the opportunity to update the list of named entities and sources\n"
#         "(only if needed) using the context below delimited by triple backticks.\n"
#         "Make sure any entities extracted are part of the context below. If not, do not add them to the list.\n"
#         "If you see any entities that are not extracted, add them.\n"
#         "The new source can be extracted at the end of the context after 'Source: '.\n"
#         "Use only the context below.\n"
#         "------------\n"
#         "{context_str}\n"
#         "------------\n"
#         "Given the new context, update the original answer to extract additional entities and sources.\n"
#         "Create a more accurate list of named entities and sources.\n"
#         "If you do update it, please update the sources as well while keeping the existing sources.\n"
#         "If the context isn't useful, return the existing answer in JSON format unchanged.\n"
#         "Output in JSON format:"
#     )
#     refine_prompt = PromptTemplate(
#         input_variables=["question", "existing_answer", "context_str"],
#         template=refine_template,
#     )

#     question_template = (
#         "Context information is below. \n"
#         "---------------------\n"
#         "{context_str}"
#         "\n---------------------\n"
#         "Given the context information and not prior knowledge, "
#         "answer the question: {question}\n"
#         "Output in JSON format:"
#     )
#     question_prompt = PromptTemplate(
#         input_variables=["context_str", "question"], template=question_template
#     )

#     chain = load_qa_with_sources_chain(
#         OpenAI(
#             temperature=0, max_tokens=-1
#         ),  # model_name="gpt-3.5-turbo",  max_tokens=2000
#         chain_type="refine",
#         return_intermediate_steps=True,
#         question_prompt=question_prompt,
#         refine_prompt=refine_prompt,
#     )
#     return chain


st.markdown(
    "This will let you ask questions about the text content in your app. Make sure it's indexed with the Language-Understanding base workflow. Instead of using OpenAI embeddings we use that base workflow embeddings AND our own vector search from our API! This will collect a shortlist of the docs and then try to summarize the shortlist into one cohesive paragraph. So it's succesptible to combining lots of unrelated information that is retrieved. "
)

user_input = get_search_query_text()

if user_input:
    docs = get_clarifai_docsearch(user_input)

    with st.expander(f"{len(docs)} Docs answering from:"):
        for idx, doc in enumerate(docs):
            st.markdown(f"### Search Result: {idx+1}")
            st.write(f"**{doc.page_content}**")
            st.write(doc.metadata)
            st.text("")

    if docs != []:
        input_list = [
            {"page_content": doc.page_content, "source": doc.metadata["source"]}
            for doc in docs
        ]

        llm_chain = load_custom_llm_chain(
            prompt_template=NER_PROMPT, model_name="gpt-3.5-turbo"
        )

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

        # Measure execution time
        entities_list = parallel_process_input(llm_chain, input_list)

        combined_entities = combine_dicts(entities_list)
        entities_help = (
            "- PER (person): Refers to individuals, including their names and titles.\n"
            " - ORG (organization): Refers to institutions, companies, government bodies, and other groups.\n"
            " - LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.\n"
            " - TIME (date or year): Refers to dates, years, and other time-related expressions.\n"
            " - MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.\n"
            " - Sources (list of sources of the text)"
        )

        # Create columns for each entity type
        st.markdown("### Entities", help=entities_help)
        columns = st.columns(len(combined_entities))
        for idx, (entity_type, entity_list) in enumerate(combined_entities.items()):
            columns[idx].info(f"**{entity_type}**")
            columns[idx].json(entity_list)

        doc_selection = st.selectbox(
            "Select search result to visualize",
            [idx if idx != 0 else "-" for idx in range(len(docs) + 1)],
            index=0,
        )

        highlighted_text_list = []
        if doc_selection and doc_selection != "-":
            # # Create color dict for each entity type
            # color_dict = {}
            # for idx, entity_type in enumerate(combined_entities.keys()):
            #     color_dict[entity_type] = f"hsl({360 * idx / len(combined_entities)}, 80%, 80%)"

            # # Create highlighted text list
            # for entity, entity_list in combined_entities.items():
            #     for word in docs[doc_selection].page_content.split():
            #         for idx_2, entity_ in enumerate(entity_list):
            #             if word in entity_:
            #                 highlighted_text_list.append((entity_list[idx_2], entity, color_dict[entity]))
            #         else:
            #             highlighted_text_list.append(
            #                 (
            #                     word,
            #                     "",
            #                     "#fff",
            #                 )
            #             )

            # annotated_text(*highlighted_text_list)

            doc_selection = int(doc_selection) - 1
            st.markdown(f"### {docs[doc_selection].metadata['source']}")

            if st.button("Summarize"):
                st.write(get_summarization_output(docs, doc_selection))

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
                        retrieval_qa_chat_chain = create_retrieval_qa_chat_chain(
                            split_texts
                        )

                        if "generatedcl2" not in st.session_state:
                            st.session_state["generatedcl2"] = []

                        if "pastcl2" not in st.session_state:
                            st.session_state["pastcl2"] = []

                        def get_search_query_text():
                            input_text = st.text_input(
                                "You: ",
                                placeholder=f"Hello, what questions do you have about content in this app?",
                                key="input_clqa2",
                            )
                            return input_text

                        user_input = get_search_query_text()

                        if user_input:
                            output = retrieval_qa_chat_chain.run(user_input)
                            st.session_state.pastcl2.append(user_input)
                            st.session_state.generatedcl2.append(output)

                        if st.session_state["generatedcl2"]:
                            for i in range(
                                len(st.session_state["generatedcl2"]) - 1, -1, -1
                            ):
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
