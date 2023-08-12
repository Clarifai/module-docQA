import concurrent.futures
import json
from typing import Any, List

import pandas as pd
import requests
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.struct_pb2 import Struct
from langchain import LLMChain, PromptTemplate
from langchain.chains import AnalyzeDocumentChain, ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import ClarifaiEmbeddings
from langchain.llms import Clarifai
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Clarifai as ClarifaiDB

USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"
EMBED_USER_ID = "openai"
EMBED_APP_ID = "embed"
EMBED_MODEL_ID = "text-embedding-ada"

# EMBED_USER_ID = "clarifai"
# EMBED_APP_ID = "main"
# EMBED_MODEL_ID = "multilingual-text-embedding"

EMBED_USER_ID = "salesforce"
EMBED_APP_ID = "blip"
EMBED_MODEL_ID = "multimodal-embedder-blip-2"


def get_search_query_text():
  input_text = st.text_input(
      "Search Query: ",
      placeholder=f"Type words or sentences to search for in your app",
      key="input_clqa",
      help="This will search for similar text in your Clarifai app.",
  )
  return input_text


def is_success(response):
  if response.status.code != status_code_pb2.SUCCESS:
    return False
  return True


def is_mixed_success(response):
  if response.status.code != status_code_pb2.MIXED_STATUS:
    return False
  return True


def calculate_expected_batch_number(inputs: List[Any], batch_size: int) -> int:
  """Function to calculate the expected number of batches

    Args:
        inputs (List[Any]): List of inputs
        batch_size (int): The number of items in the batch

    Returns:
        int: expected batch number
    """
  expected_batch_nums = ((len(inputs) // batch_size)
                         if len(inputs) % batch_size == 0 else (len(inputs) // batch_size + 1))
  return expected_batch_nums


def url_to_text(url):
  try:
    response = requests.get(url)
    response.encoding = response.apparent_encoding
  except Exception as e:
    print(f"Error: {e}")
    response = None
  return response.text if response else ""


def search_with_metadata(stub, userDataObject, search_metadata_key, search_metadata_value):
  search_metadata = Struct()
  search_metadata.update({search_metadata_key: search_metadata_value})
  post_searches_response = stub.PostSearches(
      service_pb2.PostSearchesRequest(
          user_app_id=userDataObject,
          query=resources_pb2.Query(ands=[
              resources_pb2.And(input=resources_pb2.Input(data=resources_pb2.Data(
                  metadata=search_metadata)))
          ]),
      ),)

  if post_searches_response.status.code != status_code_pb2.SUCCESS:
    print(post_searches_response.status)
    raise st.error(
        Exception("Post searches failed, status: " + post_searches_response.status.description))
  else:
    return post_searches_response


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
    input_dict_list.append(input_dict)

  return input_dict_list


@st.cache_data(persist=True)
def get_clarifai_docsearch(user_input, number_of_docs):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = create_stub(auth)
  userDataObject = auth.get_user_app_id_proto()

  docsearch = ClarifaiDB(
      user_id=userDataObject.user_id,
      app_id=userDataObject.app_id,
      pat=auth._pat,
      number_of_docs=number_of_docs)

  print("Searching for: %s" % user_input)
  docs = docsearch.similarity_search(user_input)
  return docs


# Function to load custom llm chain
def load_custom_llm_chain(prompt_template, model_name):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm_chatgpt = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  prompt = PromptTemplate(template=prompt_template, input_variables=["page_content"])
  llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)
  return llm_chain


@st.cache_data(
    persist=True, hash_funcs={
        list: lambda l: "".join([str(x.page_content) for x in l])
    })
def get_embeddings(pat, documents):
  embedder = ClarifaiEmbeddings(
      pat=pat, user_id=EMBED_USER_ID, app_id=EMBED_APP_ID, model_id=EMBED_MODEL_ID)
  texts = [doc.page_content for doc in documents]
  embeddings = embedder.embed_documents(texts)
  return embeddings


def create_retrieval_qa_chat_chain(split_texts):
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  documents = text_splitter.create_documents(split_texts)
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  texts = [doc.page_content for doc in documents]
  embedder = ClarifaiEmbeddings(
      pat=pat, user_id=EMBED_USER_ID, app_id=EMBED_APP_ID, model_id=EMBED_MODEL_ID)
  embeddings = get_embeddings(pat, documents)

  blah = zip(texts, embeddings)
  vectorstore = FAISS.from_embeddings([a for a in blah], embedder)

  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
  retrieval_qa_chat_chain = ConversationalRetrievalChain.from_llm(
      Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID),
      vectorstore.as_retriever(),
      memory=memory,
      chain_type="stuff",
      return_source_documents=False,
      get_chat_history=lambda h: h)
  return retrieval_qa_chat_chain


# Function that gets the texts and stitches them to create a full text from Clarifai app
@st.cache_data(persist=True, hash_funcs={list: lambda l: "".join([str(x) for x in l])})
def get_full_text(docs, doc_selection):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = create_stub(auth)
  userDataObject = auth.get_user_app_id_proto()
  print("Searching for: %s" % docs[doc_selection].metadata["source"])
  post_searches_response = search_with_metadata(
      stub,
      userDataObject,
      search_metadata_key="source",
      search_metadata_value=docs[doc_selection].metadata["source"],
  )
  search_input_df = pd.DataFrame(process_post_searches_response(post_searches_response))
  search_input_df = search_input_df.sort_values(["page_number", "page_chunk_number"])
  search_input_df.reset_index(drop=True, inplace=True)
  # st.session_state.search_input_df = search_input_df
  full_text = "\n".join(search_input_df.text.to_list())
  return full_text, search_input_df


# Function that gets summarization output using LLM chain
@st.cache_data(persist=True)
def get_summarization_output(full_text):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
  summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
  text_summary = summarize_document_chain.run(full_text)
  return text_summary


def combine_dicts(list_of_dicts):
  result = {}
  for dict_ in list_of_dicts:
    for key, value in dict_.items():
      result.setdefault(key, set()).update(value)
  return {k: sorted(list(v)) for k, v in result.items()}


def get_openai_output(_llm_chain, input):
  chat_output = _llm_chain(input)
  return chat_output


@st.cache_data(persist=True)
def extract_entities(llm_output):
  if isinstance(llm_output, dict) and len(llm_output) == 6:
    return llm_output
  elif isinstance(llm_output, str):
    entity_dict = {}
    llm_output = llm_output.lower().strip()
    if "output" in llm_output[:20]:
      llm_output = llm_output.split("output:")[1].strip()
    try:
      entity_dict = json.loads(llm_output)
    except Exception as e:
      st.error(f"output: {llm_output}\nerror: {e}")
    return entity_dict


@st.cache_data(persist=True)
def process_input(_llm_chain, input):
  chat_output = get_openai_output(_llm_chain, input["page_content"])
  entities = extract_entities(chat_output["text"])

  if entities == {}:
    return entities
  else:
    entities["SOURCES"] = [input["source"]]
  return entities


@st.cache_data(persist=True)
def parallel_process_input(_llm_chain, input_list):
  entities_list = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_entities = [executor.submit(process_input, _llm_chain, input) for input in input_list]
    for future in concurrent.futures.as_completed(future_entities):
      entities = future.result()
      entities_list.append(entities)
  return entities_list
