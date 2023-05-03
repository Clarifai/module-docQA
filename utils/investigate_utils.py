import requests
from io import BytesIO
import pandas as pd
import streamlit as st
from tqdm.notebook import tqdm
from google.protobuf.json_format import MessageToJson, MessageToDict
from typing import Any, List, Tuple, Optional, Dict, Iterator, Iterable
from pathlib import Path
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed


## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
from clarifai_grpc.grpc.api.service_pb2 import GetInputCountRequest, StreamInputsRequest

from google.protobuf.struct_pb2 import Struct


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
    expected_batch_nums = (
        (len(inputs) // batch_size) if len(inputs) % batch_size == 0 else (len(inputs) // batch_size + 1)
    )
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
            query=resources_pb2.Query(
                ands=[
                    resources_pb2.And(
                        input=resources_pb2.Input(data=resources_pb2.Data(metadata=search_metadata))
                    )
                ]
            ),
        ),
    )

    if post_searches_response.status.code != status_code_pb2.SUCCESS:
        print(post_searches_response.status)
        raise st.error(
            Exception("Post searches failed, status: " + post_searches_response.status.description)
        )
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
