from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.client.input import Inputs
from google.protobuf.json_format import ParseDict
from stqdm import stqdm
import hashlib


def post_texts_with_geo(
    st, stub, userDataObject, text_list, metadata_list, geo_points_list
):
    assert len(text_list) == len(metadata_list)

    batch_size = 32

    for chunking_idx in stqdm(
        range(0, len(text_list), batch_size), desc="Chunking pdf into text inputs"
    ):
        text_batch = text_list[chunking_idx : chunking_idx + batch_size]
        metadata_batch = metadata_list[chunking_idx : chunking_idx + batch_size]
        geo_points_batch = geo_points_list[chunking_idx : chunking_idx + batch_size]

        inputs = []
        for idx, text in enumerate(text_batch):
            input_obj = Inputs(logger_level="ERROR")
            id = hashlib.md5(text.encode("utf-8")).hexdigest()
            input_protos = input_obj.get_text_input(
                input_id=id,
                raw_text=text,
                geo_info=[geo_points_batch[idx]["lon"],
                          geo_points_batch[idx]["lat"]
                          ]
             )
            
            ParseDict(metadata_batch[idx], input_protos.data.metadata)
            inputs.append(input_protos)
        
        #SDK (but returns only jobID)
        input_obj = Inputs(logger_level="ERROR",
                           user_id=userDataObject.user_id,
                           app_id = userDataObject.app_id)
        
        post_inputs_request=input_obj.upload_inputs(inputs)

    #return input_protos


def post_texts(st, stub, userDataObject, text_list, metadata_list):
    assert len(text_list) == len(metadata_list)
    batch_size = 32
    for chunking_idx in stqdm(
        range(0, len(text_list), batch_size), desc="Chunking pdf into text inputs"
    ):
        text_batch = text_list[chunking_idx : chunking_idx + batch_size]
        metadata_batch = metadata_list[chunking_idx : chunking_idx + batch_size]

        inputs = []
        for idx, text in enumerate(text_batch):
            input_obj = Inputs(logger_level="ERROR")
            id = hashlib.md5(text.encode("utf-8")).hexdigest()
            input_protos = input_obj.get_text_input(
                input_id=id,
                raw_text=text
            )

            inp = resources_pb2.Input(
                data=resources_pb2.Data(
                    text=resources_pb2.Text(raw=text),
                )
            )

            ParseDict(metadata_batch[idx], input_protos.data.metadata)
            inputs.append(input_protos)

        #SDK (but returns only jobID, although function is not widely used)
        input_obj = Inputs(logger_level="ERROR",
                           user_id=userDataObject.user_id, 
                           app_id = userDataObject.app_id)
        
        post_inputs_request=input_obj.upload_inputs(inputs)
    #return input_protos

def word_counter(text):
    return len(text.split())


def split_into_chunks(string, text_chunk_size):
    words = string.split()
    chunks = []
    chunk = ""
    
    # Split into chunks
    for idx, word in enumerate(words):
        if idx % text_chunk_size == 0 and idx > 0:
            chunks.append(chunk.strip())
            chunk = ""
        chunk += word + " "
    
    # Add last chunk to list by concatenating with last chunk in the list
    if chunk and len(chunks) > 0:
        chunks[-1] += chunk.strip()
        # chunks.append(chunk.strip())
        
    # If there's only one chunk, return list with one chunk
    elif chunk:
        chunks.append(chunk.strip())
    return chunks
