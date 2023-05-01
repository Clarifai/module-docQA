from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import ParseDict
from stqdm import stqdm


def post_texts(st, stub, userDataObject, text_list, metadata_list):

    assert len(text_list) == len(metadata_list)

    batch_size = 32

    for chunking_idx in stqdm(range(0, len(text_list), batch_size), desc="Chunking pdf into text inputs"):
        text_batch = text_list[chunking_idx : chunking_idx + batch_size]
        metadata_batch = metadata_list[chunking_idx : chunking_idx + batch_size]
        inputs = []
        for idx, text in enumerate(text_batch):
            inp = resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))
            ParseDict(metadata_batch[idx], inp.data.metadata)
            inputs.append(inp)

        request = service_pb2.PostInputsRequest(user_app_id=userDataObject, inputs=inputs)

        response = stub.PostInputs(request)
        # print(response)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("PostInputs request failed: %r" % response)

    return response


def word_counter(text):
    return len(text.split())


def split_into_chunks(s, text_chunk_size):
    words = s.split()
    chunks = []
    chunk = ""
    for idx, word in enumerate(words):
        if idx % text_chunk_size == 0 and idx > 0:
            chunks.append(chunk.strip())
            chunk = ""
        chunk += word + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks
