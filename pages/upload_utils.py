from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import ParseDict
from stqdm import stqdm


def post_texts(st, stub, userDataObject, texts, metas):

    assert len(texts) == len(metas)

    batch_size = 16

    for i in stqdm(range(0, len(texts), batch_size), desc="Chunking pdf into text inputs"):
        minibatch = texts[i : i + batch_size]
        minimetas = metas[i : i + batch_size]
        inputs = []
        for j, t in enumerate(minibatch):
            inp = resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=t)))
            ParseDict(minimetas[j], inp.data.metadata)
            inputs.append(inp)

        request = service_pb2.PostInputsRequest(user_app_id=userDataObject, inputs=inputs)

        response = stub.PostInputs(request)
        # print(response)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("PostInputs request failed: %r" % response)

    return response
