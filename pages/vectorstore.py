from __future__ import annotations

import os
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
import streamlit as st


class Clarifai(VectorStore):
    """Wrapper around Clarifai AI platform's vector store.

    To use, you should have the ``clarifai`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Clarifai
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Clarifai("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        user_id: str,
        app_id: str,
        pat: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize with Clarifai client."""
        try:
            pass
        except ImportError:
            raise ValueError(
                "Could not import clarifai python package. " "Please install it with `pip install clarifai`."
            )

        if pat is None:
            if "CLARIFAI_PAT" not in os.environ:
                raise ValueError(
                    "Could not find CLARIFAI_PAT in your environment. "
                    "Please set that env variable with a valid personal access token from https://clarifai.com/settings/security."
                )
            pat = os.environ["CLARIFAI_PAT"]

        from clarifai.auth.helper import ClarifaiAuthHelper, DEFAULT_BASE
        from clarifai.client import create_stub

        if api_base is None:
            api_base = DEFAULT_BASE

        auth = ClarifaiAuthHelper(user_id=user_id, app_id=app_id, pat=pat, base=api_base)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        self._stub = stub
        self._auth = auth
        self._userDataObject = userDataObject

        # # Check if the collection exists, create it if not
        # if collection_name in [col.name for col in self._client.list_collections()]:
        #     self._collection = self._client.get_collection(name=collection_name)
        # else:
        #     # create app in the user_id account.
        #     self._collection = self._client.create_collection(
        #         name=collection_name,
        #         embedding_function=self._embedding_function.embed_documents
        #         if self._embedding_function is not None
        #         else None,
        #     )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.add(metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Clarifai.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most simmilar to the query text.
        """
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2
        from google.protobuf import json_format
        import requests

        # traceback.print_stack()
        post_annotations_searches_response = self._stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                user_app_id=self._userDataObject,
                searches=[
                    resources_pb2.Search(
                        query=resources_pb2.Query(
                            ranks=[
                                resources_pb2.Rank(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            text=resources_pb2.Text(raw=query),
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=4),
            )
        )

        if post_annotations_searches_response.status.code != status_code_pb2.SUCCESS:
            # st.json(
            #     json_format.MessageToJson(
            #         post_annotations_searches_response, preserving_proto_field_name=True))
            raise Exception(
                "Post searches failed, status: " + post_annotations_searches_response.status.description
            )

        print("Search result:")
        hits = post_annotations_searches_response.hits
        st.write("Found %d documents from API" % len(hits))

        docs = []
        for hit in hits:
            # Only return results with a score above 0.85
            if hit.score < 0.85:
                break
            metadata = json_format.MessageToDict(hit.input.data.metadata)
            t = requests.get(hit.input.data.text.url).text
            # TODO(zeiler): generalize this to return the whole metadata.
            if "source" in metadata:
                source = str(metadata.get("source"))
                text_start_index = str(metadata.get("text_start_index"))

            else:
                source = hit.input.id
                text_start_index = "NA"
            print(
                "\tScore %.2f for annotation: %s off input: %s, source: %s, t: %s"
                % (hit.score, hit.annotation.id, hit.input.id, source, t[:125])
            )
            docs.append(
                Document(
                    page_content=t, metadata={"source": f"{source} - {text_start_index}", "score": hit.score}
                )
            )

        return docs

    def delete_collection(self) -> None:
        """Delete the collection."""
        raise NotImplementedError("todo")
        self._client.delete_collection(self._collection.name)

    def persist(self) -> None:
        """Persist the collection.

        This can be used to explicitly persist the data to disk.
        It will also be called automatically when the object is destroyed.
        """
        raise NotImplementedError("This does not apply to Clarifai")
        if self._persist_directory is None:
            raise ValueError("You must specify a persist_directory on" "creation to persist the collection.")
        self._client.persist()

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            documents (List[Document]): List of documents to add.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        raise NotImplementedError("not yet ready")
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        raise NotImplementedError("not yet ready")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
