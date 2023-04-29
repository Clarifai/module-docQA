"""Python file to serve as the frontend"""
import PyPDF2
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub

from pages.upload_utils import post_texts

st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
st.title("Upload PDF as text chunks")
st.markdown(
    "This will chunk up the PDF into text pages and upload them to our platform. This also fills in the data.metadata.source = to the page number."
)


def word_counter(text):
    return len(text.split())


def split_into_chunks(s, text_chunk_size):
    words = s.split()
    chunks = []
    chunk = ""
    for i, word in enumerate(words):
        if i % text_chunk_size == 0 and i > 0:
            chunks.append(chunk.strip())
            chunk = ""
        chunk += word + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks


text_chunk_size = st.number_input("Text chunk size", min_value=100, max_value=3000, value=300, step=100)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="qapdf")

if uploaded_file:

    reader = PyPDF2.PdfReader(uploaded_file)

    texts_list = []
    metadata_list = []
    text_chunks = []
    for page_idx, page in enumerate(reader.pages):
        texts_list.append(page.extract_text())
        # metadata_list.append(
        #     {
        #         "page_number": page_idx,
        #     }
        # )

    full_text = " ".join(texts_list)
    text_chunks = split_into_chunks(full_text, text_chunk_size)
    # text_chunks = [full_text[i : i + text_chunk_size] for i in range(0, len(full_text), text_chunk_size)]
    print("length of text chunks", len(text_chunks))

    # # Update metadata list
    # metadata_list = [metadata.update({
    #         "source": "Holistic - Paper",
    #         "text_length": len(text_chunks[idx]),
    #         "text_start_index": len(" ".join(text_chunks[:idx]))}) for metadata in metadata_list)]

    metadata_list = [
        {
            "source": "Tom Clancy - The Hunt for Red October",
            "text_length": word_counter(text_chunks[idx]),
            "text_start_index": word_counter(" ".join(text_chunks[:idx])),
        }
        for idx in range(len(text_chunks))
    ]

    assert len(text_chunks) * text_chunk_size >= word_counter(
        full_text
    ), "Text chunks should be able to cover the full text"

    post_texts(st, stub, userDataObject, text_chunks, metadata_list)
