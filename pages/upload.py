"""Python file to serve as the frontend"""
import pandas as pd
import PyPDF2
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub

from utils.upload_utils import post_texts, split_into_chunks, word_counter

st.set_page_config(page_title="Upload App", page_icon=":robot:")

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
st.title("Upload PDF as text chunks")
st.markdown(
    "This will chunk up the PDF into text pages and upload them to our platform. This also fills in the `data.metadata.source = {name of document}`\
        and `data.metadata.page_number = {page number of document}`."
)


text_chunk_size = st.number_input(
    "Text chunk size", min_value=100, max_value=3000, value=500, step=100
)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="qapdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    try:
        document_title = reader.metadata.title.split(".")[0]
    except AttributeError:
        document_title = uploaded_file.name.split(".")[0]

    text_chunks = []
    page_number_list = []
    page_chunk_number_list = []
    prev_page_text = ""
    for page_idx, page in enumerate(reader.pages):
        print("default page_idx: ", page_idx)

        # if page_idx == 6:
        #     break

        current_page_text = page.extract_text()
        page_text = prev_page_text + current_page_text

        # Check if text is smaller the text_chunk_size, if so, add it to the previous page text holder
        if (word_counter(page_text) < text_chunk_size) and (
            page_idx != len(reader.pages) - 1
        ):
            prev_page_text += current_page_text
            continue
        else:
            prev_page_text = ""

        print(page_text)
        page_text_chunks = split_into_chunks(page_text, text_chunk_size)
        text_chunks.extend(page_text_chunks)
        page_number_list.extend([page_idx] * len(page_text_chunks))
        page_chunk_number_list.extend([idx for idx in range(len(page_text_chunks))])
        print("length of text chunks", len(text_chunks))

    # # Save list to a txt file for debugging
    # with open(f"{uploaded_file_name}.txt", "w") as f:
    #     for item in text_chunks:
    #         f.write("%s\n" % item)

    print("len(text_chunks): ", len(text_chunks))
    print("len(page_number_list): ", len(page_number_list))
    assert len(text_chunks) == len(
        page_number_list
    ), "Text chunks and page numbers should be the same length"
    metadata_list = [
        {
            "source": f"{document_title}",
            "text_length": word_counter(text_chunks[idx]),
            "page_number": page_number_list[idx],
            "page_chunk_number": page_chunk_number_list[idx],
        }
        for idx in range(len(text_chunks))
    ]
    print(text_chunks[-1])
    print(metadata_list[-1])

    metadata_df = pd.DataFrame(metadata_list)
    metadata_sorted_df = metadata_df.sort_values(["page_number", "page_chunk_number"])
    st.dataframe(metadata_sorted_df)

    post_texts(st, stub, userDataObject, text_chunks, metadata_list)
