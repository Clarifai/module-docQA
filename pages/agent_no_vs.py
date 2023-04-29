from __future__ import annotations

import streamlit as st
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.llms import OpenAI

# docsearch = Clarifai.from_documents(texts, embeddings, collection_name="state-of-union")

st.title("Prompt engineering")
st.markdown("This demo shows how flexible but also critical getting the right prompt is.")

input_text = st.text_input(
    "You: ",
    placeholder=f"Ask a question about calling the Clarifai API with data?",
    key="input_clqa")

template = """Question: {question}

Answer: Let's think step by step."""

# # Gets it when it's quoted.
# template = """Question: {question}

# Answer: parse out the data that is being requested to send to the API.
# Output: the parsed out data."""

# template = """You are an agent that writes python code for the streamlit package. Always import
# streamlit as st. Given a description of a user interface, return the code that would allow streamlit
# to make the user interface. Here are some examples:

# EXAMPLE 1:
# ===================
# Description: add a text box for credit card information.
# Data: dog.jpeg
# Model: general
# Workflow:
# ===================

# """

template = """Please extract the data and specific model or workflow from the question that is provided. Make sure the output is formatted with new lines. Here are some
examples:

EXAMPLE 1:
===================
Question: send dog.jpeg to the general model.
Data: dog.jpeg
Model: general
Workflow:
===================

EXAMPLE 2:
===================
Question: send I love you too to the moderation model.
Data: I love you too
Model: moderation>
Workflow:
===================

EXAMPLE 3:
===================
Question: use the logo model on something.png
Data: something.png
Model: logo
Workflow:
===================

EXAMPLE 4:
===================
Question: run demographics workflow on https://samples.clarifai.com/metro-north.jpg
Data: https://samples.clarifai.com/metro-north.jpg
Model:
Workflow: demographics
===================

Question: {question}
Data: """

prompt = PromptTemplate(template=template, input_variables=["question"])

st.write("Prompt that will be used")
with st.expander("prompt"):
  st.write(prompt.template)

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')

llm_chain = LLMChain(prompt=prompt, llm=llm)

output = llm_chain.run(input_text)

print(output)
st.write("Answer:")
output = "Data: " + output
st.write(output)
# for o in output.split('\n'):
#   st.write(o)
