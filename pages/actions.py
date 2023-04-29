from __future__ import annotations

import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.llms import OpenAI
from langchain.tools.python.tool import PythonREPLTool

# docsearch = Clarifai.from_documents(texts, embeddings, collection_name="state-of-union")

input_text = st.text_area(
    "You: ",
    placeholder=f"Hello, what questions do you have about content in this app?",
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

# template = """Please extract the data and specific model or workflow from the question that is provided. Make sure the output is formatted with new lines. Here are some
# examples:

# EXAMPLE 1:
# ===================
# Question: send dog.jpeg to the general model.
# Data: dog.jpeg
# Model: general
# Workflow:
# ===================

# EXAMPLE 2:
# ===================
# Question: send I love you too to the moderation model.
# Data: I love you too
# Model: moderation
# Workflow:
# ===================

# EXAMPLE 3:
# ===================
# Question: use the logo model on something.png
# Data: something.png
# Model: logo
# Workflow:
# ===================

# EXAMPLE 4:
# ===================
# Question: run demographics workflow on https://samples.clarifai.com/metro-north.jpg
# Data: https://samples.clarifai.com/metro-north.jpg
# Model:
# Workflow: demographics
# ===================

# Question: {question}
# Data: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')


def my_run(inp):

  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = create_stub(auth)
  userDataObject = auth.get_user_app_id_proto()

  if "," not in inp:
    return "the data and the model or workflow should be split by a comma"
  # st.write("XXXXXXXXXXXXXXXXXXXXXXXXXXX Input to the action:")
  # st.markdown(inp)
  # print(inp)
  data = inp.split('data:')[1].split(',')[0]
  # st.write(data)
  if "model:" in inp:
    model_id = inp.split(',')[1].split('model:')[1]
    # st.write("MMMMMMMMMMMMMMMMMMMMM")
    # st.write(inp.split(',')[1].split('model:'))
    # st.write(model_id)
  if "data:" not in inp:
    return "The data: was not part of the action input. Please parse the question to get the data."
    # raise ValueError(
    #     "The data: was not part of the action input. Please parse the question to get the data.")
  if "model:" not in inp and "workflow:" not in inp:
    return "The model: and workflow: were not part of the action input. Please retry."
  # if "workflow:" not in inp:
  #   return "The workflow: was not part of the action input. Please parse the question to get the workflow_id."

  #   raise ValueError(
  #       "The model: was not part of the action input. Please parse the question to get the model_id."
  #   )
  # if "workflow:" not in inp:
  #   raise ValueError(
  #       "The data: was not part of the action input. Please parse the question to get the workflow_id."
  #   )

  request = service_pb2.PostModelOutputsRequest(
      user_app_id=userDataObject,
      # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
      model_id=model_id,
      inputs=[resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=data)))])
  # if version_id is not None:
  #   request.version_id = version_id

  response = stub.PostModelOutputs(request)
  # print(response)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelOutputs request failed: %r" % response)
  j = json_format.MessageToJson(response, preserving_proto_field_name=True)
  st.write("API response in intermediate step:")
  st.json(j)

  j2 = json_format.MessageToJson(response.outputs[0].data, preserving_proto_field_name=True)
  return j2


# Putting into the description of the tools
tools = [
    Tool(
        name="Clarifai",
        func=my_run,
        description=
        "A useful tool for interacting with the Clarifai API to send data to models or workflows. Input should have the data from the original query parsed out and provided as data:image.jpeg as an example. The model or workflow from the original query should also be parsed out and passed in as model:model_id or workflow: workflow_id. So the finally input should be data:data,model:model_id,workflow:workflow_id where data, model_id and workflow_id are parsed from the question. "
    ),
    PythonREPLTool()
]
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# output = agent.run(input_text)

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""

# PREFIX = """You are an agent designed to write and execute python code to answer questions.
# You have access to a python REPL, which you can use to execute python code.
# If you get an error, debug your code and try again.
# Only use the output of your code to answer the question.
# You might know the answer without running any code, but you should still run the code to get the answer.
# If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
# """

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad"])

st.write("This is the overall prompt:")
print(prompt.template)
st.markdown(prompt.template)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
output = agent_executor.run(input_text)

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# output = llm_chain.run(input_text)

st.write(output)
# for o in output.split('\n'):
#   st.write(o)
