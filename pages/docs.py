import glob

# import streamlit as st
# from clarifai.auth.helper import ClarifaiAuthHelper
# from clarifai.client import create_stub
from langchain.text_splitter import MarkdownTextSplitter
from stqdm import stqdm

# from pages.upload_utils import post_texts

path = '/Users/zeiler/work/other/docs/'

# auth = ClarifaiAuthHelper.from_streamlit(st)
# stub = create_stub(auth)
# userDataObject = auth.get_user_app_id_proto()

allt = {}
visited = {}
for filename in stqdm(glob.iglob(path + '**/**', recursive=True), desc="Iterating over files"):
  if filename.endswith(".md"):
    if 'changelog' in filename:
      continue
    if filename in visited:
      print(f"Already visited {filename}")
      continue
    visited[filename] = True
    print(filename)
    with open(filename) as f:
      t = f.read()
      if t in allt:
        print(f"Found this text before in {allt[t]} and currently in {filename}, skipping...")
        continue
        import pdb
        pdb.set_trace()
      allt[t] = filename
      print(len(t))
      text_splitter = MarkdownTextSplitter(chunk_size=1500)
      splits = text_splitter.split_text(t)
      print(len(splits))
      if t.find("Tabs") > 0:
        print(t)
        import pdb
        pdb.set_trace()

      metas = [{'file': filename, 'split': i} for i in range(len(splits))]
      # post_texts(st, stub, userDataObject, splits, metas)

# for root, dirs, files in os.walk("."):
#   path = root.split(os.sep)
#   # print((len(path) - 1) * '---', os.path.basename(root))
#   for file in files:
#     if file.endswith(".md"):
#       print(len(path) * '---', file)

# loader = WebBaseLoader("https://docs.clarifai.com/")
# data = loader.load()

# st.write(data)
