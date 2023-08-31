# Module Document Q&A
This is a demo app showcasing the usefullness of Retrieval Augmented Text Generation. This module includes 4 main pages:
1. **Upload**: This page takes a pdf, parses it into chunks, and uploads it to a Clarifai app with metadata to track the source and page/chunk number.
2. **Upload with Geo**: This page takes a pdf, parses it into chunks, then extracts relevant locations using a LLM and uploads the text chunks and geo data to a Clarifai app.
3. **Investigate**: This page showcases different use cases such as semantic search for document retrieval, followed with NER (named-entity recognition), document summarization, and a "chat with the document" experience.
4. **Geo Search**: This page uses Clarifai semantic search and geo data search to retrieve relevant documents within a designated geo location, which can then used to ground the LLM generation for a summarization task.
