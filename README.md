# Module Geoint
This is a demo app showcasing the usefullness of Retrieval Augmented Text Generation. This module includes 4 main pages:
1. **Upload**: This page takes a pdf, chuncks it and uploads it to a Clarifai app
2. **Upload with** Geo: This page takes a pdf, chuncks it, extracts the location using a LLM and uploads it the the text and geo data to a Clarifai app
3. **Investigate**: This page uses Clarifai semantic search to retrieve relevant documents. The retrieved documents are then used to ground the LLM generation for tasks such as NER, Summarization and Chat with PDF
4. **Geo Search**: This page uses Clarifai semantic and geo data search to retrieve relevant document within a designated geo location. The retrieved documents are then used to ground the LLM generation for a summarization task.
