
RAG pipeline --
->Indexing phase: Perform chunking on document and convert chunks to vectors--> store vectors in vector DB like pinecone.
->Retrieval phase: 
-Convert user query to vector embedding.
-Vector similarity search on user vector embeddings in vector DB.