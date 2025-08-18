# VISIONDOC AI
A GenAI system that interprets and semantically links embedded images within narrative documents for visual question answering and retrieval.

1.	Accepts input formats: PDF, DOCX.
2.	Uses a vision-language model (GEMMA3) to extract semantic meaning from images.
3.	Implements chunking logic that binds pre-image and post-image text with image metadata.
4.	Stores enriched chunks in a vector store (FAISS).
5.	Enables semantic search and retrieval using vector similarity.
6.	Builds a chatbot using retrieval-augmented generation (RAG) over vector data.
7.	Ensures chunk provenance (page, image position) is preserved.
8.	Implements access control and role-based user permissions.
