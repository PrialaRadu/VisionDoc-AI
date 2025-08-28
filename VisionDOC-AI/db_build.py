# =========================
#  Module: Vector DB Build
# =========================
import os

import box
import yaml
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_vector_index(file):
    """
        Builds a vector index from a file name.
        param: file (name of the document file)
        """
    # Prepares nearby_text and description for the vector database
    loader = JSONLoader(
        file_path=f"extraction/data/{file}/metadata.json",
        jq_schema='.[] | {text: (.description + "\n" + .nearby_text + "\n" + .filename), metadata: {image_path: .image_path, filename: .filename}}',
        text_content=False
    )
    docs = loader.load()
    # Prepares the embeddings for sentence transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore_path = "vectorstore/db_faiss"

    # Check if the FAISS DB already exists
    if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        # Load existing vector store and add new docs
        vectordb = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
    else:
        # Create new vector store
        vectordb = FAISS.from_documents(docs, embeddings)

    # Save updated vector store
    vectordb.save_local(vectorstore_path)

def index_all_documents():
    files = os.listdir('extraction/data')
    for file in files:
        build_vector_index(file)

class ImageRetriever:
    def __init__(self, vectorstore_path="vectorstore/db_faiss"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore_path = vectorstore_path
        self.vector_store = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_image(self, query):
        """
        Receives a user query and retrieves the best matching image info.
        Returns a tuple: (image_path, description, filename, page)
        """
        results = self.vector_store.similarity_search(query, k=1)

        pth = ''
        description = ''
        filename = ''
        page = 0

        for doc in results:
            content = json.loads(doc.page_content)
            description = content['text']
            pth = content['metadata']['image_path']
            filename = content['metadata']['filename']

            if filename.endswith('.pdf'):
                # Extract page number from path assuming a naming convention
                page = int(pth.split('\\')[-1].split('_')[0][-1])
            else:
                page = 0

        return pth, description, filename, page


if __name__ == "__main__":
    index_all_documents()