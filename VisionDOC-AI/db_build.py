# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

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
        file_path=f"data/{file}/metadata.json",
        jq_schema='.[] | {text: (.nearby_text + "\n" + .description), metadata: {image_path: .image_path}}',
        text_content=False
    )
    docs = loader.load()
    # Prepares the embeddings for sentence transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.from_documents(docs, embeddings)
    # Removes any previous vectorstore
    shutil.rmtree("vectorstore/db_faiss", ignore_errors=True)
    # Saves data in a new vectorstore
    vectordb.save_local("vectorstore/db_faiss")


def get_image(query, file):
    """
    Receives the user query and retrieves the best image from the vector store, returning its path and description
    param: query (user query for interrogation)
    param: file (name of the document file)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    # Loads the local vectorstore using the embeddings
    vector_store = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    # Retrieves the result of the desired image
    results = vector_store.similarity_search(query, k=1)

    pth = ''
    description = ''
    # Retrieves the description and path of the desired image
    for doc in results:
        content = json.loads(doc.page_content)
        description = content['text']
        pth = content['metadata']['image_path']

    return pth, description


if __name__ == "__main__":
    build_vector_index('Cayenne_Turbo_2006.pdf')