# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
import json
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def load_documents_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for item in data:
        text = item.get("metadata", {}).get("nearby_text", "")
        image_path = item.get("image_path", "")
        if not text.strip():
            continue
        doc = Document(page_content=text, metadata={"image_path": image_path})
        docs.append(doc)

    return docs

def get_images_from_vector_store(query, k=1):
    docs = load_documents_from_json("porsche_2006.json")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(docs, embeddings)
    results = vector_store.similarity_search(query, k=k)
    return [(doc.metadata.get("image_path", "UNKNOWN"), doc.page_content) for doc in results]

def get_image(query):
    loader = JSONLoader(
        file_path="porsche_2006.json",
        jq_schema='.[] | {text: (.metadata.nearby_text + "\n" + .metadata.description), metadata: {image_path: .image_path}}',
        text_content=False
    )
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("vector_store_index")
    vector_store.load_local("vector_store_index", embeddings, allow_dangerous_deserialization=True)
    results = vector_store.similarity_search(query, k=1)

    pth = ''
    description = ''
    for doc in results:
        content = json.loads(doc.page_content)
        description = content['text']
        pth = content['metadata']['image_path']

    return pth, description


if __name__ == "__main__":
    pass