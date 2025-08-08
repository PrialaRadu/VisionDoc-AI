# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
import json

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)


def get_image(query):
    loader = JSONLoader(
        file_path="data2.json",
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


# def llama_chat():
#     llm = LlamaCpp(
#         model_path="models/llama-2-7b-chat.ggmlv3.q4_1.bin",
#         temperature=0.7,
#         max_tokens=512,
#         top_p=1
#     )
#
#     template = ""
#     prompt = PromptTemplate(template=template, input_variables=["text"])
#     chain = LLMChain(prompt=prompt, llm=llm)
#     start_time = monotonic()
#     # desc = chain.run(docs)
#     print(f"Run time: {monotonic() - start_time} seconds")
#     # print(desc)
#     return None


if __name__ == "__main__":
    query = "Show me the image with the woman and the kid"
    get_image(query)
