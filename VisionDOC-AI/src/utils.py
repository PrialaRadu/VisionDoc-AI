'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from src.prompts import qa_template
from src.llm import build_llm

def load_config():
    with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
        return box.Box(yaml.safe_load(ymlfile))

def set_qa_prompt(template_str):
    return PromptTemplate(template=template_str, input_variables=['context', 'question'])

def build_retrieval_qa(llm, prompt, vectordb, vector_count, return_sources):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k': vector_count}),
        return_source_documents=return_sources,
        chain_type_kwargs={'prompt': prompt}
    )

def setup_dbqa(qa_template, build_llm_func):
    cfg = load_config()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectordb = FAISS.load_local(
        cfg.DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = build_llm_func()
    prompt = set_qa_prompt(qa_template)

    return build_retrieval_qa(
        llm,
        prompt,
        vectordb,
        vector_count=cfg.VECTOR_COUNT,
        return_sources=cfg.RETURN_SOURCE_DOCUMENTS
    )
