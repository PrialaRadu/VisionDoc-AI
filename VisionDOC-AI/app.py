# app.py
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from src.utils import setup_dbqa
from db_build import get_image, build_vector_index
from role_access.access_permissions import access
from src.prompts import qa_template
from src.llm import build_llm
from PIL import Image

# Load env
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(page_title="VisionDOC-AI", layout="centered")

# Load LLM pipeline
@st.cache_resource
def load_dbqa():
    return setup_dbqa(qa_template, build_llm)

dbqa = load_dbqa()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("🤖 VisionDOC-AI")
st.markdown("Ask questions about documents and get relevant images and answers — like chatting with your document!")

# Chat history displayed at the top
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["query"])

    with st.chat_message("assistant"):
        st.image(Image.open(msg["image_path"]), caption="Image found", use_container_width=True)
        st.markdown(f"🖹 **Filename:** {msg['filename']}")
        if msg["filename"].endswith(".pdf"):
            st.markdown(f"🖹 **Page:** {msg['page']}")
        st.markdown(f"🖹 **Description:** {msg['description'].splitlines()[-1][:300]}")
        st.markdown(msg["response"])

# User input (chat input is at the bottom)
query = st.chat_input("What would you like to know?")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🔍 Searching relevant image..."):
        path, description, filename, page = get_image(query)

    # Display assistant response
    with st.chat_message("assistant"):
        st.image(Image.open(path), caption="Image found", use_container_width=True)
        st.markdown(f"🖹 **Filename:** {filename}")
        if filename.endswith(".pdf"):
            st.markdown(f"🖹 **Page:** {page}")
        st.markdown(f"🖹 **Description:** {description.split('\n')[0]}")
        prompt = (
            f"The user asked: '{query}'.\n"
            f"Here’s the description you can use: '{description.split('\n')[0]}'.\n\n"
            f"Write a descriptive response that explains what the image shows. "
            f"Start the response with a phrase like 'The image you asked for is'. "
            f"Make sure the explanation is clear, informative, and not just a label.\n"
        )

        with st.spinner("Generating LLM response..."):
            response = dbqa.invoke({'query': prompt})
            assistant_response = response['result']
        st.markdown(f"🤖: {assistant_response}")
