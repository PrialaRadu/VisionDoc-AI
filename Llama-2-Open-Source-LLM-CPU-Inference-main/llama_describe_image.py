import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=100)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def get_description_llama(file_path):
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)

    llm = ChatOllama(model="gemma3:4b", temperature=0)
    chain = prompt_func | llm | StrOutputParser()
    query_chain = chain.invoke(
        {"text": "Give me a fast image description in about 50 words. Respond only with the description, without your input", "image": image_b64}
    )

    return query_chain