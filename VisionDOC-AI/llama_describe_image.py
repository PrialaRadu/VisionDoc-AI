import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


llm = ChatOllama(model="gemma3", temperature=0)
output_parser = StrOutputParser()


def convert_to_base64(pil_image: Image.Image) -> str:
    """
    Converts a PIL Image to a base64 encoded string.
    param: PIL.Image (image in PIL format)

    returns: base64 encoded string
    """
    with BytesIO() as buffer:
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prompt_func(data: dict) -> list[HumanMessage]:
    """
    Converts data information into an easy-to-understand message
    """
    return [HumanMessage(content=[
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{data['image']}"},
        {"type": "text", "text": data["text"]}
    ])]


def get_chain():
    return prompt_func | llm | output_parser


chain = get_chain()


def get_description_llama(file_path: str) -> str:
    """
    Creates a llama description of an image:
    param: file_path (the path to the image file)

    returns: the llama description for the image
    """
    # Converts the image into base64
    with Image.open(file_path) as img:
        img = img.convert("RGB")  # Ensure consistency
        image_b64 = convert_to_base64(img)

    # Invokes the chain using prompt text and image in base64
    return chain.invoke({
        "text": (
            "Describe the image in about 50 words. Keep the focus on the main object."
            "Do not include phrases like 'This image shows'. Respond with the description only."
        ),
        "image": image_b64
    })