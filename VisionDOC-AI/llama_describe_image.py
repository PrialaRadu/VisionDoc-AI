import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


llm = ChatOllama(model="gemma3", temperature=0)
output_parser = StrOutputParser()


def convert_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to base64 JPEG."""
    with BytesIO() as buffer:
        pil_image.save(buffer, format="JPEG", quality=85)  # ⬅️ lower quality saves memory & is faster
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prompt_func(data: dict) -> list[HumanMessage]:
    """Build the message payload for the image + prompt."""
    return [HumanMessage(content=[
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{data['image']}"},
        {"type": "text", "text": data["text"]}
    ])]


# 🔁 2. Reuse chain setup
def get_chain():
    return prompt_func | llm | output_parser


chain = get_chain()


# ✅ 3. Efficient wrapper
def get_description_llama(file_path: str) -> str:
    """Generate image description from file path."""
    with Image.open(file_path) as img:
        img = img.convert("RGB")  # Ensure consistency
        image_b64 = convert_to_base64(img)

    return chain.invoke({
        "text": (
            "Describe this image in about 25 words. Be specific and concise. Keep the description on the main object"
            "Do not include phrases like 'This image shows'. Respond with the description only."
        ),
        "image": image_b64
    })