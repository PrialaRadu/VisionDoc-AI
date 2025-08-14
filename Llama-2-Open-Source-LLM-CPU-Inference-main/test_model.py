import torch
import clip
from PIL import Image
import os

from db_build import get_images_from_vector_store

def get_best_image_from_clip(image_input, query):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(image_input, str) and os.path.isdir(image_input):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = [os.path.join(image_input, f)
                       for f in os.listdir(image_input)
                       if os.path.splitext(f)[1].lower() in valid_exts]
    elif isinstance(image_input, list):
        image_paths = image_input
    else:
        raise ValueError("image_input trebuie sa fie o lista de cai sau un director valid.")

    if not image_paths:
        raise ValueError("Nu s-au gasit imagini valide in director.")

    images = [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_paths]
    image_input = torch.cat(images, dim=0)

    text_tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).squeeze()

    best_match_index = similarity.argmax().item()
    return image_paths[best_match_index]

def search_image(query, k_vector=1):
    image_results = get_images_from_vector_store(query, k=k_vector)
    image_paths = [img for img, _ in image_results]

    best_image = get_best_image_from_clip(image_paths, query)

    for path, description in image_results:
        if path == best_image:
            return best_image, description

    return best_image, None