import os
import timeit
import box
import yaml
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
from db_build import get_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from role_access.access_permissions import access
from src.prompts import qa_template
from src.llm import build_llm
from db_build import build_vector_index


# Load environment variables from .env file
load_dotenv(find_dotenv())


# Import config vars
with open('role_access/users.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def display_image(path):
    """
    Displays an image to the user.
    param: path (the path to the image file)

    returns: None
    """
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def main_program():
    start = timeit.default_timer()
    # print("   Setup DBQA...")
    dbqa = setup_dbqa(qa_template, build_llm)
    # print(f"[TIME] Setup: {timeit.default_timer() - start:.2f} seconds") if role == 'tester' else None

    files = os.listdir('extraction/data')
    for file in files:
        print(file)
        build_vector_index(file)
    print(f"[TIME] Vector Build: {timeit.default_timer() - start:.2f} seconds")
    while True:
        query = input("Enter your image query: ")
        if query.lower() == 'exit':
            break

        start = timeit.default_timer()
        path, description, filename, page = get_image(query)
        print(f"[TIME] Image Retrieval: {timeit.default_timer() - start:.2f} seconds")

        display_image(path)

        print(f"Description: {description.split("\n")[-1]}")
        print(f"Path: {path}")
        print(f"Filename: {filename}")
        print(f"Page: {page}") if page != 0 else None

        prompt = (
            f"The user asked: '{query}'.\n"
            f"Here’s the description you can use: '''{description}'''.\n\n"
            f"Write a descriptive response (around 40 words) that explains what the image shows. "
            f"Start the response with a phrase like 'The image you asked for is', 'What you were looking for is', or 'Here’s the information you need:'. "
            f"Make sure the explanation is clear, informative, and not just a label.\n"
            f"After that, mention where the image was found: either just the filename if page = 0, or 'filename, on page X' otherwise.\n\n"
            f"The filename is: {filename}\n"
            f"The page is: {page}"
        )
        print("   Retrieving LLM response...")
        start = timeit.default_timer()
        response = dbqa.invoke({'query': prompt})
        print(response['result'])
        print(f"[TIME] LLM Response: {timeit.default_timer() - start:.2f} seconds")
        print()


if __name__ == "__main__":
    main_program()
