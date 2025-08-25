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
    dbqa = setup_dbqa(qa_template, build_llm)
    role = access()
    file = input("Enter document name (e.g.: 'Cayenne_Turbo_2006.pdf'): ")
    while True:
        query = input("Enter your image query: ")
        if query.lower() == 'exit':
            break

        build_vector_index(file)
        path, description = get_image(query, file)

        display_image(path)

        if role == 'admin':
            print(f"Description: {description.split("\n")[-1]}")
            print(f"Path: {path}")

        prompt = (
            f"The user asked: '{query}'.\n"
            f"Here's the description: '{description.split('\n')[-1]}'.\n"
            f"Craft a descriptive response using the whole description. Start your response with something like "
            f"'The description you asked for is ', or alternatively, use phrases like "
            f"'What you were looking for is ' or 'Here’s the information you need:'.\n"
        )
        print("**Retrieving LLM response...**")
        response = dbqa.invoke({'query': prompt})
        print(response['result'])
        print()


if __name__ == "__main__":
    main_program()
