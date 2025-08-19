import box
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import timeit
from db_build import get_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from image_extraction import retrieve_images
from access_permissions import access



# Load environment variables from .env file
load_dotenv(find_dotenv())


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def main_program():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='Retrieve the image with the car interior',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()
    start = timeit.default_timer()

    role = access()

    dbqa = setup_dbqa()
    print(f"Step 0 - Setup DBQA: {timeit.default_timer() - start:.2f} seconds") if role == 'admin' else None

    # retrieve_images("data/Porsche_US Cayenne_Turbo_2006.pdf", "porsche_2006")
    # print(f"Step 1 - Extract Images: {timeit.default_timer() - start:.2f} seconds") if role == 'admin' else None

    if "image" in args.input.lower():
        path, description = get_image(args.input)
        print(f"Step 2 - Search Image: {timeit.default_timer() - start:.2f} seconds") if role == 'admin' else None

        img = mpimg.imread(path)
        imgplot = plt.imshow(img)
        plt.show()

        prompt = (
            f"The user asked: '{args.input}'. "
            f"Respond with: '{description}'. "
            "Respond exactly with: 'The image you requested is [description]'. "
            "Replace [description] with the actual description. Do not add anything else."
        )

        response = dbqa({'query': prompt})
        print(f"Step 4 - LLM response time: {timeit.default_timer() - start:.2f} seconds") if role == 'admin' else None
        print(response['result'])

        timeit.default_timer()
        print(f"\nTotal execution time: {timeit.default_timer() - start:.2f} seconds") if role == 'admin' else None


def cli():
    # role = access()
    while True:
        query = input("Enter your image query: ")
        if query.lower() == 'exit':
            break

        start = timeit.default_timer()
        path, description = get_image(query)

        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


        print(f"Description: {description.split("\n")[-1]}")
        print(f"Execution time: {timeit.default_timer() - start:.2f} seconds\n")


if __name__ == "__main__":
    cli()
