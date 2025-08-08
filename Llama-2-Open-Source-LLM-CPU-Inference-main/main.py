import box
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import timeit

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from db_build import get_image


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input',
#                         type=str,
#                         default='How much is the minimum guarantee payable by adidas?',
#                         help='Enter the query to pass into the LLM')
#     args = parser.parse_args()
#
#     # Setup DBQA
#     start = timeit.default_timer()
#     dbqa = setup_dbqa()
#     response = dbqa({'query': args.input})
#     end = timeit.default_timer()
#
#     print(f'\nAnswer: {response["result"]}')
#     print('='*50)
#
#     # Process source documents
#     source_docs = response['source_documents']
#     for i, doc in enumerate(source_docs):
#         print(f'\nSource Document {i+1}\n')
#         print(f'Source Text: {doc.page_content}')
#         print(f'Document Name: {doc.metadata["source"]}')
#         print(f'Page Number: {doc.metadata["page"]}\n')
#         print('='* 60)
#
#     print(f"Time to retrieve response: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='Retrieve the image the bird staying on a branch, with a blue sky',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()
    total_start = timeit.default_timer()

    step1_start = timeit.default_timer()
    dbqa = setup_dbqa()

    step1_end = timeit.default_timer()
    print(f"Step 1 - DBQA setup time: {step1_end - step1_start:.2f} seconds")

    if "image" in args.input.lower():
        step2_start = timeit.default_timer()
        path, description = get_image(args.input)
        step2_end = timeit.default_timer()
        print(f"Step 2 - Image retrieval time: {step2_end - step2_start:.2f} seconds")

        # step3_start = timeit.default_timer()
        # prompt = (
        #     f"The user asked: '{args.input}'. "
        #     f"The image description is: '{description}'. "
        #     "Respond exactly with: 'The image your requested is [description]'. "
        #     "Replace [description] with the actual description. Do not add anything else."
        # )
        # response = dbqa({'query': prompt})
        #
        # step3_end = timeit.default_timer()
        # print(f"Step 3 - LLM response time: {step3_end - step3_start:.2f} seconds")



        step4_start = timeit.default_timer()
        img = mpimg.imread(path)
        imgplot = plt.imshow(img)
        plt.show()
        step4_end = timeit.default_timer()
        print(f"Step 4 - Image display time: {step4_end - step4_start:.2f} seconds")

        print(f"Your image description is: {description}")
        # print(f"\n{response['result']}")

        total_end = timeit.default_timer()
        print(f"\nTotal execution time: {total_end - total_start:.2f} seconds")





