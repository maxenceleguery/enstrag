import re
import sys
import os
import json
import random

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enstrag import get_args, verify_execution
from enstrag.rag import RagAgent
from enstrag.models import get_pipeline, RagEmbedding
from enstrag.data import VectorDB, Parser, FileDocument

EVALUATION_PROMPT = """###Task Description:
For the given instruction, you are asked to evaluate the response based on the reference answer, on a score from 0 to 5 (0 being completely false answer, 5 being the exact answer). You must only output the score as an integer, after the 'Answer score' tag.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer:
{reference_answer}

###Answer score (between 0 and 5):
"""

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def evaluate_rag(agent, dataset):
    table_data = []

    eval_pipe = get_pipeline("Ministral-8B-Instruct-2410")

    for data in random.sample(dataset, 10):
        question = data["Question"]
        expected_answer = data["Answer"]
        expected_chunk = data["Chunks"][0]["chunk"]
        expected_chunk_page = data["Chunks"][0]["metadata"][0]["page"]
        # expected_chunk_page = data["Chunks"][0]["metadata"]["page"] si final_dataset.json

        generated_answer, retrieved_context, sources, best_chunk = agent.answer_question(question)

        # print(f"Question: {question}")
        # print(f"Expected Answer: {expected_answer}")
        # print(f"Generated Answer: {generated_answer}")
        # print(f"Expected Chunk: {expected_chunk}")
        print(f"Best Chunk: {best_chunk}")
        print()

        # Extract the page number from the best chunk string
        best_chunk_page_match = re.search(r'page (\d+)', best_chunk[2])
        best_chunk_page = best_chunk_page_match.group(1) if best_chunk_page_match else "N/A"

        # table_data.append((expected_chunk_page, best_chunk_page))

        # Benchmarking by a judge agent
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=question,
            response=generated_answer,
            reference_answer=expected_answer,
        )
        # print(f"\neval_prompt:\n", eval_prompt)
        eval_result = eval_pipe(eval_prompt, return_full_text=True, max_new_tokens=2)
        print(eval_result)
        score = int(re.findall(r'\d+', eval_result[0]['generated_text'].split("###Answer score (between 0 and 5):\n")[1])[0])

        table_data.append((expected_chunk_page, best_chunk_page, score))

    # Display the table
    print(f"{'Expected Chunk Page':<20} {'Best Chunk Page':<20} {'Score':<20}")
    print("-" * 60)
    for row in table_data:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20}")

if __name__ == "__main__":
    args = get_args()
    verify_execution()

    print("Importing packages...")

    llm_folder = args.llm_folder
    embedding_folder = args.embedding_folder
    persist_directory = args.persist_dir

    embedding = RagEmbedding(embedding_folder)
    db = VectorDB(embedding, persist_directory=persist_directory)

    if args.reset:
        print("Resetting database...")
        db.db.reset_collection()

    db.add_documents(
        Parser.get_documents_from_filedocs([
            #FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", None, "ML Bishop", "Machine learning"),
            #FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", None, "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
            FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", None, "CLASSICAL MECHANICS", "Physics"),
            #FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", None, "Convex Optimization", "Maths"),
            #"https://www.damtp.cam.ac.uk/user/tong/qft/qft.pdf",
            #FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", None, "Basic Algebraic Geometry", "Maths"),
            #FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", None, "Organic Chemistry", "Chemistry"),
            #"https://arxiv.org/pdf/1706.03762",
            #"https://arxiv.org/pdf/2106.09685"
        ])
    )

    agent = RagAgent(
        pipe=get_pipeline(llm_folder),
        db=db,
    )

    # Load the dataset
    dataset = load_dataset('/home/ensta/ensta-matteo.denis/ProjetIA/enstrag/enstrag/metrics/evaluation_datasets/dataset_classical_mechanics.json')

    # Evaluate the RAG agent
    evaluate_rag(agent, dataset)