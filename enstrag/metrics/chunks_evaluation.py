import re
import sys
import os
import json

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enstrag import get_args, verify_execution
from enstrag.rag import RagAgent
from enstrag.models import get_pipeline, RagEmbedding
from enstrag.data import VectorDB, Parser, FileDocument

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)
    
def count_common_words(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1.intersection(words2)
    return len(common_words), len(words1)

def evaluate_rag(agent, dataset):
    score = 0

    for data in dataset:
        question = data["Question"]
        expected_answer = data["Answer"]
        expected_chunk = data["Chunks"][0]["chunk"]
        expected_chunk_page = data["Chunks"][0]["metadata"][0]["page"]

        # Clean the expected chunk using Parser.clean_text
        cleaned_expected_chunk = Parser.clean_text(expected_chunk)
        
        # Print the cleaned expected chunk
        print(f"Expected Chunk: {cleaned_expected_chunk}")

        generated_answer, retrieved_context, sources, best_chunk = agent.answer_question(question)

        # print(f"Question: {question}")
        # print(f"Expected Answer: {expected_answer}")
        # print(f"Generated Answer: {generated_answer}")
        # print(f"Expected Chunk: {expected_chunk}")
        print(f"Best Chunk: {best_chunk[2]}")
        print()

        # Count the number of common words and total words in cleaned_expected_chunk
        common_word_count, total_words = count_common_words(cleaned_expected_chunk, best_chunk[2])
        percentage_common_words = (common_word_count / total_words) * 100
        print(f"Percentage of common words: {percentage_common_words:.2f}%")

if __name__ == "__main__":
    args = get_args()
    verify_execution()

    print("Importing packages...")

    llm_folder = args.llm_folder
    embedding_folder = args.embedding_folder
    persist_directory = args.persist_dir

    embedding = RagEmbedding(embedding_folder)
    db = VectorDB(embedding, persist_directory=persist_directory)

    # Reset the database unconditionally
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
    dataset = load_dataset('/home/ensta/ensta-joyeux/enstrag/dataset/dataset_classical_mechanics.json')

    # Evaluate the RAG agent
    evaluate_rag(agent, dataset)

    print("Hello world")