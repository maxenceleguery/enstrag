import re
import sys
import os
import json
import csv

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

def evaluate_rag(agent, dataset, dataset_name, output_file):
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for data in dataset:
            question = data["Question"]
            expected_answer = data["Answer"]
            expected_chunk = data["Chunks"][0]["chunk"]

            # Clean the expected chunk using Parser.clean_text
            cleaned_expected_chunk = Parser.clean_text(expected_chunk)

            generated_answer, retrieved_context, sources, best_chunk, chunks = agent.answer_question_for_evaluation(question)
            classified_chunks = best_chunk[3]

            # Count the number of common words and total words in cleaned_expected_chunk
            common_word_count, total_words = count_common_words(cleaned_expected_chunk, best_chunk[2])
            best_chunk_percentage = (common_word_count / total_words) * 100

            chunk_percentages = []
            for chunk in classified_chunks[1:4]:
                chunk_text = chunk
                common_word_count, total_words = count_common_words(cleaned_expected_chunk, chunk_text)
                percentage_common_words = (common_word_count / total_words) * 100
                chunk_percentages.append(percentage_common_words)

            # Write the results to the CSV file
            writer.writerow([
                dataset_name,
                best_chunk_percentage,
                chunk_percentages[0] if len(chunk_percentages) > 0 else 0,
                chunk_percentages[1] if len(chunk_percentages) > 1 else 0,
                chunk_percentages[2] if len(chunk_percentages) > 2 else 0
            ])

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
            FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", None, "ML Bishop", "Machine learning"),
            #FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", None, "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
            FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", None, "CLASSICAL MECHANICS", "Physics"),
            #FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", None, "Convex Optimization", "Maths"),
            FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", None, "Basic Algebraic Geometry", "Maths"),
            FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", None, "Organic Chemistry", "Chemistry"),
        ])
    )

    agent = RagAgent(
        pipe=get_pipeline(llm_folder),
        db=db,
    )

    # List of dataset file paths
    dataset_filepaths = [
        '/home/ensta/ensta-joyeux/enstrag/dataset/dataset_machine_learning.json',
        '/home/ensta/ensta-joyeux/enstrag/dataset/dataset_classical_mechanics.json',
        '/home/ensta/ensta-joyeux/enstrag/dataset/dataset_algebraic_geometry.json',
        '/home/ensta/ensta-joyeux/enstrag/dataset/dataset_chemistry.json'

    ]

    # Evaluate the RAG agent for each dataset and write results to a CSV file
    output_file = '/home/ensta/ensta-joyeux/enstrag/enstrag/metrics/results.csv'

    # Write the header once
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Best Chunk Percentage', 'Chunk 1 Percentage', 'Chunk 2 Percentage', 'Chunk 3 Percentage'])

    for dataset_filepath in dataset_filepaths:
        dataset = load_dataset(dataset_filepath)
        dataset_name = os.path.splitext(os.path.basename(dataset_filepath))[0].replace('dataset_', '').replace('_', ' ').upper()
        evaluate_rag(agent, dataset, dataset_name, output_file)