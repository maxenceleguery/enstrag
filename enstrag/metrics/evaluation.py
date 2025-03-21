import json
from enstrag import get_args, verify_execution
from enstrag.rag import RagAgent
from enstrag.models import get_pipeline, RagEmbedding
from enstrag.data import VectorDB, Parser, FileDocument

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def evaluate_rag(agent, dataset):
    correct_answers = 0
    total_questions = len(dataset)

    for data in dataset:
        question = data["Question"]
        expected_answer = data["Answer"]
        expected_chunk = data["Chunks"][0]["chunk"]

        generated_answer, retrieved_context, sources, best_chunk = agent.answer_question(question)

        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Expected Chunk: {expected_chunk}")
        print(f"Best Chunk: {best_chunk[2]}")
        print()

        if generated_answer.strip() == expected_answer.strip() and best_chunk[2].strip() == expected_chunk.strip():
            correct_answers += 1

    accuracy = correct_answers / total_questions
    print(f"Accuracy: {accuracy * 100:.2f}%")

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
            FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", "ML Bishop", "Machine learning"),
            FileDocument("https://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/StoicaM05.pdf", "SPECTRAL ANALYSIS OF SIGNALS", "Physics"),
            FileDocument("https://www.math.toronto.edu/khesin/biblio/GoldsteinPooleSafkoClassicalMechanics.pdf", "CLASSICAL MECHANICS", "Physics"),
            FileDocument("https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "Convex Optimization", "Maths"),
            #"https://www.damtp.cam.ac.uk/user/tong/qft/qft.pdf",
            FileDocument("http://students.aiu.edu/submissions/profiles/resources/onlineBook/Z6W3H3_basic%20algebra%20geometry.pdf", "Basic Algebraic Geometry", "Maths"),
            FileDocument("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-SAMPLE_9ADraVJ.pdf", "Organic Chemistry", "Chemistry"),
            #"https://arxiv.org/pdf/1706.03762",
            #"https://arxiv.org/pdf/2106.09685"
        ])
    )

    agent = RagAgent(
        pipe=get_pipeline(llm_folder),
        db=db,
    )

    # Load the dataset
    dataset = load_dataset('/path/to/dataset.json')

    # Evaluate the RAG agent
    evaluate_rag(agent, dataset)