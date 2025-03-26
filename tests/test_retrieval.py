from enstrag import get_args
args = get_args()

#from enstrag.models import RagEmbedding
#from enstrag.data import VectorDB, Parser, FileDocument
from enstrag.rag import RagAgent

def test_retrieval():
    embedding_folder = args.embedding_folder
    persist_directory = args.persist_dir

    """
    embedding = RagEmbedding(embedding_folder)
    db = VectorDB(embedding, persist_directory=persist_directory)

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
        ], get_pages_num=False)
    )

    chunks = db.get_context_from_query("What is a gaussian distribution ?")
    """
    chunks = [
        {'text': 'The Gaussian is the distribution that maximizes the entropy for a given variance\n(or covariance). Any linear transformation of a Gaussian random variable is againGaussian. The marginal distribution of a multivariate Gaussian with respect to a\nsubset of the variables is itself Gaussian, and similarly the conditional distribution is\nalso Gaussian. The conjugate prior for is the Gaussian, the conjugate prior for \nis the Wishart, and the conjugate prior for (,)is the Gaussian-Wishart.', 'name': 'ML Bishop', 'url': 'http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf', 'path': '/home/ensta/ensta-leguery/enstrag_folder/pdfs/ML_Bishop.pdf'},
        {'text': 'This model provides us with a particular example of a Gaussian process. In gen-\neral, a Gaussian process is dened as a probability distribution over functions y(x)\nsuch that the set of values of y(x)evaluated at an arbitrary set of points x1,...,xN\njointly have a Gaussian distribution. In cases where the input vector xis two di-\nmensional, this may also be known as a Gaussian random eld . More generally, a\nstochastic process y(x)is specied by giving the joint probability distribution for', 'name': 'ML Bishop', 'url': 'http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf', 'path': '/home/ensta/ensta-leguery/enstrag_folder/pdfs/ML_Bishop.pdf'},
        {'text': 'where ()is the digamma function dened by (B.25). The gamma distribution is\nthe conjugate prior for the precision (inverse variance) of a univariate Gaussian. For\na/greaterorequalslant1the density is everywhere nite, and the special case of a=1is known as the\nexponential distribution.\nGaussian\nThe Gaussian is the most widely used distribution for continuous variables. It is also\nknown as the normal distribution. In the case of a single variable x(,)it is', 'name': 'ML Bishop', 'url': 'http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf', 'path': '/home/ensta/ensta-leguery/enstrag_folder/pdfs/ML_Bishop.pdf'},
        {'text': 'using the multinomial distribution (2.34) with K=2 .\n2.3.\n The Gaussian Distribution\nThe Gaussian, also known as the normal distribution, is a widely used model for the\ndistribution of continuous variables. In the case of a single variable x, the Gaussian\ndistribution can be written in the form\nN(x|,2)=1\n(22)1/2exp/braceleftbigg\n1\n22(x)2/bracerightbigg\n(2.42)\nwhere is the mean and 2is the variance. For a D-dimensional vector x,t h e\nmultivariate Gaussian distribution takes the form\nN(x|,)=1\n(2)D/ 21', 'name': 'ML Bishop', 'url': 'http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf', 'path': '/home/ensta/ensta-leguery/enstrag_folder/pdfs/ML_Bishop.pdf'},
        {'text': 'Null text', 'name': 'Null name', 'url': 'null url', 'path': 'Null path'},
    ]
    for chunk in chunks:
        print(chunk)
    print(RagAgent.choose_best_document(chunks))


if __name__ == "__main__":
    test_retrieval()