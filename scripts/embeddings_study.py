from enstrag.models import RagEmbedding
from enstrag.data import Parser, FileDocument, RecursiveCharacterTextSplitter
import numpy as np

embedding_folder = "all-MiniLM-L6-v2"

embedding = RagEmbedding(embedding_folder)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)

docs = Parser.get_documents_from_filedocs([
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

labels = list(set([doc.metadata["name"] for doc in docs]))
c = []

splits = text_splitter.split_documents(docs)
print(len(splits))

for split in splits:
    c.append(labels.index(split.metadata["name"]))

splits_embed = embedding.embed_documents([split.page_content for split in splits])
splits_embed = np.array(splits_embed)
print(splits_embed.shape)

from sklearn.manifold import TSNE

splits_embed_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100).fit_transform(splits_embed)

print(splits_embed_2d.shape)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))

for g in np.unique(c):
    i = np.where(c == g)
    plt.scatter(splits_embed_2d[i,0], splits_embed_2d[i,1], label=labels[g])

#plt.scatter(splits_embed_2d[:,0], splits_embed_2d[:,1], c=c, label=labels)
plt.legend()

fig.savefig("scripts/tsne.png")
print("End")