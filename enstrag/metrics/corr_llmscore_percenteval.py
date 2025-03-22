import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
file_path = 'results_with_score.csv'
df = pd.read_csv(file_path, sep=',', header=0)
df.columns = df.columns.str.strip()

# Tracer le graphique
plt.figure(figsize=(8, 5))
plt.scatter(df['Best Chunk Percentage'], df['Score'], alpha=0.7)
plt.title('Relation entre le Best Chunk Percentage et le Score donné par le LLM externe')
plt.xlabel('Best Chunk Percentage (%)')
plt.ylabel('Score')
plt.grid(True)

# Enregistrer le graphique
plt.savefig('corr_llmscore_percenteval.png')
