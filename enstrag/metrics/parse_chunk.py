import re
import matplotlib.pyplot as plt

def extract_chunks(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    pattern = r"""
    Expected\sChunk:\s(?P<expected>.*?)\n
    Best\sChunk:\s(?P<best>.*?)\n
    Percentage\sof\scommon\swords:\s(?P<percentage>[\d\.]+%)
    """

    matches = re.finditer(pattern, data, re.DOTALL | re.VERBOSE)
    result = []

    for match in matches:
        expected = match.group('expected').strip()
        best = match.group('best').strip()
        percentage = match.group('percentage').strip()
        result.append([percentage, best, expected])

    return result

# Exemple d'utilisation
file_path = 'test.txt'
chunks = extract_chunks(file_path)
for chunk in chunks:
    print(chunk[0], "\n", chunk[1], "\n\n", chunk[2], "\n", "\n\n")

# Données
labels = ['77.27', '25.00', '57.14', '65.57', '80.00', '40.51', '26.15', '85.71', 
          '67.74', '61.54', '28.89', '31.25', '60.98', '21.57', '22.81', '16.33', 
          '94.44', '10.00', '11.63', '28.95', '47.83', '20.83']
values = [77.27, 25.00, 57.14, 65.57, 80.00, 40.51, 26.15, 85.71, 
          67.74, 61.54, 28.89, 31.25, 60.98, 21.57, 22.81, 16.33, 
          94.44, 10.00, 11.63, 28.95, 47.83, 20.83]
categories = ['OUI', 'NON', 'OUI', 'OUI', 'OUI', 'NON', 'NON', 'OUI', 
              'OUI', 'OUI', 'NON', 'NON', 'OUI', 'NON', 'NON', 'NON', 
              'OUI', 'NON', 'NON', 'NON', 'OUI', 'NON']

# Définir les couleurs
colors = ['green' if cat == 'OUI' else 'red' for cat in categories]

# Création de l'histogramme
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color=colors)

# Ajouter une ligne en pointillé à 42 %
plt.axhline(y=42, color='gray', linestyle='--', linewidth=1.5, label='Seuil à 42%')

# Ajouter une légende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Le chunk trouvé correspond'),
    Patch(facecolor='red', label='Ne correspond pas')
]
plt.legend(handles=legend_elements, loc='upper right')

# Mise en forme
plt.ylim(0, 100)  # Échelle de 0 à 100%
plt.xticks(rotation=45, ha='right')
plt.ylabel('Scoring en pourcentage (%)')
plt.title('Correspondance des chunks')

# Sauvegarde du graphique
plt.savefig('histogramme_chunks.png', dpi=300, bbox_inches='tight')

# Affichage
plt.show()
