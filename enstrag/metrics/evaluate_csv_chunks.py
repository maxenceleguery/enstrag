import pandas as pd

def evaluate_chunks(file_name, mode, threshold):
    df = pd.read_csv(file_name)

    results = {}
    grouped = df.groupby('Dataset')

    if mode == 'best_chunk_only':
        for dataset, group in grouped:
            results[dataset] = [row['Best Chunk Percentage'] > threshold for _, row in group.iterrows()]

    elif mode == 'all_chunks':
        for dataset, group in grouped:
            results[dataset] = [
                any(row[col] > threshold for col in ['Best Chunk Percentage', 'Chunk 1 Percentage', 'Chunk 2 Percentage', 'Chunk 3 Percentage'])
                for _, row in group.iterrows()
            ]

    else:
        raise ValueError("Mode must be 'best_chunk_only' or 'all_chunks'")

    return results