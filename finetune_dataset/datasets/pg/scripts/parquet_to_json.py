import pandas as pd


# open parquet file

def open_parquet_file(file_path):
    return pd.read_parquet(file_path)

pq = open_parquet_file('finetune_dataset/test.parquet')

# convert to json and save file
def convert_to_json(pq, file_path):
    pq.to_json(file_path)

convert_to_json(pq, 'finetune_dataset/test.json')