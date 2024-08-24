import os
import pandas as pd


def files_to_completion_dataset(file_dir, max_chars_per_file=(1000 * 3)):
    # create a list to store file contents
    data = []

    # iterate over all the files in the directory and add content to the list
    for filename in os.listdir(file_dir):
        with open(os.path.join(file_dir, filename), "r") as file:
            text = file.read()
            current_text = ""
            for word in text.split():
                if len(current_text) < max_chars_per_file:
                    current_text += word + " "
                else:
                    data.append({"text": current_text})
                    current_text = ""
            if current_text:
                data.append({"text": current_text})
    # create a pandas dataframe from the list
    df = pd.DataFrame(data)
    return df


def save_pandas(file_name, df):
    outdir = "finetune_dataset/datasets/completion"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    file_path = os.path.join(outdir, file_name)
    df.to_parquet(file_path)


def save_json(file_name, df):
    outdir = "finetune_dataset/datasets/completion"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    file_path = os.path.join(outdir, file_name)
    df.to_json(file_path)


files_dir = "finetune_dataset/datasets/pg_essays"
completion_dataset = files_to_completion_dataset(files_dir)
save_pandas("pg_essays_split.parquet", completion_dataset)
save_json("pg_essays_split.json", completion_dataset)
