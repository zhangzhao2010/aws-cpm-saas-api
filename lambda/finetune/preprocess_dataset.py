import os
from cpm_live.dataset import build_dataset, shuffle_dataset
import shutil
from tqdm import tqdm
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)

    args = parser.parse_args()
    return args


def reformat_data(data):
    """set your data format"""
    return data

def preprocess_dataset(source_dir, output_dir):
    files = os.listdir(source_dir)
    for ds in files:
        with build_dataset("tmp", "data") as dataset:
            with open(os.path.join(source_dir, ds), "r", encoding="utf-8") as fin:
                for line in tqdm(fin.readlines(), desc=os.path.join(source_dir, ds)):
                    data = json.loads(line)
                    dataset.write(reformat_data(data))
        shuffle_dataset(
            "tmp",
            os.path.join(output_dir, ds.split(".")[0]),
            progress_bar=True,
            output_name='ccpm_data'
        )
        shutil.rmtree("tmp")
    return

def main():
    args = get_args()
    files = os.listdir(args.input)
    for ds in files:
        with build_dataset("tmp", "data") as dataset:
            with open(os.path.join(args.input, ds), "r", encoding="utf-8") as fin:
                for line in tqdm(fin.readlines(), desc=os.path.join(args.input, ds)):
                    data = json.loads(line)
                    dataset.write(reformat_data(data))
        shuffle_dataset(
            "tmp",
            os.path.join(args.output_path, ds.split(".")[0]),
            progress_bar=True,
            output_name=args.output_name
        )
        shutil.rmtree("tmp")
    return


if __name__ == "__main__":
    main()
