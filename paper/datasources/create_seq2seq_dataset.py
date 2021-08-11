import json
import random
import re
from pathlib import Path
import pandas as pd

from src.datasources import unfun, humicroedit

regression_folder = Path("../../data/processed/regression-v2/json")
output_folder = Path("../../data/processed/seq2seq/")

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def get_all_seq2seq_with_headline_in_regression(all_data, regression_data_name):
    with open(
        regression_folder / (regression_data_name + ".json")
    ) as regression_dev_file:
        regression_dev = json.load(regression_dev_file)
        regression_dev_headlines = {h["headline"] for h in regression_dev}
        filtered_data = [
            h
            for h in all_data
            if h["from"] in regression_dev_headlines
            or h["to"] in regression_dev_headlines
        ]
        return filtered_data


def get_splitted_unfun():
    unfun_all = unfun.get_full_unfun_seq2seq_dataset()
    unfun_dev = get_all_seq2seq_with_headline_in_regression(unfun_all, "dev")
    unfun_test = get_all_seq2seq_with_headline_in_regression(unfun_all, "test")

    # Calculate which lines are not allowed
    unfun_dev_and_test = unfun_dev + unfun_test
    dev_and_test_lines = {h["from"] for h in unfun_dev_and_test}.union(
        h["to"] for h in unfun_dev_and_test
    )

    unfun_train = []
    for elem in unfun_all:
        if (
            elem["from"] not in dev_and_test_lines
            and elem["to"] not in dev_and_test_lines
        ):
            unfun_train.append(elem)

    return unfun_train, unfun_dev, unfun_test


def remove_whitespace(line):
    return _RE_COMBINE_WHITESPACE.sub(
        " ", line.replace("\n", "").replace("\t", " ")
    ).strip()


def shuffle_and_output(name: str, original_data: list, augment_casing=True):
    # Augment data with casing
    data = list(original_data)

    # Normalise headline spacings
    for d in data:
        d["from"] = remove_whitespace(d["from"])
        d["to"] = remove_whitespace(d["to"])

    if augment_casing:
        for d in original_data:
            from_line = d["from"]
            to_line = d["to"]
            if from_line != from_line.lower() or to_line != to_line.lower():
                data.append({"from": from_line.lower(), "to": to_line.lower()})
            if from_line != from_line.title() or to_line != to_line.title():
                data.append({"from": from_line.title(), "to": to_line.title()})

    # Shuffle
    random.seed(42)
    random.shuffle(data)

    print("Size of ", name, len(data))

    # Output json
    with open(output_folder / "json" / (name + ".json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(output_folder / "splitted" / (name + ".sentences"), "w") as sentences_f:
        sentences_f.writelines("\n".join([d["from"] for d in data]))

    with open(output_folder / "splitted" / (name + ".labels"), "w") as sentences_f:
        sentences_f.writelines("\n".join([str(d["to"]) for d in data]))


def create_full_regression_datasets():
    # Load Humicroedit
    (
        humicro_train,
        humicro_dev,
        humicro_test,
    ) = humicroedit.get_all_datasets_seq2seq()
    print(
        "splitted humicroedit sizes",
        len(humicro_train),
        len(humicro_dev),
        len(humicro_test),
    )

    # Load Unfun
    unfun_train, unfun_dev, unfun_test = get_splitted_unfun()
    print("splitted unfun sizes", len(unfun_train), len(unfun_dev), len(unfun_test))

    # Merge them together into train, dev & test
    train = unfun_train + humicro_train
    dev = unfun_dev + humicro_dev
    test = unfun_test + humicro_test

    print("splitted total sizes", len(train), len(dev), len(test))

    shuffle_and_output("train", train)
    shuffle_and_output("dev", dev)
    shuffle_and_output("test", test)


if __name__ == "__main__":
    create_full_regression_datasets()
