import json
import random
import re
from pathlib import Path
import pandas as pd

from src.datasources import unfun, humicroedit

output_folder = Path("../../data/processed/regression-v2/")

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def create_split(full):
    random.seed(42)
    random.shuffle(full)
    test_length = int(0.1 * len(full))
    dev_length = int(0.1 * len(full))
    train_length = len(full) - test_length - dev_length

    train = full[:train_length]
    dev = full[train_length : train_length + dev_length]
    test = full[train_length + dev_length :]

    return train, dev, test


def get_splitted_unfun(prohibited_headlines):
    return create_split(
        [
            h
            for h in unfun.get_full_unfun_regression_dataset()
            if h["headline"].lower() not in prohibited_headlines
        ]
    )


def get_splitted_sarc(prohibited_headlines):
    sarc_file = Path("../../data/raw/sarcasm_headlines/Sarcasm_Headlines_Dataset.json")
    sarc_full = []
    with open(sarc_file) as f:
        for line in f:
            row = json.loads(line)
            headline = row["headline"]
            if headline.lower() not in prohibited_headlines:
                sarc_full.append(
                    {"headline": row["headline"], "score": row["is_sarcastic"]}
                )

    return create_split(sarc_full)


def get_splitted_onionornot(prohibited_headlines):
    onion_or_not_file = Path("../../data/raw/onionornot/OnionOrNot.csv")
    oon_csv = pd.read_csv(onion_or_not_file)
    oon_full = []

    for index, row in oon_csv.iterrows():
        headline = row["text"]
        if headline.lower() not in prohibited_headlines:
            oon_full.append({"headline": headline, "score": row["label"]})

    return create_split(oon_full)


def shuffle_and_output(name: str, original_data: list, augment_casing=True):
    # Augment data with casing
    data = list(original_data)

    # Normalise headline spacings
    for d in data:
        d["headline"] = _RE_COMBINE_WHITESPACE.sub(
            " ", d["headline"].replace("\n", "").replace("\t", " ")
        ).strip()

    if augment_casing:
        for d in original_data:
            headline = d["headline"]
            if headline != headline.lower():
                data.append({"headline": headline.lower(), "score": d["score"]})
            if headline != headline.title():
                data.append({"headline": headline.title(), "score": d["score"]})

    # Shuffle
    random.seed(42)
    random.shuffle(data)

    print("Size of ", name, len(data))

    # Output json
    with open(output_folder / "json" / (name + ".json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(output_folder / "splitted" / (name + ".sentences"), "w") as sentences_f:
        sentences_f.writelines("\n".join([d["headline"] for d in data]))

    with open(output_folder / "splitted" / (name + ".labels"), "w") as sentences_f:
        sentences_f.writelines("\n".join([str(d["score"]) for d in data]))


def create_full_regression_datasets():
    # Load Humicroedit
    (
        humicro_train,
        humicro_dev,
        humicro_test,
    ) = humicroedit.get_all_datasets_regression()
    print(
        "splitted humicroedit sizes",
        len(humicro_train),
        len(humicro_dev),
        len(humicro_test),
    )
    prohibited_headlines = {
        h["headline"].lower() for h in humicro_train + humicro_dev + humicro_test
    }

    # Load Unfun
    unfun_train, unfun_dev, unfun_test = get_splitted_unfun(prohibited_headlines)
    print("splitted unfun sizes", len(unfun_train), len(unfun_dev), len(unfun_test))

    # Load Sarcasm Headlines
    sarc_train, sarc_dev, sarc_test = get_splitted_sarc(prohibited_headlines)
    print("splitted sarc sizes", len(sarc_train), len(sarc_dev), len(sarc_test))

    # Extend prohibited headlines
    prohibited_headlines = prohibited_headlines.union(
        [h["headline"].lower() for h in sarc_train + sarc_dev + sarc_test]
    )

    # Load OnionOrNot Headlines
    oon_train, oon_dev, oon_test = get_splitted_onionornot(prohibited_headlines)
    print("splitted OnionOrNot sizes", len(oon_train), len(oon_dev), len(oon_test))

    # Merge them together into train, dev & test
    train = unfun_train + humicro_train + sarc_train + oon_train
    dev = unfun_dev + humicro_dev + sarc_dev + oon_dev
    test = unfun_test + humicro_test + sarc_test + oon_test

    print("splitted total sizes", len(train), len(dev), len(test))

    shuffle_and_output("train", train)
    shuffle_and_output("dev", dev)
    shuffle_and_output("test", test)


if __name__ == "__main__":
    create_full_regression_datasets()
