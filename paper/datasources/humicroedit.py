from pathlib import Path

import pandas as pd
import numpy as np
import sys
import os

here = Path(__file__).parent
humicroedit_folder = here / "../../data/raw/semeval-2020-task-7-dataset/subtask-1/"
humicroedit_max_grade = 3


def calculate_edit_endpoints(original_headline):
    return original_headline.index("<"), original_headline.index("/>") + 2


def calculate_edited_headline(original_headline: str, edit: str):
    start_edit, end_edit = calculate_edit_endpoints(original_headline)
    return original_headline[:start_edit] + edit + original_headline[end_edit:]


def get_real_headline(original_headline: str):
    start_edit, end_edit = calculate_edit_endpoints(original_headline)
    return (
        original_headline[:start_edit]
        + original_headline[start_edit + 1 : end_edit - 2]
        + original_headline[end_edit:]
    )


def get_data(data_name: str):
    return pd.read_csv((humicroedit_folder / (data_name + ".csv")).absolute())


def generate_humicroedit_regression(data_name: str):

    result = []

    # Humicron dataset
    csv_data = get_data(data_name)

    # Add all the edited headlines
    for index, row in csv_data.iterrows():
        original_headline = row["original"]
        edited_headline = calculate_edited_headline(original_headline, row["edit"])
        mean_grade = row["meanGrade"]
        normalised_grade = mean_grade / humicroedit_max_grade
        result.append({"headline": edited_headline, "score": normalised_grade})

        # Also add the original headlines
        result.append({"headline": get_real_headline(original_headline), "score": 0})

    return result


def get_all_datasets_regression():
    humicroedit_train = generate_humicroedit_regression("train")
    funlines_train = generate_humicroedit_regression("train_funlines")
    humicroedit_dev = generate_humicroedit_regression("dev")
    humicroedit_test = generate_humicroedit_regression("test")

    return humicroedit_train + funlines_train, humicroedit_dev, humicroedit_test


def generate_humicroedit_seq2seq(data_name: str, include_id=False):

    result = []

    # Humicron dataset
    csv_data = get_data(data_name)

    # Add all the edited headlines
    for index, row in csv_data.iterrows():
        original_headline = row["original"]

        # Calculate real and satire
        real_headline = get_real_headline(original_headline)
        edited_headline = calculate_edited_headline(original_headline, row["edit"])

        element = {"from": real_headline, "to": edited_headline}

        if include_id:
            element["id"] = row["id"]

        result.append(element)

    return result


def get_all_datasets_seq2seq():
    humicroedit_train = generate_humicroedit_seq2seq("train")
    funlines_train = generate_humicroedit_seq2seq("train_funlines")
    humicroedit_dev = generate_humicroedit_seq2seq("dev")
    humicroedit_test = generate_humicroedit_seq2seq("test")

    return humicroedit_train + funlines_train, humicroedit_dev, humicroedit_test


if __name__ == "__main__":
    get_all_datasets_seq2seq()
