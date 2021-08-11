import math
from pathlib import Path

import pandas as pd
import numpy as np
import sys
import os

unfun_folder = Path("../../data/raw/unfun/")
original_headlines_csv = pd.read_csv("../../data/raw/unfun/headlines_original.csv")
edited_headlines_csv = pd.read_csv("../../data/raw/unfun/headlines_unfunned.csv")
ratings_csv = pd.read_csv("../../data/raw/unfun/ratings.csv")


def get_processed_scores():
    scores = {}

    # Convert all ratings to dictionary by headline_id
    for index, row in ratings_csv.iterrows():
        headline_id = row["headline_id"]
        rating = row["rating"]

        if math.isnan(rating):
            continue

        if headline_id not in scores:
            scores[headline_id] = [rating]
        else:
            scores[headline_id].append(rating)

    # Average out all ratings
    avg_scores = {}
    for headline_id in scores:
        score_list = scores[headline_id]
        avg_scores[headline_id] = sum(score_list) / len(score_list)

    return avg_scores


def is_mostly_uppercased(title):
    words = title.split(" ")
    uppercased_words = len([w for w in words if w.upper() == w])
    return uppercased_words >= len(words) / 2


def normalise(title):
    if is_mostly_uppercased(title) or title.lower() == title:
        title = title.title()
    return title


def get_full_unfun_regression_dataset():

    all_data = []
    avg_scores = get_processed_scores()

    # Read all original headlines
    for index, row in original_headlines_csv.iterrows():

        headline_id = row["id"]
        title = row["title"]
        truth_type = row["truth_type"]

        # Check if it has an average score thanks to user ratings
        if headline_id in avg_scores:
            score = avg_scores[headline_id]
        else:
            # Otherwise use 0 if headline, 1 if real satire
            score = 0 if truth_type == "real" else 1

        all_data.append({"headline": title, "score": score})

    # Read all edited headlines
    for index, row in edited_headlines_csv.iterrows():

        headline_id = row["id"]
        title = row["title"]

        # Check if it has an average score thanks to user ratings
        if (
            title is not None
            and not isinstance(title, float)
            and len(title.strip()) > 0
            and headline_id in avg_scores
        ):
            score = avg_scores[headline_id]

            # Title casing to make edit not blatantly obvious
            title = normalise(title)

            all_data.append({"headline": title, "score": score})

    return all_data


def get_full_unfun_seq2seq_dataset():
    all_data = []

    # Read all edited headlines
    for index, row in edited_headlines_csv.iterrows():

        headline_id = row["id"]
        unfunned_title = row["title"]

        # Check if it has an average score thanks to user ratings
        if (
            unfunned_title is not None
            and not isinstance(unfunned_title, float)
            and len(unfunned_title.strip()) > 0
        ):
            # Title casing to make edit not blatantly obvious
            unfunned_title = normalise(unfunned_title)

            original_title_id = row["original_headline_id"]
            original_title_index = original_headlines_csv.index[
                original_headlines_csv["id"] == original_title_id
            ].tolist()
            original_fun_headline = original_headlines_csv.iloc[original_title_index[0]]["title"]

            all_data.append({"from": unfunned_title, "to": original_fun_headline})

    return all_data


if __name__ == "__main__":
    get_full_unfun_seq2seq_dataset()
