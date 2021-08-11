import json
import random
from pathlib import Path

from src.datasources import create_headline_evolutions_dataset

here = Path(__file__).parent
processed_output_file = here / "../../data/evaluation/doccano/competitions.jsonl"

random.seed(42)


def create_line_source(datapoint, source: str):
    return {
        "line": datapoint["gen_" + source],
        "source": source
    }


def create_competition(d, options):
    random.shuffle(options)

    return {
        "text": "Which satirical headline is funnier? \n\n\n\na. " + options[0]["line"] + "\n\nb. " + options[1][
            "line"],
        "meta": {
            "a": options[0],
            "b": options[1],
            "row": d,
        }
    }


def create_doccano_dataset():
    datapoints = create_headline_evolutions_dataset.get_all_generated_headlines_from_file()
    datapoints += create_headline_evolutions_dataset.get_all_generated_headlines_from_file(
        create_headline_evolutions_dataset.extra_evolved_headlines_output_path)


    processed = []

    for d in datapoints:
        human = create_line_source(d, "human")
        galmet = create_line_source(d, "galmet")
        baseline = create_line_source(d, "baseline")

        hg = create_competition(d, [human, galmet])
        bg = create_competition(d, [baseline, galmet])
        processed.append(hg)
        processed.append(bg)

    random.seed(42)
    random.shuffle(processed)

    if not processed_output_file.parent.exists():
        processed_output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(processed_output_file, "w+") as output_file:
        for p in processed:
            element_json = json.dumps(p, ensure_ascii=False)
            output_file.write(element_json.replace("\n", "") + "\n")


if __name__ == '__main__':
    create_doccano_dataset()
