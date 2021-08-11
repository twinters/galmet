import json
import pickle
from pathlib import Path

import src.genetictransformer as genetictransformer
from src.baseline import BaselineTextModifier
from src.datasources import humicroedit
from src.galmet import Galmet

here = Path(__file__).parent
evaluation_output_folder = here / "../../data/evaluation/generated"
evolved_headlines_output_path = evaluation_output_folder / "evolved_headlines.jsonl"
extra_evolved_headlines_output_path = evaluation_output_folder / "pieter_generated.jsonl"
logbooks_folder = evaluation_output_folder / "logbooks"


def load_models():
    masker = genetictransformer.load_masker("../../models/roberta-satire-v0.1/")
    regression = genetictransformer.load_regression("../../models/regression-v0.2/")
    galmet = Galmet(
        masker=masker,
        regression=regression,
        crossover_prob=0.2,
        total_mutation_prob=0.8,
        population_size=50,
        nb_generations=30,
        goal_fitness=0.99,
        max_edit_distance=7,
        max_elites=6,
        elite_duplicates=3,
        mutate_token_prob=0.7,
        add_token_prob=0.05,
        remove_token_prob=0.05,
    )
    baseline = BaselineTextModifier(masker, allowed_edits=7)

    return galmet, baseline


def get_all_generated_headlines_from_file(evolved_headlines_path=evolved_headlines_output_path):
    if evolved_headlines_path.exists():
        with open(evolved_headlines_path, "r") as already_generated_file:
            return [json.loads(jline) for jline in list(already_generated_file)]
    return []


def generate_galmet(galmet, id, sentence):
    best, hof, logbook = galmet.evolve_text(sentence)

    # Safe logbook
    with open(logbooks_folder / (str(id) + ".pickle"), "wb+") as pickle_out:
        pickle.dump(logbook, pickle_out)

    # Safe hof
    with open(logbooks_folder / (str(id) + ".txt"), "w+") as hof_file:

        hof_file.write("Chosen: " + best + "\n\n")

        for h in reversed(hof):
            edit_distance = abs(-h.fitness.values[1])
            hof_file.write(
                str(int(edit_distance))
                + "\t"
                + ("%f" % h.fitness.values[0])
                + "\t"
                + galmet.masker.detokenize_sentence(h)
                + "\n"
            )

    return best


def generate_headlines():

    galmet, baseline = load_models()
    print("Loaded models")

    humicroedit_test = humicroedit.generate_humicroedit_seq2seq("test", include_id=True)

    # Check which ones are already generated
    done_ids = {a["id"] for a in get_all_generated_headlines_from_file()}

    with open(evolved_headlines_output_path, "a+") as evolved_output_file:
        for row in humicroedit_test:
            # Check if already done
            if row["id"] in done_ids:
                print("Skipping", row["id"])
                continue

            # Generate line
            original = row["from"]
            element = {
                "id": row["id"],
                "original": row["from"],
                "gen_human": row["to"],
                "gen_galmet": generate_galmet(galmet, row["id"], original),
                "gen_baseline": baseline.modify_sentence(original)
                # "gen_galmet": "Galmet generation",
                # "gen_baseline": "Baseline generation",
            }

            done_ids.add(id)

            # Write to evolved headlines
            element_json = json.dumps(element, ensure_ascii=False)
            evolved_output_file.write(element_json.replace("\n", "") + "\n")
            evolved_output_file.flush()

            print("Finished", str(row["id"]))


if __name__ == "__main__":
    generate_headlines()
