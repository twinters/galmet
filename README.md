# GALMET
GALMET is a Genetic Algorithm using Language Models for Evolving Text.
It is capable of evolving one text into another, guided by a BERT-based transformer model.

In our main example, we use a finetuned RoBERTa MLM model for mutating the text in the mutator operators of the genetic algorithm, and another finetuned RoBERTa model that guesses if the text is a real headline (0) or satire (1) or any value in between (for edited headlines with a funniness rating).

More information can be found in our paper "Survival of the Wittiest: Evolving Satire with Language Models", published on the Twelfth International Conference on Computational Creativity, ICCC’21.

## How to run

GALMET requires a Python runtime environment.
If you want to run the satire evolution example in [notebooks/galmet-satire-example.ipynb](https://github.com/twinters/galmet/blob/0.0.1/notebooks/galmet-satire-example.ipynb), perform the following steps:

1. Create a new virtual environment .
2. Install requirements.txt using `pip install -r requirements.txt`.
3. Download `robeta-satire-mlm` and `roberta-satire-regression` from the [releases page](https://github.com/twinters/galmet/releases/tag/0.0.1), and put them in a `models/` folder at the top-level of the repository.
4. Run `jupyter notebook` in command line in the repository folder.
5. Navigate to the `galmet-satire-example.ipynb` and run it. 