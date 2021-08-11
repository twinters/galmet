import random
from typing import List

import torch
import numpy as np
from deap import tools
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    RobertaTokenizer,
    RobertaForSequenceClassification, AutoModelForMaskedLM,
)


def _clone(ids: Tensor):
    return ids.clone().detach().float()


def tokenize_sentence(tokenizer, sentence: str):
    return tokenizer.encode(sentence, return_tensors="pt")


def detokenize_sentence(tokenizer, pt: Tensor, include_limiters=False):
    converted_tokens = tokenizer.convert_ids_to_tokens(pt[0].tolist())
    converted_string = tokenizer.convert_tokens_to_string(converted_tokens)
    if not include_limiters:
        if converted_string.startswith("<s>"):
            converted_string = converted_string[3:]
        if converted_string.endswith("</s>"):
            converted_string = converted_string[: len(converted_string) - 4]
    return converted_string


def pad_and_get_input_mask(
    tokenizer, block_size, mask_padding_with_zero, tokenized_text
):

    input_mask = [1 if mask_padding_with_zero else 0] * len(tokenized_text)
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    while len(tokenized_text) < block_size:
        tokenized_text.append(pad_token)
        input_mask.append(0 if mask_padding_with_zero else 1)

    return {
        "input_ids": Tensor(tokenized_text[0:block_size]),
        "attention_mask": Tensor(input_mask[0:block_size]),
    }


def long_tensor_creator(*args, **kwargs):
    return Tensor(*args, **kwargs).long()


class MaskPredictor:
    """ Class that supports all kinds of genetic operators to do with an MLM model"""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def tokenize_sentence(self, sentence: str):
        return tokenize_sentence(self.tokenizer, sentence)

    def detokenize_sentence(self, pt: Tensor, include_limiters=False):
        return detokenize_sentence(self.tokenizer, pt, include_limiters)

    # %%

    def predict_mask(self, mask_token_index: Tensor, input_ids: Tensor):
        token_logits = self.model(input_ids)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits, dim=1)

        # Pick the top 10 words TODO: throw out
        # topk = 3
        # top = torch.topk(mask_token_logits, topk, dim=1)
        # top_tokens = zip(top.indices[0].tolist(), top.values[0].tolist())
        top_tokens = enumerate(mask_token_logits)

        return top_tokens

    def mutate_token(
        self, original_input_ids, index, creator=long_tensor_creator, allow_same=True
    ):

        # Clone, to not override actual vector
        input_ids = creator(_clone(original_input_ids))

        mask_token_index = Tensor([index]).long()
        original_token = _clone(input_ids[0][index])
        input_ids[0][index] = self.tokenizer.mask_token_id

        # Get the scores for all tokens
        tokens = list(self.predict_mask(mask_token_index, input_ids))
        scores = [score.tolist() for _, score in tokens][0]

        # Decrease odds of actual token to zero
        if not allow_same:
            scores[original_token.long().item()] = 0

        # Set nonsensical tokens to 0
        for special_id in self.tokenizer.all_special_ids:
            scores[special_id] = 0

        # Select a token and fill it in
        chosen_token = random.choices(population=range(len(scores)), weights=scores)
        if len(chosen_token):
            input_ids[0][index] = chosen_token[0]
            return input_ids
        else:
            print("Error, no token found", chosen_token)
            return input_ids

    def mutate_token_randomly(
        self, original_input_ids: Tensor, creator=long_tensor_creator, allow_same=True
    ):

        # Check if there are tokens to mutate
        max_idx = len(original_input_ids[0]) - 2
        if max_idx < 1:
            return original_input_ids

        # Pick a random mask and place it over the input sentence
        random_mask_idx = random.randint(1, max_idx)

        return self.mutate_token(
            original_input_ids,
            index=random_mask_idx,
            creator=creator,
            allow_same=allow_same,
        )

    def add_token_randomly(
        self, original_input_ids: Tensor, creator=long_tensor_creator
    ):

        # Clone, to not override actual vector
        input_ids = _clone(original_input_ids)

        # Pick a random mask position
        random_mask_idx = random.randint(1, len(input_ids[0]) - 1)
        mask_token_index = Tensor([random_mask_idx]).long()

        # Insert the mask into the sentence
        enlarged_ids = creator(torch.zeros(1, len(input_ids[0]) + 1))
        enlarged_ids[0] = torch.cat(
            [
                input_ids[0][:random_mask_idx],
                Tensor([self.tokenizer.mask_token_id]),
                input_ids[0][random_mask_idx:],
            ]
        )

        # Get the scores for all tokens
        tokens = list(self.predict_mask(mask_token_index, enlarged_ids))
        scores = [score.tolist() for _, score in tokens][0]

        # Select a token and fill it in
        chosen_token = random.choices(population=range(len(scores)), weights=scores)
        if len(chosen_token):
            enlarged_ids[0][random_mask_idx] = chosen_token[0]
            return enlarged_ids
        else:
            print("Error, no token found", chosen_token)
            return input_ids

    def remove_token_randomly(
        self, original_input_ids: Tensor, creator=long_tensor_creator, min_length=1
    ):

        # Clone, to not override actual vector
        input_ids = _clone(original_input_ids)

        # Pick a random mask position
        max_ind = len(input_ids[0]) - 2
        if max_ind <= min_length:
            return input_ids
        random_mask_idx = random.randint(1, max_ind)

        # Insert the mask into the sentence

        reduced_ids = creator(torch.zeros(1, len(input_ids[0]) - 1))
        reduced_ids[0] = Tensor(
            torch.cat(
                [
                    input_ids[0].float()[:random_mask_idx],
                    input_ids[0][random_mask_idx + 1 :],
                ],
            )
        )

        return reduced_ids

    def create_individual(self, tokenized_sentence: Tensor):
        return self.mutate_token_randomly(tokenized_sentence)

    def create_population_with_mutations(self, sentence, n):
        start_sentence_tokenized = self.tokenize_sentence(sentence)
        return tools.initRepeat(
            list, lambda: self.create_individual(start_sentence_tokenized), n=n
        )

    def create_population_creator(self, toolbox):
        def create_popopulation(sentence, n):
            start_sentence_tokenized = self.tokenize_sentence(sentence)
            return tools.initRepeat(
                list, lambda: toolbox.individual(start_sentence_tokenized), n=n
            )

        return create_popopulation


class ComparativeModel:
    """ Class for executing selection mechanisms using a model that compares text pairs"""

    def __init__(self, tokenizer, model, left_label=1, batch_size=10):
        self.tokenizer = tokenizer
        self.model = model
        self.left_label = left_label
        self.batch_size = batch_size

    def tokenize_sentence(self, sentence: str):
        return tokenize_sentence(self.tokenizer, sentence)

    def detokenize_sentence(self, pt: Tensor, include_limiters=False):
        return detokenize_sentence(self.tokenizer, pt, include_limiters)

    def encode_pair(self, left, right, block_size=512, mask_padding_with_zero=True):
        tokenized_text = self.tokenizer.encode(
            left[0].tolist(),
            text_pair=right[0].tolist(),
            truncation=True,
            max_length=block_size,
            padding=False
            # padding='max_length'
        )

        return pad_and_get_input_mask(
            tokenizer=self.tokenizer,
            block_size=block_size,
            mask_padding_with_zero=mask_padding_with_zero,
            tokenized_text=tokenized_text,
        )

    def encode_pairs(
        self,
        left_sentences,
        right_sentences,
        block_size=512,
        mask_padding_with_zero=True,
    ):
        return [
            self.encode_pair(left, right, block_size, mask_padding_with_zero)
            for left, right in zip(left_sentences, right_sentences)
        ]

    def compare(self, left: List, right: List) -> List:
        predicted_ids = []

        dataset = self.encode_pairs(left, right)

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):

                # Put batch on GPU
                if torch.cuda.is_available():
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to("cuda:0").long()

                # Calculate predictions
                results = self.model(**inputs)

                # Map to a concrete prediction & log
                predicted_ids.extend(results.logits.argmax(axis=1))

        return [
            left[idx] if pred == self.left_label else right[idx]
            for idx, pred in enumerate(predicted_ids)
        ]

    def select_using_comparison(self, population, selection_rounds=1):

        # Select random opponents
        for i in range(selection_rounds):
            # Create match up
            opponents_idx = [
                random.randint(0, len(population) - 1) for _ in range(len(population))
            ]
            opponents = [population[idx] for idx in opponents_idx]

            # Determine winners
            population = self.compare(population, opponents)

        return population


class RegressionModel:
    def __init__(self, tokenizer, model, batch_size=20):
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size

    def tokenize_sentence(self, sentence: str):
        return tokenize_sentence(self.tokenizer, sentence)

    def detokenize_sentence(self, pt: Tensor, include_limiters=False):
        return detokenize_sentence(self.tokenizer, pt, include_limiters)

    def encode(self, tokenized_sentence, block_size=512, mask_padding_with_zero=True):
        tokenized_text = self.tokenizer.encode(
            tokenized_sentence[0].long().tolist(),
            truncation=True,
            max_length=block_size,
            padding=False,
        )
        return pad_and_get_input_mask(
            tokenizer=self.tokenizer,
            block_size=block_size,
            mask_padding_with_zero=mask_padding_with_zero,
            tokenized_text=tokenized_text,
        )

    def score(self, population: List[Tensor]):
        dataset = [self.encode(p) for p in population]
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            all_results = []
            for i, inputs in enumerate(dataloader):

                # Put batch on GPU
                if torch.cuda.is_available():
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to("cuda:0").long()

                # Make the model predict the id of the label for every sentence
                results = self.model(**inputs)
                all_results.extend(results.logits)

            # Turn the prediction into a human readable label
            predicted_scores = [item.item() for item in all_results]
        return predicted_scores

    def evaluate_invalid_fitness(self, offspring, fitnessClass):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [
            ind
            for ind in offspring
            if not hasattr(ind, "fitness") or not ind.fitness.valid
        ]
        fitnesses = self.score(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            if not hasattr(ind, "fitness"):
                ind.fitness = fitnessClass()
            ind.fitness.values = [fit]

        return invalid_ind

    def create_evaluator(self, fitnessClass):
        def evaluate(offspring):
            return self.evaluate_invalid_fitness(offspring, fitnessClass)

        return evaluate


class LongTensorIndividual(Tensor):
    """Using this because otherwise a float will be returned"""

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).long()


def cx_one_point_tensor(ind1: Tensor, ind2: Tensor):
    c_ind1, c_ind2 = tools.cxOnePoint(ind1[0].tolist(), ind2[0].tolist())
    return Tensor([c_ind1]).long(), Tensor([c_ind2]).long()


def mate(toolbox, offspring, crossover_prob):
    population_size = len(offspring)
    random.shuffle(offspring)
    for i in range(population_size - 1):
        child1 = offspring[i]
        child2 = offspring[i + 1]
        if random.random() < crossover_prob:
            mutated_child1, mutated_child2 = toolbox.mate(child1, child2)
            offspring[i] = mutated_child1
            offspring[i + 1] = mutated_child2

    return offspring


def mutate(toolbox, offspring, mutation_prob):
    for idx, mutant in enumerate(offspring):
        if random.random() < mutation_prob:
            offspring[idx] = toolbox.mutate(mutant)
    return offspring


def _get_random_position_of_word(individual, word):
    return random.choice((individual[0] == word).nonzero(as_tuple=False).tolist())[0]


def _get_content_words_of_individual(individual):
    return set(individual[0].tolist()[1 : len(individual[0]) - 2])


def _cross_individuals(
    ind1: Tensor,
    ind2: Tensor,
    ind1_to: int,
    ind2_from: int,
    creator=long_tensor_creator,
):
    new_size = ind1_to + len(ind2[0]) - ind2_from
    crossed = creator(torch.zeros(1, new_size))
    crossed[0] = Tensor(torch.cat([ind1[0].float()[:ind1_to], ind2[0][ind2_from:]]))
    return crossed


def crossover_on_mutual_word(ind1: Tensor, ind2: Tensor, creator=long_tensor_creator):
    values1 = _get_content_words_of_individual(ind1)
    values2 = _get_content_words_of_individual(ind2)
    overlapping_words = list(values1.intersection(values2))

    # If there are overlapping words, cross on one of them
    if len(overlapping_words) > 0:
        chosen_word = random.choice(overlapping_words)
        ind1_position = _get_random_position_of_word(ind1, chosen_word)
        ind2_position = _get_random_position_of_word(ind2, chosen_word)

        new_1 = _cross_individuals(
            ind1, ind2, ind1_position, ind2_position, creator=creator
        )
        new_2 = _cross_individuals(
            ind2, ind1, ind2_position, ind1_position, creator=creator
        )

        return new_1, new_2

    return ind1, ind2


def create_crossover_on_mutual_word(creator):
    def curried(ind1: Tensor, ind2: Tensor):
        return crossover_on_mutual_word(ind1, ind2, creator=creator)

    return curried


def load_masker(model_name, max_length=512):
    mlm_tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=max_length
    )
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    # if torch.cuda.is_available():
    #     mlm_model.to('cuda:0')
    mlm_model.eval()

    # Create Genetic MLM model
    masker = MaskPredictor(mlm_tokenizer, mlm_model)
    return masker


def load_regression(model_name, max_length=512, batch_size=20):
    regression_tokenizer = RobertaTokenizer.from_pretrained(
        model_name, model_max_length=max_length
    )
    regression_model = RobertaForSequenceClassification.from_pretrained(
        model_name, return_dict=True
    )
    if torch.cuda.is_available():
        regression_model.to("cuda:0")
    else:
        print("Cuda not available")
    regression_model.eval()
    return RegressionModel(regression_tokenizer, regression_model, batch_size=batch_size)
