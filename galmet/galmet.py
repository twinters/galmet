""" Genetic Algorithm using Language Models for Evolving Text """

# Helper functions
import random

import editdistance
import torch
from deap.tools import ParetoFront

import src.genetictransformer as gt

from deap import tools, creator, base
import numpy as np
from torch import Tensor


def calculate_edit_distance(start_sentence, current_sentence):
    start = start_sentence[0].tolist()
    now = current_sentence[0].tolist()

    distance = editdistance.eval(start, now)
    return distance


def eq_individual(ind1, ind2):
    return ind1.shape == ind2.shape and torch.all(ind1.eq(ind2))


# def max_fitness(population):
#     return max([ind.fitness.values[0] for ind in population])


class LongTensorIndividual(Tensor):
    """Using this because otherwise a float will be returned"""

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Tensor):
            return super().__new__(cls, args[0].float(), **kwargs).long()
        return super().__new__(cls, *args, **kwargs).long()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        own_fitness = self.fitness.values if self.fitness else None
        deepcopy_tensor = super().__deepcopy__(memo)
        deepcopy = LongTensorIndividual(deepcopy_tensor.float())
        deepcopy.fitness = creator.FitnessMax()
        if own_fitness:
            deepcopy.fitness.values = tuple(own_fitness)
        return deepcopy


# Keep note of the evolution
def create_stats():
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_distance = tools.Statistics(key=lambda ind: int(abs(ind.fitness.values[1])))

    mstats = tools.MultiStatistics(fitness=stats_fit, distance=stats_distance)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "distance", "best"
    logbook.chapters["fitness"].header = "min", "avg", "max"
    logbook.chapters["distance"].header = "min", "avg", "max"

    return mstats, logbook


class Galmet:
    def __init__(
        self,
        masker,
        regression,
        crossover_prob=0.15,
        total_mutation_prob=0.8,
        population_size=50,
        nb_generations=50,
        goal_fitness=1,
        max_edit_distance=6,
        max_elites=6,
        elite_duplicates=3,
        mutate_token_prob=0.7,
        add_token_prob=0.1,
        remove_token_prob=0.1,
    ):
        self.masker = masker
        self.regression = regression

        self.toolbox = self.create_toolbox()

        self.crossover_prob = crossover_prob
        self.total_mutation_prob = total_mutation_prob
        self.population_size = population_size
        self.nb_generations = nb_generations
        self.goal_fitness = goal_fitness
        self.max_edit_distance = max_edit_distance
        self.max_elites = max_elites
        self.elite_duplicates = elite_duplicates
        self.mutate_token_prob = mutate_token_prob
        self.add_token_prob = add_token_prob
        self.remove_token_prob = remove_token_prob

    def create_toolbox(self):

        # Create toolbox for DEAP
        toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1))
        creator.create("Individual", LongTensorIndividual, fitness=creator.FitnessMax)
        toolbox.register(
            "individual",
            lambda tokenized_sentence: creator.Individual(tokenized_sentence.tolist()),
        )
        toolbox.register("population", self.masker.create_population_creator(toolbox))

        # toolbox.register("select", comparator.select_using_comparison)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", gt.create_crossover_on_mutual_word(creator.Individual))
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("tokenize", self.masker.tokenize_sentence)
        return toolbox

    def has_low_enough_edit_distance(self, ind):
        return int(abs(ind.fitness.values[1])) <= self.max_edit_distance

    def get_individuals_with_highest_score_fitness(self, population):
        max_fit = max([ind.fitness.values[0] for ind in population])
        best_sentences = [ind for ind in population if ind.fitness.values[0] == max_fit]
        return best_sentences

    def get_best(self, population):
        max_valid_fit = max(
            [
                ind.fitness.values[0]
                for ind in population
                if self.has_low_enough_edit_distance(ind)
            ]
        )
        if max_valid_fit:
            # Filter out all sentences that are above goal fitness, or have the maximum valid fitness
            sentences_above_threshold = [
                ind
                for ind in population
                if (
                    ind.fitness.values[0] > self.goal_fitness
                    or ind.fitness.values[0] == max_valid_fit
                )
                and self.has_low_enough_edit_distance(ind)
            ]

            lowest_edit_distance = min(
                int(abs(ind.fitness.values[1])) for ind in sentences_above_threshold
            )
            # Filter out best sentences with higher edit distance
            lowest_edit_distance_above_score_threshold = [
                ind
                for ind in sentences_above_threshold
                if int(abs(ind.fitness.values[1])) == lowest_edit_distance
            ]
            best_sentences = self.get_individuals_with_highest_score_fitness(
                lowest_edit_distance_above_score_threshold
            )

        # If none found, do without edit distance constraint
        else:
            print("None found with edit distance constraint")
            best_sentences = self.get_individuals_with_highest_score_fitness(population)

        return self.masker.detokenize_sentence(best_sentences[0])

    def get_elites(self, hof):
        elites = list(reversed(hof))
        if len(elites) > self.max_elites:
            elites = elites[: self.max_elites]
        return elites

    def found_suitable(self, population):
        suitables = [
            p
            for p in population
            if p.fitness.values[0] >= self.goal_fitness
            and self.has_low_enough_edit_distance(p)
        ]
        return len(suitables) > 0

    def mutate(self, ind: Tensor):
        if random.random() < self.mutate_token_prob:
            ind = self.masker.mutate_token_randomly(ind, creator=creator.Individual)

        if random.random() < self.add_token_prob:
            ind = self.masker.add_token_randomly(ind, creator=creator.Individual)

        if random.random() < self.remove_token_prob:
            ind = self.masker.remove_token_randomly(ind, creator=creator.Individual)

        return ind

    def evaluate(self, population, start_sentence):
        # First normal evaluation
        changed = self.regression.evaluate_invalid_fitness(
            population, creator.FitnessMax
        )

        # Then tweak the fitnesses based on distance to start sentence
        for c in changed:
            c.fitness.values = (
                c.fitness.values[0],
                -calculate_edit_distance(start_sentence, c),
            )

        return changed

    def evolve_text(
        self,
        start_sentence,
        stats_creator=create_stats,
        print_logbook=False,
        print_final_population=False,
    ):
        # Variable keeping track of the number of generations & hall of fame
        g = 0
        # hof = ParetoFront(similar=lambda x, y: torch.all(x.eq(y)))
        hof = ParetoFront(similar=eq_individual)

        # Initialize population
        tokenized_start_sentence = self.toolbox.tokenize(start_sentence)
        pop = self.toolbox.population(start_sentence, n=self.population_size)

        # Evaluate the entire population
        self.toolbox.evaluate(pop, tokenized_start_sentence)
        hof.update(pop)

        # Statistics
        stats, logbook = stats_creator()

        # Begin the evolution
        while not self.found_suitable(population=pop) and g < self.nb_generations:
            # A new generation
            g = g + 1

            # Select the next generation individuals
            # Get best one for each edit distance
            elites = self.get_elites(hof)
            offspring = self.elite_duplicates * elites + self.toolbox.select(
                pop, len(pop) - self.elite_duplicates * len(elites)
            )

            # Apply crossover and mutation on the offspring
            gt.mate(self.toolbox, offspring, self.crossover_prob)

            # Mutate the individuals
            gt.mutate(self.toolbox, offspring, self.total_mutation_prob)

            changed_fitness = self.toolbox.evaluate(offspring, tokenized_start_sentence)
            hof.update(offspring)

            pop[:] = offspring

            # Statistics
            record = stats.compile(pop)
            logbook.record(
                gen=g, evals=len(changed_fitness), best=self.get_best(pop), **record
            )
            if print_logbook:
                print(logbook.stream)

        if print_final_population:
            pop = sorted(pop, key=lambda x: (-x.fitness.values[1], x.fitness.values[0]))
            print(
                "\nFinal population!\n",
                "\n".join(
                    str(int(abs(t.fitness.values[1])))
                    + ": "
                    + ("%f" % t.fitness.values[0])
                    + ": "
                    + self.masker.detokenize_sentence(t)
                    for t in pop
                ),
                sep="",
            )
        return self.get_best(hof), hof, logbook
