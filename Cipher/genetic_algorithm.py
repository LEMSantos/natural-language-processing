from operator import itemgetter
from random import sample, random, randint
from string import ascii_lowercase as letters
from typing import List, Callable, Dict

import numpy as np

from utils import decode_message, get_cipher_map


class GeneticAlgorithm:

    def __init__(self,
        fitness_func: Callable[[str], float],
        population_size: int=20,
        evolve_factor: int=3,
        evolve_probability: float=0.6,
        keep_best: int=2,
    ):
        self.__fitness_func = fitness_func
        self.__population_size = population_size
        self.__evolve_factor = evolve_factor
        self.__evolve_probability = evolve_probability
        self.__keep_best = keep_best

        self.__generation = self.__get_random_generation()

    def __get_random_generation(self):
        return [
            sample(letters, k=len(letters))
            for _ in range(self.__population_size)
        ]

    def __evolve_generation(
        self, generation: List[List[str]]
    ) -> List[List[str]]:
        evolved_generation = []

        for individual in generation:
            p = random()

            if p > self.__evolve_probability:
                evolved_generation.append(individual)
                continue

            new_ind = individual.copy()

            for _ in range(self.__evolve_factor):

                i = randint(0, len(new_ind) - 1)
                j = randint(0, len(new_ind) - 1)

                new_ind[i], new_ind[j] = new_ind[j], new_ind[i]

            evolved_generation.append(new_ind)

        return evolved_generation

    def search(self, encoded_message: str, iterations: int=1000) -> List[str]:
        for iteration in range(iterations):
            fitness = []

            for individual in self.__generation:
                cipher_map = get_cipher_map(individual)
                decoded_message = decode_message(encoded_message, cipher_map)

                fitness.append(self.__fitness_func(decoded_message))

            sorted_by_fitness = [
                x for x, _ in sorted(
                    zip(self.__generation, fitness),
                    key=itemgetter(1),
                    reverse=True,
                )
            ]

            best_individuals = sorted_by_fitness[:self.__keep_best]
            evolved_generation = self.__evolve_generation(sorted_by_fitness)

            self.__generation = best_individuals + evolved_generation[:-self.__keep_best]

            if iteration % 200 == 0:
                msg = (
                    f'iter: {iteration} '
                    f'mean fitness: {np.mean(fitness)} '
                    f'best so far: {max(fitness)}'
                )

                print(msg)

        return sorted_by_fitness[0]
