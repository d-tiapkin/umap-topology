import numpy as np
import gudhi as g
import logging
import time
import os

logger = logging.getLogger(name="filtration")

def witness_complex_persistence(points, numb_landmarks, alpha, dim, field):
    start_time = time.time()
    landmarks = g.pick_n_random_points(
        points=points,
        nb_points=numb_landmarks)
    witness_complex = g.EuclideanWitnessComplex(
        witnesses=points,
        landmarks=landmarks)
    simplex_tree = witness_complex.create_simplex_tree(
        max_alpha_square=alpha**2,
        limit_dimension=dim)

    logger.info("Build complex in {} sec".format(time.time() - start_time))
    logger.info("{} vertices, {} simplices".format(
        simplex_tree.num_vertices(),
        simplex_tree.num_simplices()))

    start_time = time.time()
    simplex_tree.initialize_filtration()
    simplex_tree.persistence(homology_coeff_field=field)
    logger.info("Build persistence in {} sec".format(time.time() - start_time))

    return simplex_tree


def write_persistence(simplex_tree, file_name):
    simplex_tree.write_persistence_diagram(file_name)


def build_and_write_persistence(points, numb_landmarks, alpha, dim, field, file_name):
    write_persistence(witness_complex_persistence(points, numb_landmarks, alpha, dim, field), file_name)
