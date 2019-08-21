#!/usr/bin/env python3
import numpy as np
import gudhi as g
import logging
import time
import os


def witness_complex_persistence(points, numb_landmarks, alpha, dim, logger):
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

    logger.info("Process: {}. Build complex in {} sec".format(os.getpid(), time.time() - start_time))
    logger.info("Process: {}. {} vertices, {} simplices".format(
        os.getpid(),
        simplex_tree.num_vertices(),
        simplex_tree.num_simplices()))

    start_time = time.time()
    simplex_tree.initialize_filtration()
    simplex_tree.persistence()
    logger.info("Process: {}. Build persistence in {} sec".format(os.getpid(), time.time() - start_time))

    return simplex_tree


def main(file_samples, file_diagram, numb_samples, numb_landmarks, alpha, dim):
    logger = logging.basicConfig(filename="filtration.log", level=logging.INFO)
    logger = logging.getLogger()

    points = np.genfromtxt(file_samples, max_rows=numb_samples)
    logger.info("Generating witness complex with {} points from {} using {} landmarks".format(
        numb_samples,
        file_samples,
        numb_landmarks))
    logger.info("Paramters: alpha={}, dim={}".format(alpha, dim))
    simplex_tree = witness_complex_persistence(points, numb_landmarks, alpha, dim, logger)
    simplex_tree.write_persistence_diagram(file_diagram)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building witness complex and calculate perstence homology of its filtration')
    parser.add_argument('--input', help='Input file with samples')
    parser.add_argument('--output', help='Output file with diagram')
    parser.add_argument('--samples', help='Number of samples', type=int)
    parser.add_argument('--landmarks', help='Number of landmarks', type=int)
    parser.add_argument('--alpha', help='Alpha', type=float)
    parser.add_argument('--dim', help='Dimension of manifold', type=int)
    args = parser.parse_args()
    main(args.input, args.output, args.samples, args.landmarks, args.alpha, args.dim)
