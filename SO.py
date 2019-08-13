#!/usr/bin/env python3
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import umap
import gudhi as g

SAMPLES = 100000
LANDMARKS = 5000


def generate_witness_simplex_tree(points, numb_landmarks, a_square, dim):
    landmarks = g.pick_n_random_points(points=points, nb_points=numb_landmarks)
    witness_complex = g.EuclideanWitnessComplex(witnesses=points, landmarks=landmarks)
    return witness_complex.create_simplex_tree(max_alpha_square=a_square, limit_dimension=dim)


def SO2():
    points= np.genfromtxt('samples/SO2.csv', max_rows=SAMPLES)
    simplex_tree = generate_witness_simplex_tree(points, LANDMARKS, 0.1, 2)
    simplex_tree.initialize_filtration()

    print(simplex_tree.num_simplices())
    print(simplex_tree.num_vertices())

    simplex_tree.write_persistence_diagram('filtration/SO2_base.filt')


def SO3():
    # TODO: make it
    pass


def SO4():
    # TODO: make it
    pass


def main():
    SO2()
    SO3()
    SO4()


if __name__ == '__main__':
    main()
