#!/usr/bin/env python3
# coding: utf-8

import base
import numpy as np

import sphere


class GeneratorSO2(sphere.GeneratorS1):
    def __init__(self):
        # base matrices were simplified to a vector view
        # Absent of a divider is not a error
        self.base_matrix1 = np.array(
            [ 1, 0 ,
              0, 1 ])
        self.base_matrix2 = np.array(
            [  0, 1,
              -1, 0 ])

    def sample(self):
        a,b = sphere.GeneratorS1.sample(self)
        return a * self.base_matrix1 + b * self.base_matrix2


def main():
    GeneratorSO2().sample_file(int(5e6), '/home/unkoll/UMAP/samples/SO2.csv')

if __name__ == '__main__':
    main()
