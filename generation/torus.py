#!/usr/bin/env python3
# coding: utf-8

import base
import numpy as np


def Indicator(x, x_lower, x_upper):
    return x_lower <= x <= x_upper


class GeneratorT2(base.GeneratorMCMC):
    def __init__(self, r, R, sigma, x0):
        manifold_func = lambda x: np.array(((R + r*np.cos(x[1]))*np.cos(x[0]), (R + r*np.cos(x[1]))*np.sin(x[0]), r*np.sin(x[1])))
        density = lambda x: (R + r * np.cos(x[1])) * float(Indicator(x[0], 0, 2*np.pi) *Indicator(x[1], -np.pi, np.pi))
        base.GeneratorMCMC.__init__(self, density, manifold_func, 2, sigma, x0)


import sphere

class GeneratorT(sphere.GeneratorS1):
    def __init__(self, n):
        self.dim = n

    def sample(self):
        return np.concatenate([ sphere.GeneratorS1.sample(self) for i in range(self.dim) ])


def main():
    #GeneratorT2(r=0.3, R=1.0, sigma=0.2, x0=np.array([0,0])).sample_file(int(5e6), '/home/unkoll/UMAP/samples/T2_R3.csv')
    GeneratorT(2).sample_file(int(5e6), '/home/unkoll/UMAP/samples/T2.csv')
    GeneratorT(3).sample_file(int(5e6), '/home/unkoll/UMAP/samples/T3.csv')
    GeneratorT(4).sample_file(int(5e6), '/home/unkoll/UMAP/samples/T4.csv')


if __name__ == '__main__':
    main()
