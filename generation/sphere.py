#!/usr/bin/env python3
# coding: utf-8

import base
import numpy as np


class GeneratorS(base.Generator):
    def __init__(self, n):
        self.dim = n

    def sample_unsafe(self):
        rand = np.random.rand(self.dim+1)
        x = 2 * rand - np.ones_like(rand)
        if np.linalg.norm(x) > 1:
            return None
        return x / np.linalg.norm(x)

    def sample(self):
        x = self.sample_unsafe()
        while x is None:
            x = self.sample_unsafe()
        return x


class GeneratorS1(base.Generator):
    def sample(self):
        u = np.random.rand() * 2 * np.pi
        return np.array([ np.cos(u), np.sin(u) ])


def main():
    GeneratorS1().sample_file(int(5e6), '/home/unkoll/UMAP/samples/S1.csv')
    GeneratorS(2).sample_file(int(5e6), '/home/unkoll/UMAP/samples/S2.csv')
    GeneratorS(3).sample_file(int(5e6), '/home/unkoll/UMAP/samples/S3.csv')
    GeneratorS(4).sample_file(int(5e6), '/home/unkoll/UMAP/samples/S4.csv')


if __name__ == '__main__':
    main()
