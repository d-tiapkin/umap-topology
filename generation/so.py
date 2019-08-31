#!/usr/bin/env python3
# coding: utf-8

import base
import numpy as np

import sphere
import sympy as sp
from sympy.abc import x, y, z


import scipy.stats as stats


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


class GeneratorSO3(base.GeneratorMCMC):
    def __init__(self, sigma, x0):
        U = sp.Matrix([
            [sp.cos(x) * sp.cos(y),     -sp.sin(y),     sp.sin(x) * sp.cos(y)],
            [sp.cos(x) * sp.sin(y),     sp.cos(y),      sp.sin(x) * sp.sin(y) ],
            [-sp.sin(x),                0.0,            sp.cos(x)]])

        D = sp.Matrix([
            [sp.cos(z),     -sp.sin(z),     0.0],
            [sp.sin(z),     sp.cos(z),      0.0],
            [0.0,             0.0,          1.0]])

        V = U @ D @ U.T

        Vx = np.array(V.diff(x))
        Vy = np.array(V.diff(y))
        Vz = np.array(V.diff(z))

        G = sp.Matrix([Vx.reshape((9)), Vy.reshape((9)), Vz.reshape((9))])

        func = lambda val: np.array(V.subs(x, val[0]).subs(y, val[1]).subs(z, val[2])).reshape((9))
        def dens_func(val):
            if not(-np.pi/2.0 <= val[0]  <= np.pi/2.0 and 0 <= val[1]  <= np.pi and 0 <= val[2] <= 2 * np.pi):
                return 0.0
            Gx = G.subs(x, val[0])
            Gxy = Gx.subs(y, val[1])
            G_subs = Gxy.subs(z, val[2])
            return sp.sqrt((G_subs @ G_subs.T).det())

        base.GeneratorMCMC.__init__(self, dens_func, func, 3, sigma, x0)

def main():
    #GeneratorSO2().sample_file(int(5e6), '/home/unkoll/UMAP/samples/SO2.csv')
    GeneratorSO3(0.25, [np.pi/6.0, np.pi/6.0, np.pi/6.0]).sample_file(int(1e5), '/home/unkoll/UMAP/samples/SO3.csv')


if __name__ == '__main__':
    main()
