# coding: utf-8

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import progressbar as pb

class Generator(object):
    def sample(self):
        pass

    def sample_many(self, cnt):
        res = []
        for _ in range(cnt):
            res.append(self.sample())
        return np.array(res)

    def sample_file(self, cnt, path):
        widgets = [pb.Percentage(),
                ' ', pb.Bar(),
                ' ', pb.ETA()]
        pbar = pb.ProgressBar(widgets=widgets)
        pbar.maxval = cnt
        pbar.start()
        with open(path, 'w') as f:
            for _ in range(cnt):
                x = self.sample()
                f.write('\t'.join([str(coord) for coord in x]) + '\n')
                pbar.update(_+1)


class GeneratorMCMC(Generator):
    def __init__(self, density, manifold_func, dim, sigma, x0):
        # Density should be accurate to proportionality coefficient
        self.density = density
        self.manifold = manifold_func
        self.walking_density = lambda x: stats.multivariate_normal(mean=x, cov=sigma * np.eye(dim))
        self.prev = x0

    def sample(self):
        # Based on Metropolis-Hastings symmetric algorithm
        u = np.random.random()
        x = self.walking_density(self.prev).rvs()
        if u < min(1, self.density(x)/self.density(self.prev)):
            self.prev = x
            return self.manifold(x)
        return self.manifold(self.prev)
