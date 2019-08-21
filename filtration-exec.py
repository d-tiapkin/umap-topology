#!/usr/bin/env python3
import numpy as np
import logging

import lib.filtration as filt


def main(file_samples, file_diagram, numb_samples, numb_landmarks, alpha, dim):
    logging.basicConfig(filename="filtration.log", level=logging.INFO)
    logger = logging.getLogger(name="main")

    points = np.genfromtxt(file_samples, max_rows=numb_samples)
    logger.info("Generating witness complex with {} points from {} using {} landmarks".format(
        numb_samples,
        file_samples,
        numb_landmarks))
    logger.info("Paramters: alpha={}, dim={}".format(alpha, dim))
    build_and_write_persistence(points, numb_landmarks, alpha, dim, file_diagram)


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
