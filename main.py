#!/usr/bin/env python3
import lib.filtration as filt

import numpy as np
import multiprocessing
import umap

import yaml
import logging

#TODO: logging

DEFAULT_LEVEL = logging.DEBUG
formatter = logging.Formatter("%(levelname)s: %(asctime)s - %(name)s - %(process)s - %(message)s")

def asynch_persistence(points, config, pool, waiting_processes):
    if not config.skip:
        waiting_processes.append(
            pool.apply_async(filt.build_and_write_persistence, (
            points=         points,
            numb_landmarks= config.landmarks,
            alpha=          config.alpha,
            dim=            config.dim,
            file_name=      config.file)))


def main(config):
    logging.BasicConfig(filename="experiment.log", level=logging.INFO)
    logger = logging.getLogger("main")
    logger.info("Started...")

    points = np.genfromtxt(config.data_path, max_rows=config.samples)
    waiting_processes = []

    with multiprocessing.Pool(processes=config.nproc) as pool:
        asynch_persistence(points, config.persistence, pool, waiting_processes)

        # TODO: logging
        for reduction in config.dimension_reduction:
            reducer = umap.UMAP(
                n_neighbors=            reduction.umap.neighbors,
                n_components=           reduction.umap.dimension,
                n_epochs=               reduction.umap.epochs,
                learning_rate=          reduction.umap.learning_rate,
                min_dist=               reduction.umap.min_dist,
                spread=                 reduction.umap.spread,
                random_state=           reduction.umap.random_state,
                transform_seed=         reduction.umap.transform_seed,
                local_connectivity=     reduction.umap.local_connectivity,
                repulsion_strength=     reduction.umap.repulsion_strength,
                negative_sample_rate=   reduction.umap.negative_sample_rate
            )
            low_dimensional = pool.apply_async(reducer.fit_transform, (points))
            np.savetxt(reduction.umap.filename, low_dimensional.get())
            asynch_persistence(low_dimensional.get(), reduction.umap.persistence, pool, waiting_processes)

    for process in waiting_processes:
        process.get()


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    main(config)

