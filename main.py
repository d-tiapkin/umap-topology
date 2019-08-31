#!/usr/bin/env python3
import lib.filtration as filt

import numpy as np
import multiprocessing
import multiprocessing_logging
import umap

import yaml
import logging
import sys
import time

def calc_persistence(points, config):
    if config is None or config['skip']:
        return
    logger = logging.getLogger("calc_persistence")

    logger.info("Start calculating persistence...")
    filt.build_and_write_persistence(
        points[:config['samples']],
        config['landmarks'],
        config['alpha'],
        config['dim'],
        config['field'],
        config['file'])


def calc_reduction(points, config):
    if config is None:
        return
    if config['umap'] is None:
        return

    print(config)
    config = config['umap']

    start_time = time.time()
    logger = logging.getLogger("calc_reduction")
    logger.info("Start calculating dimension reduction...")
    reducer = umap.UMAP(
        n_neighbors=            config['neighbors'],
        n_components=           config['dimension'],
        n_epochs=               config['epochs'],
        learning_rate=          config['learning_rate'],
        min_dist=               config['min_dist'],
        spread=                 config['spread'],
        random_state=           config['random_state'],
        transform_seed=         config['transform_seed'],
        local_connectivity=     config['local_connectivity'],
        repulsion_strength=     config['repulsion_strength'],
        negative_sample_rate=   config['negative_sample_rate']
    )
    low_dimensional = reducer.fit_transform(points)
    logger.info("Reduced dimension in {} sec".format(time.time() - start_time))
    np.savetxt(config['file'], low_dimensional)
    calc_persistence(low_dimensional, config['persistence'])


def main(config):
    logging.basicConfig(
        filename='log.log',
        #stream=sys.stdout,
        format=u'%(levelname)s: %(asctime)s - %(name)s - %(process)s - %(message)s',
        level=logging.DEBUG)
    multiprocessing_logging.install_mp_handler()

    logger = logging.getLogger("main")
    logger.info('Started...')

    points = np.genfromtxt(config['data_path'], max_rows=config['samples'])
    logger.info('Get {} points. Start calculation...'.format(len(points)))

    pool = multiprocessing.Pool(processes=config['nproc'])
    pool.apply_async(calc_persistence, [points, config['persistence']])
    for reducing_config in config['dimension_reduction']:
        pool.apply_async(calc_reduction, [points, reducing_config])
    pool.close()
    pool.join()


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    main(config)

