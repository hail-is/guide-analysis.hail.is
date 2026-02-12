import os
import tempfile
import logging

import numpy as np
from scipy.stats import norm

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.info('loading zscores')
zscores = np.load('zscores.npy', mmap_mode='r')
with tempfile.TemporaryFile(dir=os.getcwd()) as tmpf:
    logging.info('mapping big array')
    l10pv = np.memmap(tmpf, dtype=np.float64, mode='w+', shape=zscores.shape)
    logging.info('copy zscores to data')
    np.copyto(l10pv, zscores)
    logging.info('data = abs(zscores)')
    np.abs(l10pv, out=l10pv)
    logging.info('data = norm.cdf(data)')
    for ix, data in enumerate(l10pv):
        l10pv[ix, :] = norm.cdf(data)
    logging.info('data = 1 - data')
    np.subtract(1, l10pv, out=l10pv)
    logging.info('data = 2 * data')
    np.multiply(2, l10pv, out=l10pv)
    logging.info('data = clip(data)')
    np.clip(l10pv, a_min=1e-300, a_max=None, out=l10pv)
    logging.info('data = log10(data)')
    np.log10(l10pv, out=l10pv)
    logging.info('data = -data')
    np.negative(l10pv, out=l10pv)
    logging.info('data = clip(data)')
    np.clip(l10pv, a_min=0, a_max=None, out=l10pv)

    logging.info('open out file')
    reall10pv = np.lib.format.open_memmap(
        'log10pvalues.npy', mode='w+', shape=l10pv.shape, dtype=np.float32
    )

    logging.info('copy out data')
    np.copyto(reall10pv, l10pv)
    logging.info('flushing out data')
    reall10pv.flush()

logging.info('attempt load data')
np.load('log10pvalues.npy', mmap_mode='r')
