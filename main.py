import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import logsumexp

#we have a data file generated from concatemer data. Let's restrict ourselves to chromosome 8 for now (sneakily interested in seeing that tyfonas junction)

data = pd.read_csv('data_example')

data = data[data.seqnames=='chr8']
data['binid'] = data.binid.astype(int)
nbins = int(max(data.binid))

concatemers = list(data.groupby(by='cid')['binid'].apply(list).values)
concatemers = [l for l in concatemers if len(l)>1]

coords = np.random.rand((nbins,2))*nbins #2d position of every bin, on a uniform grid [0,nbin]x[0,nbin]

def concat_energy(coords,concats):
	#given coordinate data, evaluate the energy contribution from a single concatemer
	energy = 0
	for concat in concats
		distances = pdist(coords[concat])
                distsqmax = logsumexp(distances**2)
		energy += distsqmax


