import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
#from jax.scipy.optimize import minimize
from scipy.optimize import minimize

#we have a data file generated from concatemer data. Let's restrict ourselves to chromosome 8 for now (sneakily interested in seeing that tyfonas junction)

def load_concats(filename = 'data_example_skinny',chr_focus = False):
    data = pd.read_csv(filename)
    
    if chr_focus:
        data = data[data.seqnames==chr_focus]

    data['binid'] = data.binid.astype(int)

    bid = list(np.unique(data.binid))
    newbids = list(range(len(bid)))
    res = {bid[i]: newbids[i] for i in range(len(bid))}
    numbin = max(newbids)

    data['nbid'] = data.binid.map(res)

    concatemers = list(data.groupby(by='cid')['nbid'].apply(list).values)
    concatemers = [jnp.array(l) for l in concatemers if len(l)>1]
    return concatemers,numbin

def minrep(data,numbin):   
    init_coords = np.random.rand(numbin,2)*numbin #2d position of every bin, on a uniform grid [0,nbin]x[0,nbin]
    funtomin = lambda x: concat_energy(x,data)
    init_flat = init_coords.reshape(init_coords.size,)
    jacob = jax.jacrev(funtomin)
    hess = jax.hessian(funtomin)
    mincoord = minimize(funtomin,init_flat,jac = jacob,hess = hess)
    return mincoord


def concat_energy(coords,concats):
    #given coordinate data, evaluate the energy contribution from a single concatemer
    energy = 0
    coords = coords.reshape(int(coords.size/2),2)
    for concat in concats:
        distances = my_pdist(coords[concat])
        distsqmax = jnp.log(jnp.sum(jnp.exp(distances**2))) #JAX-compatible logsumexp()
        energy += distsqmax
    return energy

def my_pdist(X): #copied from scipy.spatial.distance, to make it Jax-compatible
    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = jnp.empty(out_size,)
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm = dm.at[k].set(jnp.linalg.norm(X[i]-X[j]))
            k += 1
    return dm

