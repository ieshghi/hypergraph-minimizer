import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
#from jax.scipy.optimize import minimize
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from jax.nn import logsumexp
from scipy.special import comb
import multiprocessing as mlp
#we have a data file generated from concatemer data. Let's restrict ourselves to chromosome 8 for now (sneakily interested in seeing that tyfonas junction)


pool = mlp.Pool()
def load_concats(filename = 'data_example_skinny',chr_focus = False):
    data = pd.read_csv(filename,nrows=1000)
    
    if chr_focus:
        data = data[data.seqnames==chr_focus]

    data['binid'] = data.binid.astype(int)

    bid = list(np.unique(data.binid))
    newbids = list(range(len(bid)))
    res = {bid[i]: newbids[i] for i in range(len(bid))}
    numbin = max(newbids)+1

    data['nbid'] = data.binid.map(res)

    concatemers = list(data.groupby(by='cid')['nbid'].apply(list).values)
    concatemers = [jnp.array(l) for l in concatemers if len(l)>1]
    return concatemers,numbin

def getfuncs(data): 
    energyfun = lambda x: concat_energies(x,data)
    negforcefun = jax.jacrev(energyfun)
    hessfun = jax.hessian(energyfun)
    return energyfun,negforcefun,hessfun 

def concat_energies(coords,concats): #this is where all the speeding up can happen
    #given coordinate data, evaluate the energy contribution from a single concatemer
    coords = coords.reshape(int(coords.size/2),2)
    #fun = lambda x: concat_energy(coords,x)
    energy = 0
    distlist = map(lambda x:(my_pdist(coords[x])-1),concats)
    distsqlist = map(lambda x:(logsumexp(x**2)*len(x)),distlist)
    energy = sum(distsqlist)
    
    #for cat in concats:
    #    distances = (my_pdist(coords[cat]) - 1) #want an equilibrium distance of 1
    #    distsqmax = logsumexp(distances**2) #JAX-compatible logsumexp()
    #    energy += comb(len(cat),2)*distsqmax
    return energy
#    all_ens = pool.map(fun,concats)
#    return sum(all_ens)

def concat_energy(coords,cat):
    distances = (my_pdist(coords[cat]) - 1) #want an equilibrium distance of 1
    distsqmax = logsumexp(distances**2) #JAX-compatible logsumexp()
    return comb(len(cat),2)*distsqmax

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

def lazy_test():
    cats,nb = load_concats(filename='data_example_skinny',chr_focus='chr8')
    initc = np.random.rand(nb*2)*nb
    energy,negforce,hess = getfuncs(cats)
    print(energy(initc))
  
    step = 0.01
    coords = initc.copy()
    eps = 1e-1
    delta = 1
    coords_hist = []
    save_frame = 1
    i = 0
    en_prev = 0
    while delta > eps:
        dc = (-1)*step*negforce(coords)
        coords += dc
        if i%save_frame==0:
            coords_hist.append(coords.reshape(int(coords.size/2),2))
        en = energy(coords)
        delta = abs(en-en_prev)
        en_prev = en.copy()
        print(en,delta)
        i+=1
    coords = coords.reshape(int(coords.size/2),2)
    return coords_hist

def animate_soln(coords_hist,cats,fname='bla'):
    nt = len(coords_hist)
    
    fig,ax = plt.subplots()
    line, = ax.plot(coords_hist[0][:,0],coords_hist[0][:,1],'.') 
#    catlines = []
#    for cat in cats:
#        a, = ax.plot(coords_hist[0][cat][:,0],coords_hist[0][cat][:,1],'-') 
#        catlines.append(a)
    
#    print(catlines[0])
    def animate(i):
        print('Frame '+str(i)+'/'+str(nt))
        line.set_data(coords_hist[i][:,0],coords_hist[i][:,1])
#        for i in range(len(cats)):
#            cat = cats[i]
#            catline = catlines[i]
#            catline.set_data(coords_hist[i][cat][:,0],coords_hist[i][cat][:,1],'-')
         
        return line,
    
    anim = FuncAnimation(fig, animate,frames=nt, blit=True)
    
    anim.save('movies/'+fname+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

