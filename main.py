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

#we have a data file generated from concatemer data. Let's restrict ourselves to chromosome 8 for now (sneakily interested in seeing that tyfonas junction)

def load_concats(filename = 'data_example_skinny',chr_focus = False):
    data = pd.read_csv(filename)
    
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
    energyfun = lambda x: concat_energy(x,data)
    negforcefun = jax.jacrev(energyfun)
    hessfun = jax.hessian(energyfun)
    return energyfun,negforcefun,hessfun 

def concat_energy(coords,concats):
    #given coordinate data, evaluate the energy contribution from a single concatemer
    energy = 0
#    coords = coords.reshape(int(coords.size/2),2)
    for concat in concats:
        distances = my_pdist(coords[concat]) - 1 #want an equilibrium distance of 1
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


def lazy_test():
    cats,nb = load_concats(filename='data_test',chr_focus='a')
    initc = np.random.rand(nb*2)*nb
    energy,negforce,hess = getfuncs(cats)
    print(energy(initc))
  
    step = 0.05
    nstep = 1000
    coords = initc.copy()
    eps = 1e-10
    en = 1
    coords_hist = []
    save_frame = 10
    i = 0
    while en > eps:
        dc = (-1)*step*negforce(coords)
        coords += dc
        if i%save_frame==0:
            coords_hist.append(coords.reshape(int(coords.size/2),2))
        en = energy(coords)
        print(en)
        i+=1
    coords = coords.reshape(int(coords.size/2),2)
    return coords_hist

def animate_soln(coords_hist,cats,fname='bla'):
    nt = len(coords_hist)
    
    fig,ax = plt.subplots()
    line, = ax.plot(coords_hist[0][:,0],coords_hist[0][:,1],'.') 
    catlines = []
    for cat in cats:
        a, = ax.plot(coords_hist[0][cat][:,0],coords_hist[0][cat][:,1],'-') 
        catlines.append(a)
    
    print(catlines[0])
    def animate(i):
        print('Frame '+str(i)+'/'+str(nt))
        line.set_data(coords_hist[i][:,0],coords_hist[i][:,1])
#        for i in range(len(cats)):
#            cat = cats[i]
#            catline = catlines[i]
#            catline.set_data(coords_hist[i][cat][:,0],coords_hist[i][cat][:,1],'-')
         
        return line,catlines
    
    anim = FuncAnimation(fig, animate,frames=nt, blit=True)
    
    anim.save('movies/'+fname+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

