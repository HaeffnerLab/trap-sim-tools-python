mport pickle
import numpy as np

#Load:
files = [ open( 'atrap_cct_simulation_{}.pkl'.format(j) ) for j in range(1,8) ]
pkls = [ pickle.load(f) for f in files ]

#Concatenate:
xlen = len(pkls[0].X)
ylen = len(pkls[0].Y)
pkl_output = pkls[0]
for el in range(1,24):
    el_name = 'EL_DC_{}'.format(el)
    el_pot_tuple  = tuple( [ pkl[el_name] for pkl in pkls ] )
    pkl_output[el_name] = np.concatenate( el_pot_tuple, axis=2 )
    
#Save output:
f = open('data.pkl','w')
pickle.dump(pkl_output,f)
