'''
Reads vtk files from Kwame pipeline into numpy arrays
'''

import sys
import numpy as np
import pickle

def main():
    filename = sys.argv[1]

    with open(filename) as fn:
        for i in range(5):
            fn.readline()
        
        num_V = int(str.split(fn.readline())[1])
        V = np.zeros((num_V, 3))   
        
        for i in range(num_V):
            l = fn.readline()
            V[i][0] = float(str.split(l)[0])
            V[i][1] = float(str.split(l)[1])
            V[i][2] = float(str.split(l)[2])

        num_F = int(str.split(fn.readline())[1])
        F = np.zeros((num_F, 3))

        for i in range(num_F):
            l = fn.readline()
            F[i][0] = int(str.split(l)[1])
            F[i][1] = int(str.split(l)[2])
            F[i][2] = int(str.split(l)[3])

        num_W = int(str.split(fn.readline())[1])
        W = np.zeros(num_W)
        fn.readline()
        fn.readline()
	 
        for i in range(num_W):
            W[i] = float(fn.readline())

    path = 'hippocampus/thicknessMap/dataframes/brain2/'

    if sys.argv[2] == 'midsurface':
        with open(path + sys.argv[3], "wb") as output:
            pickle.dump(V, output)

        with open(path + sys.argv[4], "wb") as output:
            pickle.dump(F, output)

    else:
        with open(path + sys.argv[3], "wb") as output:
            pickle.dump(W, output)

    print(V)
    print(F)
    print(W)
    return V, F, W
	    
if __name__ == "__main__":
    main()	
    
