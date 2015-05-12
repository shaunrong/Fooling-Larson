#!/usr/bin/env python
from larson.GSOM import GSOM
import numpy as np
import scipy.misc.pilutil as smp
from random import randint, uniform

SCALE = 10 #easy number to see
N = 3 #pixels are composed of R,G,B values

class pixelMap:
    def __init__(self,alpha=0.1,lam=10,tou=0.001,n=N):
        self.SOM = GSOM(alpha,lam,tou,n)
        self.n = n
        self.iter = 0

    def __convertToPixelMap(self,colormap, scale):
        if self.n != 3:
            raise ValueError("The vectors aren't of size 3, cannot make RGB map.")
        return 256*np.kron(colormap, np.ones((scale,scale,1)))    

    def update(self,feature):
        self.SOM.update(feature)

    def update_random(self):
        vector = np.random.rand(self.n)
        self.update(vector)

    def update_controlled(self):
        vector = np.arange(self.n,dtype=float)
        for i in xrange(self.n):
            if ((self.iter/2**i) % 2) == 0:
                vector[i] = uniform(0.,0.2)
            else:
                vector[i] = uniform(0.8,1.0)
        self.iter = (self.iter + 1) % (2**self.n)
        self.update(vector)

    def show(self):
        img = smp.toimage(self.__convertToPixelMap(self.SOM.map,SCALE))
        img.show()

    @property
    def converged(self):
        return self.SOM.converged

if __name__ == '__main__':
    pm = pixelMap()
    i = 0
    while(not pm.converged):
        if i % 500 == 0:
            pm.show()
            raw_input()
        i += 1
        if i % 100 == 0:
            print i
        pm.update_controlled()
    pm.show()
        
