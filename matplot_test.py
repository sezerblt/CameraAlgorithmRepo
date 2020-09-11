import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as pimg

file="C:\\Users\\rootx\\Desktop\\logo.jpg"

myimage = pimg.imread(file)

#plt.imshow(myimage)
plt.hist(myimage.ravel(),bins=256,range=(0.0,1.0),fc='k',ec='k')
plt.show()
#-----------------

