import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
X,Y=np.meshgrid(x,y)
Z=np.exp(-X**2-Y**2)
plt.contourf(X,Y,Z,cmap='jet')