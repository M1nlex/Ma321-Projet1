import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(8,6))
ax3d = plt.axes(projection="3d")

xdata = np.linspace(-100,100,1000)
ydata = np.linspace(-100,100,1000)
X,Y = np.meshgrid(xdata,ydata)
Z = X**2 + Y**2 -X*Y - X - Y
#Z = np.sin(X)*np.cos(Y)

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(X, Y, Z,cmap='plasma')
ax3d.set_title('Surface')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

plt.show()
