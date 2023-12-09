import numpy as np
import matplotlib.pyplot as plt 

x_min, x_max = -1, 1
y_min, y_max = -1, 1
step = 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
print(xx)
print(yy)
Z = np.sin(xx**2 + yy**2)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
plt.show()
