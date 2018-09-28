import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = x*0
y2 = -x

plt.plot(x, y, 'b--', linewidth=2)
plt.fill([-10, 10, 10, -10], [0, 0, -10, -10], facecolor='none', alpha=0.5, hatch='|', edgecolor='b')

plt.plot(y, x, 'g--', linewidth=2)
plt.fill([-10, 0, 0, -10], [10, 10, -10, -10], facecolor='none', alpha=0.5, hatch='-', edgecolor='g')

plt.plot(x, y2, 'r', linewidth=2)
plt.fill([-10, 10, 10], [10, 10, -10], facecolor='none', alpha=0.5, hatch='/', edgecolor='r')

plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()