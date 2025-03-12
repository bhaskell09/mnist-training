import matplotlib.pyplot as plt
import numpy as np
import matplotlib

current_backend = matplotlib.get_backend()
print(f"Current matplotlib backend: {current_backend}")

x = np.linspace(0, 10, 100)

plt.plot(x, x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = x')
plt.show()
