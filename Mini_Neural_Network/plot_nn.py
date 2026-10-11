import numpy as np
import matplotlib.pyplot as plt

from main import MLP, make_spirals

X,y = make_spirals()
net = MLP([2, 45, 45, 3])
history = net.fit(X, y, epochs=3000, lr=1.2, log_every=500)

epochs = [h[0] for h in history]
losses = [h[1] for h in history]
accs = [h[2] for h in history]

xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = net.predict(grid).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.contourf(xx, yy, alpha=0.3, cmap="viridis", levels=[-0.5, 0.5, 1.5, 2.5])
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=25)
ax1.set_title("Decision boundary (3-class spiral)")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")

ax2.plot(epochs, losses, color="tab:red", label="loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Cross-entropy loss", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.grid(alpha=0.3)
ax3 = ax2.twinx()
ax3.plot(epochs, accs, color="tab:blue", label="accuracy")
ax3.set_ylabel("Accuracy", color="tab:blue")
ax3.tick_params(axis="y", labelcolor="tab:blue")
ax3.set_ylim(0, 1.05)
ax2.set_title("Training progress")

plt.tight_layout()
plt.savefig("nn_results.png", dpi=150)
print("Saved nn_results.png")