import pandas as pd, matplotlib.pyplot as plt
clean = pd.read_csv("clean.csv")
noisy = pd.read_csv("noisy.csv")
plt.plot(clean["update"], clean["eval_return"], label="MAPPO, perfect perception")
plt.plot(noisy["update"], noisy["eval_return"], label="MAPPO, noisy perception")
plt.axhline(94, linestyle="--", color="gray", label="random")
plt.axhline(185, linestyle="--", color="black", label="greedy, no noise")
plt.axhline(157, linestyle=":", color="black", label="greedy, noise 0.3")
plt.xlabel("update"); plt.ylabel("eval return"); plt.legend(); plt.show()