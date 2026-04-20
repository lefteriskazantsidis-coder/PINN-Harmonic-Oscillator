# 🧲 Physics-Informed Neural Network — Simple Harmonic Oscillator

A PyTorch implementation of a **Physics-Informed Neural Network (PINN)** that reconstructs the full trajectory of a simple harmonic oscillator from only **5 noisy measurements**, by embedding Newton's 2nd Law directly into the training loss.

---

## 📐 The Physics

A mass **m** attached to a spring with constant **k** obeys Newton's 2nd Law:

$$m\ddot{x} + kx = 0$$

The analytical solution is:

$$x(t) = A\cos(\omega t + \phi), \quad \omega = \sqrt{\frac{k}{m}}$$

**Parameters used:** `m = 1 kg`, `k = 4 N/m`, `ω = 2 rad/s`, `A = 1 m`

---

## 🧠 The Challenge

We have only **5 sparse, noisy measurements** of `x(t)` in the first 60% of the period. Can a neural network recover the full trajectory — including the region with no data?

| Approach | Uses | Result |
|---|---|---|
| Standard NN | Data only | Fails — overfits the 5 points |
| **PINN** | Data + Physics law | ✅ Recovers the true solution |

---

## ⚙️ Loss Function

The PINN is trained by minimizing a composite loss:

$$\mathcal{L} = \underbrace{\frac{1}{N_d}\sum(x_{NN} - x_{data})^2}_{\mathcal{L}_{data}} + \lambda\underbrace{\frac{1}{N_f}\sum(m\ddot{x}_{NN} + kx_{NN})^2}_{\mathcal{L}_{physics}}$$

- **Data loss** — fit the 5 observations
- **Physics loss** — enforce the ODE at collocation points via automatic differentiation (`torch.autograd`)

---

## 🏗️ Architecture

```
t  →  [Linear(1→64), tanh]  →  [Linear(64→64), tanh] × 2  →  [Linear(64→1)]  →  x(t)
```

- `tanh` activations are used because they are **infinitely differentiable** — required for computing `ẍ` via `torch.autograd.grad`
- Xavier initialization for stable convergence

---

## 📦 Requirements

```bash
pip install torch numpy matplotlib
```

| Library | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| NumPy | ≥ 1.23 |
| Matplotlib | ≥ 3.5 |

---

## 🚀 Usage

Open and run the notebook:

```bash
jupyter notebook PINN_HarmonicOscillator.ipynb
```

The notebook is self-contained and runs sequentially. No external data files required.

---

## 📊 Results

The PINN successfully reconstructs `x(t)` across the full time domain, including the unobserved region, by leveraging the physical constraint. The standard neural network (trained on data only) diverges outside the observed window.

---

## 📁 File Structure

```
├── PINN_HarmonicOscillator.ipynb   # Main notebook
└── README.md
```

---

## 🔗 References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.
