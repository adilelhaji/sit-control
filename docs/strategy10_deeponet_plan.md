# Strategy 10 — DeepONet Implementation Plan

**Status:** Documentation-only. To be implemented *only if* TFG schedule allows
after Strategy 9 (Deep RL) is complete and the empirical Chapter 4 results
(strategies 1–5, 8) have been generated.

**Estimated effort:** 1–2 weeks, of which ≈42 h are offline dataset generation
running GEKKO on a Latin Hypercube sample of Θ.

---

## 1. Architecture

The operator $\mathcal{G}: \Theta \to L^\infty([0,T];[0, U_{\max}])$ is approximated
by a DeepONet (Lu, Jin & Karniadakis 2021):

$$\mathcal{G}_\Phi(\theta)(t) \;=\; \sum_{k=1}^{p} b_k(\theta;\Phi_b)\,\tau_k(t;\Phi_\tau)$$

Two MLPs:

- **Branch net** $b: \mathbb{R}^9 \to \mathbb{R}^p$. Input: nine biological parameters
  $\theta = (\beta_E, \delta_E, \delta_M, \delta_F, \delta_s, \nu_E, \nu, \gamma_s, K)$.
- **Trunk net** $\tau: \mathbb{R} \to \mathbb{R}^p$. Input: time $t \in [0, T]$.
- Latent dimension: $p = 64$.
- Hidden layers: 3 × 128 ReLU.

---

## 2. File scaffold to create

```
src/sit_control/
├── operator_net.py            # DeepONet PyTorch module
├── operator_dataset.py        # LHS sampling + GEKKO data generation
├── operator_train.py          # Training loop (MSE loss)
└── operator_evaluate.py       # Test-set RMSE + per-parameter error analysis

scripts/
├── run_operator_data.py       # Offline: generate Θ → u* dataset via GEKKO
├── run_operator_training.py   # Train DeepONet on saved dataset
└── run_operator_eval.py       # Evaluate on held-out test set

tests/
├── test_operator_net.py       # Forward pass shapes; gradient flow
└── test_operator_dataset.py   # LHS uniformity; serialization

results/operator/
├── dataset_lhs.npz            # Training data (input θ, output u*(t))
├── deeponet_model.pt          # Trained weights
└── deeponet_metrics.json      # Test-set RMSE, per-θ-axis errors
```

---

## 3. Dataset generation

**Algorithm:**

```
1. Sample N = 5000 points θ_i ~ LHS([0.7, 1.3] × θ_0)  via scipy.stats.qmc.LatinHypercube
2. For each θ_i:
       a. Build a BiologicalParameters(θ_i) instance
       b. Solve L¹ via GekkoOptimiser(params=θ_i).solve_L1(config)
       c. Resample u_i*(·) on a uniform grid {t_j}_{j=1}^{M=150}
       d. Save (θ_i, u_i*) to dataset_lhs.npz
3. Split: 80 % train, 10 % validation, 10 % test (fixed seed).
```

**Cost:** 5000 × ~30 s ≈ 42 h offline (single CPU). Parallelizable across cores
to ~6 h on 8 cores. **Plan to run overnight on the development machine.**

**Failure handling:** GEKKO may fail to converge for ~1–3 % of LHS points (extreme
θ regions). Retry once with N_collocation=200 → 100; otherwise discard sample.

---

## 4. Training

**Loss:**
$$\mathcal{L}(\Phi) = \frac{1}{N_{\text{train}} \cdot M}
\sum_{i=1}^{N_{\text{train}}} \sum_{j=1}^{M}
\bigl(\mathcal{G}_\Phi(\theta_i)(t_j) - u^*_{\theta_i}(t_j)\bigr)^2$$

**Optimizer:** Adam, lr=$10^{-3}$, weight decay $10^{-5}$, scheduler ReduceLROnPlateau.
**Batch size:** 64 functions per batch (each function = full $u^*(\cdot)$ on the grid).
**Epochs:** 200, early stop on val RMSE plateau.
**Compute:** ≈10 min on a single GPU (≈1 h on CPU).

---

## 5. Evaluation protocol

- **Primary metric:** test-set RMSE between $u^*_\theta(\cdot)$ and $\mathcal{G}_\Phi(\theta)(\cdot)$
  in absolute units (mosquitos/day).
- **Sensitivity check:** per-parameter mean error $\mathbb{E}_{\theta_i}\,|\Delta u^*|$
  segmented by which axis of θ varies most.
- **Cost recovery:** $J_\Phi(\theta) = \int_0^T \mathcal{G}_\Phi(\theta)(t)\,dt$;
  compare against $J^*_1(\theta)$ from GEKKO. Target: $|J_\Phi - J^*_1| / J^*_1 < 5\,\%$.
- **Deployment speed:** wall-clock per inference (single forward pass): expected
  $\sim 10\,\mu$s vs. 30 s for GEKKO.

---

## 6. Integration with the main CLI

After implementation, add two subcommands to `main.py`:

```bash
python main.py operator-data --config configs/almeida2022.yaml --n-samples 5000
python main.py operator-train --dataset results/operator/dataset_lhs.npz
python main.py operator-eval --model results/operator/deeponet_model.pt
```

---

## 7. Dependencies to add to pyproject.toml (under [project.optional-dependencies])

```toml
operator = [
    "torch>=2.0",
    "scipy>=1.12",          # already present, for LHS via scipy.stats.qmc
    "matplotlib>=3.8",      # already present
]
```

(`gekko>=1.0` already present in main deps.)

---

## 8. Decision gate

Implement **only if all of the following are true:**

1. Strategy 9 (Deep RL) is implemented, trained, and a paragraph of results
   is written for the Capítulo 5 of the TFG.
2. The empirical Chapter 4 tables (verify, strategies, sensitivity) have been
   filled with `J(u*)` numerical results from GEKKO.
3. TFG submission deadline is at least 6 weeks away.

If any of (1)–(3) fail, **leave Strategy 10 as the argued proposal in
`docs/estrategias_sit.md`** and cite this plan in Chapter 5 ("Trabajo Futuro")
as the roadmap for a postdoctoral / PhD continuation of the work.
