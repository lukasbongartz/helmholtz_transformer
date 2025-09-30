# run_hfe_experiments.py
# -------------------------------------------------------------
# End-to-end: theoretical vs simulated vs empirical H & F on a HF transformer
# Everything is measured at AFTER-LN checkpoints (LN1, LN2).
# -------------------------------------------------------------
import os
import math
import argparse
from typing import List

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

import hfe

def make_causal_mask(B: int, N: int, device, dtype=torch.float32) -> torch.Tensor:
    mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).expand(B, -1, -1).contiguous()

def plot_three(series_emp, series_sim, title, ylabel, series_theo=None, savepath=None):
    xs = list(range(len(series_emp)))
    plt.figure(figsize=(8.2, 4.3))
    plt.plot(xs, series_emp, label="Empirical (model forward)", linewidth=2)
    plt.plot(xs, series_sim, label="Simulated FPE", linestyle="--")
    if series_theo is not None:
        plt.plot(xs, series_theo, label="Theoretical Attempt", linestyle=":")
    plt.xlabel("Micro-step (LN1, LN2 per block)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_series(xs, ys_list, labels, title, ylabel, savepath=None):
    plt.figure(figsize=(8.2, 4.3))
    for ys, lab in zip(ys_list, labels):
        plt.plot(xs, ys, label=lab)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--prompts", type=str, nargs="*", default=[
        "A quick brown fox jumps over the lazy dog.",
        "In a distant galaxy, scientists discovered a new form of matter.",
        "The theorem follows by a straightforward application of the divergence theorem.",
        "Neural scaling laws suggest predictable improvements with compute.",
    ])
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"])
    ap.add_argument("--save_dir", type=str, default="hfe_out")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)

    # Sanity experiments
    ap.add_argument("--sanity_diffusion", action="store_true", help="Run pure diffusion sanity test on S^{D-1}")
    ap.add_argument("--sanity_ou", action="store_true", help="Run quadratic OU sanity test with closed-form comparator")
    ap.add_argument("--sanity_steps", type=int, default=20, help="Number of blocks/steps for sanity tests")
    ap.add_argument("--sanity_nu", type=float, default=1e-3, help="Constant nu for sanity tests")
    ap.add_argument("--sanity_B", type=int, default=2, help="Batch size for sanity tests")
    ap.add_argument("--sanity_N", type=int, default=32, help="Sequence length for sanity tests")
    # Diagnostics
    ap.add_argument("--diag", action="store_true", help="Run in-model thermodynamic diagnostics (E,H,F,D, stationarity)")
    # Epsilon experiments
    ap.add_argument("--eps_exp", action="store_true", help="Run epsilon (temperature) experiments")
    ap.add_argument("--eps_values", type=float, nargs="*", default=[0.5, 1.0, 2.0],
                    help="Constant epsilon multipliers to scale per-layer nu")
    ap.add_argument("--eps_anneal", type=float, nargs=2, metavar=("EPS0","EPS1"), default=None,
                    help="Linearly anneal epsilon from EPS0 to EPS1 across micro-steps")
    ap.add_argument("--eps_fd", type=float, default=1e-3, help="Finite-diff delta for dF/dε at fixed state")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading {args.model} on {args.device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)
    model.eval()

    enc = tokenizer(args.prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    input_ids = enc["input_ids"].to(args.device)
    attn_mask = enc["attention_mask"].to(args.device)
    B, N = input_ids.shape
    causal_mask = make_causal_mask(B, N, device=args.device, dtype=torch_dtype)

    # ====== 0) Collect AFTER-LN states ======
    print("[INFO] Collecting LN1/LN2 states (after LayerNorms)")
    X0, ln1_list, ln2_list = hfe.collect_after_ln_states(model, input_ids, attn_mask)

    # ====== 1) Build ∇U_t, U_t and ν_t (ν_t estimated at LN1) ======
    print("[INFO] Building (∇U_t, U_t) and estimating ν_t per layer ...")
    gradU_ts = hfe.make_gradU_list(model)
    U_ts = hfe.make_U_list(model, n_quad=8)
    nu_ts = [hfe.nu(b, X, attn_mask=causal_mask)
             for b, X in zip(hfe._get_blocks(model), ln1_list)]

    # ====== 2) THEORETICAL (OU linearization) ======
    print("[INFO] Theoretical OU path ...")
    theo = hfe.theoretical_path_OU(model, X0, nu_ts, dt=args.dt)
    # theo = hfe.theoretical_path_closed_form(model, X0, nu_ts, dt=args.dt)

    # ====== 3) SIMULATED (FPE path) ======
    print("[INFO] FPE simulation ...")
    X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_ts, dt=args.dt,
                     micro_steps_per_block=2, return_all=True)  # [2L,B,N,D]
    # Measure empirical H & F along the simulated path using layer-indexed U_t, ν_t
    sim_H, sim_F = [], []
    L = len(gradU_ts)
    for k in range(2*L):
        t = k // 2
        Xk = X_path[k]
        Hk = float(hfe.empirical_entropy_KL(Xk))
        Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
        Fk = Ek - nu_ts[t] * Hk
        sim_H.append(Hk); sim_F.append(Fk)

    # ====== 4) EMPIRICAL (forward-only, at LN1/LN2) ======
    print("[INFO] Empirical from forward LNs ...")
    emp = hfe.empirical_from_forward(ln1_list, ln2_list, U_ts, nu_ts)

    # ====== 5) Plots ======
    print("[INFO] Plotting ...")
    plot_three(emp["entropy"], sim_H,
               # theo["entropy"],
               title="Entropy across depth (LN1/LN2 per block)",
               ylabel="Negentropy -H(ρ)",
               savepath=os.path.join(args.save_dir, "entropy_emp_sim_theo.png"))
    # print([len(t) for t in [emp["free"], sim_F, theo["free"]]])
    exclude = 20
    plot_three(emp["free"][:exclude], sim_F[:exclude],
               # theo["free"][:exclude],
               title="Free energy across depth (LN1/LN2 per block)",
               ylabel="Free energy F",
               savepath=os.path.join(args.save_dir, "free_energy_emp_sim_theo.png"))

    # Save scalars
    torch.save({
        "nu_ts": nu_ts,
        "entropy_emp": torch.tensor(emp["entropy"]),
        "free_emp": torch.tensor(emp["free"]),
        "entropy_sim": torch.tensor(sim_H),
        "free_sim": torch.tensor(sim_F),
        "entropy_theo": torch.tensor(theo["entropy"]),
        "free_theo": torch.tensor(theo["free"]),
    }, os.path.join(args.save_dir, "summary_emp_sim_theo.pt"))

    print("[DONE] Saved outputs to", args.save_dir)
    print("ν_t:", [f"{v:.3g}" for v in nu_ts])

    # =====================
    #  Thermodynamic diagnostics (in-model)
    # =====================
    if args.diag:
        print("[DIAG] Running in-model thermodynamic diagnostics ...")
        # fixed ε → use the same nu_ts used in stepping/scoring
        L = len(gradU_ts)
        dt = args.dt
        # build path as before
        X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_ts, dt=dt,
                         micro_steps_per_block=2, return_all=True)
        E_series, H_series, F_series, D_series = [], [], [], []
        # per micro-step, pair with layer t = k//2
        for k in range(X_path.shape[0]):
            t = k // 2
            Xk = X_path[k]
            Hk = float(hfe.empirical_entropy_KL(Xk))
            Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
            Fk = Ek - float(nu_ts[t]) * Hk
            # Corrected dissipation (Gaussian closure):
            # D_true ≈ E[||P_X(∇U_t(X) − ν Σ_tan^{-1}(X − μ))||^2]
            g = gradU_ts[t](Xk)
            mu_k, Sigma_k = hfe.tangent_mean_and_cov(Xk)
            # Regularize inverse in high-D
            eps_reg = 1e-6
            # project (X - μ) to tangent and apply Σ^{-1}
            Xm = Xk - mu_k
            _, Pk = hfe.tangent_basis(mu_k)
            Xm_tan = torch.einsum("ij,bnd->bni", Pk, Xm)  # [B,N,D]
            Sigma_reg = Sigma_k + eps_reg * torch.eye(Sigma_k.shape[0], device=Sigma_k.device, dtype=Sigma_k.dtype)
            Sigma_inv = torch.inverse(Sigma_reg)
            Sigma_inv_Xm = torch.einsum("ij,bnj->bni", Sigma_inv, Xm_tan)
            g_corr = g - float(nu_ts[t]) * Sigma_inv_Xm
            g_corr_tan = hfe.tangent_project(Xk, g_corr)
            # scale by D for normalization
            Dk = float((g_corr_tan * g_corr_tan).sum(dim=-1).mean() / Xk.shape[-1])
            E_series.append(Ek); H_series.append(Hk); F_series.append(Fk); D_series.append(Dk)
        # discrete dissipation ΔF + dt·D
        DF_plus_D = []
        for k in range(1, len(F_series)):
            t = (k-1) // 2
            DF_plus_D.append(F_series[k] - F_series[k-1] + dt * D_series[k-1])
        # plots
        xs = list(range(len(F_series)))
        plot_series(xs, [F_series], ["F"],
                    title="Diagnostics: Free energy per micro-step (expect ↓)",
                    ylabel="F = E - nu H",
                    savepath=os.path.join(args.save_dir, "diag_F_series.png"))
        plot_series(xs, [E_series, H_series], ["E", "H"],
                    title="Diagnostics: Energy and Entropy",
                    ylabel="value",
                    savepath=os.path.join(args.save_dir, "diag_EH_series.png"))
        plot_series(list(range(len(DF_plus_D))), [DF_plus_D], ["ΔF + dt·D"],
                    title="Diagnostics: Onsager inequality (≤ O(dt^2))",
                    ylabel="ΔF + dt·D",
                    savepath=os.path.join(args.save_dir, "diag_onsager.png"))
        # stationarity at late layers: average and variance of P_X ∇U_t(X)
        t_last = L-1
        X_last = X_path[-1]
        g_last = gradU_ts[t_last](X_last)
        g_last_tan = hfe.tangent_project(X_last, g_last)
        stat_mean = float(g_last_tan.mean().abs())
        stat_var = float(g_last_tan.var())
        torch.save({
            "E_series": torch.tensor(E_series),
            "H_series": torch.tensor(H_series),
            "F_series": torch.tensor(F_series),
            "D_series": torch.tensor(D_series),
            "DF_plus_D": torch.tensor(DF_plus_D),
            "stationarity_mean_abs": stat_mean,
            "stationarity_var": stat_var,
        }, os.path.join(args.save_dir, "diagnostics.pt"))
        print(f"[DIAG] stationarity mean|g_tan|={stat_mean:.3e}, var={stat_var:.3e}")

    # =====================
    #  Epsilon (temperature) experiments
    # =====================
    if args.eps_exp:
        print("[EPS] Running epsilon experiments ...")
        L = len(gradU_ts)
        dt = args.dt
        # Base nu per layer (estimated at LN1)
        base_nu = [float(v) for v in nu_ts]

        # ---- Constant epsilon multipliers ----
        const_runs = {}
        for eps_mul in args.eps_values:
            nu_scaled = [eps_mul * v for v in base_nu]
            X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_scaled, dt=dt,
                             micro_steps_per_block=2, return_all=True)
            F_series = []
            for k in range(X_path.shape[0]):
                t = k // 2
                Xk = X_path[k]
                Hk = float(hfe.empirical_entropy_KL(Xk))
                Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
                Fk = Ek - nu_scaled[t] * Hk
                F_series.append(Fk)
            const_runs[eps_mul] = F_series
        # plot const eps F series overlay
        xs = list(range(len(next(iter(const_runs.values()))))) if const_runs else []
        if xs:
            ys_list = [const_runs[m] for m in sorted(const_runs.keys())]
            labels = [f"eps×{m:g}" for m in sorted(const_runs.keys())]
            plot_series(xs, ys_list, labels,
                        title="Constant epsilon: F per micro-step",
                        ylabel="F = E - nu H",
                        savepath=os.path.join(args.save_dir, "eps_const_F_series.png"))

        # ---- Annealing epsilon ----
        if args.eps_anneal is not None:
            eps0, eps1 = args.eps_anneal
            K = 2 * L
            schedule = [eps0 + (eps1 - eps0) * (k / max(K - 1, 1)) for k in range(K)]
            # build a callable nu_t(X) that uses per-step multiplier at micro-step k
            nu_calls = []
            for t in range(L):
                # two micro-steps per block → pick schedule[2*t] and schedule[2*t+1]; use their avg for step-size
                mul_t = 0.5 * (schedule[2*t] + schedule[2*t+1])
                nu_calls.append(float(mul_t) * base_nu[t])
            X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_calls, dt=dt,
                             micro_steps_per_block=2, return_all=True)
            F_series, D_series = [], []
            for k in range(X_path.shape[0]):
                t = k // 2
                Xk = X_path[k]
                Hk = float(hfe.empirical_entropy_KL(Xk))
                Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
                Fk = Ek - nu_calls[t] * Hk
                F_series.append(Fk)
                g = gradU_ts[t](Xk)
                g_tan = hfe.tangent_project(Xk, g)
                Dk = float((g_tan * g_tan).sum(dim=-1).mean())
                D_series.append(Dk)
            # finite-diff ∂_ε F_ε at fixed states (evaluate F at ε±δ using same X_path),
            # reusing the same base ν used for stepping/scoring to keep consistency
            d_eps = args.eps_fd
            dF_dEps = []
            for k in range(X_path.shape[0]):
                t = k // 2
                Xk = X_path[k]
                Hk = float(hfe.empirical_entropy_KL(Xk))
                Ek = float(hfe.empirical_energy_U(Xk, U_ts[t]))
                eps_ref = schedule[k]
                # F(ε±δ) = E - (ε±δ)·ν_base·H at fixed Xk
                F_plus = Ek - (eps_ref + d_eps) * base_nu[t] * Hk
                F_minus = Ek - (eps_ref - d_eps) * base_nu[t] * Hk
                dF_dEps.append((F_plus - F_minus) / (2.0 * d_eps))
            # Compare dF/dt vs -D + (∂_ε F) ε_dot
            eps_dot = [(schedule[min(k+1, K-1)] - schedule[k]) / dt for k in range(K)]
            dFdt_est = [ (F_series[min(k+1, K-1)] - F_series[k]) / dt for k in range(K) ]
            rhs = [ -D_series[k] + dF_dEps[k] * eps_dot[k] for k in range(K) ]
            plot_series(list(range(K)), [dFdt_est, rhs], ["dF/dt", "-D + (dF/dε) ε̇"],
                        title="Annealing: dF/dt vs -D + (∂εF) ε̇",
                        ylabel="value",
                        savepath=os.path.join(args.save_dir, "eps_anneal_balance.png"))
            torch.save({
                "schedule": torch.tensor(schedule),
                "F_series": torch.tensor(F_series),
                "D_series": torch.tensor(D_series),
                "dF_dEps": torch.tensor(dF_dEps),
                "dFdt_est": torch.tensor(dFdt_est),
                "rhs": torch.tensor(rhs),
            }, os.path.join(args.save_dir, "eps_anneal.pt"))

    # =====================
    #  Sanity: Pure diffusion
    # =====================
    if args.sanity_diffusion:
        print("[SANITY] Pure diffusion on S^{D-1} ...")
        # Use model hidden dim from collected states
        D = ln1_list[0].shape[-1]
        torch.manual_seed(args.seed)
        # Concentrated around a pole μ with small noise
        mu0 = torch.zeros(D, device=args.device, dtype=torch_dtype); mu0[0] = 1.0
        noise = 0.1 * torch.randn(args.sanity_B, args.sanity_N, D, device=args.device, dtype=torch_dtype)
        X0 = hfe.normalize_sphere(mu0.view(1,1,-1).expand(args.sanity_B, args.sanity_N, -1) + noise)

        L = args.sanity_steps
        zero_grad = lambda x: torch.zeros_like(x)
        gradU_ts = [zero_grad for _ in range(L)]
        nu_ts_const = [float(args.sanity_nu) for _ in range(L)]

        X_path = hfe.fpe(X0, gradU_ts=gradU_ts, nu_ts=nu_ts_const, dt=args.dt,
                         micro_steps_per_block=2, return_all=True)
        H_series = [float(hfe.empirical_entropy_KL(X_path[k])) for k in range(X_path.shape[0])]
        mu_norm = []
        for k in range(X_path.shape[0]):
            mu_k, _ = hfe.tangent_mean_and_cov(X_path[k])
            mu_norm.append(float(mu_k.norm()))
        xs = list(range(len(H_series)))
        plot_series(xs, [H_series], ["Entropy"],
                    title="Pure diffusion: entropy should increase",
                    ylabel="H",
                    savepath=os.path.join(args.save_dir, "sanity_diffusion_entropy.png"))
        plot_series(xs, [mu_norm], ["||mu||"],
                    title="Pure diffusion: mean norm should decrease",
                    ylabel="||mu||",
                    savepath=os.path.join(args.save_dir, "sanity_diffusion_mu_norm.png"))
        print("[SANITY] Diffusion entropy (first,last):", H_series[0], H_series[-1])

    # =====================
    #  Sanity: Quadratic OU vs closed-form
    # =====================
    if args.sanity_ou:
        print("[SANITY] Quadratic OU test (Σ vs Lyapunov, F monotone) ...")
        D = ln1_list[0].shape[-1]
        torch.manual_seed(args.seed + 1)
        X0 = torch.randn(args.sanity_B, args.sanity_N, D, device=args.device, dtype=torch_dtype)
        X0 = hfe.normalize_sphere(X0)

        # Construct symmetric PSD H ambient
        R = torch.randn(D, D, device=args.device, dtype=torch_dtype)
        H_amb = (R.t() @ R) / float(D)
        H_amb = 0.5 * (H_amb + H_amb.t())

        def gradU_quad(x):
            # grad of 1/2 x^T H x is H x
            x_flat = x.reshape(-1, D)
            g = x_flat @ H_amb.t()
            return g.view_as(x)

        L = args.sanity_steps
        gradU_ts = [gradU_quad for _ in range(L)]
        nu_ts_const = [float(args.sanity_nu) for _ in range(L)]

        # Linear tangent-space simulator (no renorm): propagate μ,Σ in tangent linearly
        mu_lin, Sigma_lin = hfe.tangent_mean_and_cov(X0)
        m = D - 1
        theo_traces, sim_traces, F_sim = [], [], []
        Sigma_th = Sigma_lin.clone()
        # Simulate explicit Euler in tangent for A and M
        def step_A(mu_curr, Sigma_curr, nu_curr):
            _, P = hfe.tangent_basis(mu_curr)
            return mu_curr, 0.5 * (Sigma_curr + Sigma_curr.t()) + (2.0 * nu_curr * args.dt) * P
        def step_M(mu_curr, Sigma_curr):
            gmu = gradU_quad(mu_curr.unsqueeze(0)).squeeze(0)
            gmu = hfe.tangent_project(mu_curr, gmu)
            mu_next = mu_curr - args.dt * gmu
            # OU covariance linearized: Σ <- Σ - (HΣ+ΣH)dt
            _, P = hfe.tangent_basis(mu_next)
            Htan = P @ H_amb @ P; Htan = 0.5 * (Htan + Htan.t())
            Sigma_next = Sigma_curr - args.dt * (Htan @ Sigma_curr + Sigma_curr @ Htan)
            Sigma_next = 0.5 * (Sigma_next + Sigma_next.t())
            return mu_next, Sigma_next

        # Theoretical covariance via Lyapunov, projecting H to tangent at current mean
        # Helper to compute free energy pieces at a state
        def energy_free(mu_curr, Sigma_curr, nu_curr):
            # Tangent-projected H at current mean
            _, P = hfe.tangent_basis(mu_curr)
            Htan = P @ H_amb @ P
            Htan = 0.5 * (Htan + Htan.t())
            H_gauss = hfe.gaussian_entropy_from_cov(Sigma_curr, m)
            E = 0.5 * float((mu_curr @ (H_amb @ mu_curr))) + 0.5 * float(torch.trace(Htan @ Sigma_curr))
            F = E - nu_curr * H_gauss
            return Htan, H_gauss, E, F

        # Walk theoretical and linear-sim at each micro-step
        k = 0
        for t in range(L):
            # micro-step A (diffusion)
            _, P = hfe.tangent_basis(mu_lin)
            Sigma_th = Sigma_th + (2.0 * nu_ts_const[t] * args.dt) * P
            Sigma_th = 0.5 * (Sigma_th + Sigma_th.t())
            Htan, H_gauss, E, F = energy_free(mu_lin, Sigma_th, nu_ts_const[t])
            theo_traces.append(float(torch.trace(Sigma_th)))
            # linear sim A
            mu_lin, Sigma_lin = step_A(mu_lin, Sigma_lin, nu_ts_const[t])
            sim_traces.append(float(torch.trace(Sigma_lin)))
            _, H_gauss_k, E_k, F_k = energy_free(mu_lin, Sigma_lin, nu_ts_const[t])
            F_sim.append(F_k)

            # micro-step M (drift)
            # Theoretical Lyapunov at updated mean
            gmu = gradU_quad(mu_lin.unsqueeze(0)).squeeze(0)
            gmu = hfe.tangent_project(mu_lin, gmu)
            mu_lin_next = mu_lin - args.dt * gmu
            mu_lin_next = mu_lin_next  # no renorm in linear test
            Htan, _, _, _ = energy_free(mu_lin_next, Sigma_th, nu_ts_const[t])
            Sigma_th = hfe.lyapunov_step(Sigma_th, Htan, nu_ts_const[t], args.dt)
            Htan, H_gauss, E, F = energy_free(mu_lin_next, Sigma_th, nu_ts_const[t])
            theo_traces.append(float(torch.trace(Sigma_th)))
            # linear sim M
            mu_lin, Sigma_lin = step_M(mu_lin, Sigma_lin)
            sim_traces.append(float(torch.trace(Sigma_lin)))
            _, H_gauss_k, E_k, F_k = energy_free(mu_lin, Sigma_lin, nu_ts_const[t])
            F_sim.append(F_k)

        xs = list(range(len(sim_traces)))
        plot_series(xs, [sim_traces, theo_traces], ["Sim trace Σ", "Theo trace Σ"],
                    title="Quadratic OU: covariance trace vs closed-form",
                    ylabel="trace(Σ)",
                    savepath=os.path.join(args.save_dir, "sanity_ou_cov_trace.png"))
        # Free energy series (sim)
        plot_series(list(range(len(F_sim))), [F_sim], ["F_sim"],
                    title="Quadratic OU: free energy (expect monotone ↓)",
                    ylabel="F = E - nu * H",
                    savepath=os.path.join(args.save_dir, "sanity_ou_free_energy.png"))

if __name__ == "__main__":
    main()

