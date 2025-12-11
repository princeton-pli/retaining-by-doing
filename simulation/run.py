"""
python -m simulation.run --help
python -m simulation.run --gain_thresholds "0.9,0.9,0.9"
python -m simulation.run --gain_thresholds "0.9,0.9,0.9" --save pdf
python -m simulation.run --gain_thresholds "0.9,0.9,0.9" --save gif
python -m simulation.run --gain_thresholds "0.9,0.9,0.9" --save teaser
"""
import time
from pathlib import Path
from functools import partial

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import fire
from rich import print

from simulation.utils import load_module

np.random.seed(0)


def setup_variables(setting):
    global xs, w1, m1, s1, w2, m2, s2, p1_vals, p2_vals, area_p1, area_p2, overlap1_init, overlap2_init

    xs = np.linspace(setting.X_MIN, setting.X_MAX, setting.N_X)
    w1, m1, s1 = setting.MIXTURE[0]
    w2, m2, s2 = setting.MIXTURE[1]
    
    p1_vals = w1 * norm.pdf(xs, m1, s1)
    p2_vals = w2 * norm.pdf(xs, m2, s2)
    area_p1, area_p2 = np.trapz(p1_vals, xs), np.trapz(p2_vals, xs)
    
    q1_vals_init = setting.PI1 * norm.pdf(xs, setting.MU1, setting.SIG1)
    q2_vals_init = (1-setting.PI1) * norm.pdf(xs, setting.MU2, setting.SIG2)
    overlap1_init = np.trapz(np.minimum(p1_vals, q1_vals_init), xs)/area_p1
    overlap2_init = np.trapz(np.minimum(p2_vals, q2_vals_init), xs)/area_p2
    
    p_integral = np.trapz(p1_vals + p2_vals, xs)
    print(f"p's density adds up to: {p_integral}")
    print(f"Initial overlaps - Mode 1: {overlap1_init:.3f}, Mode 2: {overlap2_init:.3f}")
    print('setup_global_variables(): variables initialized')

    return dict(
        xs=xs, w1=w1, m1=m1, s1=s1, w2=w2, m2=m2, s2=s2, 
        p1_vals=p1_vals, p2_vals=p2_vals,
        area_p1=area_p1, area_p2=area_p2,
        overlap1_init=overlap1_init, overlap2_init=overlap2_init,
    )


def p_pdf(x, w1, m1, s1, w2, m2, s2, setting):
    return sum(w * norm.pdf(x, m, s) for (w, m, s) in setting.MIXTURE)


def sample_p_right(n, w1, m1, s1, w2, m2, s2, setting):
    _, mr, sr = setting.MIXTURE[1]
    return np.random.normal(mr, sr, size=n)


def q_pdf(x, pi1, mu1, s1, mu2, s2, setting):
    return pi1 * norm.pdf(x, mu1, s1) + (1-pi1) * norm.pdf(x, mu2, s2)


def sample_q_right(n, pi1, mu1, s1, mu2, s2, setting):
    return np.random.normal(mu2, s2, size=n)


def sample_q_left(n, pi1, mu1, s1, mu2, s2, setting):
    return np.random.normal(mu1, s1, size=n)


def sample_q(n, pi1, mu1, s1, mu2, s2, setting):
        return pi1 * np.random.normal(mu1, s1, size=n) + (1 - pi1) * np.random.normal(mu2, s2, size=n)


def fit_forward_kl(init_variables, setting, learning_rate):
    pi1, mu1, s1 = setting.PI1, setting.MU1, setting.SIG1
    mu2, s2      = setting.MU2, setting.SIG2

    xs = init_variables['xs']
    p1_vals = init_variables['p1_vals']
    p2_vals = init_variables['p2_vals']
    area_p1 = init_variables['area_p1']
    area_p2 = init_variables['area_p2']
    overlap1_init = init_variables['overlap1_init']
    overlap2_init = init_variables['overlap2_init']

    hist = []
    acc1_fwd = []
    drop_fwd = []
    acc2_fwd = []
    gain_fwd = []

    hist.append((pi1, mu1, s1, mu2, s2))

    for _ in range(setting.NUM_ITERS):
        Xp = sample_p_right(setting.NUM_SAMPLES, w1, m1, s1, w2, m2, s2, setting)
        N1 = norm.pdf(Xp, mu1, s1)
        N2 = norm.pdf(Xp, mu2, s2)
        qx = pi1 * N1 + (1-pi1) * N2 + 1e-12

        g_pi1 = -np.mean((N1 - N2) / qx)
        r1, r2 = pi1 * N1/qx, (1-pi1) * N2/qx
        g_mu1  = -np.mean(r1 * (Xp-mu1)/(s1**2))
        g_mu2  = -np.mean(r2 * (Xp-mu2)/(s2**2))
        g_s1   = -np.mean(r1 * (-(1/s1)+(Xp-mu1)**2/(s1**3)))
        g_s2   = -np.mean(r2 * (-(1/s2)+(Xp-mu2)**2/(s2**3)))

        pi1  -= learning_rate * g_pi1 * setting.SLOW_FACTOR
        mu1  -= learning_rate * g_mu1 * setting.SLOW_FACTOR
        mu2  -= learning_rate * g_mu2
        s1   -= learning_rate * g_s1 * setting.SLOW_FACTOR
        s2   -= learning_rate * g_s2

        pi1 = np.clip(pi1, 1e-3, 1-1e-3)
        s1, s2 = max(s1, 1e-3), max(s2, 1e-3)

        hist.append((pi1, mu1, s1, mu2, s2))

        q1_vals = pi1 * norm.pdf(xs, mu1, s1)
        q2_vals = (1-pi1) * norm.pdf(xs, mu2, s2)
        current_overlap1 = np.trapz(np.minimum(p1_vals, q1_vals), xs)/area_p1
        current_overlap2 = np.trapz(np.minimum(p2_vals, q2_vals), xs)/area_p2
        
        drop = current_overlap1 - overlap1_init
        gain = current_overlap2 - overlap2_init

        drop_fwd.append(drop)
        gain_fwd.append(gain)
        acc1_fwd.append(current_overlap1)
        acc2_fwd.append(current_overlap2)
    q_final = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
    q_final_integral = np.trapz(q_final, xs)
    print(f"Forward KL: final q's probability density integrates to: {q_final_integral}")

    return hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd


def fit_reinforce(init_variables, setting, learning_rate):
    """
    • Draws X ~ N(mu2, s2²)   (right component only).
    • Reward  r(x) = log p₂(x) with p₂ the *target* right-hand Gaussian.
    • Advantage is centred & st-dev-normalised (GRPO style).
    • Updates θ with Monte-Carlo ∇  E_q2 [ A(x) ∇θ log qθ(x) ].
    • Returns the same (hist, drop, gain, acc1, acc2) lists as other fit_*().
    """

    pi1, mu1, s1 = setting.PI1, setting.MU1, setting.SIG1
    mu2, s2      = setting.MU2, setting.SIG2
    print(f"Initial params: pi1: {pi1:.3f}, mu1: {mu1:.3f}, s1: {s1:.3f}, mu2: {mu2:.3f}, s2: {s2:.3f}")

    w2, m2_tgt, s2_tgt = setting.MIXTURE[1]

    xs = init_variables['xs']
    p1_vals = init_variables['p1_vals']
    p2_vals = init_variables['p2_vals']
    area_p1 = init_variables['area_p1']
    area_p2 = init_variables['area_p2']
    overlap1_init = init_variables['overlap1_init']
    overlap2_init = init_variables['overlap2_init']

    hist, drop_hist, gain_hist = [], [], []
    acc1_hist, acc2_hist       = [], []

    hist = [(pi1, mu1, s1, mu2, s2)]

    for _ in range(setting.NUM_ITERS):
        Xq = sample_q_right(setting.NUM_SAMPLES, pi1, mu1, s1, mu2, s2, setting)

        q1 = pi1 * norm.pdf(Xq, mu1, s1)
        q2 = (1 - pi1) * norm.pdf(Xq, mu2, s2)
        qx = q1 + q2 + 1e-12
        r1 = q1 / qx

        p2 = norm.pdf(Xq, m2_tgt, s2_tgt)

        reward = np.log(p2 / q2 / setting.REINFORCE_BETA + 1e-12)

        importance_reweighting = qx / (q2 + 1e-12)
        
        score_pi1 = importance_reweighting * (norm.pdf(Xq, mu1, s1) - norm.pdf(Xq, mu2, s2)) / qx
        score_mu1 = importance_reweighting * r1          * (Xq - mu1) / (s1 ** 2)
        score_mu2 = importance_reweighting * (1 - r1)    * (Xq - mu2) / (s2 ** 2)
        score_s1  = importance_reweighting * r1          * (-(1 / s1) + (Xq - mu1) ** 2 / s1 ** 3)
        score_s2  = importance_reweighting * (1 - r1)    * (-(1 / s2) + (Xq - mu2) ** 2 / s2 ** 3)

        g_pi1 = np.mean(reward * score_pi1)
        g_mu1 = np.mean(reward * score_mu1)
        g_mu2 = np.mean(reward * score_mu2)
        g_s1  = np.mean(reward * score_s1)
        g_s2  = np.mean(reward * score_s2)

        pi1 += learning_rate * g_pi1 * setting.SLOW_FACTOR
        mu1 += learning_rate * g_mu1 * setting.SLOW_FACTOR
        mu2 += learning_rate * g_mu2
        s1  += learning_rate * g_s1 * setting.SLOW_FACTOR
        s2  += learning_rate * g_s2

        pi1 = np.clip(pi1, 1e-3, 1 - 1e-3)
        s1  = max(s1, 1e-3)
        s2  = max(s2, 1e-3)

        hist.append((pi1, mu1, s1, mu2, s2))

        q1_vals = pi1 * norm.pdf(xs, mu1, s1)
        q2_vals = (1 - pi1) * norm.pdf(xs, mu2, s2)
        curr_ov1 = np.trapz(np.minimum(p1_vals, q1_vals), xs) / area_p1
        curr_ov2 = np.trapz(np.minimum(p2_vals, q2_vals), xs) / area_p2

        drop_hist.append(curr_ov1 - overlap1_init)
        gain_hist.append(curr_ov2 - overlap2_init)
        acc1_hist.append(curr_ov1)
        acc2_hist.append(curr_ov2)

    q_final = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
    print(f"Reverse-KL-2 (q2-sampling): final q integrates to {np.trapz(q_final, xs):.6f}")
    g_density = norm.pdf(Xq, mu2, s2)
    Z_est     = np.mean(np.exp(reward) * qx / g_density)
    print(f"∫ q(x) * exp(r(x)) dx ≈ {Z_est:.6f} (target 1.0)")
    print(f"Final params: pi1: {pi1:.3f}, mu1: {mu1:.3f}, s1: {s1:.3f}, mu2: {mu2:.3f}, s2: {s2:.3f}")
    return hist, drop_hist, gain_hist, acc1_hist, acc2_hist


def animate(i, num_total_iters, init_variables, setting, axs, fwd_results, rev_results, gain_thresholds=None):
    xs = init_variables['xs']
    w1, m1, s1, w2, m2, s2 = init_variables['w1'], init_variables['m1'], init_variables['s1'], init_variables['w2'], init_variables['m2'], init_variables['s2']

    for ax in axs:
        ax.clear()
        px = p_pdf(xs, w1, m1, s1, w2, m2, s2, setting)
        px /= np.trapz(px, xs)
        ax.plot(xs, px, 'k--')

    for fwd_idx, (fwd_hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd) in enumerate(fwd_results):
        ax = axs[fwd_idx]
        
        if gain_thresholds and fwd_idx < len(gain_thresholds):
            threshold = gain_thresholds[fwd_idx]
            stop_iter = len(fwd_hist) - 1
            for j, gain_val in enumerate(gain_fwd):
                if gain_val >= threshold:
                    stop_iter = j
                    break
            current_iter = min(i, stop_iter)
        else:
            current_iter = min(i, len(fwd_hist)-1)
        
        pi1, mu1, s1, mu2, s2 = fwd_hist[current_iter]
        qf = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
        qf /= np.trapz(qf, xs)
        ax.plot(xs, qf, 'r-', lw=2)
        ax.set_title(f"Forward KL | LR={setting.FWD_LR[fwd_idx]}")
        ax.text(setting.TEXT_POSITION_T[0], 0.5, f"T'", transform=ax.transAxes, fontsize=20)
        ax.text(setting.TEXT_POSITION_T_PRIME[0], 0.5, f"T", transform=ax.transAxes, fontsize=20)
        ax.text(0.05, 0.9, f"Iteration: {current_iter+1}", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.8, f"Acc(T) ={acc2_fwd[current_iter]:.2f} / Gain={gain_fwd[current_iter]:.2f}", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.7, f"Acc(T')={acc1_fwd[current_iter]:.2f} / Drop={-drop_fwd[current_iter]:.2f}", transform=ax.transAxes, fontsize=12)

    for rev_idx, (rev_hist, drop_rev, gain_rev, acc1_rev, acc2_rev) in enumerate(rev_results):
        ax = axs[len(fwd_results) + rev_idx]
        
        if gain_thresholds and (len(fwd_results) + rev_idx) < len(gain_thresholds):
            threshold = gain_thresholds[len(fwd_results) + rev_idx]
            stop_iter = len(rev_hist) - 1
            for j, gain_val in enumerate(gain_rev):
                if gain_val >= threshold:
                    stop_iter = j
                    break
            current_iter = min(i, stop_iter)
        else:
            current_iter = min(i, len(rev_hist)-1)
        
        pi1, mu1, s1, mu2, s2 = rev_hist[current_iter]
        qr = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
        qr /= np.trapz(qr, xs)
        ax.plot(xs, qr, 'b-', lw=2)
        ax.set_title(f"Reverse KL | LR={setting.REV_LR[rev_idx]}")
        ax.text(setting.TEXT_POSITION_T[0], 0.5, f"T'", transform=ax.transAxes, fontsize=20)
        ax.text(setting.TEXT_POSITION_T_PRIME[0], 0.5, f"T", transform=ax.transAxes, fontsize=20)
        ax.text(0.05, 0.9, f"Iteration: {current_iter+1}", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.8, f"Acc(T) ={acc2_rev[current_iter]:.2f} / Gain={gain_rev[current_iter]:.2f}", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.7, f"Acc(T')={acc1_rev[current_iter]:.2f} / Drop={-drop_rev[current_iter]:.2f}", transform=ax.transAxes, fontsize=12)

    for ax in axs:
        ax.set_xlim(setting.X_MIN, setting.X_MAX)
        ax.set_ylim(0, setting.Y_MAX)
    
    return axs


def plot_overlaid_iterations(
    init_variables,
    setting,
    fwd_results,
    rev_results,
    save_path=None,
    frame_skip: int = 30,
    plot_ticks: bool = False,
    gain_thresholds=None,
):
    """
    Overlays selected iterations with a color gradient:
      • Forward KL: orange → red
      • Reverse KL: blue   → purple

    Args:
        xs:            precomputed grid (np.linspace).
        setting:       loaded setting object.
        fwd_results:   list of (hist, drop, gain, acc1, acc2) per FWD_LR.
        rev_results:   list of (hist, drop, gain, acc1, acc2) per REV_LR.
        save_path:     if set, will save figure there.
        frame_skip:    only plot every `frame_skip` iterations.
        gain_thresholds: if provided, only plot up to when each subplot reaches its threshold.
    """
    num_iters = setting.NUM_ITERS
    n_fwd    = len(fwd_results)
    n_rev    = len(rev_results)
    num_plots= n_fwd + n_rev

    xs = init_variables['xs']
    w1 = init_variables['w1']
    m1 = init_variables['m1']
    s1 = init_variables['s1']
    w2 = init_variables['w2']
    m2 = init_variables['m2']
    s2 = init_variables['s2']

    fig, axs = plt.subplots(
        1,
        num_plots,
        figsize=(4 * num_plots + 7, 4) if not plot_ticks else (4 * num_plots + 5, 4),
        squeeze=True,
    )
    if num_plots == 1:
        axs = [axs]

    px = p_pdf(xs, w1, m1, s1, w2, m2, s2, setting)
    px /= np.trapz(px, xs)
    for ax in axs:
        ax.plot(xs, px, 'k--', label="p(x)", linewidth=3)

    start_fwd = np.array(to_rgb('red'))
    end_fwd   = np.array(to_rgb('yellow'))
    start_rev = np.array(to_rgb('#0096c7'))
    end_rev   = np.array(to_rgb('#5a189a'))

    def interp_color(start, end, t):
        return start + (end - start) * t 

    for idx, (hist, drop_fwd, gain_fwd, _, _) in enumerate(fwd_results):
        ax = axs[idx]
        
        if gain_thresholds and idx < len(gain_thresholds):
            threshold = gain_thresholds[idx]
            max_iter = len(hist) - 1
            for j, gain_val in enumerate(gain_fwd):
                if gain_val >= threshold:
                    max_iter = j
                    break
        else:
            max_iter = len(hist) - 1
        
        for i, (pi1, mu1, s1, mu2, s2) in enumerate(hist):
            if i > max_iter or (i % frame_skip and i != max_iter):
                continue
            t_lin = i / (max_iter if max_iter > 0 else 1)
            t = t_lin ** 0.7
            c = interp_color(start_fwd, end_fwd, t)
            q = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
            q /= np.trapz(q, xs)
            ax.plot(xs, q, color=c, linewidth=4)
        
        ax.text(0.16, 0.9, f'Forward KL w/ LR={setting.FWD_LR[idx]:.2f}', transform=ax.transAxes, fontsize=23)
        safe_idx = min(max_iter, len(drop_fwd) - 1) if drop_fwd else 0
        ax.text(0.15, 0.73, f'Drop={-drop_fwd[safe_idx]:.2f}', transform=ax.transAxes, fontsize=20)
        ax.text(0.63, 0.40, f'Gain={gain_fwd[safe_idx]:.2f}', transform=ax.transAxes, fontsize=20)
        ax.set_xlim(setting.X_MIN + 1.0, setting.X_MAX - 1.0)
        ax.set_ylim(0, setting.Y_MAX - 0.3)

        if not plot_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax.tick_params(axis='both', which='major', labelsize=14)

    for jdx, (hist, drop_rev, gain_rev, _, _) in enumerate(rev_results, start=n_fwd):
        ax = axs[jdx]
        
        if gain_thresholds and jdx < len(gain_thresholds):
            threshold = gain_thresholds[jdx]
            max_iter = len(hist) - 1
            for j, gain_val in enumerate(gain_rev):
                if gain_val >= threshold:
                    max_iter = j
                    break
        else:
            max_iter = len(hist) - 1
        
        for i, (pi1, mu1, s1, mu2, s2) in enumerate(hist):
            if i > max_iter or (i % frame_skip and i != max_iter):
                continue
            t_lin = i / (max_iter if max_iter > 0 else 1)
            t = t_lin ** 0.7
            c = interp_color(start_rev, end_rev, t)
            q = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
            q /= np.trapz(q, xs)
            ax.plot(xs, q, color=c, linewidth=4)
        ax.text(0.16, 0.9, f'Reverse KL w/ LR={setting.REV_LR[jdx-n_fwd]:.2f}', transform=ax.transAxes, fontsize=23)
        safe_idx = min(max_iter, len(drop_rev) - 1) if drop_rev else 0
        ax.text(0.15, 0.73, f'Drop={-drop_rev[safe_idx]:.2f}', transform=ax.transAxes, fontsize=20)
        ax.text(0.63, 0.40, f'Gain={gain_rev[safe_idx]:.2f}', transform=ax.transAxes, fontsize=20)
        ax.set_xlim(setting.X_MIN + 1.0, setting.X_MAX - 1.0)
        ax.set_ylim(0, setting.Y_MAX - 0.3)
        
        if not plot_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout(rect=[0,0,1,1])
    fig.savefig(save_path, dpi=150)


def plot_params(fwd_results, rev_results, setting, save_path):
    """
    Plot the progression of the parameters (pi1, mu2, sigma2) and the gain/drop over training steps.
    Each gets its own subplot. All runs (forward/reverse, all learning rates) are overlaid.
    """
    param_names = [r'$\pi_1$', r'$\mu_2$', r'$\sigma_2$', 'Drop', 'Gain']
    param_indices = [0, 3, 4]  # indices for pi1, mu2, sigma2 in hist
    num_params = 3
    num_metrics = 2  # drop, gain
    num_plots = num_params + num_metrics
    num_fwd = len(fwd_results)
    num_rev = len(rev_results)

    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    for i, (hist, drop, gain, _, _) in enumerate(fwd_results):
        hist = np.array(hist)  # shape: (num_steps+1, 5)
        for p, idx in enumerate(param_indices):
            axs[p].plot(hist[:, idx], label=f'FWD lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')
        axs[3].plot(drop, label=f'FWD drop lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')
        axs[4].plot(gain, label=f'FWD gain lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')

    for i, (hist, drop, gain, _, _) in enumerate(rev_results):
        hist = np.array(hist)  # shape: (num_steps+1, 5)
        for p, idx in enumerate(param_indices):
            axs[p].plot(hist[:, idx], label=f'PG lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')
        axs[3].plot(drop, label=f'PG drop lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')
        axs[4].plot(gain, label=f'PG gain lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')

    for p in range(num_plots):
        axs[p].set_title(param_names[p])
        axs[p].set_xlabel('Training step')
        axs[p].set_ylabel(param_names[p])
        axs[p].legend()
        axs[p].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_single_frame(results, setting, frame_idx, save_path, mode):
    """
    Plot a single frame of the animation at the specified frame_idx.
    Just plots the curve without any formatting.
    """
    print(f"save_path: {save_path}")
    if mode == 'fwd':
        color = 'r'
    elif mode == 'rev':
        color = 'b'
    else:
        raise ValueError(f"Invalid mode: {mode}")

    init_variables = setup_variables(setting)
    xs = init_variables['xs']
    w1, m1, s1, w2, m2, s2 = init_variables['w1'], init_variables['m1'], init_variables['s1'], init_variables['w2'], init_variables['m2'], init_variables['s2']
    
    px = p_pdf(xs, w1, m1, s1, w2, m2, s2, setting)
    px /= np.trapz(px, xs)
    
    if results:
        hist, _, _, _, _ = results[0]
        if frame_idx < len(hist):
            pi1, mu1, s1, mu2, s2 = hist[frame_idx]
            q = q_pdf(xs, pi1, mu1, s1, mu2, s2, setting)
            q /= np.trapz(q, xs)
            
            plt.figure(figsize=(8, 6))
            plt.plot(xs, px, 'k--', linewidth=4)  # True distribution p(x)
            plt.plot(xs, q, color, linewidth=4)    # Learned distribution q(x)

            plt.xlim(setting.X_MIN, setting.X_MAX)
            plt.ylim(-0.02, setting.Y_MAX - 0.20)
            plt.xticks([])
            plt.yticks([])
            
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()


def main(**kwargs):
    if kwargs.get('help', False):
        print('python -m simulation.run --setting <setting_name>')
        print('python -m simulation.run --gain_thresholds 0.5,0.3')
        print('List of available setting names:')
        from simulation.utils import list_class_objects
        from simulation.settings import BaseSetting
        list_class_objects('simulation.settings', BaseSetting)
        return

    setting_name = kwargs.get('setting', 'TwoModeSetting')
    setting = load_module('simulation.settings', setting_name)
    
    gain_thresholds = None
    if kwargs.get('gain_thresholds'):
        gain_thresholds_str = kwargs['gain_thresholds']
        if isinstance(gain_thresholds_str, str):
            gain_thresholds = [float(x.strip()) for x in gain_thresholds_str.split(',')]
        elif isinstance(gain_thresholds_str, (list, tuple)):
            gain_thresholds = [float(x) for x in gain_thresholds_str]
        print(f"Gain thresholds: {gain_thresholds}")

    fwd_results = []
    for fwd_lr in setting.FWD_LR:
        init_variables = setup_variables(setting)
        fwd_hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd = fit_forward_kl(init_variables, setting, fwd_lr)
        fwd_results.append((fwd_hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd))
    
    rev_results = []
    for rev_lr in setting.REV_LR:
        init_variables = setup_variables(setting)
        rev_hist, drop_rev, gain_rev, acc1_rev, acc2_rev = fit_reinforce(init_variables, setting, rev_lr)
        rev_results.append((rev_hist, drop_rev, gain_rev, acc1_rev, acc2_rev))

    num_plots = len(setting.FWD_LR) + len(setting.REV_LR)
    if num_plots == 2:
        col_num, row_num = 2, 1
        col_size, row_size = 10, 4
    elif num_plots == 3:
        col_num, row_num = 3, 1
        col_size, row_size = 12, 4
    elif num_plots == 4:
        col_num, row_num = 4, 1
        col_size, row_size = 14, 4
    else:
        raise ValueError(f"Number of plots must be 2 or 3, got {num_plots}")

    save_mode = kwargs.get('save', None)
    if save_mode == 'pdf':
        plot_overlaid_iterations(
            init_variables,
            setting,
            fwd_results,
            rev_results,
            save_path=f"plots/simulation/overlay_setting={setting_name}.pdf",
            frame_skip=10,
            plot_ticks=False,
            gain_thresholds=gain_thresholds,
        )
    elif save_mode == 'params':
        save_path = f"plots/simulation/params_setting={setting_name}.pdf"
        plot_params(fwd_results, rev_results, setting, save_path)
    elif save_mode == 'teaser':
        save_path = f"plots/simulation/teaser/fwd-0.png"
        plot_single_frame(fwd_results, setting, 0, save_path, 'fwd')
        save_path = f"plots/simulation/teaser/fwd-1.png"
        plot_single_frame(fwd_results, setting, 1, save_path, 'fwd')
        save_path = f"plots/simulation/teaser/fwd-50.png"
        plot_single_frame(fwd_results, setting, 50, save_path, 'fwd')
        save_path = f"plots/simulation/teaser/rev-0.png"
        plot_single_frame(rev_results, setting, 0, save_path, 'rev')
        save_path = f"plots/simulation/teaser/rev-1.png"
        plot_single_frame(rev_results, setting, 1, save_path, 'rev')
        save_path = f"plots/simulation/teaser/rev-50.png"
        plot_single_frame(rev_results, setting, 50, save_path, 'rev')
    else:
        fig, axs = plt.subplots(row_num, col_num, figsize=(col_size, row_size))
        fig.suptitle(setting_name, fontsize=16)

        def calculate_max_frames():
            if not gain_thresholds:
                return setting.NUM_ITERS
            
            max_frames = 0
            for fwd_idx, (_, _, gain_fwd, _, _) in enumerate(fwd_results):
                if fwd_idx < len(gain_thresholds):
                    threshold = gain_thresholds[fwd_idx]
                    for j, gain_val in enumerate(gain_fwd):
                        if gain_val >= threshold:
                            max_frames = max(max_frames, j + 1)
                            break
                    else:
                        max_frames = max(max_frames, len(gain_fwd))
                else:
                    max_frames = max(max_frames, len(gain_fwd))
            
            for rev_idx, (_, _, gain_rev, _, _) in enumerate(rev_results):
                plot_idx = len(fwd_results) + rev_idx
                if plot_idx < len(gain_thresholds):
                    threshold = gain_thresholds[plot_idx]
                    for j, gain_val in enumerate(gain_rev):
                        if gain_val >= threshold:
                            max_frames = max(max_frames, j + 1)
                            break
                    else:
                        max_frames = max(max_frames, len(gain_rev))
                else:
                    max_frames = max(max_frames, len(gain_rev))
            
            return max_frames

        num_total_iters = calculate_max_frames()
        print(f"Animation will run for {num_total_iters} frames")
        
        anim = animation.FuncAnimation(
            fig,
            partial(
                animate,
                num_total_iters=num_total_iters,
                init_variables=init_variables,
                setting=setting,
                axs=axs,
                fwd_results=fwd_results,
                rev_results=rev_results,
                gain_thresholds=gain_thresholds,
            ),
            frames=num_total_iters,
            interval=setting.INTERVAL
        )
        plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.95])
        if save_mode == 'gif':
            writer = PillowWriter(fps=10)
            save_path = f"plots/simulation/{setting_name}_fwd={','.join(f'{lr:.2f}' for lr in setting.FWD_LR)}_rev={','.join(f'{lr:.2f}' for lr in setting.REV_LR)}.gif"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, writer=writer)
        else:
            plt.show()


if __name__ == '__main__':
    fire.Fire(main)
