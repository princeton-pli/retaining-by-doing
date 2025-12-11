"""
Single-Gaussian version of your forward/reverse KL playground.

Usage examples (same spirit as your old script):

# interactive anim (matplotlib window)
python -m simulation.run_single_mode --gain_thresholds "0.9,0.9"

# save overlaid curves to PDF
python -m simulation.run_single_mode --gain_thresholds "0.9,0.9" --save pdf

# save animation to GIF
python -m simulation.run_single_mode --save gif

# use custom setting
python -m simulation.run_single_mode --setting SingleModeSetting

Only one q-component: q(x) = N(mu, sig). We update (mu, sig) only.
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
    
    q_vals_init = norm.pdf(xs, setting.MU1, setting.SIG1)
    overlap1_init = np.trapz(np.minimum(p1_vals, q_vals_init), xs)/area_p1
    overlap2_init = np.trapz(np.minimum(p2_vals, q_vals_init), xs)/area_p2
    
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


def q_pdf(x, mu1, s1, setting):
    return norm.pdf(x, mu1, s1)


def sample_q(n, mu1, s1, setting):
    return np.random.normal(mu1, s1, size=n)


def fit_forward_kl(init_variables, setting, learning_rate):
    mu1, s1 = setting.MU1, setting.SIG1

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

    hist.append((mu1, s1))

    for _ in range(setting.NUM_ITERS):
        Xp = sample_p_right(setting.NUM_SAMPLES, w1, m1, s1, w2, m2, s2, setting)
        qx = norm.pdf(Xp, mu1, s1) + 1e-12

        g_mu1 = -np.mean((Xp - mu1) / (s1**2))
        g_s1 = -np.mean(-(1/s1) + (Xp - mu1)**2 / (s1**3))

        mu1 -= learning_rate * g_mu1
        s1 -= learning_rate * g_s1

        s1 = max(s1, 1e-3)

        hist.append((mu1, s1))

        q_vals = norm.pdf(xs, mu1, s1)
        current_overlap1 = np.trapz(np.minimum(p1_vals, q_vals), xs)/area_p1
        current_overlap2 = np.trapz(np.minimum(p2_vals, q_vals), xs)/area_p2
        
        drop = current_overlap1 - overlap1_init
        gain = current_overlap2 - overlap2_init

        drop_fwd.append(drop)
        gain_fwd.append(gain)
        acc1_fwd.append(current_overlap1)
        acc2_fwd.append(current_overlap2)
    
    q_final = q_pdf(xs, mu1, s1, setting)
    q_final_integral = np.trapz(q_final, xs)
    print(f"Forward KL: final q's probability density integrates to: {q_final_integral}")

    return hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd


def fit_reinforce(init_variables, setting, learning_rate):
    """
    • Draws X ~ N(mu1, s1²)   (single component).
    • Reward  r(x) = log p₂(x) with p₂ the *target* right-hand Gaussian.
    • Updates θ with Monte-Carlo ∇  E_q [ A(x) ∇θ log qθ(x) ].
    • Returns the same (hist, drop, gain, acc1, acc2) lists as other fit_*().
    """

    mu1, s1 = setting.MU1, setting.SIG1
    print(f"Initial params: mu1: {mu1:.3f}, s1: {s1:.3f}")

    w2, m2_tgt, s2_tgt = setting.MIXTURE[1]

    xs = init_variables['xs']
    p1_vals = init_variables['p1_vals']
    p2_vals = init_variables['p2_vals']
    area_p1 = init_variables['area_p1']
    area_p2 = init_variables['area_p2']
    overlap1_init = init_variables['overlap1_init']
    overlap2_init = init_variables['overlap2_init']

    hist, drop_hist, gain_hist = [], [], []
    acc1_hist, acc2_hist = [], []

    hist = [(mu1, s1)]

    for _ in range(setting.NUM_ITERS):
        Xq = sample_q(setting.NUM_SAMPLES, mu1, s1, setting)

        qx = norm.pdf(Xq, mu1, s1) + 1e-12

        p2 = norm.pdf(Xq, m2_tgt, s2_tgt)

        reward = np.log(p2 / qx / setting.REINFORCE_BETA + 1e-12)
        
        score_mu1 = (Xq - mu1) / (s1 ** 2)
        score_s1 = -(1 / s1) + (Xq - mu1) ** 2 / s1 ** 3

        g_mu1 = np.mean(reward * score_mu1)
        g_s1 = np.mean(reward * score_s1)

        mu1 += learning_rate * g_mu1
        s1 += learning_rate * g_s1

        s1 = max(s1, 1e-3)

        hist.append((mu1, s1))

        q_vals = norm.pdf(xs, mu1, s1)
        curr_ov1 = np.trapz(np.minimum(p1_vals, q_vals), xs) / area_p1
        curr_ov2 = np.trapz(np.minimum(p2_vals, q_vals), xs) / area_p2

        drop_hist.append(curr_ov1 - overlap1_init)
        gain_hist.append(curr_ov2 - overlap2_init)
        acc1_hist.append(curr_ov1)
        acc2_hist.append(curr_ov2)

    q_final = q_pdf(xs, mu1, s1, setting)
    print(f"Reverse-KL: final q integrates to {np.trapz(q_final, xs):.6f}")
    print(f"Final params: mu1: {mu1:.3f}, s1: {s1:.3f}")
    return hist, drop_hist, gain_hist, acc1_hist, acc2_hist


def fit_reverse_kl(init_variables, setting, learning_rate):
    """
    Reverse KL divergence: KL(q||p) = E_q[log q(x) - log p(x)]
    We minimize this by sampling from q and computing gradients.
    """
    mu1, s1 = setting.MU1, setting.SIG1

    xs = init_variables['xs']
    p1_vals = init_variables['p1_vals']
    p2_vals = init_variables['p2_vals']
    area_p1 = init_variables['area_p1']
    area_p2 = init_variables['area_p2']
    overlap1_init = init_variables['overlap1_init']
    overlap2_init = init_variables['overlap2_init']

    hist = []
    acc1_rev = []
    drop_rev = []
    acc2_rev = []
    gain_rev = []

    hist.append((mu1, s1))

    for _ in range(setting.NUM_ITERS):
        Xq = sample_q(setting.NUM_SAMPLES, mu1, s1, setting)
        
        qx = norm.pdf(Xq, mu1, s1) + 1e-12
        _, m2_tgt, s2_tgt = setting.MIXTURE[1]
        px = norm.pdf(Xq, m2_tgt, s2_tgt) + 1e-12
        
        log_q = np.log(qx)
        log_p = np.log(px)
        
        score_mu1 = (Xq - mu1) / (s1 ** 2)
        score_s1 = -(1 / s1) + (Xq - mu1) ** 2 / s1 ** 3
        
        g_mu1 = np.mean(score_mu1 * (log_q - log_p))
        g_s1 = np.mean(score_s1 * (log_q - log_p))

        mu1 -= learning_rate * g_mu1
        s1 -= learning_rate * g_s1

        s1 = max(s1, 1e-3)

        hist.append((mu1, s1))

        q_vals = norm.pdf(xs, mu1, s1)
        current_overlap1 = np.trapz(np.minimum(p1_vals, q_vals), xs)/area_p1
        current_overlap2 = np.trapz(np.minimum(p2_vals, q_vals), xs)/area_p2
        
        drop = current_overlap1 - overlap1_init
        gain = current_overlap2 - overlap2_init

        drop_rev.append(drop)
        gain_rev.append(gain)
        acc1_rev.append(current_overlap1)
        acc2_rev.append(current_overlap2)
    
    q_final = q_pdf(xs, mu1, s1, setting)
    q_final_integral = np.trapz(q_final, xs)
    print(f"Reverse KL: final q's probability density integrates to: {q_final_integral}")

    return hist, drop_rev, gain_rev, acc1_rev, acc2_rev


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
        
        mu1, s1 = fwd_hist[current_iter]
        qf = q_pdf(xs, mu1, s1, setting)
        qf /= np.trapz(qf, xs)
        qf *= setting.PI1
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
        
        mu1, s1 = rev_hist[current_iter]
        qr = q_pdf(xs, mu1, s1, setting)
        qr /= np.trapz(qr, xs)
        qr *= setting.PI1
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
        figsize=(4 * num_plots + 9, 4) if not plot_ticks else (4 * num_plots + 5, 4),
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
        
        fwd_frames = [0, 1, 20, 50, 100, 200, 300, 350, 380, 384]
        for i, (mu1, s1) in enumerate(hist):
            if i not in fwd_frames:
                continue
            t_lin = i / (max_iter if max_iter > 0 else 1)
            t = t_lin ** 0.7
            c = interp_color(start_fwd, end_fwd, t)
            q = q_pdf(xs, mu1, s1, setting)
            q /= np.trapz(q, xs)
            q *= setting.PI1
            ax.plot(xs, q, color=c, linewidth=4)
        
        ax.text(0.37, 0.9, f'Forward KL', transform=ax.transAxes, fontsize=30)
        safe_idx = min(max_iter, len(drop_fwd) - 1) if drop_fwd else 0
        ax.text(0.02, 0.82, f'Old Mode', transform=ax.transAxes, fontsize=24)
        ax.text(0.02, 0.70, f'Drop={-drop_fwd[safe_idx]:.2f}', transform=ax.transAxes, fontsize=24)
        ax.text(0.58, 0.63, f'Target Mode', transform=ax.transAxes, fontsize=24)
        ax.text(0.58, 0.50, f'Gain={gain_fwd[safe_idx]:.2f}', transform=ax.transAxes, fontsize=24)
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
        
        for i, (mu1, s1) in enumerate(hist):
            if i > max_iter or (i % frame_skip and i != max_iter):
                continue
            t_lin = i / (max_iter if max_iter > 0 else 1)
            t = t_lin ** 0.7
            c = interp_color(start_rev, end_rev, t)
            q = q_pdf(xs, mu1, s1, setting)
            q /= np.trapz(q, xs)
            q *= setting.PI1
            ax.plot(xs, q, color=c, linewidth=4)
        ax.text(0.37, 0.9, f'Reverse KL', transform=ax.transAxes, fontsize=30)
        safe_idx = min(max_iter, len(drop_rev) - 1) if drop_rev else 0
        ax.text(0.02, 0.82, f'Old Mode', transform=ax.transAxes, fontsize=24)
        ax.text(0.02, 0.70, f'Drop={-drop_rev[safe_idx]:.2f}', transform=ax.transAxes, fontsize=24)
        ax.text(0.58, 0.63, f'Target Mode', transform=ax.transAxes, fontsize=24)
        ax.text(0.58, 0.50, f'Gain={gain_rev[safe_idx]:.2f}', transform=ax.transAxes, fontsize=24)
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
    Plot the progression of the parameters (mu1, sigma1) and the gain/drop over training steps.
    Each gets its own subplot. All runs (forward/reverse, all learning rates) are overlaid.
    """
    param_names = [r'$\mu_1$', r'$\sigma_1$', 'Drop', 'Gain']
    param_indices = [0, 1]  # indices for mu1, sigma1 in hist
    num_params = 2
    num_metrics = 2  # drop, gain
    num_plots = num_params + num_metrics
    num_fwd = len(fwd_results)
    num_rev = len(rev_results)

    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    for i, (hist, drop, gain, _, _) in enumerate(fwd_results):
        hist = np.array(hist)  # shape: (num_steps+1, 2)
        for p, idx in enumerate(param_indices):
            axs[p].plot(hist[:, idx], label=f'FWD lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')
        axs[2].plot(drop, label=f'FWD drop lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')
        axs[3].plot(gain, label=f'FWD gain lr={setting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')

    for i, (hist, drop, gain, _, _) in enumerate(rev_results):
        hist = np.array(hist)  # shape: (num_steps+1, 2)
        for p, idx in enumerate(param_indices):
            axs[p].plot(hist[:, idx], label=f'PG lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')
        axs[2].plot(drop, label=f'PG drop lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')
        axs[3].plot(gain, label=f'PG gain lr={setting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+num_fwd}')

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
            mu1, s1 = hist[frame_idx]
            q = q_pdf(xs, mu1, s1, setting)
            q /= np.trapz(q, xs)
            q *= setting.PI1
            
            plt.figure(figsize=(10, 6))
            plt.plot(xs, px, 'k--', linewidth=4)
            plt.plot(xs, q, color, linewidth=4)

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
        print('python -m simulation.run_single_mode --setting <setting_name>')
        print('python -m simulation.run_single_mode --gain_thresholds 0.5,0.3')
        print('List of available setting names:')
        from simulation.utils import list_class_objects
        from simulation.settings import SingleModeSetting
        list_class_objects('simulation.settings', SingleModeSetting)
        return

    setting_name = kwargs.get('setting', 'SingleModeSetting')
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
            frame_skip=30,
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
