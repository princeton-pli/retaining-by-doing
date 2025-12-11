"""
python -m simulation.sweep
python -m simulation.sweep --save
"""
import fire
from rich import print
import matplotlib.pyplot as plt
from simulation.settings import TwoModeSetting
from simulation.run import (
    setup_variables,
    fit_forward_kl,
    fit_reinforce,
)


def plot(results, **kwargs):
    """
    Plot the results.
    - Three subplots (one for each right_mode)
    - x-axis: training steps
    - y-axis: drop, gain
    """
    num_modes = len(results)
    fig, axs = plt.subplots(1, num_modes, figsize=(6 * num_modes, 5), sharey=True)
    if num_modes == 1:
        axs = [axs]

    for idx, (right_mode, res) in enumerate(results.items()):
        setting = res['setting']
        print(right_mode)
        print(setting)
        ax = axs[idx]
        for i, fwd in enumerate(res['fwd']):
            ax.plot(fwd['drop'], label=f'FWD drop lr={TwoModeSetting.FWD_LR[i]:.3f}', linestyle='--', color=f'C{i}')
            ax.plot(fwd['gain'], label=f'FWD gain lr={TwoModeSetting.FWD_LR[i]:.3f}', linestyle='-', color=f'C{i}')
        for i, rev in enumerate(res['rev']):
            ax.plot(rev['drop'], label=f'PG drop lr={TwoModeSetting.REV_LR[i]:.3f}', linestyle='--', color=f'C{i+2}')
            ax.plot(rev['gain'], label=f'PG gain lr={TwoModeSetting.REV_LR[i]:.3f}', linestyle='-', color=f'C{i+2}')
        ax.set_title(f"Dist. of mean between q_new(x) and p_new(x): {right_mode[1] - setting.MU2:.2f}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Drop / Gain")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'plots/simulation/mean_dist.pdf')
        print(f"Saved to plots/simulation/mean_dist.pdf")
    else:
        plt.show()


def main(**kwargs):
    setting = TwoModeSetting()

    right_modes = [
        (0.25, 4.5, 0.7),
        (0.25, 5.5, 0.7),
        (0.25, 6.5, 0.7),
    ]

    results = dict()
    for right_mode in right_modes:
        setting.MIXTURE[1] = right_mode
        init_variables = setup_variables(setting)

        fwd_results = []
        for fwd_lr in TwoModeSetting.FWD_LR:
            fwd_hist, drop_fwd, gain_fwd, acc1_fwd, acc2_fwd = fit_forward_kl(init_variables, setting, fwd_lr)
            fwd_results.append(dict(
                drop=drop_fwd,
                gain=gain_fwd,
            ))
        
        rev_results = []
        for rev_lr in TwoModeSetting.REV_LR:
            rev_hist, drop_rev, gain_rev, acc1_rev, acc2_rev = fit_reinforce(init_variables, setting, rev_lr)
            rev_results.append(dict(
                drop=drop_rev,
                gain=gain_rev,
            ))

        results[right_mode] = dict(
            setting=setting,
            fwd=fwd_results,
            rev=rev_results,
        )
    plot(results, **kwargs)


if __name__ == '__main__':
    fire.Fire(main)
