"""Compare saved molecular forecasts with conditional paired-gene intervals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/slp11-transition'
OUTPUT = BASE / 'human-essential-count-response-comparison-v1'
JOINT = BASE / 'human-essential-count-shared-context-development-evaluation-v2'
RANK = BASE / 'human-essential-count-response-rank32-seed731-v1'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'report.json').exists():
        raise FileExistsError('comparison already exists')
    for path, expected in (
        (JOINT / 'report.json', 'ca6438891609689cd2b00b0d9987f5ba44ff1270b1a3a3cb076b488a8bd25e07'),
        (RANK / 'report.json', 'f1f97a9cb5d4b782db969f6ed0aa83a4ab39ea81b6534e575253252fe9bc49af'),
    ):
        if digest(path) != expected:
            raise ValueError('fixed report changed')
    protocol = {
        'purpose': 'Descriptive paired-gene bootstrap intervals after aggregate development results; not an independent confirmation or new selection rule.',
        'resamples': 10000, 'seed': 731, 'interval': '2.5 and 97.5 percentiles',
        'estimand': 'Ratio of equal-gene candidate MSE to full static ridge MSE, paired by gene within source.',
        'conditionalOn': 'This adaptive development cohort, single fitted checkpoint and source. No between-seed, biological-replicate or new-context uncertainty is estimated.',
        'sourceSha256': digest(Path(__file__)),
    }
    (OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    joint_report = json.loads((JOINT / 'report.json').read_text())
    rank_report = json.loads((RANK / 'report.json').read_text())
    joint_path = JOINT / joint_report['perGeneScores']['path']
    rank_path = RANK / rank_report['perGeneMetrics']['path']
    if digest(joint_path) != joint_report['perGeneScores']['sha256'] or digest(rank_path) != rank_report['perGeneMetrics']['sha256']:
        raise ValueError('per-gene artifact changed')
    labels = ['Anchored mean', 'K562-only count', 'Joint count', 'Rank-32 response']
    keys = ['anchored_mean_prediction', 'k562_only_prediction', 'joint_prediction', 'rank32_prediction']
    colors = ['#94a3b8', '#a07854', '#476a8c', '#087f73']
    report = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    with np.load(joint_path, allow_pickle=False) as joint, np.load(rank_path, allow_pickle=False) as rank:
        for ax, source, title in zip(axes, ('k562', 'rpe1'), ('K562 · 305 genes', 'RPE1 · 360 genes')):
            genes = joint[f'{source}_gene_ids']
            if not np.array_equal(genes, rank[f'{source}_gene_ids']):
                raise ValueError('gene cohorts differ')
            reference = joint[f'{source}_static_ridge_prediction_mse']
            candidates = [joint[f'{source}_{key}_mse'] if key != 'rank32_prediction'
                          else rank[f'{source}_{key}_mse'] for key in keys]
            rng = np.random.default_rng(731)
            ratios = np.empty((10000, len(keys)))
            for left in range(0, 10000, 200):
                index = rng.integers(0, len(genes), size=(min(200, 10000 - left), len(genes)))
                denominator = reference[index].mean(1)
                for column, candidate in enumerate(candidates):
                    ratios[left:left + len(index), column] = candidate[index].mean(1) / denominator
            point = np.array([candidate.mean() / reference.mean() for candidate in candidates])
            low, high = np.quantile(ratios, [.025, .975], axis=0)
            report[source] = {key: {'mseRatioToRidge': float(value), 'pairedGene95Interval': [float(lo), float(hi)]}
                              for key, value, lo, hi in zip(keys, point, low, high)}
            for row, (value, lo, hi, color) in enumerate(zip(point, low, high, colors)):
                ax.plot([lo, hi], [row, row], color=color, linewidth=2.5)
                ax.scatter(value, row, color=color, s=65, zorder=3)
                ax.annotate(f'{100 * (value - 1):+.2f}%', (value, row), xytext=(0, 12),
                            textcoords='offset points', ha='center', fontsize=9, color=color)
            ax.axvline(1, color='#334155', linestyle='--', linewidth=1)
            ax.set_title(title, loc='left', fontweight='bold', pad=17)
            ax.set_xlabel('MSE / static ridge MSE (lower is better)', labelpad=10)
            ax.set_yticks(range(4), labels)
            ax.set_ylim(3.5, -.6)
            ax.grid(axis='x', color='#e2e8f0')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
    fig.suptitle('Compact response state improves measured-panel prediction', x=.02, ha='left', fontweight='bold', fontsize=14)
    fig.text(.02, .02, 'Adaptive development · 95% paired-gene bootstrap intervals · No independent confirmation', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .055, 1, .91))
    fig.savefig(OUTPUT / 'comparison.png', dpi=180)
    fig.savefig(OUTPUT / 'comparison.svg')
    plt.close(fig)
    report['limitations'] = protocol['conditionalOn']
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
