"""Export 120 RMSE/R² line charts from existing independent training results.

Place this script next to app.py and train_all.py; run: python export_120_line_charts.py
Requires matplotlib and results/{appliances,parkinsons,student}.json.
Each chart shows the metric for feature counts 1..K, with K highlighted.
"""
from pathlib import Path
import json
import math
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
except ImportError:
    print('Missing matplotlib. Install it with: python -m pip install matplotlib')
    sys.exit(1)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
OUTPUT = BASE / 'line_charts_120'
PLANS = {
    'appliances': [6, 12, 17, 22, 27],
    'parkinsons': [4, 8, 12, 16, 19],
    'student': [6, 12, 18, 25, 32],
}
LABELS = {
    'appliances': 'Appliances Energy',
    'parkinsons': 'Parkinsons Telemonitoring',
    'student': 'Student Performance',
}
METHODS = {
    'f_test': 'F-test',
    'rfe': 'RFE',
    'lasso': 'Lasso',
    'elasticnet': 'ElasticNet',
}
METRICS = {
    'rmse': ('RMSE', 'Lower is better', '#9467bd'),
    'r2': ('R-squared (R²)', 'Higher is better', '#2575b9'),
}


def load_metrics(dataset):
    file = RESULTS / f'{dataset}.json'
    if not file.is_file():
        raise FileNotFoundError(f'Missing {file}. Run: python train_all.py')
    with file.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data.get('metrics'), list):
        raise ValueError(f'Invalid result file: {file} has no metrics list')
    return data['metrics']


def validate(plan, lookup, dataset):
    missing = [(method, k) for method in METHODS for k in range(1, max(plan) + 1)
               if (method, k) not in lookup]
    if missing:
        available = [k for (method, k) in lookup if method == 'f_test']
        highest = max(available, default=0)
        raise ValueError(
            f'{dataset}: saved metrics only reach k={highest} or are incomplete. '
            f'Requires all k=1..{max(plan)} for all four methods. '
            'Check app.py maxk and rerun: python train_all.py'
        )


def make_plot(dataset, method, k, metric, lookup):
    title, hint, color = METRICS[metric]
    xs = list(range(1, k + 1))
    ys = [float(lookup[(method, n)][metric]) for n in xs]
    if not all(math.isfinite(v) for v in ys):
        raise ValueError(f'{dataset}/{method}/{metric}: non-finite metric')

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#ffffff')
    ax.plot(xs, ys, color=color, lw=2.7, marker='o', ms=5,
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=.8)
    # Highlight the user's requested feature count, like the provided sample.
    ax.scatter([k], [ys[-1]], s=165, color='#10b981',
               edgecolor='white', linewidth=2, zorder=6)
    ax.annotate(f'k={k}: {ys[-1]:,.4f}', (k, ys[-1]), xytext=(-12, 15),
                textcoords='offset points', ha='right', fontsize=10,
                fontweight='bold', color='#047857')
    ax.set_title(f'{title} by Number of Features\n'
                 f'{LABELS[dataset]}  |  {METHODS[method]}  |  k = 1–{k}',
                 fontsize=14, fontweight='bold', color='#17233b', pad=18)
    ax.text(0, 1.005, hint, transform=ax.transAxes, fontsize=9.5,
            color='#64748b', va='bottom')
    ax.set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_xlim(.75, k + .6)
    if k <= 12:
        ax.set_xticks(xs)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.3f}'))
    ax.grid(True, which='major', color='#e9edf3', linewidth=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color('#cbd5e1')
    ax.tick_params(colors='#566174')
    ax.margins(y=.16)
    out_dir = OUTPUT / dataset / METHODS[method].lower().replace('-', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f'{dataset}_{method}_k{k:02d}_{metric}.png'
    fig.tight_layout(pad=2.3)
    fig.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return filename


def main():
    # Validate all three data files before outputting any charts.
    lookups = {}
    for dataset, plan in PLANS.items():
        rows = load_metrics(dataset)
        lookup = {(r['method'], int(r['k'])): r for r in rows}
        validate(plan, lookup, dataset)
        lookups[dataset] = lookup
    count = 0
    for dataset, plan in PLANS.items():
        print(f'\n[{LABELS[dataset]}]')
        for k in plan:
            for method in METHODS:
                for metric in METRICS:
                    path = make_plot(dataset, method, k, metric, lookups[dataset])
                    print(f'  {path.relative_to(OUTPUT)}')
                    count += 1
    print(f'\nCompleted: {count} images. Folder: {OUTPUT}')
    assert count == 120


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
