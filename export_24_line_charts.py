"""Export 24 RMSE/R² line charts from existing training results.

Each dataset has exactly one RMSE chart and one R² chart for each feature-selection method.
X-axis only uses the five user-specified feature counts for that dataset.
Total charts = 3 datasets * 4 methods * 2 metrics = 24.

Place this script next to app.py and train_all.py; run: python export_24_line_charts.py
Requires matplotlib and results/{appliances,parkinsons,student}.json.
"""
from pathlib import Path
import json
import math
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    print('Missing matplotlib. Install it with: python -m pip install matplotlib')
    sys.exit(1)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
OUTPUT = BASE / 'line_charts_24'
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
    'rmse': ('RMSE by Number of Features', 'RMSE (Lower is Better)', 'Lower is better.', '#9467bd'),
    'r2': ('R-squared by Number of Features', 'R-squared (Higher is Better)', 'Higher is better.', '#2575b9'),
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
    missing = [(method, k) for method in METHODS for k in plan if (method, k) not in lookup]
    if missing:
        available = sorted({k for (method, k) in lookup if method == 'f_test'})
        raise ValueError(
            f'{dataset}: missing some required (method, k) results. '\
            f'Required k = {plan}. Available sample k values = {available[:10]}... '\
            'Please confirm app.py maxk and rerun: python train_all.py'
        )


def make_plot(dataset, method, metric, lookup):
    feature_points = PLANS[dataset]
    chart_title, y_label, hint, color = METRICS[metric]
    xs = feature_points
    ys = [float(lookup[(method, k)][metric]) for k in xs]
    if not all(math.isfinite(v) for v in ys):
        raise ValueError(f'{dataset}/{method}/{metric}: non-finite metric')

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#ffffff')

    ax.plot(xs, ys, color=color, lw=2.7, marker='o', ms=6,
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=.9)

    # Highlight all five requested feature-count points.
    ax.scatter(xs, ys, s=140, color='#10b981', edgecolor='white', linewidth=2, zorder=6)
    for x, y in zip(xs, ys):
        ax.annotate(f'({x}, {y:.4f})', (x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8.8,
                    fontweight='bold', color='#047857')

    ax.set_title(
        f'{chart_title}\n{LABELS[dataset]}  |  {METHODS[method]}',
        fontsize=14, fontweight='bold', color='#17233b', pad=18
    )
    ax.text(0, 1.005, hint, transform=ax.transAxes, fontsize=9.5,
            color='#64748b', va='bottom')

    ax.set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
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
    filename = out_dir / f'{dataset}_{method}_{metric}.png'
    fig.tight_layout(pad=2.3)
    fig.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return filename


def main():
    lookups = {}
    for dataset, plan in PLANS.items():
        rows = load_metrics(dataset)
        lookup = {(r['method'], int(r['k'])): r for r in rows}
        validate(plan, lookup, dataset)
        lookups[dataset] = lookup

    count = 0
    for dataset in PLANS:
        print(f'\n[{LABELS[dataset]}]')
        for method in METHODS:
            for metric in METRICS:
                path = make_plot(dataset, method, metric, lookups[dataset])
                print(f'  {path.relative_to(OUTPUT)}')
                count += 1

    print(f'\nCompleted: {count} images. Folder: {OUTPUT}')
    assert count == 24


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
