from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from app import DATASETS, load_or_train


# 你指定的 feature 數量
FEATURE_PLAN = {
    'appliances': [9, 18, 27],
    'parkinsons': [7, 13, 19],
    'student': [11, 22, 32],
}

METHOD_LABELS = {
    'f_test': 'F-test',
    'rfe': 'RFE',
    'lasso': 'Lasso',
    'elasticnet': 'ElasticNet'
}

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / 'metric_plots_18'
OUT_DIR.mkdir(exist_ok=True)


def sanitize_name(text: str) -> str:
    return (
        text.lower()
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
    )


def get_rows_for_k(metrics, k):
    rows = [r for r in metrics if r['k'] == k]
    # 固定方法順序
    method_order = ['f_test', 'rfe', 'lasso', 'elasticnet']
    rows = sorted(rows, key=lambda x: method_order.index(x['method']))
    return rows


def save_metric_plot(dataset_title, dataset_key, k, rows, metric_key, ylabel, filename_suffix):
    methods = [METHOD_LABELS[r['method']] for r in rows]
    values = [r[metric_key] for r in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, values)
    plt.title(f'{dataset_title}\n{ylabel} Comparison (Top {k} Features)')
    plt.xlabel('Feature Selection Method')
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{v:.4f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()

    file_name = f'{dataset_key}_k{k}_{filename_suffix}.png'
    save_path = OUT_DIR / file_name
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'Saved: {save_path}')


def main():
    total = 0

    for dataset_key, ks in FEATURE_PLAN.items():
        cfg = DATASETS[dataset_key]
        dataset_title = cfg['title']

        print(f'\n=== {dataset_title} ===')

        # 讀取或重新訓練該資料集
        result = load_or_train(dataset_key, force=True)
        metrics = result['metrics']

        available_k = sorted(set(r['k'] for r in metrics))
        max_k = max(available_k)

        for k in ks:
            if k not in available_k:
                print(f'[WARNING] {dataset_key}: k={k} 不存在，可用最大值為 {max_k}，跳過。')
                continue

            rows = get_rows_for_k(metrics, k)

            # RMSE 圖
            save_metric_plot(
                dataset_title=dataset_title,
                dataset_key=dataset_key,
                k=k,
                rows=rows,
                metric_key='rmse',
                ylabel='RMSE',
                filename_suffix='rmse'
            )
            total += 1

            # R-squared 圖
            save_metric_plot(
                dataset_title=dataset_title,
                dataset_key=dataset_key,
                k=k,
                rows=rows,
                metric_key='r2',
                ylabel='R-squared',
                filename_suffix='r2'
            )
            total += 1

    print(f'\n完成，共輸出 {total} 張圖。')
    print(f'輸出資料夾：{OUT_DIR}')


if __name__ == '__main__':
    main()