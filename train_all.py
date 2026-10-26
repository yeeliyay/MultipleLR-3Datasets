"""Independently train all three datasets and save three separate JSON result files."""
from app import DATASETS, load_or_train, result_path

if __name__ == '__main__':
 for key, config in DATASETS.items():
  print(f'\n=== {config["title"]} ===', flush=True)
  try:
   r=load_or_train(key,force=True)
   print(f'Completed: {result_path(key)}', flush=True)
   print(f'Test RMSE: {r["best"]["rmse"]:.4f}; R2: {r["best"]["r2"]:.4f}',flush=True)
  except Exception as exc:
   print(f'Failed {key}: {exc}',flush=True)
