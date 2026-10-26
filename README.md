# AIoT DIC2 · Multiple Linear Regression Explorer (3 Kaggle datasets)

A locally executable Flask + Plotly dashboard inspired by [AIoT-DIC2-MultipleLR](https://github.com/huanchen1107/AIoT-DIC2-MultipleLR). It is a separate implementation, not a copy of the original source. All plotted metrics are computed from real CSV files. **No fabricated prediction results are bundled.**

## Dataset source & feature count (original fields, before encoding)

| Data | Kaggle reference | Original input features | Target | Notes |
|---|---|---:|---|---|
| Appliance Energy Prediction | [Kaggle source](https://www.kaggle.com/competitions/machine-learning-sb1) | 28 fields + timestamp (time not counted as predictive feature) | Appliances | Remove rv1, rv2 random controls; 26 modeled inputs; chronological 80:20 split. |
| Parkinsons Telemonitoring | [Kaggle source](https://www.kaggle.com/datasets/leilahasan/parkinson-dataset) | 20 input variables after removing subject identifier and competing motor_UPDRS target | total_UPDRS | Group-wise 80:20 subject split prevents person leakage. |
| Student Performance (Math) | [Kaggle source](https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set) | 32 original inputs | G3 | Original 33 columns including target. Random 80:20 split. |

Feature count is for the *original predictor fields*, not one-hot expanded dimensions. The student category features expand to more encoded columns. The exact raw source count for Parkinsons is 22 columns total: subject#, 20 non-identifier fields including motor_UPDRS and total_UPDRS; thus **19 candidate features** after excluding subject# and motor_UPDRS, and excluding the target. (The code computes and displays the exact number.) Its 19 inputs satisfy the >13 and <40 constraint. 

The original UCI public repositories are also provided by `download_data.py` solely to facilitate repeatable downloads of the same underlying datasets. For strictly Kaggle-sourced submission, download the Kaggle copy and place the following files into `data/`:

- `data/energydata_complete.csv`
- `data/parkinsons_updrs.data`
- `data/student-mat.csv`

**Note**: The Kaggle competition linked for Appliance Energy has a competition-specific split; the dashboard uses the original complete underlying dataset with its own 80:20 chronological split. Verify that the CSV fields match the expected source before submission.

## Quick start (Windows CMD)

```bat
cd D:\AIOTHW2\MultipleLR_3Datasets
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python download_data.py
python train_all.py
python app.py
```

Open `http://127.0.0.1:5000`.

Internet is required once for UCI download and for loading Plotly from CDN. If automatic download fails, download the files from Kaggle and place them in `data/`.

## Assignment compliance: CRISP-DM

1. **Business Understanding:** Appliance electricity forecasting, disease-monitoring score estimation (not medical diagnosis), student grade forecasting.
2. **Data Understanding:** Inspect row counts, predictor count, target distribution and missing values; visible per dataset on web.
3. **Data Preparation:** Split train/test first, impute missing numeric/categorical features, standardize numeric features, one-hot encode categories. No train/test preprocessing leakage.
4. **Modeling:** Ordinary least squares linear regression. Compare F-test, RFE, Lasso, ElasticNet feature rankings; select k from 1 to 20 (or smaller if needed), using train-only 3-fold CV, then fit OLS.
5. **Evaluation:** Test RMSE, MAE, R², RMSE/R² by feature count, actual-vs-predicted, 95% prediction interval, residual analysis and observed PI coverage. Dashboard charts can be saved via Plotly's camera icon.
6. **Deployment:** Flask API and frontend for local use. Static GitHub Pages alone **cannot** execute Flask modeling; deploy Python backend on a Python-compatible host if public hosting is required.

## Plots

- Test RMSE vs number of selected features for four rankings
- Test R² vs number of selected features for four rankings
- Holdout actual vs predicted with **95% prediction interval**
- Predicted vs actual scatter and ideal y=x line
- Residual vs predicted diagnostics
- Feature ranking progression table and best-model coefficients table

### Prediction intervals

Used approximation: `prediction ± t(0.975, n-k-1) × sqrt(s² (1 + xᵀ(XᵀX)^+x))`, where the design matrix includes an intercept and predictors are fitted with ordinary least squares. The interval is for **individual future observations** under classical independent homoskedastic normal error assumptions. This is not the confidence interval of the conditional mean. For temporal and longitudinal data these assumptions can be violated. The coverage reported is descriptive, not an inferential guarantee.

### Model-selection limitation

The chosen k/method is based on CV *only on training data*, and the test set remains held out. Feature ranking is determined once on all training data before computing train CV scores; therefore this is an **approximate/non-nested CV** for hyperparameter choice, and may be optimistic. A rigorous nested CV would refit feature selection inside each fold. Avoid choosing the model by looking at the test curves, which are provided only for educational demonstration.

### Dataset limitations

- Appliance energy is sampled over time, so a time-based split is used. Subsequent prediction intervals ignore temporal dependence.
- Parkinsons: subjects are held out as groups for evaluation, but PI still ignores repeated-measures covariance.
- Student G1 and G2 are available before G3; if predicting prior to any exams, exclude them and refit the model.

## HTTP API

- `GET /api/analyze/appliances`
- `GET /api/analyze/parkinsons`
- `GET /api/analyze/student`
- `POST /api/refresh/<dataset_key>`

All APIs compute on-demand from local files; initial execution might take some time.

## 三個資料集獨立訓練與下拉切換

1. `python download_data.py`：取得 CSV（若已手動下載可跳過）。
2. `python train_all.py`：**獨立訓練三次**，產生 `results/appliances.json`、`results/parkinsons.json`、`results/student.json`。任一資料集失敗不會中止其他資料集。
3. `python app.py`：瀏覽 `http://127.0.0.1:5000`，在頁面左上的 `01 / DATASET` 下拉選單切換。
4. 切換時，網頁重新讀取專屬 `/api/analyze/<key>` 結果並清空舊圖，顯示對應資料集的 RMSE、R²、MAE、圖表、特徵、模型係數與 CRISP-DM。
5. `重新計算` 僅重新訓練當前所選的資料集，並更新對應的 JSON。若沒有事先執行 `train_all.py`，第一次選到資料集時也會自動進行一次訓練。

**圖表解讀提示：** RMSE/R² 上方選擇的特徵方法和 k 用於比較線圖與表格；下方預測圖／預測區間／殘差圖使用該資料集訓練集 CV 所選的推薦模型。不同資料集的預測結果絕不混合。
