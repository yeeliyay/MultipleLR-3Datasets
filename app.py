from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

BASE=Path(__file__).resolve().parent
app=Flask(__name__)
DATASETS={
 'appliances':{'title':'Appliances Energy Prediction','filename':'energydata_complete.csv','target':'Appliances','exclude':['date'],'kaggle':'https://www.kaggle.com/competitions/machine-learning-sb1','task':'依環境溫濕度、天氣與照明資訊預測家電用電量 (Wh)','split':'time','note':'date 為時間索引；rv1、rv2 是隨機對照變數，不列入模型。原始數據另有 28 個候選特徵，去除兩個隨機變數後使用 26 個。'},
 'parkinsons':{'title':'Parkinsons Telemonitoring','filename':'parkinsons_updrs.data','target':'total_UPDRS','exclude':['subject#','motor_UPDRS'],'kaggle':'https://www.kaggle.com/datasets/leilahasan/parkinson-dataset','task':'由聲學測量與基本資料估計 total_UPDRS','split':'group','note':'subject# 僅供依受試者分組，motor_UPDRS 與預測目標高度相關，為避免標籤洩漏排除。原始資料有 19 個候選輸入特徵（排除識別欄、另一 UPDRS 目標及預測目標）。'},
 'student':{'title':'Student Performance (Math)','filename':'student-mat.csv','target':'G3','exclude':[],'kaggle':'https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set','task':'以學生的背景、學習習慣及前期成績預測期末成績 G3','split':'random','note':'使用 32 個原始輸入欄位（不含 G3），類別變數採 One-Hot。G1/G2 是前期成績，僅適用於已取得前期成績的預測情境。'}
}
CACHE={}
RESULTS_DIR=BASE/'results'
RESULTS_DIR.mkdir(exist_ok=True)

def result_path(key):
 return RESULTS_DIR/f'{key}.json'

def load_or_train(key, force=False):
 if not force and key in CACHE:return CACHE[key]
 path=result_path(key)
 if not force and path.exists():
  with path.open(encoding='utf-8') as fp: result=json.load(fp)
 else:
  result=analyze(key)
  tmp=path.with_suffix('.json.tmp')
  with tmp.open('w',encoding='utf-8') as fp: json.dump(result,fp,ensure_ascii=False)
  tmp.replace(path)
 CACHE[key]=result
 return result

def load_data(key):
 cfg=DATASETS[key]; p=BASE/'data'/cfg['filename']
 if not p.exists():
  raise FileNotFoundError(f'尚未找到 {p.name}。請先執行 python download_data.py，或將 Kaggle 下載的原始 CSV 放進 data/。')
 df=pd.read_csv(p,sep=';' if key=='student' else ',')
 if cfg['target'] not in df.columns: raise ValueError(f'缺少目標欄位 {cfg["target"]}')
 df=df.dropna(subset=[cfg['target']]).copy()
 if key=='appliances': df=df.sort_values('date')
 return df

def feature_rank(X,y,method):
 n=X.shape[1]
 if method=='f_test':
  with np.errstate(all='ignore'): vals=f_regression(X,y)[0]
  score=np.nan_to_num(vals,nan=0,posinf=1e12,neginf=0)
 elif method=='rfe':
  model=RFE(LinearRegression(),n_features_to_select=1,step=max(1,n//12)).fit(X,y)
  score=-model.ranking_.astype(float)
 elif method in ('lasso','elasticnet'):
  est=(LassoCV(cv=3,random_state=42,max_iter=5000) if method=='lasso' else ElasticNetCV(cv=3,random_state=42,max_iter=5000)).fit(X,y)
  score=np.abs(est.coef_)
 else: raise ValueError('Unknown method')
 return np.argsort(-score,kind='stable')

def analyze(key):
 df=load_data(key); cfg=DATASETS[key]
 xcols=[c for c in df.columns if c not in [cfg['target'],*cfg['exclude']]]
 Xraw=df[xcols].copy(); y=pd.to_numeric(df[cfg['target']],errors='coerce').to_numpy(float)
 if cfg['split']=='time':
  cut=int(len(df)*.8); tr=np.arange(cut);te=np.arange(cut,len(df))
 elif cfg['split']=='group':
  from sklearn.model_selection import GroupShuffleSplit
  tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(Xraw,y,groups=df['subject#']))
 else: tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=42)
 nums=Xraw.select_dtypes(include='number').columns.tolist(); cats=[c for c in xcols if c not in nums]
 pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)],sparse_threshold=0)
 Xtr=pre.fit_transform(Xraw.iloc[tr]); Xte=pre.transform(Xraw.iloc[te]); names=pre.get_feature_names_out().tolist(); yy=y[tr];yt=y[te]
 # Exactly one fixed test set, feature selection trained exclusively on training subset.
 maxk = min(len(xcols), len(names), max(1, len(tr)-2)); ks=list(range(1,maxk+1))
 methods=['f_test','rfe','lasso','elasticnet']; records=[]; fits={}
 for method in methods:
  rank=feature_rank(Xtr,yy,method)
  for k in ks:
   ids=rank[:k]; reg=LinearRegression().fit(Xtr[:,ids],yy)
   pred=reg.predict(Xte[:,ids]);train_pred=reg.predict(Xtr[:,ids])
   row={'method':method,'k':k,'rmse':float(np.sqrt(mean_squared_error(yt,pred))),'mae':float(mean_absolute_error(yt,pred)),'r2':float(r2_score(yt,pred)),'train_rmse':float(np.sqrt(mean_squared_error(yy,train_pred))),'features':[names[i] for i in ids]}
   records.append(row);fits[(method,k)]=(ids,reg)
 # Select model with CV *training only*, not test set; evaluation is unbiased conditional on this procedure.
 from sklearn.model_selection import KFold, cross_val_score
 # Rank from full training data is used for illustrative progression; best-k via train-only CV on fixed rankings.
 evaluations=[]
 # CV speed limit: sample only training rows, never held-out test rows.
 cv_sample=np.random.default_rng(42).choice(len(yy),size=min(3500,len(yy)),replace=False)
 for method in methods:
  rank=feature_rank(Xtr,yy,method)
  for k in ks:
   # 5-fold quick CV on train, no test selection. Rankings still computed on whole training set;
   # selection CV is approximate; final metrics are held-out and untouched.
   Xsel=Xtr[cv_sample][:,rank[:k]]
   cv=KFold(n_splits=3,shuffle=True,random_state=42)
   rm=-cross_val_score(LinearRegression(),Xsel,yy[cv_sample],cv=cv,scoring='neg_root_mean_squared_error').mean()
   evaluations.append((rm,method,k))
 cv_rm,method,k=min(evaluations);ids,reg=fits[(method,k)]
 pred=reg.predict(Xte[:,ids]);r=yy-reg.predict(Xtr[:,ids]);dof=max(1,len(yy)-len(ids)-1)
 # Approximate OLS PI (normal independent errors, constant variance), intercept included.
 design=np.column_stack([np.ones(len(yy)),Xtr[:,ids]]); new=np.column_stack([np.ones(len(yt)),Xte[:,ids]])
 inv=np.linalg.pinv(design.T@design);lever=np.einsum('ij,jk,ik->i',new,inv,new)
 sigma2=float(np.sum(r*r)/dof); margin=stats.t.ppf(.975,dof)*np.sqrt(sigma2*(1+np.clip(lever,0,None)))
 order=np.argsort(pred)
 plotids=np.linspace(0,len(order)-1,min(250,len(order))).round().astype(int);samples=order[plotids]
 coefs=[{'feature':names[i],'beta':float(b)} for i,b in zip(ids,reg.coef_)]
 result={'key':key,'dataset':cfg,'count':len(df),'original_features':len(xcols),'encoded_features':len(names),'train':len(tr),'test':len(te),'metrics':records,'cv_rmse':float(cv_rm),'best':{'method':method,'k':k,'features':[names[i] for i in ids],'intercept':float(reg.intercept_),'coefs':coefs,'rmse':float(np.sqrt(mean_squared_error(yt,pred))),'mae':float(mean_absolute_error(yt,pred)),'r2':float(r2_score(yt,pred)),'pi_coverage':float(np.mean((yt>=pred-margin)&(yt<=pred+margin))),'pi_mean_width':float(np.mean(2*margin))},'plot':{'actual':yt[samples].tolist(),'predicted':pred[samples].tolist(),'lower':(pred-margin)[samples].tolist(),'upper':(pred+margin)[samples].tolist(),'residual':(yt-pred)[samples].tolist()},'target_stats':{'mean':float(np.mean(y)),'std':float(np.std(y)),'min':float(np.min(y)),'max':float(np.max(y))},'missing':int(df.isna().sum().sum())}
 return result

@app.route('/')
def home(): return render_template('index.html',datasets=DATASETS)
@app.route('/api/analyze/<key>')
def get_analysis(key):
 if key not in DATASETS:return jsonify(error='Unknown dataset'),404
 try:
  return jsonify(load_or_train(key))
 except Exception as e:return jsonify(error=f'{type(e).__name__}: {e}'),400
@app.route('/api/refresh/<key>',methods=['POST'])
def refresh(key):

 if key not in DATASETS:return jsonify(error='Unknown dataset'),404
 try:return jsonify(load_or_train(key,force=True))
 except Exception as e:return jsonify(error=f'{type(e).__name__}: {e}'),400
if __name__=='__main__':app.run(host='127.0.0.1',port=int(os.getenv('PORT','5000')),debug=False)

@app.cli.command('train-all')
def train_all():
 """Train the three datasets separately and persist per-dataset outputs."""
 for key in DATASETS:
  print(f'>>> Training dataset: {key}',flush=True)
  try:
   output=load_or_train(key,force=True)
   print(f'    Saved: {result_path(key)} (RMSE={output["best"]["rmse"]:.4f})',flush=True)
  except Exception as ex:
   print(f'    ERROR {key}: {ex}',flush=True)
