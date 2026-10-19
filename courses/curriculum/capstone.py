#!/usr/bin/env python3
"""Auditable, CPU-friendly reference workflow for the three supplied capstones.

Default runs never read final test labels. --final performs the locked comparison.
The validation split has its own seed (--split-seed, 42 by default), so runs with
different training seeds (--seed) differ only in initialization and batch order
and are always scored on the same validation rows.
Outputs are local teaching artifacts, not a claim of production readiness.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'course-matplotlib'))
import matplotlib
if __name__=='__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, average_precision_score, confusion_matrix)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'neural_networks_course'/'Day_3_Capstone'/'exercises'
SPECS={
 'esp': {'folder':'01_esp_gas_lock','train':'esp_train.csv','test':'esp_test.csv',
   'labels':'esp_test_actual_gas_lock.csv','target':'GAS_LOCK',
   'features':['PIP','PDP','Q','THP','Mamps','PIT'],'task':'classification'},
 'log': {'folder':'02_log_curves','train':'Log curve - Train.csv','test':'Log curve - Test.csv',
   'labels':'Log curve - Labels.csv','target':'PE',
   'features':['Depth','GR','ILD_log10','DeltaPHI','PHIND','RELPOS'],'task':'regression'},
 'sensor': {'folder':'03_sensor_signals','train':'Sensor_train.csv','test':'Sensor_test.csv',
   'labels':'Sensor_target.csv','target':'target',
   'features':[f'col{i}' for i in range(1,11)],'task':'classification'}
}

def prepare(project,seed=42):
    """Load, split and preprocess the training file; seed only controls the validation split."""
    spec=SPECS[project]
    df=pd.read_csv(DATA/spec['folder']/spec['train'])
    if df[spec['target']].isna().any():
        raise ValueError('Missing targets require a documented exclusion before splitting.')
    X=df[spec['features']]
    y=df[spec['target']].to_numpy()
    if project=='sensor':
        if not set(np.unique(y)) <= {-1,1}:raise ValueError('Unexpected sensor labels')
        y=(y==1).astype(int)
    if spec['task']=='classification' and set(np.unique(y))!={0,1}:
        raise ValueError('Expected binary target values 0 and 1')
    if project=='esp':
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=2,random_state=seed).split(X,y,groups=df.ENTITY))
        assert not set(df.ENTITY.iloc[tr]) & set(df.ENTITY.iloc[va])
        policy='Hold out two ESP entities. This evaluates unseen entities, not warning lead time.'
    elif project=='log':
        cut=int(.8*len(df));tr=np.arange(cut);va=np.arange(cut,len(df))
        policy='Last 20% by source-row order. Missing well IDs prevent a verified cross-well claim.'
    else:
        tr,va=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=seed)
        policy='Stratified row holdout; independence is unverified because asset/time metadata is missing.'
    prep=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    Xt=prep.fit_transform(X.iloc[tr]).astype('float32');Xv=prep.transform(X.iloc[va]).astype('float32')
    assert np.isfinite(Xt).all() and np.isfinite(Xv).all()
    target_scaler=None
    if spec['task']=='regression':
        target_scaler=StandardScaler().fit(y[tr,None]);yt=target_scaler.transform(y[tr,None]).ravel();yv=target_scaler.transform(y[va,None]).ravel()
    else:yt,yv=y[tr],y[va]
    return {'project':project,'spec':spec,'frame':df,'X':X,'y':y,'train_idx':tr,'val_idx':va,
            'Xt':Xt,'Xv':Xv,'yt':yt.astype('float32'),'yv':yv.astype('float32'),
            'prep':prep,'target_scaler':target_scaler,'split_policy':policy}

def build_model(n_features,width=32):
    return nn.Sequential(nn.Linear(n_features,width),nn.ReLU(),nn.Linear(width,1))

def predict_neural(bundle,X):
    z=bundle['prep'].transform(X).astype('float32')
    bundle['model'].eval()
    with torch.inference_mode():
        raw=bundle['model'](torch.from_numpy(z)).squeeze(-1).numpy()
    if bundle['task']=='classification':return 1/(1+np.exp(-np.clip(raw,-80,80)))
    return bundle['target_scaler'].inverse_transform(raw[:,None]).ravel()

def metrics(y,p,task,threshold=.5):
    if task=='regression':return {'MAE':float(mean_absolute_error(y,p)),
        'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
    pred=p>=threshold
    return {'average_precision':float(average_precision_score(y,p)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'F1_event':float(f1_score(y,pred,zero_division=0)),
        'F1_macro':float(f1_score(y,pred,average='macro',zero_division=0)),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def choose_threshold(y,p,project):
    average='macro' if project=='sensor' else 'binary'
    thresholds=np.linspace(.05,.95,37)
    scores=[f1_score(y,p>=t,average=average,zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])

def final_data(project):
    spec=SPECS[project];path=DATA/spec['folder']
    test=pd.read_csv(path/spec['test']);labels=pd.read_csv(path/spec['labels'])
    if project in ['esp','sensor']:
        if test.uuid.duplicated().any() or labels.uuid.duplicated().any():raise ValueError('Duplicate UUIDs')
        joined=test[['uuid']].merge(labels,on='uuid',how='left',validate='one_to_one',sort=False)
        if len(joined)!=len(test) or joined[spec['target']].isna().any():raise ValueError('Label alignment failed')
        y=joined[spec['target']].to_numpy();ids=test.uuid.to_numpy()
    else:
        if list(labels.columns)!=['PE'] or len(labels)!=len(test):raise ValueError('Positional PE labels do not match')
        y=labels.PE.to_numpy();ids=np.arange(len(test))
    if project=='sensor':y=(y==1).astype(int)
    return test[spec['features']],y,ids

def run(project,epochs=40,width=32,lr=.001,seed=42,final=False,out=None,split_seed=42):
    if epochs<1:raise ValueError('epochs must be positive')
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(1)
    data=prepare(project,split_seed);task=data['spec']['task']
    estimator=Ridge(alpha=1) if task=='regression' else LogisticRegression(max_iter=1500,random_state=seed)
    baseline=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),estimator)
    baseline.fit(data['X'].iloc[data['train_idx']],data['y'][data['train_idx']])
    model=build_model(data['Xt'].shape[1],width)
    criterion=nn.MSELoss() if task=='regression' else nn.BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(data['Xt']),torch.from_numpy(data['yt'])),
        batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');best_state=None;best_epoch=0;stale=0;history=[]
    for epoch in range(epochs):
        model.train();total=0
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(xb).squeeze(-1),yb)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=loss.item()*len(xb)
        model.eval()
        with torch.inference_mode():vl=criterion(model(torch.from_numpy(data['Xv'])).squeeze(-1),torch.from_numpy(data['yv'])).item()
        history.append({'epoch':epoch+1,'train_loss':total/len(data['Xt']),'val_loss':vl})
        if vl<best-1e-7:best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(best_state)
    bundle={'prep':data['prep'],'target_scaler':data['target_scaler'],'model':model,'task':task}
    Xv=data['X'].iloc[data['val_idx']];yv=data['y'][data['val_idx']]
    pn=predict_neural(bundle,Xv)
    pb=baseline.predict(Xv) if task=='regression' else baseline.predict_proba(Xv)[:,1]
    thresholds={'baseline':.5,'neural':.5} if task=='regression' else {
        'baseline':choose_threshold(yv,pb,project),'neural':choose_threshold(yv,pn,project)}
    result={'project':project,'task':task,'seed':seed,'split_seed':split_seed,'width':width,'learning_rate':lr,
        'epochs_requested':epochs,'best_epoch':best_epoch,'feature_names':data['spec']['features'],
        'split_policy':data['split_policy'],'train_rows':len(data['train_idx']),'validation_rows':len(data['val_idx']),
        'thresholds':thresholds,'validation_is_used_for_selection':True,
        'validation':{k:metrics(yv,p,task,thresholds[k]) for k,p in [('baseline',pb),('neural',pn)]},
        'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                    'scikit-learn':sklearn.__version__}}
    out=Path(out) if out else ROOT/'output'/'capstone'/project
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(out/'learning_curve.csv',index=False)
    pd.DataFrame({'source_row':np.r_[data['train_idx'],data['val_idx']],
        'partition':['train']*len(data['train_idx'])+['validation']*len(data['val_idx'])}).to_csv(out/'split.csv',index=False)
    predictions=pd.DataFrame({'source_row':data['val_idx'],
        'baseline_prediction':pb,'neural_prediction':pn,'target_encoded':yv})
    if project=='esp':
        predictions['ENTITY']=data['frame'].ENTITY.iloc[data['val_idx']].to_numpy()
    if project=='sensor':
        predictions['uuid']=data['frame'].uuid.iloc[data['val_idx']].to_numpy()
    predictions.to_csv(out/'validation_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Training');ax.plot([h['epoch'] for h in history],[h['val_loss'] for h in history],label='Validation');ax.set(xlabel='Epoch',ylabel='MSE on standardized target' if task=='regression' else 'Binary cross-entropy');ax.legend();fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=180);plt.close(fig)
    joblib.dump({'preprocessing':data['prep'],'target_scaler':data['target_scaler'],'baseline':baseline},out/'preprocessing_and_baseline.joblib')
    torch.save(model.state_dict(),out/'neural_weights.pt')
    # A real reload check verifies the saved preprocessing and parameter state.
    saved=joblib.load(out/'preprocessing_and_baseline.joblib')
    reloaded=build_model(len(data['spec']['features']),width)
    reloaded.load_state_dict(torch.load(out/'neural_weights.pt',map_location='cpu',weights_only=True))
    check={'prep':saved['preprocessing'],'target_scaler':saved['target_scaler'],'model':reloaded,'task':task}
    np.testing.assert_allclose(predict_neural(check,Xv.iloc[:8]),pn[:8],rtol=1e-6,atol=1e-6)
    result['reload_prediction_check']='passed'
    if final:
        Xf,yf,ids=final_data(project);nf=predict_neural(bundle,Xf);bf=baseline.predict(Xf) if task=='regression' else baseline.predict_proba(Xf)[:,1]
        result['final_test']={k:metrics(yf,p,task,thresholds[k]) for k,p in [('baseline',bf),('neural',nf)]}
        pd.DataFrame({'uuid' if project!='log' else 'source_row':ids,'baseline_prediction':bf,'neural_prediction':nf,'target_encoded':yf}).to_csv(out/'final_predictions.csv',index=False)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('project',choices=list(SPECS));p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--width',type=int,default=32);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42,help='seed for initialization and batch order')
    p.add_argument('--split-seed',type=int,default=42,dest='split_seed',help='seed for the validation split; keep it fixed when comparing settings')
    p.add_argument('--final',action='store_true')
    p.add_argument('--out',type=Path)
    run(**vars(p.parse_args()))
