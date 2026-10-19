"""Figures shared by several modules; slides reference them by names without a module prefix.

Module-specific figures live in figs/<key>.py.
"""
from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register
import matplotlib
import matplotlib.ticker
import pandas as pd
from scipy.special import expit, softmax
from sklearn.datasets import make_moons, make_blobs, make_classification, load_digits
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, mean_squared_error, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

KEY = 'core'

# Provenance kind for the original shared figures (see figlib.save).
_SCHEMATIC = {'bias_variance', 'sequence_window', 'training_curves'}
_DATA = {'log_profile', 'sensor_profile', 'esp_profile'}
_save = save


def save(name, fig, source='Deterministic synthetic teaching example generated with random seed 42.', kind=None, **kw):
    if kind is None:
        kind = 'schematic' if name in _SCHEMATIC else 'data' if name in _DATA else 'computed'
    return _save(name, fig, source, kind=kind, **kw)

def boundaries(name, models, titles):
    X,y=make_moons(n_samples=220,noise=.23,random_state=42)
    xx,yy=np.meshgrid(np.linspace(-1.75,2.7,540),np.linspace(-1.25,1.8,390))
    grid=np.c_[xx.ravel(),yy.ravel()]
    fig,axes=sub(len(models))
    for ax,model,title in zip(axes,models,titles):
        model.fit(X,y)
        zz=model.predict(grid).reshape(xx.shape)
        ax.contourf(xx,yy,zz,levels=[-.5,.5,1.5],cmap=BG)
        ax.contour(xx,yy,zz,levels=[.5],colors=['#8193A6'],linewidths=.8)
        points(ax,X,y);ax.set_title(title);ax.set_xlim(-1.75,2.7);ax.set_ylim(-1.25,1.8)
        ax.set_xlabel('Feature 1');ax.set_ylabel('Feature 2' if ax is axes[0] else '')
    save(name,fig)

def regression():
    rng=np.random.default_rng(42)
    x=np.sort(rng.uniform(-1,1,32));truth=lambda z:1+z+1.6*z*z
    y=truth(x)+rng.normal(0,.24,len(x));xx=np.linspace(-1,1,300)
    fig,axes=sub(3)
    for ax,degree in zip(axes,[0,2,12]):
        model=make_pipeline(PolynomialFeatures(degree),LinearRegression()).fit(x[:,None],y)
        ax.scatter(x,y,s=25,c=B,label='Training samples',zorder=3)
        ax.plot(xx,model.predict(xx[:,None]),color=O,label='Fitted prediction')
        ax.plot(xx,truth(xx),'--',color=T,label='Generating function',lw=1.6)
        ax.set_ylim(-.2,4.3);ax.set_title(f'Polynomial degree {degree}')
        clean(ax,'Input x','Target y' if ax is axes[0] else None)
    axes[0].legend(loc='upper left',fontsize=12)
    save('regression_fit',fig)
    fit=LinearRegression().fit(x[:,None],y);pred=fit.predict(x[:,None])
    fig,axes=sub(2)
    axes[0].scatter(x,y,c=B,s=28,label='Observed');axes[0].plot(x,pred,color=O,label='Linear fit')
    for a,b,c in zip(x,y,pred):axes[0].plot([a,a],[b,c],color=G,alpha=.5,lw=1)
    axes[0].legend();axes[0].set_title('Observed values and fitted line')
    axes[1].scatter(x,y-pred,c=B,s=28);axes[1].axhline(0,color=O,ls='--')
    axes[1].set_title('Residual pattern')
    clean(axes[0],'Input x','Target y');clean(axes[1],'Input x','Observed − predicted')
    save('regression_residuals',fig)
    fig,axes=sub();ax=axes[0]
    c=np.linspace(.3,8,200);bias=3/(c+.3);var=.035*c*c;noise=np.full_like(c,.3)
    for v,label,col,ls in [(bias,'Squared bias',B,'-'),(var,'Variance',O,'-'),(noise,'Noise',G,':'),(bias+var+noise,'Expected squared error',T,'--')]:
        ax.plot(c,v,label=label,c=col,ls=ls)
    clean(ax,'Model complexity (schematic)','Error (schematic)');ax.legend(ncol=2);save('bias_variance',fig,'Schematic analytical curves; not estimated from data.')
    fig,axes=sub();ax=axes[0]
    a,b=np.meshgrid(np.linspace(-3,3,100),np.linspace(-2,2,100));z=a*a+5*b*b
    ax.contour(a,b,z,levels=[.1,.5,1,2,4,8,12,18,25],colors=[CMAP_SEQ(v) for v in np.linspace(.3,.95,9)],linewidths=1.3)
    w=np.array([2.6,1.5]);track=[w.copy()]
    for _ in range(15):w=w-.16*np.array([2*w[0],10*w[1]]);track.append(w.copy())
    track=np.array(track);ax.plot(track[:,0],track[:,1],'-o',color=O,markersize=5,label='Gradient updates')
    ax.scatter([0],[0],s=130,marker='*',color=T,label='Minimum',zorder=5)
    clean(ax,'Parameter 1','Parameter 2');ax.legend(frameon=True,framealpha=.95,facecolor='white',edgecolor='none');save('gd_path',fig,'Exact quadratic loss L = w1 squared + 5 w2 squared; learning rate 0.16.')
    X=rng.normal(size=(120,5));X[:,1]=X[:,0]+rng.normal(0,.2,120);target=X@np.array([2,1,-1,.5,0])+rng.normal(0,1,120)
    X=StandardScaler().fit_transform(X);alphas=np.logspace(-3,4,70)
    coefs=np.array([Ridge(alpha=a).fit(X,target).coef_ for a in alphas])
    fig,axes=sub();ax=axes[0]
    for i in range(5):ax.semilogx(alphas,coefs[:,i],label=f'Feature {i+1}',color=COLORS[i])
    ax.axhline(0,color=G,lw=.8);clean(ax,'Ridge penalty alpha (log scale)','Coefficient');ax.legend(ncol=3);save('ridge_path',fig)

def classification():
    boundaries('knn_boundaries',[KNeighborsClassifier(n_neighbors=k) for k in [1,7,31]],['K = 1','K = 7','K = 31'])
    boundaries('svm_kernels',[make_pipeline(StandardScaler(),SVC(kernel='linear',C=1)),make_pipeline(StandardScaler(),SVC(kernel='rbf',C=3,gamma=1))],['Linear kernel','RBF kernel'])
    boundaries('tree_depth',[DecisionTreeClassifier(max_depth=d,random_state=42) for d in [2,5,None]],['Depth 2','Depth 5','Unrestricted depth'])
    boundaries('ensemble_boundaries',[DecisionTreeClassifier(random_state=42),RandomForestClassifier(n_estimators=100,min_samples_leaf=4,random_state=42),GradientBoostingClassifier(n_estimators=70,max_depth=2,random_state=42)],['Single tree','Random forest','Gradient boosting'])
    X,y=make_blobs(n_samples=50,centers=[[-1,-.7],[1,.7]],cluster_std=.62,random_state=42)
    model=SVC(kernel='linear',C=1).fit(X,y)
    fig,axes=sub();ax=axes[0];points(ax,X,y,45)
    xx=np.linspace(-2.5,2.5,150);w=model.coef_[0];b=model.intercept_[0]
    for margin,style in [(0,'-'),(-1,'--'),(1,'--')]:ax.plot(xx,(margin-b-w[0]*xx)/w[1],style,c=G,lw=1.8)
    sv=model.support_vectors_;ax.scatter(sv[:,0],sv[:,1],s=135,facecolors='none',edgecolors=T,lw=1.7,label='Support vectors')
    ax.set_xlim(-2.6,2.6);ax.set_ylim(-2.3,2.3);clean(ax,'Feature 1','Feature 2');ax.legend();save('svm_margin',fig)

def ensembles():
    fig,axes=sub();ax=axes[0];n=np.arange(1,101)
    for rho,col in zip([0,.2,.6],COLORS):
        ax.plot(n,rho+(1-rho)/n,c=col);endlabel(ax,100,rho+(1-rho)/100,f'correlation ρ = {rho}',col,fs=13)
    ax.set_xlim(-2,128);ax.set_xticks(range(0,101,20));ax.spines['bottom'].set_bounds(0,100);clean(ax,'Number of models','Variance of average (unit base variance)');save('ensemble_variance',fig,'Analytical equal-variance, equal-correlation formula.')
    rng=np.random.default_rng(42);samples=np.array([np.bincount(rng.integers(0,8,8),minlength=8) for _ in range(3)])
    x=np.linspace(-2,2,120);y=np.sin(2*x)+rng.normal(0,.18,120);xx=np.linspace(-2,2,300)
    fig,axes=sub(3)
    for ax,n in zip(axes,[1,10,70]):
        m=GradientBoostingRegressor(n_estimators=n,learning_rate=.1,max_depth=2,random_state=42).fit(x[:,None],y)
        ax.scatter(x,y,s=13,c=B,alpha=.5);ax.plot(xx,m.predict(xx[:,None]),color=O);ax.set_title(f'{n} cumulative tree' + ('s' if n > 1 else ''));clean(ax,'Input x','Target y' if ax is axes[0] else None);ax.set_ylim(-1.7,1.7)
    save('boosting_steps',fig)
    X,y=make_classification(n_samples=600,n_features=5,n_informative=3,n_redundant=1,shuffle=False,random_state=42)
    tr,va,yt,yv=train_test_split(X,y,test_size=.35,random_state=42,stratify=y)
    m=RandomForestClassifier(n_estimators=120,min_samples_leaf=3,random_state=42).fit(tr,yt)
    imp=permutation_importance(m,va,yv,n_repeats=15,random_state=42,scoring='balanced_accuracy')
    fig,axes=sub();ax=axes[0];labels=['Signal 1','Signal 2','Signal 3','Redundant','Irrelevant']
    ax.barh(labels,imp.importances_mean,xerr=imp.importances_std,color=[B,B,B,T,O],height=.6,capsize=4)
    ax.axvline(0,color=G,lw=.8);ax.invert_yaxis();clean(ax,'Validation balanced-accuracy drop after shuffling');save('importance',fig)
    xx=rng.uniform(-2,2,(220,1));yy=np.sin(3*xx[:,0])+rng.normal(0,.35,len(xx));tr,va,yt,yv=train_test_split(xx,yy,test_size=.4,random_state=42)
    fig,axes=sub();ax=axes[0]
    for rate,col in zip([.03,.1,.4],COLORS):
        m=GradientBoostingRegressor(n_estimators=220,learning_rate=rate,max_depth=3,random_state=42).fit(tr,yt)
        err=[mean_squared_error(yv,p) for p in m.staged_predict(va)]
        ax.plot(np.arange(1,221),err,c=col);endlabel(ax,220,err[-1],f'learning rate {rate}',col,fs=13)
    ax.set_xlim(-4,272);ax.set_xticks(range(0,201,50));ax.spines['bottom'].set_bounds(0,220);clean(ax,'Number of boosting stages','Validation MSE');save('gb_learning',fig,'Synthetic sin(3x) data with noise, seed 42; 60/40 train/validation split; depth-3 trees.')

def clusters():
    X,_=make_blobs(n_samples=150,centers=[[-2,0],[1,2],[2,-1]],cluster_std=.55,random_state=42)
    ks=range(2,9);fits=[KMeans(n_clusters=k,n_init=10,random_state=42).fit(X) for k in ks]
    fig,axes=sub(2);axes[0].plot(list(ks),[m.inertia_ for m in fits],'-o',color=B);axes[1].plot(list(ks),[silhouette_score(X,m.labels_) for m in fits],'-o',color=T)
    clean(axes[0],'Number of clusters K','Inertia');clean(axes[1],'Number of clusters K','Mean silhouette');save('elbow_silhouette',fig)
    X,y=make_moons(n_samples=230,noise=.065,random_state=42)
    fig,axes=sub(2)
    for ax,m,title in zip(axes,[KMeans(n_clusters=2,n_init=10,random_state=42),DBSCAN(eps=.2,min_samples=6)],['K-means: nearest centroid','DBSCAN: connected density']):
        points(ax,X,m.fit_predict(X));ax.set_title(title);clean(ax,'Feature 1','Feature 2')
    save('clusters_shapes',fig)

def reduction():
    rng=np.random.default_rng(42)
    fig,axes=sub();ax=axes[0];dims=[2,5,10,25,50,100,250];sp=[]
    for d in dims:
        distances=np.linalg.norm(rng.normal(size=(3000,d))-rng.normal(size=(3000,d)),axis=1);sp.append(np.std(distances)/np.mean(distances))
    ax.semilogx(dims,sp,'-o',c=B);ax.set_xticks(dims,[str(d) for d in dims]);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());clean(ax,'Number of independent features (log scale)','Distance standard deviation / mean');save('distance_concentration',fig)
    X=rng.multivariate_normal([0,0],[[3,2],[2,2]],80);m=PCA(1).fit(X);z=m.transform(X);rec=m.inverse_transform(z)
    fig,axes=sub();ax=axes[0];ax.scatter(X[:,0],X[:,1],c=B,s=28,label='Original observations');ax.scatter(rec[:,0],rec[:,1],c=O,s=18,label='PC1 reconstruction')
    for i in range(0,len(X),4):ax.plot([X[i,0],rec[i,0]],[X[i,1],rec[i,1]],'--',c=G,lw=1)
    v=m.components_[0];line=np.array([-8.,8.]);ax.plot(line*v[0]+m.mean_[0],line*v[1]+m.mean_[1],c=T,lw=2,label='First principal direction')
    clean(ax,'Feature 1','Feature 2');ax.legend(loc='upper left',fontsize=11);ax.set_xlim(-5.3,5.3);ax.set_ylim(-4.6,4.3);ax.set_aspect('equal',adjustable='box');save('pca_projection',fig)
    latent=rng.normal(size=(300,3));X=latent@rng.normal(size=(3,8))+rng.normal(0,.3,(300,8));m=PCA().fit(StandardScaler().fit_transform(X));ratio=m.explained_variance_ratio_
    fig,axes=sub();ax=axes[0];k=np.arange(1,9);ax.bar(k,ratio,.7,color=B,label='Individual component');ax.plot(k,np.cumsum(ratio),'-o',color=O,label='Cumulative');ax.axhline(.9,ls=':',c=G,label='90% reference');clean(ax,'Principal component','Explained variance ratio');ax.set_ylim(0,1.05);ax.legend();save('pca_variance',fig)
    digits=load_digits();X=digits.data/16;idx=3
    fig,axes=sub(4,height=3.3)
    for ax,k in zip(axes,[None,4,12,32]):
        if k is None:im=X[idx];title='Original'
        else:m=PCA(k,random_state=42).fit(X);im=m.inverse_transform(m.transform(X[[idx]]))[0];title=f'{k} components'
        ax.imshow(im.reshape(8,8),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title);ax.set_axis_off()
    save('pca_reconstruction',fig,'sklearn bundled handwritten digits; PCA fitted to the local dataset; illustrative reconstruction.')
    ix=np.random.default_rng(42).choice(len(X),450,replace=False);Z=StandardScaler().fit_transform(X[ix]);y=digits.target[ix]
    fig,axes=sub(3)
    for ax,perplex in zip(axes,[5,30,80]):
        emb=TSNE(perplexity=perplex,max_iter=750,learning_rate='auto',init='pca',random_state=42).fit_transform(Z)
        ax.scatter(emb[:,0],emb[:,1],c=y,cmap='tab10',s=13,alpha=.8);ax.set_title(f'Perplexity {perplex}');ax.set_xlabel('Embedding coordinate 1');ax.set_ylabel('Coordinate 2' if ax is axes[0] else '');ax.set_xticks([]);ax.set_yticks([])
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',linestyle='',color=plt.get_cmap('tab10')(i),label=str(i)) for i in range(10)]
    fig.legend(handles=handles,title='Observed digit label (colour only)',loc='outside lower center',ncol=10,fontsize=11,title_fontsize=12)
    save('tsne_perplexity',fig,'450 fixed sampled sklearn digit images, standardized; TSNE seeds 42; labels colour only, never used in fitting.')

def anomalies():
    rng=np.random.default_rng(42);X=np.r_[rng.normal([-1,0],[.4,.65],(160,2)),rng.normal([1.3,.5],[.22,.3],(80,2))]
    normal=rng.normal(.28,.11,1500);event=rng.normal(.57,.15,180)
    fig,axes=sub();ax=axes[0];bins=np.linspace(0,1,28)
    ax.hist(normal,bins=bins,density=True,alpha=.6,color=B,label='Ordinary cases');ax.hist(event,bins=bins,density=True,alpha=.6,color=O,label='Events');ax.axvline(.46,color=T,ls='--',label='Chosen threshold');clean(ax,'Illustrative anomaly score','Density');ax.legend();save('anomaly_scores',fig,'Simulated score distributions, not a fitted detector or calibrated probabilities.')
    X,y=make_classification(n_samples=1600,n_features=8,n_informative=5,weights=[.94,.06],class_sep=1,random_state=42)
    tr,va,yt,yv=train_test_split(X,y,test_size=.4,random_state=42,stratify=y);m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(tr,yt);p=m.predict_proba(va)[:,1]
    precision,recall,_=precision_recall_curve(yv,p);ap=average_precision_score(yv,p)
    fig,axes=sub();ax=axes[0];ax.step(recall,precision,where='post',c=B,label=f'Logistic baseline, AP = {ap:.3f}');ax.axhline(yv.mean(),c=O,ls='--',label=f'Positive prevalence = {yv.mean():.1%}');ax.set_xlim(0,1);ax.set_ylim(0,1.03);clean(ax,'Recall','Precision');ax.legend();save('pr_curve',fig)
    thresholds=np.linspace(0,1,101);cost=[]
    for t in thresholds:
        pred=p>=t;fp=((pred==1)&(yv==0)).sum();fn=((pred==0)&(yv==1)).sum();cost.append(fp+12*fn)
    fig,axes=sub();ax=axes[0];ax.plot(thresholds,cost,c=B,label='FP cost 1, FN cost 12');best=int(np.argmin(cost));ax.scatter(thresholds[best],cost[best],c=O,s=80,zorder=4,label=f'Validation minimum at {thresholds[best]:.2f}');clean(ax,'Probability threshold','Illustrative total cost');ax.legend();save('threshold_cost',fig,'Fixed synthetic validation predictions; illustrative false-positive cost 1 and false-negative cost 12.')
    fig,axes=sub();ax=axes[0];major=rng.normal([0,0],[.8,.6],(70,2));minor=np.array([[1.8,1.2],[2.3,1.8],[1.4,1.9]])
    ax.scatter(major[:,0],major[:,1],c=B,s=28,label='Majority');ax.scatter(minor[:,0],minor[:,1],c=O,marker='^',s=70,label='Minority')
    a,b=minor[:2];interp=a+np.linspace(.2,.8,4)[:,None]*(b-a);ax.plot([a[0],b[0]],[a[1],b[1]],c=G,ls='--');ax.scatter(interp[:,0],interp[:,1],marker='s',c=T,s=55,label='Synthetic interpolations')
    clean(ax,'Scaled feature 1','Scaled feature 2');ax.legend();save('smote',fig,'Exact interpolation along one selected minority-neighbor segment; not a full SMOTE implementation.')

def neural():
    x=np.linspace(-6,6,400)
    for name,values,labels in [('activation',[np.maximum(0,x),expit(x),np.tanh(x)],['ReLU','Sigmoid','tanh']),('activation_derivatives',[(x>0).astype(float),expit(x)*(1-expit(x)),1-np.tanh(x)**2],['ReLU derivative','Sigmoid derivative','tanh derivative'])]:
        fig,axes=sub(3)
        for ax,val,title,col in zip(axes,values,labels,COLORS):
            if title=='ReLU derivative':  # a step: draw the two pieces without a vertical join at z = 0
                ax.plot(x[x<0],val[x<0],c=col);ax.plot(x[x>0],val[x>0],c=col)
                ax.scatter([0],[0],s=40,facecolors=col,edgecolors=col,zorder=4,linewidths=1.8)  # frameworks define the value at 0 as 0
                ax.scatter([0],[1],s=40,facecolors='white',edgecolors=col,zorder=4,linewidths=1.8)
            else:
                ax.plot(x,val,c=col)
            ax.axhline(0,c=G,lw=.5);ax.axvline(0,c=G,lw=.5);ax.set_title(title);clean(ax,'Input z','Response' if ax is axes[0] else None)
        save(name,fig,'Exact analytical functions; ReLU derivative at zero shown using a software-style zero convention.')
    fig,axes=sub(3);epoch=np.arange(1,61)
    patterns=[(.8+.5*np.exp(-epoch/15),.95+.5*np.exp(-epoch/15)),(.2+.9*np.exp(-epoch/12),.45+.6*np.exp(-epoch/12)+.00035*(epoch-15)**2),(.25+.9*np.exp(-epoch/14)+.12*np.sin(epoch*1.7),.42+.8*np.exp(-epoch/14)+.2*np.sin(epoch*1.3))]
    for ax,(train,val),title in zip(axes,patterns,['Persistent high loss','Widening validation gap','Unstable updates']):
        ax.plot(epoch,train,c=B,label='Training');ax.plot(epoch,val,c=T,ls='--',label='Validation');ax.set_title(title,fontsize=15);clean(ax,'Epoch','Loss (schematic)' if ax is axes[0] else None);ax.set_ylim(0,1.8)
    axes[0].legend();save('training_curves',fig,'Schematic learning curves, not measured training runs; shapes admit multiple diagnoses.')

def capstone():
    base=ROOT/'neural_networks_course'/'Day_3_Capstone'/'exercises'
    d=pd.read_csv(base/'02_log_curves'/'Log curve - Train.csv');fig,axes=plt.subplots(2,1,figsize=(11.8,4.2),layout='constrained',sharex=True)
    axes[0].plot(np.arange(len(d)),d.Depth,c=B,lw=1);axes[0].set_ylabel('Depth (as supplied)');axes[1].plot(np.arange(len(d)),d.PE,c=O,lw=.8);axes[1].set_ylabel('PE (as supplied)');axes[1].set_xlabel('Source row position');save('log_profile',fig,'Local Log curve - Train.csv, original row order. Units and well identifiers are not supplied in the CSV.')
    d=pd.read_csv(base/'03_sensor_signals'/'Sensor_train.csv');fig,axes=sub(2)
    for ax,label in zip(axes,[-1,1]):
        subset=d[d.target==label].iloc[:5]
        for _,row in subset.iterrows():ax.plot(range(1,11),row[[f'col{i}' for i in range(1,11)]].values,alpha=.7,lw=1.6)
        ax.set_title(f'First five rows with target {label}');clean(ax,'Feature index (not verified time)','Value (as supplied)');ax.set_xticks(range(1,11))
    save('sensor_profile',fig,'Local Sensor_train.csv; first five records of each target class; physical units and ordering semantics are not documented.')

def sequence():
    fig,axes=sub();ax=axes[0];t=np.arange(-23,7);signal=np.sin(t/4)+.04*t
    ax.plot(t[t<=0],signal[t<=0],'-o',c=B,markersize=4,label='Available history');ax.plot(t[t>=0],signal[t>=0],'--',c=G,label='Future (unavailable input)');ax.scatter([6],[signal[-1]],s=90,c=O,zorder=4,label='Target at horizon h = 6')
    ax.axvline(0,c=T,ls=':');ax.axvspan(-23,0,color=B,alpha=.06);ax.set_xticks([-23,-18,-12,-6,0,6]);clean(ax,'Steps relative to forecast origin','Illustrative signal');ax.legend(ncol=2);save('sequence_window',fig,'Schematic forecast timeline, lookback 24 observations, horizon 6 steps.')
    fig,axes=sub();ax=axes[0];steps=np.arange(0,101)
    for f,col in zip([.5,.9,.99],COLORS):ax.plot(steps,f**steps,c=col,label=f'Fixed forget value {f}')
    clean(ax,'Steps after a memory contribution','Fraction of contribution retained');ax.legend();save('recurrent_memory',fig,'Analytical powers of fixed forget factors, ignoring candidate additions; schematic LSTM memory path.')
    scores=np.array([[2,1,0,.5],[.2,1.5,.7,0],[.1,.4,1.8,.5],[0,.3,.8,1.4]])
    mask=np.triu(np.ones((4,4),bool),1);causal=scores.copy();causal[mask]=-np.inf
    fig,axes=sub(2)
    for ax,weights,title in zip(axes,[softmax(scores,axis=1),softmax(causal,axis=1)],['Unmasked self-attention','Causal self-attention']):
        ax.imshow(weights,cmap='course_seq',vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{weights[i,j]:.2f}',ha='center',va='center',fontsize=16,color='white' if weights[i,j]>.6 else '#122B46')
        ax.set_xticks(range(4),['1','2','3','4']);ax.set_yticks(range(4),['1','2','3','4']);ax.set_xlabel('Key position');ax.set_ylabel('Query position');ax.set_title(title)
    save('attention_weights',fig,'Exact row-wise softmax over an illustrative fixed score matrix, with additive negative-infinity future mask.')
    rng=np.random.default_rng(42);series=np.zeros(1000)
    for i in range(1,len(series)):series[i]=.85*series[i-1]+rng.normal(0,.4)
    model=LinearRegression().fit(series[:598,None],series[1:599]);origins=np.arange(650,970);hs=np.array([1,3,6,12,24]);base=[];linear=[]
    for h in hs:
        target=series[origins+h];p=series[origins].copy()
        for _ in range(h):p=model.predict(p[:,None])
        base.append(np.sqrt(np.mean((target-series[origins])**2)));linear.append(np.sqrt(np.mean((target-p)**2)))
    fig,axes=sub();ax=axes[0];ax.plot(hs,base,'-o',c=O,label='Persistence');ax.plot(hs,linear,'-s',c=B,label='Fitted recursive linear forecast');clean(ax,'Forecast horizon (steps)','RMSE on fixed future origins');ax.legend();save('forecast_horizon',fig,'Simulated AR(1) process, coefficient 0.85, innovation SD 0.4; linear model fitted before index 600, evaluation origins 650-969.')




# ---- shared diagrams ----
from pathlib import Path  # noqa: E402
ROLE = {0: ('Train', BLUE), 1: ('Val', TEAL), 2: ('Test', ORANGE), 3: ('', '#C9D3DD')}


def cell_row(ax, x0, y, roles, w=.78, h=.62, gap=.08, fs=12.5, texts=True):
    xs = []
    for j, r in enumerate(roles):
        name, col = ROLE[r]
        x = x0 + j * (w + gap)
        if r == 3:
            box(ax, x, y, w, h, '', '#9AA9B8', fill='white', lw=1.1, ls=(0, (2, 2)), radius=.07)
        else:
            box(ax, x, y, w, h, name if texts else '', col, fill='tint', fs=fs, weight='semibold',
                text_color=col, lw=1.4, radius=.07)
        xs.append(x)
    return xs


def group_split():
    f, ax = canvas(11.8, 4.0)
    w, gap, wgap, x0, h = .47, .045, .2, 3.1, .56
    xs = [x0 + k * (4 * w + 3 * gap + wgap) + j * (w + gap) for k in range(4) for j in range(4)]
    rows = [([0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0], 'Random rows', 'tests new rows from the same wells'),
            ([0] * 8 + [1] * 4 + [2] * 4, 'Grouped by well', 'tests wells never seen in training'),
            ([0, 0, 1, 2] * 4, 'Ordered in time', 'tests the later period of every well')]
    ys = [2.62, 1.62, .62]
    for (roles, name, desc), y in zip(rows, ys):
        label(ax, .15, y + .42, name, fs=14, weight='semibold', ha='left')
        label(ax, .15, y + .14, desc, fs=11.5, color=MUTED, ha='left')
        for x, r in zip(xs, roles):
            nm, col = ROLE[r]
            box(ax, x, y, w, h, nm, col, fill='tint', fs=11, weight='semibold', text_color=col, lw=1.3, radius=.06)
    for k, well in enumerate('ABCD'):
        a, b = xs[4 * k], xs[4 * k + 3] + w
        bracket(ax, a + .02, b - .02, 3.37, f'Well {well}', color=MUTED, fs=11.5, below=False)
        arrow(ax, (a, .36), (b, .36), color=MUTED, lw=1.2, head=9)
    label(ax, (xs[0] + xs[-1] + w) / 2, .14, 'observations within each well, in depth or time order', fs=11, color=MUTED)
    save('group_split', f, 'Schematic partition assignments for four wells with four ordered observations each.', kind='schematic')


def rolling_origin():
    f, ax = canvas(11.8, 3.2)
    x0, w, gap = 1.7, .72, .08
    folds = [[0] * 4 + [1] * 2 + [3] * 6, [0] * 6 + [1] * 2 + [3] * 4, [0] * 8 + [1] * 2 + [3] * 2]
    for k, roles in enumerate(folds):
        y = 2.2 - k * .82
        label(ax, .2, y + .28, f'Origin {k + 1}', fs=13.5, weight='semibold', ha='left')
        cell_row(ax, x0, y, roles, w=w, h=.56, gap=gap, fs=12)
        o = x0 + roles.index(1) * (w + gap) - gap / 2
        ax.plot([o, o], [y - .08, y + .64], color=INK, lw=1.6, zorder=5)
    arrow(ax, (x0, .28), (x0 + 12 * (w + gap) - gap, .28), color=MUTED, lw=1.3, head=11)
    label(ax, x0 + 6 * (w + gap), .1, 'chronological blocks; the vertical line marks each forecast origin', fs=11, color=MUTED)
    save('rolling_origin', f, 'Schematic expanding-window evaluation. Gray blocks are not used in that fold.', kind='schematic')


def confusion():
    f, ax = canvas(6.4, 5.0)
    cells = {(0, 0): (85, 'True negative', TEAL), (0, 1): (5, 'False positive', ORANGE),
             (1, 0): (3, 'False negative', ORANGE), (1, 1): (7, 'True positive', TEAL)}
    x0, y0, s = 1.75, .55, 1.72
    for (i, j), (n, name, col) in cells.items():
        x, y = x0 + j * (s + .1), y0 + (1 - i) * (s + .1)
        box(ax, x, y, s, s, '', col, fill='tint', lw=1.6, radius=.1)
        label(ax, x + s / 2, y + s * .6, str(n), fs=34, weight='semibold', color=col)
        label(ax, x + s / 2, y + s * .27, name, fs=13, color=INK)
    for j, t in enumerate(['No event', 'Event']):
        label(ax, x0 + j * (s + .1) + s / 2, y0 + 2 * s + .35, t, fs=13, weight='medium')
    for i, t in enumerate(['No event', 'Event']):
        label(ax, x0 - .18, y0 + (1 - i) * (s + .1) + s / 2, t, fs=13, weight='medium', ha='right')
    label(ax, x0 + s + .05, y0 + 2 * s + .72, 'Predicted class', fs=14, weight='semibold')
    label(ax, .28, y0 + s + .05, 'Actual class', fs=14, weight='semibold', rotation=90)
    save('confusion', f, 'Illustrative counts TN=85, FP=5, FN=3, TP=7; not model performance.', kind='schematic')


def bagging_samples():
    rng = np.random.default_rng(42)
    samples = np.array([np.bincount(rng.integers(0, 8, 8), minlength=8) for _ in range(3)])
    f, ax = canvas(11.8, 3.3)
    x0, w, gap = 2.3, 1.0, .1
    for j, ch in enumerate('ABCDEFGH'):
        label(ax, x0 + j * (w + gap) + w / 2, 2.95, f'Row {ch}', fs=12.5, color=MUTED)
    for i, row in enumerate(samples):
        y = 2.2 - i * .8
        label(ax, .2, y + .3, f'Bootstrap {i + 1}', fs=13.5, weight='semibold', ha='left')
        for j, n in enumerate(row):
            x = x0 + j * (w + gap)
            if n == 0:
                box(ax, x, y, w, .6, 'out of bag', '#8C9AA8', fill='white', fs=11, ls=(0, (2, 2)), lw=1.1,
                    text_color=MUTED, weight='normal')
            else:
                box(ax, x, y, w, .6, f'×{n}', BLUE, fill='solid' if n > 1 else 'tint', fs=14, weight='semibold')
    label(ax, x0 + 4 * (w + gap), .1, 'Each bootstrap draws 8 rows with replacement: some rows repeat, others are left out (out of bag).',
          fs=11.5, color=MUTED)
    save('bagging_samples', f, 'Deterministic synthetic teaching example; seed 42.', kind='computed')


def esp_profile():
    base = ROOT / 'neural_networks_course' / 'Day_3_Capstone' / 'exercises' / '01_esp_gas_lock'
    d = pd.read_csv(base / 'esp_train.csv')
    counts = d.groupby('ENTITY')['GAS_LOCK'].value_counts().unstack(fill_value=0)
    f, axes = sub(2, height=4.2)
    ax = axes[0]
    xs = np.arange(len(counts))
    ax.bar(xs, counts[0], color=BLUE, width=.65, label='GAS_LOCK = 0')
    ax.bar(xs, counts[1], bottom=counts[0], color=ORANGE, width=.65, label='GAS_LOCK = 1')
    ax.set_xticks(xs, counts.index, rotation=25)
    clean(ax, 'Entity', 'Rows')
    ax.set_title('Training labels by entity')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.28), ncol=2)
    missing = d[['PIP', 'PDP', 'Q', 'THP', 'Mamps', 'PIT']].isna().sum()
    axes[1].bar(missing.index, missing.values, color=TEAL, width=.6)
    for k, v in enumerate(missing.values):
        axes[1].text(k, v + .8, str(int(v)), ha='center', fontsize=12, color=INK)
    clean(axes[1], 'Input column', 'Missing cells')
    axes[1].set_title('Training input missingness')
    save('esp_profile', f, 'Local esp_train.csv; full training file, 3,243 rows, 7 entities; counts are observations, not events.', kind='data')



for _fn in [regression, classification, ensembles, clusters, reduction, anomalies, neural,
            capstone, sequence, group_split, rolling_origin, confusion, bagging_samples,
            esp_profile]:
    register(KEY)(_fn)
