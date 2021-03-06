import pandas as pd
import GPy
from pipeline import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
tqdm.monitor_interval=0

class CompleteModel(object):
    
    def __init__(self, listi, trdir, tedir=None):
        self.train = LoadData(trdir)
        self.train.filterf(listi)
        if tedir is not None:
            self.test = LoadData(tedir, train=False)
            self.test.filterf(listi)
        else:
            self.test = LoadData(trdir, train=False)
            self.test.filterf(['gaga'])
        self.ranker = None
        self.selected = None
        self.dimreducer = None
        self.kernel = None
        
    def generate_selection(self, *args, **kwargs):
        self.ranker = Ranking()
        self.ranker.compute_ranking(self.train.feats, self.train.target)
        selection = self.ranker.get_limited_selection(*args, **kwargs)
        return selection
        
    def train_model(self, limits, limit_corr=0.5, nfeats=3, n_opts=20):
        # Creating the training set:
        self.selected = self.generate_selection(limits, limit_corr=0.5)
        self.dimreducer = DimensionalityReducer()
        self.dimreducer.fitPCA(self.train.feats[self.selected], nfeats=nfeats)
        X = self.dimreducer.transformPCA(self.train.feats[self.selected])
        
        # Preparing the model:
        self.kernel = GPy.kern.RBF(nfeats, variance=1, ARD=True)
        self.model = GPy.models.GPClassification(X, self.train.target.values.reshape(-1, 1), 
                                                 kernel=self.kernel)
        
        # Optimize the model:
        for i in range(n_opts):
            self.model.optimize('bfgs', max_iters=100)
            
    def predict(self):
        X = self.dimreducer.transformPCA(self.test.feats[self.selected])
        return self.model.predict(X)
        
    def makeCVSimulations(self, *args, n_sims=100, frac_test=0.25, **kwargs):
        train = self.train.feats.copy()
        test = self.test.feats.copy()
        target = self.train.target.copy()
        scores = pd.Series()
        rocs = pd.DataFrame()
        
        for i in tqdm(range(n_sims)):
            del self.train.feats
            del self.test.feats
            del self.train.target
            sample = train.sample(frac=1-frac_test, replace=False)
            self.test.feats = train.loc[~train.index.isin(sample.index)]
            self.train.target = target.loc[sample.index]
            self.train.feats = sample
            print(self.train.feats.shape[0], self.test.feats.shape[0])
            self.train_model(*args, **kwargs)
            preds = self.predict()[0]
            scores.loc[i] = roc_auc_score(target.loc[self.test.feats.index], preds)
            rocs[i] = roc_curve(target.loc[self.test.feats.index], preds)
        self.trf = self.train.feats
        self.tef = self.test.feats
        self.train.feats = train
        self.test.feats = test
        self.train.target = target
        return scores, rocs
        
        
        