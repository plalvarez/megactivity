import pandas as pd
import GPy
from pipeline import *
from tqdm import tqdm
from sklearn.metric import roc_auc_score


class CompleteModel(object):
    
    def __init__(self, trdir, tedir):
        self.train = LoadData(trdir)
        self.test = LoadData(tedir, train=False)
        self.ranker = None
        self.selected = None
        self.dimreducer = None
        self.kernel = None
        
    def generate_selection(self, *args, **kwargs):
        self.ranker = Ranking()
        self.ranker.compute_ranking(self.train.feats, self.train.target)
        selection = self.ranker.get_limited_selection(*args, **kwargs)
        return selection
        
    def train(self, limits, limit_corr=0.5, nfeats=3, n_opts=20):
        # Creating the training set:
        self.selected = self.generate_selection(limits, limit_corr=0.5)
        self.dimreducer = DimensionalityReducer()
        self.dimreducer.fitPCA(self.train.feats[self.selected], nfeats=nfeats)
        X = self.dimreducer.transformPCA(self.train.feats[self.selected])
        
        # Preparing the model
        self.kernel = GPy.kern.RBF(nfeats, variance=1, ARD=True)
        self.model = GPy.models.GPClassification(X, self.train.target, kernel=k)
        
        # Optimize the model:
        for i in range(n_opts):
            self.model.optimize('bfgs', max_iters=100)
            
    def predict(self):
        X = self.dimreducer.transformPCA(self.test.feats[self.selected])
        return self.model.predict(X)[0]
        
    def makeCVSimulatuions(n_sims=100, n_test=10):
        pass
        
        
        
        
        