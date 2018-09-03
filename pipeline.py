import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.decomposition import SparsePCA
from sklearn.preporcessing import StandardScaler


class LoadData(object):
    
    def __init__(self, directory, train=True):
        self.data = pd.read_csv(directory)
        self.feats = None
        if train is True:
            self.target = self.data['class']
            self.target.map(1:-1, 2:1)
            self.target.reshape(-1, 1)
    
    def filterf(self, listi):
        for c in listi:
            auxi.append(self.feats[[a for a in self.feats if c in a]])
        self.feats = pd.concat(auxis, axis=1)    
        return self.feats


class Ranking(object):
    
    def __init__(self):
        self.ranking = None
        self.selection = None
        self.preselection = None
    
    def compute_ranking(self, X, y):
        pvals = -pd.Series(f_classif(X, y)[1], index=X.columns)
        mi = pd.Series(mutual_info_classif(X, y), index=X.columns)
        feature_power = pd.concat((mi, pvals), axis=1)
        feature_power['ranking mi'] = np.nan
        feature_power['ranking pvals'] = np.nan
        feature_power = feature_power.rename(columns={0: 'MI', 1: 'pvals'})
        feature_power.loc[feature_power.sort_values('MI').index, 'ranking mi'] = range(len(feature_power))
        feature_power.loc[feature_power.sort_values('pvals').index, 'ranking pvals'] = range(len(feature_power))
        feature_power['mixed_ranking'] = feature_power.iloc[:, -2:].mean(axis=1)
        self.ranking = feature_power.copy()
        
    def get_limited_selection(self, limits, limit_corr=0.5):
        sorted_rank = self.ranking.sort_values('mixed_ranking', ascending=False)
        bol = pd.Series(True, index=sorted_rank.index)
        for k in limits:
            bol = bol&~(sorted_rank>limits[k])
        self.preselection = sorted_rank.loc[bol].index[:-1]
        self.selection = [self.preselection[0]]
        for i, s in enumerate(self.preselection[1:]):
            if (feats[self.selection+[s]].corr().iloc[:-1, -1].abs()<limit_corr).sum() == len(self.selection):
                self.selection.append(s)
        return self.selection
    

class DimensionalityReducer(object):
    
    def __init__(self):
        self.sc = None
        self.pca = None
    
    def fitPCA(self, X, nfeats=3):
        self.sc = StandardScaler()
        self.pca = SparsePCA(n_components=nfeats)
        self.pca.fit(self.sc.fit_transform(X))
        
    def transformPCA(self, X):
        components = self.pca.transform(self.sc.transform(X))
        return components
