from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
strat_cat = "income_cat"

class preprocessing_split():
    def __init__(self, df =None, bin=[0,1.5,3,4.5,6,np.inf], label = [1,2,3,4,5] ):
        self.df = df 
        self.bin = bin
        self.label = label
    def split(self):
        self.df["income_cat"] = pd.cut(self.df["median_income"],
                                    bins= self.bin,
                                       labels=  self.label )
        self.split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for train_index, test_index in self.split.split(self.df, self.df["income_cat"]):
            strat_train_set = self.df.loc[train_index]
            strat_test_set = self.df.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis= 1, inplace = True)
        return strat_train_set, strat_test_set
        