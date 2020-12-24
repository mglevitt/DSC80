import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    '''
    simple_pipeline takes in a dataframe like data and returns a tuple 
    consisting of the pipeline and the predictions your model makes 
    on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    '''
    
    steps = [('log_c',FunctionTransformer(np.log)),('lr',LinearRegression())]
    pl = Pipeline(steps)
    pl.fit(data[['c2']],data['y'])
    preds = pl.predict(data[['c2']])
    return tuple([pl,preds])

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def multi_type_pipeline(data):
    '''
    multi_type_pipeline that takes in a dataframe like data and 
    returns a tuple consisting of the pipeline and the predictions 
    your model makes on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    '''
    steps = [('log_c',FunctionTransformer(np.log),['c2']),('one_hot',OneHotEncoder(),['group'])]
    c_trans = ('col_trans',ColumnTransformer(steps))
    pl = Pipeline([c_trans,('lr',LinearRegression())])
    pl.fit(data[['c1','c2','group']],data['y'])
    preds = pl.predict(data[['c1','c2','group']])
    return tuple([pl,preds])

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):
      
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        df = pd.DataFrame(X)
        means = df.groupby(X.iloc[:,0]).mean().to_dict()
        stds = df.groupby(X.iloc[:,0]).std().to_dict()
        # A dictionary of means/standard-deviations for each column, for each group.
        
        self.grps_ = (means, stds)
        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        

        # Define a helper function here?
        def normalize(row):
            out_row = []
            for i in range(len(row) - 1):
                if self.grps_[1][df.columns[i+1]][row[0]] == 0:
                    out_row.append(0)
                else:
                    out_row.append((row[i + 1] - self.grps_[0][df.columns[i+1]][row[0]]) / self.grps_[1][df.columns[i+1]][row[0]])
            
            return pd.Series(out_row)
        # X may not be a dataframe (e.g. np.array)

        df = pd.DataFrame(X)
        out_df = df.apply(normalize, axis = 1)
        for i in range(len(df.columns) - 1):
            out_df = out_df.rename(columns = {i:df.columns[i + 1]})
        return out_df


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def eval_toy_model():
    """
    hardcoded answers to question 4

    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    """

    return (7.598222156931544,5.536928354075744,5.363818490270249)


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """
    X_train, X_test, y_train, y_test = train_test_split(galton.drop(columns = 'childHeight'),galton['childHeight'])
    out = pd.DataFrame(columns = ['train_err','test_err'])
    for i in range(20):
        
        tree = DecisionTreeRegressor(max_depth = i + 1)
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_r = ((y_train - train_pred) ** 2).sum() / (len(y_train) - 1)
        test_r = ((y_test - test_pred) ** 2).sum() / (len(y_test) - 1)
        temp = pd.DataFrame({'train_err':[train_r], 'test_err':[test_r]}).set_index(pd.Series([i + 1]))
        out = pd.concat([out,temp])
    return out


def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """
    X_train, X_test, y_train, y_test = train_test_split(galton.drop(columns = 'childHeight'),galton['childHeight'])
    out = pd.DataFrame(columns = ['train_err','test_err'])
    for i in range(20):
        
        tree = KNeighborsRegressor(n_neighbors = i + 1)
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_r = ((y_train - train_pred) ** 2).sum() / (len(y_train) - 1)
        test_r = ((y_test - test_pred) ** 2).sum() / (len(y_test) - 1)
        temp = pd.DataFrame({'train_err':[train_r], 'test_err':[test_r]}).set_index(pd.Series([i + 1]))
        out = pd.concat([out,temp])
    return out

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """

    data = titanic.copy()
    def get_title(name):
        title = name.split(' ')[0]
        if title not in ['Mr.','Mrs.','Miss.','Master.']:
            title = 'other'
        return title
    def get_titles(names):
        return pd.DataFrame(names.iloc[:,0].apply(get_title))
    title_transformer = Pipeline([('get_title',FunctionTransformer(func = get_titles, validate = False)),('one_hot',OneHotEncoder())])
    step = ('std', StdScalerByGroup(), ['Pclass','Age'])
    steps = [step,('title',title_transformer,['Name']), ('one_hot',OneHotEncoder(),['Sex'])]
    
    c_trans = ('col_trans',ColumnTransformer(steps, remainder = 'passthrough'))
    pl = Pipeline([c_trans,('lr',LogisticRegression())])
    X = data.drop(columns = 'Survived')
    y = data['Survived']
    pl.fit(X,y)
    return pl

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def json_reader(file, iterations):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> isinstance(reviews, list)
    True
    >>> isinstance(labels, list)
    True
    >>> len(labels) == len(reviews)
    True
    """

    return ...


def create_classifier_multi(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """
    
    return ...


def to_binary(labels):
    """
    :Example
    >>> lst = [1, 2, 3, 4, 5]
    >>> to_binary(lst)
    >>> print(lst)
    [0, 0, 0, 1, 1]
    """
    
    return ...


def create_classifier_binary(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> to_binary(labels)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['simple_pipeline'],
    'q02': ['multi_type_pipeline'],
    'q03': ['StdScalerByGroup'],
    'q04': ['eval_toy_model'],
    'q05': ['tree_reg_perf', 'knn_reg_perf'],
    'q06': ['titanic_model']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True