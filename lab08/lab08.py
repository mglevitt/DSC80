import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    # take log and square root of the dataset
    # look at the fit of the regression line (and R^2)

    return 1

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    
    out_df = pd.DataFrame()
    def helper(x):
        return dic[x]
    for col in df.columns:
        typ = type(df[col][0])
        if typ != np.float64 and typ != np.int64:
            i = 0
            dic = {}
            for val in df[col].unique():
                dic[val] = i
                i+=1
            out_df['ordinal_' + col] = df[col].apply(helper)
    return out_df

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """
    
    out_df = pd.DataFrame()
    def helper(col):
        temp_df = pd.DataFrame()
        for val in col.unique():
            temp_df['one_hot_' + col.name + '_' + val] = (col == val).astype(int)
        return temp_df
    for col in df.columns:
        typ = type(df[col][0])
        if typ != np.float64 and typ != np.int64:
            temp = helper(df[col])
            out_df = pd.concat([out_df, temp], axis = 1)       
    return out_df


def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    dataframe of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """
    out_df = pd.DataFrame()
    def helper(x):
        return float(temp.loc[x])
    for col in df.columns:
        typ = type(df[col][0])
        if typ != np.float64 and typ != np.int64:
            temp = df.groupby(col).size()
            temp = temp/temp.sum()
            out_df['proportion_' + col] = df[col].apply(helper)
    return out_df.astype(float)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
        
    cols = []
    out_df = pd.DataFrame()
    for col in df.columns:
        typ = type(df[col][0])
        if typ == np.float64:
            cols.append(col)
    for i in range(len(cols)):
        for j in range(len(cols) - i - 1):
            out_df[cols[i] + ' * ' + cols[i + j + 1]] = df[cols[i]] * df[cols[i + j + 1]]
    return out_df

# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def comparing_performance():
    """
    Hard coded answers to comparing_performance.

    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table

    return [.8493305264354858,1548.533193061311,'x','carat * x','clarity',0.041]

# ---------------------------------------------------------------------
# Question # 6, 7, 8
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds 
        and returns a binarized carat column (an np.ndarray).

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """

        temp = Binarizer(threshold=1)
        return temp.transform(data[['carat']]) 
    
    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """

        transformer =  QuantileTransformer()
        transformer.fit(self.data[['carat']])
        return transformer.transform(data[['carat']])
    
    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """
        df = data[['x','y','z']]
        def helper(row):
            x = row[:,0]
            y = row[:,1]
            z = row[:,2]
            return z * 200 / (x + y)
        transformer = sklearn.preprocessing.FunctionTransformer(func = helper,validate = True)
        return transformer.transform(df)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['best_transformation'],
    'q02': ['create_ordinal'],
    'q03': ['create_one_hot', 'create_proportions'],
    'q04': ['create_quadratics'],
    'q05': ['comparing_performance'],
    'q06,7,8': ['TransformDiamonds']
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
