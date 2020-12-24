import os
import pandas as pd
import numpy as np
import math
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [.155,'NR']


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [3.65394511e-315, 'R','D']


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    def permutation(df, col):
        ks_lst = []
        df = df.assign(null=col.isnull())
        for _ in range(100):
            sample_cols = df['father'].sample(replace=False, frac=1).reset_index(drop=True)
            sample = df.assign(**{'father':sample_cols, 'null':col.isnull()})
            null_fathers = sample.groupby('null')['father']
            ks = ks_2samp(null_fathers.get_group(True), null_fathers.get_group(False)).statistic
            ks_lst.append(ks)
        grouped = df.groupby('null')['father']
        obs_ks = ks_2samp(grouped.get_group(True), grouped.get_group(False)).statistic
        return np.count_nonzero(np.array(ks_lst) > obs_ks) / 100
    p_vals = []
    col_names = heights.columns.drop(['child','father'])
    for col in col_names:
        p_vals.append(permutation(heights,heights[col]))
    return pd.Series(data = p_vals, index = col_names)


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [1,2,5]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    df = new_heights.copy()[['father','child']]
    df['father']=pd.qcut(df['father'],4)
    means =  df.groupby('father').mean()
    def helper(row):
        father = row['father']
        child = row['child']
        if math.isnan(child):
            for i in range(means.shape[0]):
                if father in means.index[i]:
                    child = means['child'][i]
        row['father'] = father
        row['child'] = child
        return row
    out_df = new_heights.copy()
    return out_df.transform(helper, axis = 1)['child']

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """
    freq, bins = np.histogram(child.dropna(), bins = 10)
    probs = freq/freq.sum()
    bins_width = np.diff(bins)[0]
    rand_probs = np.random.choice(bins[:-1],p = probs,size = N)
    outs = np.array([])
    for prob in rand_probs:
        outs = np.append(outs,np.random.uniform(prob,prob+bins_width))
    return outs


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    return child.fillna(pd.Series(quantitative_distribution(child, len(child))))


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    list1 = [1,2,1,1]
    list2 = ['qq.com','soundcloud.com','fc2.com', '*facebook.com','*linkedin.com','*soso.com']
    return list1, list2




# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
