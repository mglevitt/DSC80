
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3,6]

def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2, 5]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2, 4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 2


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    >>> cleaned.Reviews.dtype == int
    True
    '''
    out = df.copy()
    out['Reviews'] = df['Reviews'].astype(int)
    def clean_size(x):
        return x.strip('M').strip('k')
    out['Size'] = df['Size'].apply(clean_size)
    def clean_installs(x):
        return int(x.strip('+').replace(',',''))
    out['Installs'] = df['Installs'].apply(clean_installs)
    def clean_type(x):
        if x == 'Free':
            return 1
        else:
            return 0
    out['Type'] = df['Type'].apply(clean_type)
    def clean_price(x):
        return float(x.strip('$'))
    out['Price'] = df['Price'].apply(clean_price)
    def clean_lu(x):
        return int(x[-4:])
    out['Last Updated'] = df['Last Updated'].apply(clean_lu)
    return out

def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    year_count = cleaned.groupby('Last Updated').count()
    filtered_years = year_count[year_count['App'] >= 100].index
    filtered_df = cleaned[cleaned['Last Updated'].isin(filtered_years)]
    median_installs = filtered_df[['Last Updated','Installs']].groupby('Last Updated').median().idxmax()[0]
    
    cr_high_min_rating = cleaned[['Content Rating', 'Rating']].groupby('Content Rating').min().idxmax()[0]
    
    high_price_cat = cleaned[['Category','Price']].groupby('Category').mean().idxmax()[0]
    
    filtered_df_2 = cleaned[cleaned['Reviews'] >= 1000]
    low_avg_rating = filtered_df_2[['Category', 'Rating']].groupby('Category').mean().idxmin()[0]
    return [median_installs, cr_high_min_rating, high_price_cat, low_avg_rating]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    out = cleaned.copy()
    means = cleaned[['Category','Reviews']].groupby('Category').mean()
    stds = cleaned[['Category','Reviews']].groupby('Category').std()
    def standardize(row):
        return np.array([row[0],(row[1] - means.loc[row[0],:]['Reviews']) / (stds.loc[row[0],:]['Reviews'])])
    out['Reviews'] = cleaned[['Category','Reviews']].transform(standardize, axis = 1)['Reviews']
    return out[['Category', 'Reviews']]

def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    
    return ['equal','GAME']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    dirs = os.listdir(dirname)
    read_files = []
    for file in dirs:
        df = pd.read_csv(dirname + '/' + file)
        cols = []
        for col in df.columns:
            cols.append(col.replace('_', ' ').lower())
        df.columns = cols
        read_files.append(df)
              
    return pd.concat(read_files)
    



def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """
    out1 = max(df.groupby('current company').size())
    def check_edu(email):
        temp = str(email)
        if temp[-4:] == '.edu':
            return True
        else:
            return False
    out2 = len(list(filter(check_edu, df['email'])))
    max_out3 = 0
    out3 = ''
    out4 = 0
    for tit in df['job title']:
        temp = len(str(tit))
        if temp > max_out3:
            max_out3 = temp
            out3 = str(tit)
        fixed = str(tit).lower()
        if 'manager' in fixed:
            out4 += 1
    
    return [out1, out2, out3, out4]


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    
    directory = os.listdir(dirname)
    dfs = []
    for file in directory:
        df = pd.read_csv(dirname + "/" + file)  
        dfs.append(df.set_index('id'))
    return pd.concat(dfs, sort=True, axis=1)
    


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    def check_student(student):
        num_surveys_not_done = student.isnull().sum()
        if num_surveys_not_done > 1:
            return 0 
        else:
            return 5
    def check_class(df):
        bool_ar = (df.isnull().sum().values / df.shape[0]) <= .10
        if True in bool_ar:
            return True
        else:
            return False
    df['extra credit'] = df.apply(check_student, axis=1)
    if check_class(df) == True:
        df['extra credit'] = df['extra credit'] + 1
    return df[['name', 'extra credit']]


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    What is the most popular Procedure Type for all of the pets we have in our `pets` dataset?
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out,str)
    True
    """
    merged_df = pets.merge(procedure_history, how = 'inner',on = 'PetID')
    return merged_df.groupby('ProcedureType').count().idxmax()[0]


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    def list_func(pets):
        pets = list(pets)
        if len(pets) == 1:
            return pets[0]
        else:
            return pets
    owners_ID_pets = pets.groupby('OwnerID')['Name'].apply(list_func)
    merged_df = owners.merge(owners_ID_pets, how = 'inner', on = 'OwnerID')[['Name_x','Name_y']]
    return merged_df.set_index(['Name_x'])['Name_y']


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
​
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    merged_df = pets.merge(owners, how = 'inner', on = 'OwnerID').merge(procedure_history, how = 'inner', on = 'PetID').merge(procedure_detail, how = 'inner', on = 'ProcedureType')
    return merged_df.groupby('City').sum()['Price']



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['most_popular_procedure', 'pet_name_by_owner', 'total_cost_per_city']
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
