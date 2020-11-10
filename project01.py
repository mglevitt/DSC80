
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    keys = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']

#Create the blank dictionary
#grades_dict = dict(zip(keys, [[]]*len(keys)))

#Clean data frame to contain only what we need
    df = grades[grades.columns.drop(list(grades.filter(regex='Max')))]
    df = df[df.columns.drop(list(df.filter(regex='Lateness')))]
   # df = df[df.columns.drop(list(df.filter(regex='response')))]
    df = df.drop('PID', axis=1).drop('College', axis=1).drop('Level', axis =1)

#Create a list of column names (values for dictionary)
    columns = list(df.columns)
    grades_dict = {}
    for key in keys:
        grades_dict[key] = []
        for col in columns:
            if key in col.lower():
                grades_dict[key].append(col)
   
    remove_lst = []         
    for elem in grades_dict['project']:
        if 'checkpoint' in elem:
            remove_lst.append(elem)
    
    for elem in remove_lst:
        grades_dict['project'].remove(elem)
    
    
    return grades_dict


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    grades = grades.fillna(value=0)
    columns = list(grades.columns)

    #separate max points for normal and free response 
    cols_raw = []
    cols_max = []
    fr_cols_raw = []
    fr_cols_max = []
    for col in columns:
        if 'project' in col and 'checkpoint' not in col and 'Lateness' not in col:
            if "free" in col:
                if 'Max' in col:
                    fr_cols_max.append(col)
                else:
                    fr_cols_raw.append(col)
            else: 
                if 'Max' in col:
                    cols_max.append(col)
                else:
                    cols_raw.append(col)
    df = pd.DataFrame()
    
    for col in cols_raw:
        max_col_nm =''
        fr_col_nm =''
        max_fr_col_nm =''
        for max_col in cols_max:
            if col in max_col:
                max_col_nm = max_col
        for fr_col in fr_cols_raw:
            if col in fr_col:
                fr_col_nm = fr_col
        for max_fr_col in fr_cols_max:
            if col in max_fr_col:
                max_fr_col_nm = max_fr_col
        if fr_col_nm == '':
            df[col] = grades[col] / grades[max_col_nm]
        else:
            df[col] = (grades[col] + grades[fr_col_nm]) / (grades[max_col_nm] + grades[max_fr_col_nm])
    return df.mean(axis = 1)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe
    grades and a Series indexed by lab assignment that
    contains the number of submissions that were turned
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    def convert(x):
        temp = x.split(':')
        return (float(temp[0]) * 3600) + (float(temp[1]) * 60) + float(temp[2])
    
    columns = list(grades.columns)
    cols_late = []
    for col in columns:
        if 'Lateness' in col and "lab" in col:
            cols_late.append(col)
    grades_late = grades[cols_late]
    grades_late = grades_late.applymap(convert)
    threshold = 10000
    def count_late(x):
        if 0 < x <= threshold:
            return 1
        else:
            return 0
    grades_late = grades_late.applymap(count_late)
    grades_late = grades_late.apply(np.sum, axis = 0)
    for col in cols_late:
        grades_late = grades_late.rename({col:col[:5]})
    return grades_late


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """

    def convert(x):
        temp = x.split(':')
        return (float(temp[0]) * 3600) + (float(temp[1]) * 60) + float(temp[2])
    col = list(map(convert, col))
    threshold = 50000
    def late_pen(x):
        if 0 <= x < threshold:
            return 1
        elif x < 604800:
            return .9
        elif x < 1209600:
            return .7
        else:
            return .4
    return pd.Series(list(map(late_pen, col)))


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment,
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    grades = grades.fillna(value=0)
    lab_grades = pd.DataFrame()
    lab_names = get_assignment_names(grades)['lab']
    for lab in lab_names:
        for col in grades.columns:
            if lab in col:
                if "Max" in col:
                    max_lab_col = col
                elif "Late" in col:
                    late_lab_col = col
        lab_grades[lab] = (lateness_penalty(grades[late_lab_col]) * grades[lab] / grades[max_lab_col])
    return lab_grades


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series).

    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    def ind_grade(row):
        row = np.delete(row, np.argmin(row))
        return np.mean(row)
    return processed.apply(ind_grade, axis = 1, raw = True)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    grades = grades.fillna(value=0)
    proccessed_labs = process_labs(grades)
    lab_grades = lab_total(proccessed_labs)
    project_grades = projects_total(grades)
    
    assignment_dict = get_assignment_names(grades)
    columns = list(grades.columns)
    keys = ['midterm', 'final', 'disc', 'checkpoint']
    final_grades = pd.DataFrame()
    for key in keys:
        df = pd.DataFrame()
        assignment_names = assignment_dict[key]
        for assignment in assignment_names:
            max_col_nm = ''
            for col in columns:
                if (assignment in col) & ('Max' in col):
                    max_col_nm = col
            df[assignment] = grades[assignment] / grades[max_col_nm]
        
        final_grades[key] = df.mean(axis = 1)
    final_grades['midterm'] = final_grades['midterm'] *.15
    final_grades['final'] = final_grades['final'] *.3
    final_grades['disc'] = final_grades['disc'] *.025
    final_grades['checkpoint'] = final_grades['checkpoint'] *.025
    final_grades['lab'] = lab_grades * .2
    final_grades['projects'] = project_grades * .3
    
    return final_grades.sum(axis = 1)

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def letter(grade):
        if grade >= .9:
            return 'A'
        elif grade >= .8:
            return 'B'
        elif grade >= .7:
            return 'C'
        elif grade >= .6:
            return 'D'
        else:
            return 'F'
    return total.apply(letter)

def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    num_letter = pd.DataFrame(final_grades(total_points(grades))).groupby(0).size().sort_values(ascending = False)
    
    return (num_letter / num_letter.sum())

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    grades = grades.fillna(value=0)
    proccessed_labs = process_labs(grades)
    lab_grades = lab_total(proccessed_labs)
    project_grades = projects_total(grades)
    
    assignment_dict = get_assignment_names(grades)
    columns = list(grades.columns)
    keys = ['midterm', 'final', 'disc', 'checkpoint']
    final_grades = pd.DataFrame()
    for key in keys:
        df = pd.DataFrame()
        assignment_names = assignment_dict[key]
        for assignment in assignment_names:
            max_col_nm = ''
            for col in columns:
                if (assignment in col) & ('Max' in col):
                    max_col_nm = col
            df[assignment] = grades[assignment] / grades[max_col_nm]
        
        final_grades[key] = df.mean(axis = 1)
    final_grades['midterm'] = final_grades['midterm'] *.15
    final_grades['final'] = final_grades['final'] *.3
    final_grades['disc'] = final_grades['disc'] *.025
    final_grades['checkpoint'] = final_grades['checkpoint'] *.025
    final_grades['lab'] = lab_grades * .2
    final_grades['projects'] = project_grades * .3
    
    data = pd.DataFrame()
    data['Level'] = grades['Level']
    data['grades'] = final_grades.sum(axis = 1)
    num_srs = data.groupby('Level').size()['SR']
    sr_avg = data.groupby('Level').mean()['grades'][2]
    avgs = []
    for i in range(N):
        sample = data.sample(num_srs, replace = True)
        avgs.append(np.mean(sample['grades']))
    
    return np.count_nonzero(avgs < sr_avg) / N


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades,
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    grades = grades.fillna(value=0)
    processed_labs = process_labs(grades)
    num_rows = processed_labs.shape[0]
    num_cols = processed_labs.shape[1]
    noise = np.random.normal(0, 0.02, size=(num_rows, num_cols))
    processed_labs = processed_labs + noise
    for col in processed_labs.columns:
        processed_labs[col] = np.clip(processed_labs[col], 0, 1)
    lab_grades = lab_total(processed_labs)
    
    grades = grades.fillna(value=0)
    columns = list(grades.columns)

    #separate max points for normal and free response 
    cols_raw = []
    cols_max = []
    fr_cols_raw = []
    fr_cols_max = []
    for col in columns:
        if 'project' in col and 'checkpoint' not in col and 'Lateness' not in col:
            if "free" in col:
                if 'Max' in col:
                    fr_cols_max.append(col)
                else:
                    fr_cols_raw.append(col)
            else: 
                if 'Max' in col:
                    cols_max.append(col)
                else:
                    cols_raw.append(col)
    df = pd.DataFrame()
    
    for col in cols_raw:
        max_col_nm =''
        fr_col_nm =''
        max_fr_col_nm =''
        for max_col in cols_max:
            if col in max_col:
                max_col_nm = max_col
        for fr_col in fr_cols_raw:
            if col in fr_col:
                fr_col_nm = fr_col
        for max_fr_col in fr_cols_max:
            if col in max_fr_col:
                max_fr_col_nm = max_fr_col
        if fr_col_nm == '':
            df[col] = grades[col] / grades[max_col_nm]
        else:
            df[col] = (grades[col] + grades[fr_col_nm]) / (grades[max_col_nm] + grades[max_fr_col_nm])
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    noise = np.random.normal(0, 0.02, size=(num_rows, num_cols))
    df = df + noise
    for col in df.columns:
        df[col] = np.clip(df[col], 0, 1)
    project_grades = df.mean(axis = 1)
    
    assignment_dict = get_assignment_names(grades)
    columns = list(grades.columns)
    keys = ['midterm', 'final', 'disc', 'checkpoint']
    final_grades = pd.DataFrame()
    for key in keys:
        df = pd.DataFrame()
        assignment_names = assignment_dict[key]
        for assignment in assignment_names:
            max_col_nm = ''
            for col in columns:
                if (assignment in col) & ('Max' in col):
                    max_col_nm = col
            df[assignment] = grades[assignment] / grades[max_col_nm]
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        noise = np.random.normal(0, 0.02, size=(num_rows, num_cols))
        df = df + noise
        for col in df.columns:
            df[col] = np.clip(df[col], 0, 1)    
        final_grades[key] = df.mean(axis = 1)
    final_grades['lab'] = lab_grades * .2
    final_grades['projects'] = project_grades * .3
    final_grades['midterm'] = final_grades['midterm'] *.15
    final_grades['final'] = final_grades['final'] *.3
    final_grades['disc'] = final_grades['disc'] *.025
    final_grades['checkpoint'] = final_grades['checkpoint'] *.025
    
    
    return final_grades.sum(axis = 1)


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return [.0058,.83,[.79,.85],.058,[True,True]]


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
