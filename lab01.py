import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def median_vs_average(nums):
    """
    median takes a non-empty list of numbers,
    returning a boolean of whether the median is
    greater or equal than the average
    If the list has even length, it should return
    the mean of the two elements in the middle.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether median is greater or equal than average.

    :Example:
    >>> median_vs_average([6, 5, 4, 3, 2])
    True
    >>> median_vs_average([50, 20, 15, 40])
    False
    >>> median_vs_average([1, 2, 3, 4])
    True
    """
    cnt = 0
    ttl = 0
    lngth= len(nums)
    srtd = sorted(nums)
    md_in = (lngth - 1) // 2
    if lngth % 2 == 0:
        mdn = (srtd[md_in] + srtd[md_in + 1]) / 2
    else:
        mdn = srtd[md_in]
    for i in range(len(nums)):
        cnt += 1
        ttl += nums[i]
    avg = ttl / cnt
    if mdn >= avg:
        return True
    else:
        return False


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------
def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.
    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    lngth = len(ints)
    for i in range(lngth):
        for j in range(lngth):
            if i - j == 0:
                continue
            x = ints[i]
            y = ints[j]
            if i - j == abs(x - y):
                return True
    return False


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Data!', 3)
    'DatDaD'
    >>> n_prefixes('Marina', 4)
    'MariMarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    """
    if n == 0:
        return ""
    return s[:n] + n_prefixes(s, n - 1)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------
def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers.
    :Example:
    >>> exploded_numbers([3, 4], 2)
    ['1 2 3 4 5', '2 3 4 5 6']
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    """

    out = []
    dgts = len(str(max(ints)))
    for i in ints:
        temp = str(i).zfill(dgts)
        for t in range(1, n + 1):
            temp = str(i - t).zfill(dgts) + " " + temp + " " + str(i + t).zfill(dgts)
        out.append(temp)
    return out

# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a
    string consisting of the last character of the line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    txt = fh.read()
    lines = txt.split("\n")
    out = ""
    for line in lines:
        out += line[-1:]
    return out

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    temp = np.array(range(len(A)))
    temp = np.sqrt(temp)

    return A + temp

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """
    int_check = lambda x : round(x) == x
    return np.array(list(map(int_check, np.sqrt(A))))

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """

    return np.round(np.diff(A) / A[:-1], 2)

def arr_4(A):
    """
    Create a function arr_4 that takes in A and
    returns the day on which you can buy at least
    one share from 'left-over' money. If this never
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    try:
        return (np.cumsum(20 % A) > A).tolist().index(True)
    except:
        return -1

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a dataframe of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    try:
        num_pl = salary['Player'].nunique()
    except:
        num_pl = None
    try:
        num_tm = salary['Team'].nunique()
    except:
        num_tm = None
    try:
        ttl_sal = salary['Salary'].nunique()
    except:
        ttl_sal = None
    try:
        mx_sal = np.max(salary['Salary'])
        hghst_sal = salary[salary['Salary'] == mx_sal]['Player'].values(0)
    except:
        hghst_sal = None
    try:
        avg_bos = np.round(np.mean(salary[salary['Team'] == 'BOS']['Salary']), 2)
    except:
        avg_bos = None
    try:
        thrd_lwst_nm = salary.sort_values(by = 'Salary')['Player'].values(2)
        thrd_lwst_tm = salary.sort_values(by = 'Salary')['Team'].values(2)
        thrd_lwst = thrd_lwst_nm + ", " + thrd_lwst_tm
    except:
        thrd_lwst = None
    lst_nm = lambda x : x.split(' ')[1]
    try:
        if salary['Player'].apply(lst_nm).nunique() == len(salary['Player']):
            dplct = False
        else:
            dplct = True
    except:
        dplct = False
    try:
        hghst_sal_tm = salary[salary['Salary'] == mx_sal].values(0)(1)
        ttl_hghst_sal_tm = np.sum(salary[salary['Team'] == hghst_sal_tm]['Salary'])
    except:
        ttl_hghst_sal_tm = None

    indxs = ['num_players', 'num_teams', 'total_salary', 'highest_salary', 'avg_bos', 'third_lowest', 'duplicates', 'total_highest']
    data = [num_pl, num_tm, ttl_sal, hghst_sal, avg_bos, thrd_lwst, dplct, ttl_hghst_sal_tm]
    return pd.Series(data, index = indxs)



# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a
    properly formatted dataframe (as described in
    the question).
    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data,
    as specificed in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    with open(fp) as file:
        txt = file.read().split('\n')
        entries = [line.split(',') for line in txt]
        for entry in entries:
            if '' in entry:
                entry.remove('')
            if entry == []:
                entries.remove(entry)
        col_dic = {}
        col_nms = []
        for i in entries[0]:
            col_dic[i] = []
            col_nms.append(i)
        for entry in entries[1:]:
            for i in range(len(col_nms)):
                if i < 2:
                    col_dic[col_nms[i]].append(entry[i].strip('"'))
                elif i < 4:
                    col_dic[col_nms[i]].append(float(entry[i].strip('"')))
                else:
                    col_dic[col_nms[i]].append(entry[i].strip('"') + ',' + entry[i + 1].strip('"'))
    return pd.DataFrame(col_dic)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median_vs_average'],
    'q02': ['same_diff_ints'],
    'q03': ['n_prefixes'],
    'q04': ['exploded_numbers'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['salary_stats'],
    'q08': ['parse_malformed']
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
