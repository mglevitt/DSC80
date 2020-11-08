import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    #all flights in SAN
    def process(df):
        #print(df.columns)
        return df[(df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN')]
    pd.read_csv(infp, nrows=0).to_csv(outfp, index=False)

    L = pd.read_csv(infp, chunksize=10000, dtype = str)
    #cols_added = False
    with open(outfp,'a') as out_file:
        for df in L: 
            
            temp = process(df)
            temp.to_csv(out_file, header=False, mode = 'a')
            
    return None


def get_sw_jb(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    #all flights in SAN
    def process(df):
        #print(df.columns)
        return df[(df['AIRLINE'] == 'B6') | (df['AIRLINE'] == 'WN')]
    pd.read_csv(infp, nrows=0).to_csv(outfp, index=False)

    L = pd.read_csv(infp, chunksize=10000, dtype = str)
    #cols_added = False
    with open(outfp,'a') as out_file:
        for df in L: 
            
            temp = process(df)
            temp.to_csv(out_file, header=False, mode = 'a')
            
    return None


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """

    data_kinds= {'YEAR': 'Q', 'MONTH':'O',"DAY":'O',"DAY_OF_WEEK":'O', "AIRLINE":'N',
                 "FLIGHT_NUMER":'N', "TAIL_NUMBER":'N', "ORIGIN_AIRPORT":'N',
                 "DESTINATION_AIRPORT": 'N', "SCHEDULED_DEPARTURE":"O", "DEPARTURE_TIME":"O",
                "DEPARTURE_DELAY": "O", "TAXI_OUT":"Q", "WHEELS_OFF":"O",
                "SCHEDULED_TIME":"O", "ELAPSED_TIME": "Q", "AIR_TIME": "Q", "DISTANCE":'Q',
                "WHEELS_ON":"O", "TAXI_IN":'Q', "SCHEDULED_ARRIVAL":"O","ARRIVAL_TIME":"O",
                "ARRIVAL_DELAY":"Q", "DIVERTED":"N","CANCELLED":"N", "CANCELLATION_REASON":"N",
                "AIR_SYSTEM_DELAY":"Q", "SECURITY_DELAY":"Q", "AIRLINE_DELAY":"Q",
                "LATE_AIRCRAFT_DELAY":"Q", "WEATHER_DELAY":"Q"}
    
    return data_kinds

def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    data_types = {'YEAR': 'int', 'MONTH':'int',"DAY":'int',"DAY_OF_WEEK":'int', "AIRLINE":'str',
                 "FLIGHT_NUMER":'int', "TAIL_NUMBER":'str', "ORIGIN_AIRPORT":'str',
                 "DESTINATION_AIRPORT": 'str', "SCHEDULED_DEPARTURE":'str', "DEPARTURE_TIME":'str',
                "DEPARTURE_DELAY": 'float', "TAXI_OUT":'float', "WHEELS_OFF":'str',
                "SCHEDULED_TIME":'int', "ELAPSED_TIME": 'float', "AIR_TIME": 'float', "DISTANCE":'int',
                "WHEELS_ON":'str', "TAXI_IN":'float', "SCHEDULED_ARRIVAL":'str',"ARRIVAL_TIME":'str',
                "ARRIVAL_DELAY":'float', "DIVERTED":'bool',"CANCELLED":'bool', "CANCELLATION_REASON":'str',
                "AIR_SYSTEM_DELAY":'float', "SECURITY_DELAY":'float', "AIRLINE_DELAY":'float',
                "LATE_AIRCRAFT_DELAY":'float', "WEATHER_DELAY":'float'}
    return data_types

# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    out_df = pd.DataFrame(index = ['ARRIVING','DEPARTING'])
    arr_flights = flights[flights['DESTINATION_AIRPORT'] == 'SAN']
    dep_flights = flights[flights['ORIGIN_AIRPORT'] == 'SAN']
    count_arr = arr_flights.shape[0]
    count_dep = dep_flights.shape[0]
    mean_delay_arr = arr_flights['ARRIVAL_DELAY'].mean()
    mean_delay_dep = dep_flights['ARRIVAL_DELAY'].mean()
    med_delay_arr = arr_flights['ARRIVAL_DELAY'].median()
    med_delay_dep = dep_flights['ARRIVAL_DELAY'].median()
    long_arr = arr_flights[arr_flights['ARRIVAL_DELAY'].max() == arr_flights['ARRIVAL_DELAY']]['AIRLINE'].values[0]
    long_dep = dep_flights[dep_flights['ARRIVAL_DELAY'].max() == dep_flights['ARRIVAL_DELAY']]['AIRLINE'].values[0]
    months_arr = arr_flights.groupby('MONTH').size().sort_values(ascending = False)[:3].index.values
    months_dep = dep_flights.groupby('MONTH').size().sort_values(ascending = False)[:3].index.values
    out_df['count'] = np.array([count_arr, count_dep])
    
    final_df = pd.DataFrame({'count':[count_arr, count_dep],'mean_delay':[mean_delay_arr,mean_delay_dep]
                             , 'median_delay':[med_delay_arr,med_delay_dep]
                             ,'airline':[long_arr,long_dep]
                             ,'top_months':[months_arr,months_dep]}, index = ['ARRIVING', 'DEPARTING'])
    
    
    return final_df


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """
    ttl_flights = flights.shape[0]
    late1 = flights[flights['DEPARTURE_DELAY'] > 0][flights['ARRIVAL_DELAY'] <= 0].shape[0] / ttl_flights
    late2 = flights[flights['DEPARTURE_DELAY'] <= 0][flights['ARRIVAL_DELAY'] > 0].shape[0] / ttl_flights
    late3 = flights[flights['DEPARTURE_DELAY'] > 0][flights['ARRIVAL_DELAY'] > 0].shape[0] / ttl_flights
    data = np.array([late1,late2,late3])
    return pd.Series(data,index = ['late1','late2','late3'])


def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """
    months = flights.groupby('MONTH').count().index.values
    all_data = {}
    for month in months:
        month_flights = flights[flights['MONTH']==month]
        ttl_flights = month_flights.shape[0]
        late1 = month_flights[month_flights['DEPARTURE_DELAY'] > 0][month_flights['ARRIVAL_DELAY'] <= 0].shape[0] / ttl_flights
        late2 = month_flights[month_flights['DEPARTURE_DELAY'] <= 0][month_flights['ARRIVAL_DELAY'] > 0].shape[0] / ttl_flights
        late3 = month_flights[month_flights['DEPARTURE_DELAY'] > 0][month_flights['ARRIVAL_DELAY'] > 0].shape[0] / ttl_flights
        data = [late1,late2,late3]
        all_data[month] = data
    return pd.DataFrame.from_dict(all_data,orient = 'index', columns = ['late1','late2','late3'])


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """
    flights_2015 = flights[flights['YEAR'] == 2015]
    out = flights_2015.pivot_table(values = 'FLIGHT_NUMBER', index = 'DAY_OF_WEEK', columns = 'AIRLINE', aggfunc = 'count')
    return out


def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """
    flights_2015 = flights[flights['YEAR'] == 2015]
    out = flights_2015.pivot_table(values = 'ARRIVAL_DELAY', index = 'DAY_OF_WEEK', columns = 'AIRLINE', aggfunc = 'mean')
    return out


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    if (row['CANCELLED']== True) | (row['DIVERTED']==True):
        return True
    else:
        return False


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    return ((row['CANCELLED'] == True) | (row['DIVERTED'] == True)|(row['ARRIVAL_DELAY'] < 15))



# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """

    dist = flights.assign(is_null = flights.DEPARTURE_DELAY.isnull()).pivot_table(index = 'is_null', columns = col, aggfunc='size').apply(lambda x:x/x.sum(), axis=1)
    #dist.T.plot(kind='bar') 
    
    tvds = [] #tvds array
    for _ in range(N):
        #shuffle the col column
        shuffled_col = flights[col].sample(replace=False, frac=1).reset_index(drop=True)
        #put into a table
        shuffled = flights.assign(**{col: shuffled_col, 'is_null':flights['DEPARTURE_DELAY'].isnull()})
        #compute TVD
        shuffled = shuffled.pivot_table(index='is_null', columns = col, aggfunc='size').apply(lambda x:x/x.sum(), axis=1)
        tvd=shuffled.diff().iloc[-1].abs().sum()/2
        tvds.append(tvd)
    obs = dist.diff().iloc[-1].abs().sum()/2
    pval = np.mean(tvds >= obs)
    #plot = pd.Series(tvds).plot(kind='hist', density=True, alpha=.8, title = 'p-val')
    #plot2 = plt.scatter(obs, 0, color='red', s=40)
    return pval

def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return ['AIRLINE','DAY_OF_WEEK']


def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """

    data = ['MAR', 'MAR', 'MCAR', 'MAR']
    return pd.Series(data, index = ['CANCELLED', 'CANCELLATION_REASON', 'TAIL_NUMBER', 'ARRIVAL_TIME'])

# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """

    airlines = jb_sw[jb_sw["ORIGIN_AIRPORT"].isin(["ABQ", "BDL", "BUR", "DCA", "MSY", "PBI", "PHX", "RNO", "SJC", "SLC"])]
    #departure delay is greater than 0 
    airlines['DEPARTURE_DELAY'] = airlines['DEPARTURE_DELAY'] > 0.0
    
    return airlines.groupby('AIRLINE')['DEPARTURE_DELAY'].mean().to_frame()


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """

    airlines = jb_sw[jb_sw["ORIGIN_AIRPORT"].isin(["ABQ", "BDL", "BUR", "DCA", "MSY", "PBI", "PHX", "RNO", "SJC", "SLC"])]
    airlines['DEPARTURE_DELAY'] = airlines['DEPARTURE_DELAY'] > 0.0

    pivoted = airlines.pivot_table(values = 'DEPARTURE_DELAY', columns = 'ORIGIN_AIRPORT', index='AIRLINE', aggfunc='mean')
    return pivoted


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """

    copy1 = df.copy()
    out1 = copy1.groupby(group1)[occur].mean().to_frame()
    copy2 = df.copy()
    out2 = copy2.pivot_table(values = occur, columns = group2, index=group1, aggfunc='mean')
    prop1 = out1.iloc[0,0]
    prop2 = out1.iloc[1,0]
    out = True
    if prop1 > prop2:
        for i in range(out2.shape[1]):
            if out2.iloc[0,i] >= out2.iloc[1,i]:
                out = False
    else:
         for i in range(out2.shape[1]):
            if out2.iloc[0,i] <= out2.iloc[1,i]:
                out = False
    return out


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson']
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
