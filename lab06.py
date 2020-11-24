import os
import pandas as pd
import numpy as np
import requests
import bs4
import json
import datetime


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """
    text = '''
    <html>
        <head>
            <title>lab06_1</title>
        </head>
        <body>
            <h1>Surfing</h1>
            <img src = "data/surf_photo.jpg" alt = 'surf_photo'>
            <p> How to Surf
                <br>
                <a href = "https://www.youtube.com/watch?v=67QNw2xQlsk">learn to surf!</a>
            </p>
                <p>Surfing for the views
                    <img src = "data/sunset.JPG" alt = 'sunset'>
                </p>
            <h2>Learn How to Shred!</h2>
                <p>
                <a href = 'https://www.wikihow.com/Surf'>Steps to Surf</a>
                </p>
                <p>Look like Kelly Slater after basic steps!
                    <img src = "https://cdn1.theinertia.com/wp-content/uploads/2018/12/kelly4-670x357.jpg"' alt = 'Kelly Slater Pic'>
                </p>
                <p>How to be a pro in San Diego
                    <br>
                    <a href = 'https://www.sandiego.org/articles/surfing/mele-sailis-surf-faves.aspx'>Pro Surf Guide</a>
                    
        <h2>San Diego Surf Spot Ratings by Max Levitt</h2>

        <table style="width:100%">
          <tr>
            <th>Beach</th>
            <th>Rating</th> 
          </tr>
          <tr>
            <td>Blacks</td>
            <td>4</td>
          </tr>
          <tr>
            <td>Scripps</td>
            <td>5</td>
          </tr>
          <tr>
            <td>La Jolla Shores</td>
            <td>3</td>
          </tr>
          <tr>
            <td>Pacific Beach</td>
            <td>4</td>
          </tr>
        </table>
        </body>
    </html>
    '''
    file = open("lab06_1.html","w")
    file.write(text)
    file.close()
    return None


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    data = bs4.BeautifulSoup(text, features="html.parser")
    cleaned = data.find_all("article", attrs={"class": "product_pod"})
    ratings = []
    for x in cleaned:
        if x.find("p", attrs={"class": "star-rating Four"}) or x.find("p", attrs={"class": "star-rating Five"}):
            temp = float(list(map(str.isdigit,x.find("p", attrs={"class": "price_color"}).text[1:]))[0])
            if temp < 50:
                ratings.append(x.find("a").get("href"))
    return ratings


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    data = bs4.BeautifulSoup(text, features="html.parser")
    cat = []
    for x in data.find("ul", attrs={"class": "breadcrumb"}):
        cat.append(x)
    cat = cat[5].text.strip()
    if cat in categories:
        tbl = data.find("table", attrs={"class": "table table-striped"})
        vals = tbl.find_all(["th", "td"])
        end_vals = []
        for x in vals:
            end_vals.append(x.text)
        dict = {"Availability": end_vals[11],"Category": cat,"Description": data.find_all("p")[3].text,"Number of reviews": end_vals[13],
                "Price (excl. tax)": end_vals[5],"Price (incl. tax)": end_vals[7],"Product Type": end_vals[3],"Rating": data.find_all("p")[2].attrs.get("class")[1],
                "Tax": end_vals[9],"Title": data.find("li", attrs={"class": "active"}).text,"UPC": end_vals[1]}
        return dict
    return None


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    df = pd.DataFrame()
    for page in np.arange(1, k + 1):
        books = extract_book_links(requests.get("http://books.toscrape.com/catalogue/page-%d.html" % page).text)
        for book in books:
            book_info = get_product_info(requests.get("http://books.toscrape.com/catalogue/" + book).text, categories)
            if book_info != None:
                df = df.append(book_info, ignore_index=True)
    return df

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    stock_endpoint = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}?apikey=298a6bfcbbf37a8725002f1da9404cb9'
    start_date = datetime.datetime(year,month,1)
    if month == 12:
        end_date = datetime.datetime(year + 1,1,1)
    else:
        end_date = datetime.datetime(year,month + 1,1)
    date = pd.date_range(start = start_date, end = end_date)
    data = requests.get(stock_endpoint.format(ticker)).json()['historical']
    df = pd.DataFrame.from_dict(data)
    params={'from': date[0].strftime('%Y-%m-%d'), 'to': date[-1].strftime('%Y-%m-%d')}
    out = df[(df['date'] >= params['from']) & (df['date'] < params['to'])]
    return out


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    def tot_tran_vol(row):
        return (row['high']+row['low']) / (2*row['volume'])
    total = str(history.apply(tot_tran_vol, axis = 1).sum()) + 'B'
    close = history.iloc[0]['close']
    ope = history.iloc[-1]['open']
    percent_change =  round(((close - ope) / ope) *100, 2) 
    if percent_change >= 0:
        percent_change = '+'+str(percent_change)+'%'
    else:
        percent_change = '-'+str(percent_change)+'%'
    return tuple([percent_change, total])


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    news_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    stack = [storyid]
    in_data = requests.get(news_endpoint.format(stack.pop())).json()
    if 'kids' in in_data.keys():
            for kid in in_data['kids']:
                stack.append(kid)
    all_data =[]
    while len(stack) > 0:
        in_data = requests.get(news_endpoint.format(stack.pop())).json()
        if 'dead' in in_data.keys():
            continue
        all_data.append(in_data)
        if 'kids' in in_data.keys():
            for kid in in_data['kids']:
                stack.append(kid)
    df = pd.DataFrame(all_data)[['id','by','parent','text','time']]
    def convert_time(time):
        return pd.to_datetime(time)
    df['time'] = df['time'].apply(convert_time)
    return df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
