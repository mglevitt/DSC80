
import os
import pandas as pd
import numpy as np
import requests
import time
import re
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    
    # Get the actual book text
    
    START = '\x02' #start of paragraph
    END = '\x03' #end of paragraph 
    
    #request the ULR - allow 5 second scrape time
    def try_request(url, retry = 5):
        req = requests.get(url)
        if req.ok:
            return req
        else:
            time.sleep(2**retry)
            return try_request(url)
        
    req = try_request(url)
    text = req.text
    replaced_newlines = re.sub('(\r){1}', "", text)
    #replaced_newlines = re.sub('(\\\n){1}','\n',temp)
    start = re.search('(\*\*\* START){1}.+(\*\*\*){1}',replaced_newlines).span()[1]
    end = re.search('(\*\*\* END){1}',replaced_newlines).span()[0]
    return replaced_newlines[start:end]
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    #replace paragraph breaks 
    replaced = '\x02' + re.sub("\n(\n)+",'\x03 \x02',book_string) + "\x03"
   
    #words = re.findall('\w+|[^a-zA-Z0-9\s]+',replaced)
    #clean out new lines!
    clean_nl = re.sub("\n", " ", replaced)
    #Clean out empty list values and split by word boundaries
    cleaned = list(filter(None, re.split(r'(\b|\x02|\x03)',clean_nl)))
    #clean out spaces from values
    no_spaces = map(str.strip, cleaned) 
    #clean out empty space values
    tokens = [x for x in no_spaces if x.strip()]
    return tokens
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        return pd.Series(1/len(np.unique(tokens)), index = np.unique(tokens))
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """

        try:
            return np.prod(self.mdl.loc[list(words)])
        except:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        self.out = ''
        sample = pd.Series(np.random.choice(self.mdl.index, M, replace = True))
        def helper(x):
            self.out += x + ' '
        sample = sample.apply(helper)
        return self.out[:-1]

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        counts = pd.DataFrame({'tokens':tokens}).groupby('tokens').size()
        return counts/counts.sum()
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """

        try:
            return np.prod(self.mdl.loc[list(words)])
        except:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        self.out = ''
        sample = pd.Series(np.random.choice(self.mdl.index, M, replace = True, p = self.mdl.values))
        def helper(x):
            self.out += x + ' '
        sample = sample.apply(helper)
        return self.out[:-1]
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        
        return list(zip(*[tokens[i:] for i in range(self.N)]))
    
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        n_counts= [ngrams.count(ngram) for ngram in ngrams]
        n1_grams = [ngram[:self.N-1] for ngram in ngrams]
        n1_counts = [n1_grams.count(ngram) for ngram in n1_grams]
        final = pd.DataFrame()
        final['ngram'] = ngrams
        final['n1gram'] = n1_grams
        final['prob'] = np.array(n_counts) / np.array(n1_counts)
        return final
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        
        """
        temp = self
        n = temp.N
        dic = {n:temp}
        while n > 1:
            
            temp = self.prev_mdl
            
            try:
                n = temp.N
            except:
                n = 1
            dic[n] = temp
        
        
        n = 1
        #This initializes probablity as probability of the first word
        prob = dic[n].mdl.loc[words[0]]
        
        while n < len(words):
            try:
                #This creates a ngram of the last word and current word
                curr_ngram = tuple(words[n-1:n+1])
                #This accesses the df of bigrams
                temp_df = dic[2].mdl
                #This locates the ngram in df then returns probability
                t = temp_df[temp_df['ngram'] == curr_ngram]['prob'].iloc[0]
                prob *= t
                n += 1
            except:
                #In case of an error or a word not there this will end loop and change probability to 0
                prob = 0
                n = len(words)
        return prob

    
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        
        """

        # Use a helper function to generate sample tokens of length `length`
        
        #use recursion call prev model, np.random_choice(probs) to choose next word
        out = '\x02 '
        temp = self.mdl['n1gram'].unique()
        curr = np.random.choice(temp)[0]
        out += curr + ' '
        for i in range(M - 1):
            try:
                curr = list(self.mdl[self.mdl['n1gram'] == tuple([curr])].sample(n=1,weights = 'prob')['ngram'].iloc[0])[1]
                out += curr + ' '
            except:
                break
        return out[:-1]


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
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
