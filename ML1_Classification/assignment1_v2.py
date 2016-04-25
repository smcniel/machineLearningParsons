#Code taken and modified from Aaron Hill
# https://github.com/visualizedata/ml

# to best predict 'helpful' : (data.helpScore >= 0.9) 
# & (data.HelpfulnessDenominator > 3)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.sparse as sp
get_ipython().magic('matplotlib inline')
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('./data/Amazon.csv')
print(data.shape)
data.head(5)

def process_review():
    # review length
    data['reviewLen'] = data['Text'].str.len()

    # number of words in review
    data['wordCount'] = [len(w.split()) for w in data['Text']]
    
    # review contains capitol words
    data['numCaps'] = [np.sum([w.isupper() for w in word.split()] ) for word in data['Text']]

    # number of numbers used in review
    data['numDigits'] = [np.sum([w.isdigit() for w in word.split()] ) for word in data['Text']]

    # number of exclamation punctuation
    data['numExclam'] = [e.count("!") for e in data['Text']]

    # number of parenthesis used in review
    data['numParen'] = [p.count(")") for p in data['Text']]

    # reshape features and concatenate them
    XreviewLen = data.reviewLen.values.reshape(data.shape[0], 1)
    XwordCount = data.wordCount.values.reshape(data.shape[0], 1)
    XnumCaps = data.numCaps.values.reshape(data.shape[0], 1)
    XnumDigits = data.numDigits.values.reshape(data.shape[0], 1)
    XnumExclam = data.numExclam.values.reshape(data.shape[0], 1)
    XnumParen = data.numParen.values.reshape(data.shape[0], 1)

    X_features_rev = np.concatenate((XreviewLen, XwordCount, XnumCaps,
                                XnumDigits, XnumExclam, XnumParen),
                                axis=1)

    return X_features_rev

def extract_misc_features():
    # how many reviews for the same product
    numProductReviews = data.groupby(['ProductId'])['Id'].transform('count')
    print(numProductReviews)

    # how many times does a user give a review?
    numRepeatUsers = data.groupby(['UserId'])['ProductId'].transform('count')
    print(numRepeatUsers)

    # how many unique users?
    numUniqueUsers = data.groupby(['ProductId'])['UserId'].transform('count')

    # create new dataframe from features
    colNames = np.array(['numProductReviews', 'numRepeatUsers', 'numUniqueUsers'])
    result = np.array([numProductReviews, numRepeatUsers, numUniqueUsers]).T
    newFeatures = pd.DataFrame(result, columns = colNames)

    # add new dataframe to data
    data = pd.concat([data, newFeatures], axis=1)
    # print(data.shape)
    # data.head(5)

    # reshape features and concatenate them
    XnumProdRev = data.numProductReviews.values.reshape(data.shape[0], 1)
    XnumRepeatUsers = data.numRepeatUsers.values.reshape(data.shape[0], 1)
    XnumUniqueUsers = data.numUniqueUsers.values.reshape(data.shape[0], 1)
    XScore = data.iloc[:, 7].values.reshape(data.shape[0], 1)

    X_features_misc = np.concatenate((XnumProdRev, XnumRepeatUsers, 
                                    XnumUniqueUsers, XScore) axis=1)


    return X_features_misc 

def bag_words_review():
    # vectorize Bag of Words from review text; as sparse matrix
    # hashing here avoids fitting the data into the main memory of the computer
    hv = HashingVectorizer(n_features=2 ** 17, non_negative=True)
    X_reviews = hv.transform(data.Text)

    # convert additional features to sparse matrix and concatenate 
    # onto the bag of words sparse matrix
    XtoaddSparse = csr_matrix(Xfeatures)
    Xfinal = hstack([X_reviews, XtoaddSparse])
    X = csr_matrix(Xfinal)
    print(X.shape)


    # XScore = data.iloc[:, 7].values.reshape(data.shape[0], 1)
    # XreviewLen = data.reviewLen.values.reshape(data.shape[0], 1)
    Xtoadd = np.concatenate((XScore, XreviewLen), axis=1)

    return Xtoadd


def process_summary():
    # fix missing data
    data.Summary = data.Summary.fillna(0)
    # length of summary
    data['sumLength'] = data['Summary'].str.len()
    # punctuation
    data['sumExplanation'] = data['Summary'].str.contains('!')


    # vectorize Bag of Words from summary text; as sparse matrix
    # from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=2 ** 17, non_negative=True)
    X = hv.transform(data.Summary)

    # reshape and add features 
    XsumLength = data.sumLength.values.reshape(data.shape[0], 1)
    XsumExplanation = data.sumExplanation.values.reshape(data.shape[0], 1)
    Xtoadd = np.concatenate((XsummaryLen,
                             XsumExplanation), axis=1)
    return Xtoadd



def process_profileName():
    # how many capital letters?
    data['numCapName'] = [np.sum([w.isupper() for w in word.split()] ) for word in data['ProfileName']]
    # does it contain numbers? how many?
    data['numDigName'] = [np.sum([w.isdigit() for w in word.split()] ) for word in data['ProfileName']]
    # length of profile name
    data['proLength'] = data['ProfileName'].str.len()

    # reshape and add features 
    XcapitalName = data.capitalName.values.reshape(data.shape[0], 1)
    XnumName = data.numName.values.reshape(data.shape[0], 1)
    XquotedName = data.quotedName.values.reshape(data.shape[0], 1)
    Xtoadd = np.concatenate((XcapitalName,
                             XnumName,
                             XquotedName), axis=1)
    return Xtoadd


# work in progress. I've been calling certain functions and not others.
# finish this
def combine_features():

    XallFeatures = np.concatenate((process_summary.Xtoadd,
                                    process_profileName.Xtoadd,
                                    process_review.Xtoadd), axis=1)

    return XallFeatures


# report on training and test sets
def print_results():
    print('Error rate on training set: ')
    print((y_train != y_pred).sum() / X_train.shape[0])
    print('Accuracy rate on training set: ')
    print(1 - (y_train != y_pred).sum() / X_train.shape[0])
    print('True positive rate on training tet:')
    print(((y_train==True) & (y_pred==True)).sum() / y_train.sum())
    print('**************')
    print('Error rate on test set: ')
    print((y_test != y_pred_test).sum() / X_test.shape[0])
    print('Accuracy rate on test set: ')
    print(1 - (y_test != y_pred_test).sum() / X_test.shape[0])
    print('True positive rate on test set')
    print(((y_test==True) & (y_pred_test==True)).sum() / y_test.sum())
    print('True negative rate on test set')
    print(((y_test==False) & (y_pred_test==False)).sum() / (y_test.shape[0] - y_test.sum()))



# convert additional features to sparse matrix and concatenate onto the bag of words sparse matrix
# from scipy.sparse import csr_matrix, hstack
XtoaddSparse = csr_matrix(Xtoadd)
Xfinal = hstack([X, XtoaddSparse])
X = csr_matrix(Xfinal)


# size of feature set
print(X.shape)

# define y
y = data.iloc[:, 12].values
y.shape

# create training and test sets
# from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

# feature scaling
# from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)    # with_mean=False disables centering or scaling
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# MODEL: SVM, linear
# from sklearn import linear_model
clf = linear_model.SGDClassifier()
clf.fit(X_train_std, y_train)
y_pred = clf.fit(X_train_std, y_train).predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
print_results()

# MODEL: logistic regression
# from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='log', n_iter=50, alpha=0.00001)
clf.fit(X_train_std, y_train)
y_pred = clf.fit(X_train_std, y_train).predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
print_results()

# MODEL: Naive Bayes
# from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_std, y_train)
y_pred = clf.fit(X_train_std, y_train).predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
print_results()

# Perceptron
# from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='perceptron')
clf.fit(X_train_std, y_train)
y_pred = clf.fit(X_train_std, y_train).predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
print_results()