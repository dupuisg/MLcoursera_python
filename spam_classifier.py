import math, string, os, re, time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from nltk.stem import PorterStemmer
from collections import Counter

ps = PorterStemmer()

def process_email(filename):
    #Read email, preprocess, and return a list of its words

    f = open(filename, 'r')
    contents = f.read()

    #Remove email header - comment out if not using raw email w header
    if '\n\n' in contents:
        contents = contents[contents.find('\n\n'):]

    #Convert to lower case
    contents = contents.lower()

    #Strip html tags
    contents = re.sub(r'<[^<>]+>', '', contents)

    #Replace all numbers with 'number'
    contents = re.sub(r'\d+', 'number', contents)

    #Replace all urls with 'httpaddr'
    contents = re.sub(r'https?://[^\s]*', 'httpaddr', contents)

    #Replace all email addresses with 'emailaddr'
    contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', contents)

    #Replace all instances of dollar sign with 'dollar'
    contents = contents.replace('$', 'dollar')

    #Remove underscores
    contents = contents.replace('_', '')

    #Tokenize, removing white spaces, punctuation, non-alphanumeric
    word_list = re.split(r'\W',contents)
    word_list = [w for w in word_list if w]

    for word in word_list:
        #Word stemming
        try:
            word = ps.stem(word)
        except:
            word = ""
            continue

    f.close()
    return word_list

def create_vocablist():

    word_counts = Counter()

    spam_dir = "spam_corpus/"
    for sub_dir in os.listdir(spam_dir):
        for filename in os.listdir(spam_dir + sub_dir):
            word_list = process_email(spam_dir + sub_dir + '/' +  filename)
            word_counts.update(Counter(word_list))

    #keep words occuring 100 or more times
    vocablist = {word:word_counts[word] for word in word_counts if word_counts[word]>=100}

    f = open('vocab_list.txt', 'w')
    idx = 0
    for (word, freq) in vocablist.items():
        f.write(word + ' ' + str(idx) + '\n') 
        idx += 1
    f.close()

def get_vocablist():
   
    vocablist = {}
    f = open('vocab_list.txt', 'r')
    for line in f:
        (word, idx) = line.split()        
        vocablist[word] = int(idx)   

    f.close()
    return vocablist

##Get word indices from word_list and map to a binary feature vector
def email_features(word_list):

    vocablist = get_vocablist()
    word_indices = []

    for word in word_list:
        #Look up in vocablist and add index to word_indices
        if word in vocablist:
            word_indices += [vocablist[word]]

    x = np.zeros(len(vocablist))
    x[word_indices]= 1

    return x

def create_datasets():
## Create datasets using spam corpus
## Examples taken from spamassassin.apache.org/old/publiccorpus/ 
    outpath = "datasets_spam"    
    datapath = "spam_corpus"
    spam_examples = "spam_2"

    total = len([f for sub_dir in os.listdir(datapath) for f in os.listdir(datapath + "/" + sub_dir)])
    m = math.ceil(total*0.6)
    mval = math.ceil((total - m)/2.0)
    mtest = total - m - mval

    vocablist = get_vocablist()
    n = len(vocablist)

    X = np.zeros([total, n])
    y = np.zeros(total)

    count = 0
    for sub_dir in os.listdir(datapath):
        for name in os.listdir(datapath + "/" + sub_dir):
            if sub_dir == spam_examples:
                y[count] = 1
            word_list = process_email(datapath + "/" + sub_dir + "/" + name)
            x = email_features(word_list)
            X[count, :] = x
            count += 1

    indices = np.random.permutation(total)
    Xtrain = X[indices[:m],:]
    ytrain = y[indices[:m]]
    Xval = X[indices[m:m+mval],:]
    yval = y[indices[m:m+mval]]
    Xtest = X[indices[m+mval:],:]
    ytest = y[indices[m+mval:]]

    np.savetxt(outpath + "/" + 'X_full.dat', X)
    np.savetxt(outpath + "/"+ 'y_full.dat', y)
    np.savetxt(outpath + "/"+ 'Xtrain.dat', Xtrain)
    np.savetxt(outpath + "/"+ 'ytrain.dat', ytrain)
    np.savetxt(outpath + "/"+ 'Xval.dat', Xval)
    np.savetxt(outpath + "/"+ 'yval.dat', yval)
    np.savetxt(outpath + "/"+ 'Xtest.dat', Xtest)
    np.savetxt(outpath + "/"+ 'ytest.dat', ytest)

#---------------------------------------------------------

## Only need to do this once, comment out if already created
#create_vocablist()
#create_datasets()

datapath = "datasets_spam/"
X = np.loadtxt(datapath + 'Xtrain.dat')
y = np.loadtxt(datapath + 'ytrain.dat')
Xval = np.loadtxt(datapath + 'Xval.dat')
yval = np.loadtxt(datapath + 'yval.dat')
Xtest = np.loadtxt(datapath + 'Xtest.dat')
ytest = np.loadtxt(datapath + 'ytest.dat')

## Fit  and get training/test accuracies
clf = svm.LinearSVC(C=0.1) 
clf.fit(X,y)
training_acc = clf.score(X,y)
print 'Training accuracy: ', training_acc
test_acc = clf.score(Xtest,ytest)
print 'Test accuracy: ', test_acc

print '----------------------------------------------------'

## Determine top predictors
#  parameters with largest (+)ve values, and display corresponding words
vocablist = get_vocablist()
vocablist_reverse = {}
for word, index in vocablist.items():
    vocablist_reverse[index] = word

indices = np.argsort(-clf.coef_).flatten()
indices = indices[:15]

print 'Top predictors:'
for i in indices:
    print vocablist_reverse[i]
print '--------------------------------------------------'

#Pick a sample email and a sample spam and predict outcome
print 'Prediction for sample email.....'
word_list = process_email('email_samples/emailSample1.txt')
x = email_features(word_list)
x = x.reshape(1,-1)
prediction = clf.predict(x)
if prediction == 1:
    print 'Spam'
else:
    print 'Not spam'

print 'Prediction for second sample email.....'
word_list = process_email('email_samples/emailSample2.txt')
x = email_features(word_list)
x = x.reshape(1,-1)
prediction = clf.predict(x)
if prediction == 1:
    print 'Spam'
else:
    print 'Not spam'

print 'Prediction for spam sample.....'
word_list = process_email('email_samples/spamSample2.txt')
x = email_features(word_list)
x = x.reshape(1,-1)
prediction = clf.predict(x)
if prediction == 1:
    print 'Spam'
else:
    print 'Not spam'


