import ast
import sys
import glob
import os
import collections
import numpy as np
import string
import re
import math

os.chdir(sys.path[0])
data_array_test = np.empty((0, 2))

all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
test_by_class = collections.defaultdict(list)  # dictionary for classification data
fileout = open("percepoutput.txt", "w")
with open(sys.argv[1], "r") as file:
    wtVectorVPN = file.readline()
    wtVectorVPN = ast.literal_eval(wtVectorVPN)
    wtVectorVTD = ast.literal_eval(file.readline())
    bias1 = ast.literal_eval(file.readline())
    bias2 = ast.literal_eval(file.readline())

# defaultdict is analogous to dict() [or {}], except that for keys that do not
# yet exist (i.e. first time access), the value gets contructed using the function
# pointer (in this case, list() i.e. initializing all keys to empty lists).
test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
train_by_class = collections.defaultdict(list)

# create dictionary of all the files for classification
for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    test_by_class[class1 + class2].append(f)

    # predict function

for key, value in test_by_class.items():
    for file in value:
        text_file = open(file, "r")
        line = text_file.read()
        data_array_test = np.append(data_array_test, np.array([[str(line), str(file)]]), axis=0)


def cleanText(line):
    cleaned_str = re.sub('[^a-z\s]+', ' ', line, flags=re.IGNORECASE)  # every char except alphabets is replaced
    text = cleaned_str.lower()
    return text


''' tokenize the data by creating wordcount for each class
For each word, add to class it belongs to, if it isn’t already there, and update the number of counts.
Also add that word to the global vocabulary.'''


def toeknizeText(line):
    worddict = {}
    stopwords = ["a", "an", "able", "about", "above", "be", "the", "at", "became", "become", "becomes", "becoming",
                 "been", "before", "by", "can", "cause", "causes", "certain", "changes",
                 "example", "except", "fifth", "first", "five", "followed", "following", "follows", "for", "get",
                 "gets", "getting", "given", "gives", "go", "goes", "had", "happens", "has",
                 "have", "having", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead",
                 "itself", "keep", "keeps", "kept", "know", "knows", "lately", "later",
                 "latter", "latterly", "least", "mainly", "may", "maybe", "me", "mean", "meanwhile", "merely", "near",
                 "nearly", "need", "needs", "neither", "next", "on", "once", "one", "ones",
                 "onto", "other", "others", "ought", "our", "ours", "placed", "possible", "probably", "provides",
                 "same", "saw", "say", "soon", "see", "do", "its", 'hers', 'between', 'yourself', 'again', 'there',
                 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for',
                 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'is', 's', 'am', 'or', 'who',
                 'as', 'from', 'him', 'each', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'me',
                 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
                 'up', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
                 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why',
                 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'only', 'myself',
                 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'by', 'doing', 'it',
                 'how', 'further', 'was', 'here', 'than', "th", "rd", "st", "nd", "to", "yep", "but", "not", "one",
                 "two", "three", "first", "second", "third"]

    for word in line.split():
        if word not in stopwords:
            # complete_vocab.add(word)
            if word in worddict:
                worddict[word] += 1
            else:
                worddict[word] = 1

    return worddict


# predict function
def predict(X, value, wtVector1, wtVector2):
    featureVector = {}
    line = cleanText(X)
    # tokenize data,For each (document, label) pair, tokenize the document into words.
    ptokenize_nd = toeknizeText(line)
    for word, _ in ptokenize_nd.items():
        if word not in wtVector1:
            continue
        if word not in featureVector:
            featureVector[word] = 1
        else:
            featureVector[word] += 1
    activation1 = 0
    activation2 = 0
    for feature, count in featureVector.items():
        activation1 += (wtVector1[feature] * count)
        activation2 += (wtVector2[feature] * count)
    activation1 += bias1
    activation2 += bias2
    if activation1 >= 0:
        class1 = "positive"
    else:
        class1 = "negative"
    if activation2 >= 0:
        class2 = "truthful"
    else:
        class2 = "deceptive"
    # positive_deceptive : 0, positive_truthful :1,
    # negative_truthful : 2 , negative_deceptive : -2
    # final = max(score_pd,score_pt,score_nd,score_nt)
    value_f = 0;
    '''if c1 == "Positive" and c2 == "Deceptive":
        value_f = 0
    if  c1 == "Positive" and c2 == "Truthful":
        value_f = 1
    if  c1 == "Negative" and c2 == "Truthful":
        value_f = -2
    if  c1 == "Negative" and c2 == "Deceptive":
        value_f = 2'''
    # result1.append(value_f)
    # print(result1)
    fileout.write(class2 + ' ' + class1 + ' ' + value + '\n')


# print(data_array_test.shape)
for item in data_array_test:
    line = item[0]
    value = item[1]
    predict(line, value, wtVectorVPN, wtVectorVTD)



