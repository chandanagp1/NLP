'''
@author: chandanagp
learn class file for learning of  vanilla_perceptron and averaged_perceptron classifier.
uses clean text and tokenization for preprocessing
creates a numpy 2d-array and a probability dictionary for learning
'''
import sys
import glob
import os
import collections
import numpy as np
import string
import re
import math
from numpy import random

vanilla_vocab = {}
average_vocab = {}
data_array = np.empty((0, 2))

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
train_by_class = collections.defaultdict(list)

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    train_by_class[class1 + class2].append(f)


# Clean the data for tokeniation

def cleanText(line):
    cleaned_str = re.sub('[^a-z\s]+', ' ', line, flags=re.IGNORECASE)
    text = cleaned_str.lower()
    return text


''' tokenize the data by creating wordcount for each class
For each word, add to class it belongs to, if it isn’t already there, and update the number of counts.
Also add that word to the global vocabulary.'''


def toeknizeText(line, worddict):
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
                 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'doing', 'it',
                 'how', 'further', 'was', 'here', 'than', "th", "rd", "st", "nd", "to", "yep", "but", "not", "one",
                 "two", "three", "first", "second", "third"]

    for word in line.split():
        if word not in stopwords:
            if word not in worddict:
                worddict[word] = 0
    return worddict


'''
Creates feature dictionary for each document, called by vanilla and averaged.
'''


def createFeaturedict(line):
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
            if word not in worddict:
                worddict[word] = 1
            else:
                worddict[word] += 1
    return worddict


'''
Averaged perceptron : takes the data_array and weight vectors for each class and computes weight of each words to update the weight Vectors
Returns : Weight vectors of each class and bias value
'''


def averaged(data_array, wtVectorAPN, wtVectorATD):
    numIter = 80
    uVector1 = dict(wtVectorAPN)
    uVector2 = dict(wtVectorATD)
    bias1 = 0.0
    bias2 = 0.0
    beta1 = 0.0
    beta2 = 0.0
    act1 = 0.0
    act2 = 0.0
    c = 1

    for j in range(0, numIter):

        # shuffle the data before each iteration
        random.seed(j)
        random.shuffle(data_array)
        for i in range(0, len(data_array)):

            featureVector = createFeaturedict(data_array[i][0])
            # print(len(featureVector))
            reviewLablel1 = 0
            reviewLablel2 = 0
            # positive_deceptive : 0, positive_truthful : 1,
            # negative_truthful : 2 , negative_deceptive : -2
            # pos_neg
            if data_array[i][1] == '0' or data_array[i][1] == '1':
                reviewLablel1 = 1

            elif data_array[i][1] == '2' or data_array[i][1] == '-2':
                reviewLablel1 = -1
            # Truth_deceptive
            if data_array[i][1] == '2' or data_array[i][1] == '1':
                reviewLablel2 = 1
            elif data_array[i][1] == '0' or data_array[i][1] == '-2':
                reviewLablel2 = -1

            for feature, count in featureVector.items():
                act1 = act1 + (wtVectorAPN[feature] * count)
                act2 = act2 + (wtVectorATD[feature] * count)
            act1 += bias1
            act2 += bias2
            if reviewLablel1 * act1 <= 0:
                for feature, count in featureVector.items():
                    wtVectorAPN[feature] += reviewLablel1 * count
                    uVector1[feature] += reviewLablel1 * c * count
                bias1 += reviewLablel1
                beta1 += reviewLablel1 * c
            act1 = 0
            if reviewLablel2 * act2 <= 0:
                for feature, count in featureVector.items():
                    wtVectorATD[feature] += reviewLablel2 * count
                    uVector2[feature] += reviewLablel2 * c * count
                bias2 += reviewLablel2
                beta2 += reviewLablel2 * c
            act2 = 0
            c += 1
    for feature in wtVectorVPN:
        wtVectorAPN[feature] = wtVectorAPN[feature] - (1 / float(c) * uVector1[feature])
    for feature in wtVectorATD:
        wtVectorATD[feature] = wtVectorATD[feature] - (1 / float(c) * uVector2[feature])
    bias1 = bias1 - ((1 / float(c)) * beta1)
    bias2 = bias2 - ((1 / float(c)) * beta2)

    return wtVectorAPN, wtVectorATD, bias1, bias2


'''
Vanilla perceptron : takes the data_array and weight vectors for each class and computes weight of each words to update the weight Vectors
Returns : Weight vectors of each class and bias value
'''


def vanilla(data_array, wtVectorVPN, wtVectorVTD):
    numIter = 80
    bias1 = 0
    bias2 = 0
    act1 = 0
    act2 = 0

    for j in range(0, numIter):
        for i in range(0, len(data_array)):
            featureVector = createFeaturedict(data_array[i][0])
            # print(len(featureVector))
            reviewLablel1 = 0
            reviewLablel2 = 0
            # positive_deceptive : 0, positive_truthful : 1,
            # negative_truthful : 2 , negative_deceptive : -2
            # pos_neg
            if data_array[i][1] == '0' or data_array[i][1] == '1':
                reviewLablel1 = 1

            elif data_array[i][1] == '2' or data_array[i][1] == '-2':
                reviewLablel1 = -1
            # Truth_deceptive
            if data_array[i][1] == '2' or data_array[i][1] == '1':
                reviewLablel2 = 1
            elif data_array[i][1] == '0' or data_array[i][1] == '-2':
                reviewLablel2 = -1

            for word, count in featureVector.items():
                act1 = act1 + (wtVectorVPN[word] * count)
                act2 = act2 + (wtVectorVTD[word] * count)
            act1 += bias1
            act2 += bias2
            # print(actposneg)
            # print(actdectru)
            if reviewLablel1 * act1 <= 0:
                for word, count in featureVector.items():
                    wtVectorVPN[word] = wtVectorVPN[word] + (count * reviewLablel1)
                bias1 = bias1 + reviewLablel1
            act1 = 0

            if reviewLablel2 * act2 <= 0:
                for word, count in featureVector.items():
                    wtVectorVTD[word] = wtVectorVTD[word] + (count * reviewLablel2)
                bias2 = bias2 + reviewLablel2
            act2 = 0
    return wtVectorVPN, wtVectorVTD, bias1, bias2


# take each key from train data - gives you polarity and truthfullness
for key, value in train_by_class.items():
    # iterate through each key and get polarity to create numpy array : positive_deceptive : 0, positive_truthful :1,
    # negative_truthful : 2 , negative_deceptive : -2
    # print(key)
    if key == "positive_polaritydeceptive_from_MTurk":
        for file in value:
            # remove these two lines later for submission
            # file = file[1:]
            # file = "F:\\NLP\\Assignment1\\op_spam_training_data"+file

            text_file = open(file, "r")
            line = text_file.read()
            line = cleanText(line)
            # tokenize data,For each (document, label) pair, tokenize the document into words.
            vanilla_vocab = toeknizeText(line, vanilla_vocab)
            average_vocab = toeknizeText(line, average_vocab)
            data_array = np.append(data_array, np.array([[str(line), 0]]), axis=0)

    if key == "negative_polaritytruthful_from_Web":
        for file in value:
            # remove these two lines later for submission
            # file = file[1:]
            # file = "F:\\NLP\\Assignment1\\op_spam_training_data"+file

            text_file = open(file, "r")
            line = text_file.read()
            line = cleanText(line)
            # tokenize data,For each (document, label) pair, tokenize the document into words.
            vanilla_vocab = toeknizeText(line, vanilla_vocab)
            average_vocab = toeknizeText(line, average_vocab)
            data_array = np.append(data_array, np.array([[str(line), 2]]), axis=0)
    if key == "positive_polaritytruthful_from_TripAdvisor":
        for file in value:
            # remove these two lines later for submission
            # file = file[1:]
            # file = "F:\\NLP\\Assignment1\\op_spam_training_data"+file

            text_file = open(file, "r")
            line = text_file.read()
            line = cleanText(line)
            # tokenize data,For each (document, label) pair, tokenize the document into words.
            vanilla_vocab = toeknizeText(line, vanilla_vocab)
            average_vocab = toeknizeText(line, average_vocab)
            data_array = np.append(data_array, np.array([[str(line), 1]]), axis=0)
    if key == "negative_polaritydeceptive_from_MTurk":
        for file in value:
            # remove these two lines later for submission
            # file = file[1:]
            # file = "F:\\NLP\\Assignment1\\op_spam_training_data"+file

            text_file = open(file, "r")
            line = text_file.read()
            line = cleanText(line)
            # tokenize data,For each (document, label) pair, tokenize the document into words.
            vanilla_vocab = toeknizeText(line, vanilla_vocab)
            average_vocab = toeknizeText(line, average_vocab)
            data_array = np.append(data_array, np.array([[str(line), -2]]), axis=0)

# data_array is the numpy array which contains text and class that the text belongs to.
# print("Data read and numpy array created")

# print("Computing vanilla weights :")
wtVectorVPN = dict(vanilla_vocab)
wtVectorVTD = dict(vanilla_vocab)
wtVectorVPN, wtVectorVTD, bias1, bias2 = vanilla(data_array, wtVectorVPN, wtVectorVTD)

# print(bias1)
# print(bias2)


wtVectorAPN = dict(average_vocab)
wtVectorATD = dict(average_vocab)
wtVectorAPN, wtVectorATD, biasA1, biasA2 = averaged(data_array, wtVectorAPN, wtVectorATD)
# print(biasA1)
# print(biasA2)

os.chdir(sys.path[0])
with open("vanillamodel.txt", "w") as file:
    file.write(str(wtVectorVPN))
    file.write("\n")
    file.write(str(wtVectorVTD))
    file.write("\n")
    file.write(str(bias1))
    file.write("\n")
    file.write(str(bias2))
file.close()
with open("averagedmodel.txt", "w") as file:
    file.write(str(wtVectorAPN))
    file.write("\n")
    file.write(str(wtVectorATD))
    file.write("\n")
    file.write(str(biasA1))
    file.write("\n")
    file.write(str(biasA2))
file.close()




