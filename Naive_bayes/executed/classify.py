import sys
import glob
import os
import collections
import re
import math

probability = dict()
priors = dict()
"""stop words in english taken from http://www.ranks.nl/stopwords"""
stop_words = ["a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along",
              "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody",
              "anyone", "anything", "anywhere", "are", "area", "areas", "aren't", "around", "as", "ask", "asked",
              "asking", "asks", "at", "away", "b", "back", "backed", "backing", "backs", "be", "became", "because",
              "become", "becomes", "been", "before", "began", "behind", "being", "beings", "below", "best", "better",
              "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "can't", "case", "cases", "certain",
              "certainly", "clear", "clearly", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different",
              "differently", "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs",
              "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly",
              "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact",
              "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully",
              "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets",
              "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group",
              "grouped", "grouping", "groups", "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
              "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him",
              "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important", "in",
              "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it", "its", "it's",
              "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l",
              "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "let's", "like", "likely",
              "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member",
              "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my",
              "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest",
              "next", "no", "nobody", "non", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "numbers",
              "o", "of", "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened",
              "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "ought", "our",
              "ours", "ourselves", "out", "over", "own", "p", "part", "parted", "parting", "parts", "per", "perhaps",
              "place", "places", "point", "pointed", "pointing", "points", "possible", "present", "presented",
              "presenting", "presents", "problem", "problems", "put", "puts", "q", "quite", "r", "rather", "really",
              "right", "room", "rooms", "s", "said", "same", "saw", "say", "says", "second", "seconds", "see", "seem",
              "seemed", "seeming", "seems", "sees", "several", "shall", "shan't", "she", "she'd", "she'll", "she's",
              "should", "shouldn't", "show", "showed", "showing", "shows", "side", "sides", "since", "small", "smaller",
              "smallest", "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still",
              "such", "sure", "t", "take", "taken", "than", "that", "that's", "the", "their", "theirs", "them",
              "themselves", "then", "there", "therefore", "there's", "these", "they", "they'd", "they'll", "they're",
              "they've", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts",
              "three", "through", "thus", "to", "today", "together", "too", "took", "toward", "turn", "turned",
              "turning", "turns", "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very",
              "w", "want", "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll",
              "wells", "went", "were", "we're", "weren't", "we've", "what", "what's", "when", "when's", "where",
              "where's", "whether", "which", "while", "who", "whole", "whom", "who's", "whose", "why", "why's", "will",
              "with", "within", "without", "won't", "work", "worked", "working", "works", "would", "wouldn't", "x", "y",
              "year", "years", "yes", "yet", "you", "you'd", "you'll", "young", "younger", "youngest", "your", "you're",
              "yours", "yourself", "yourselves", "you've", "z"]

classifier = {'PT': 'truthful positive ', 'PF': 'deceptive positive ', 'NT': 'truthful negative ',
              'NF': 'deceptive negative '}


def get_naive_bayes_model():
    lines = [line.rstrip('\n') for line in open('nbmodel.txt')]
    tempPrior = lines[0].split(' ')
    priors['PT'] = tempPrior[1]
    priors['PF'] = tempPrior[2]
    priors['NT'] = tempPrior[3]
    priors['NF'] = tempPrior[4]
    for line in lines[2:]:
        val = line.split(' ')
        Obj = dict()
        Obj[val[1]] = float(val[2])
        Obj[val[3]] = float(val[4])
        Obj[val[5]] = float(val[6])
        Obj[val[7]] = float(val[8])
        probability[val[0]] = Obj


def invoke_classifier(filename):
    fd=open(filename,'r')
    text=fd.read()
    count = {'PT': math.log(float(priors['PT'])), 'PF': math.log(float(priors['PF'])),
             'NT': math.log(float(priors['NT'])), 'NF': math.log(float(priors['NF']))}


    text = text = re.sub('[^a-zA-Z0-9]', ' ', text.replace("\n", " ").replace("\r", " ").replace("&", " and "))
    words_list = text.split(' ')

    for word in words_list:
        if check_if_valid(word):
            if not word.isupper():
                word = word.lower()
            if word in probability:
                count['PT'] = count['PT'] + probability[word]['PT']
                count['PF'] = count['PF'] + probability[word]['PF']
                count['NT'] = count['NT'] + probability[word]['NT']
                count['NF'] = count['NF'] + probability[word]['NF']
    max_val = max(count.values())
    for k, v in count.items():
        if (max_val == v):
            class_type = k;
            break
    return class_type


def check_if_valid(word):
    if not word:
        return False
    if word in stop_words:
        return False;
    return True


get_naive_bayes_model()
output_file = open('nboutput.txt', 'w')

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
# print(all_files)

test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated


for f in all_files:

  class1, class2, fold, fname = f.split('/')[-4:]

  if fold == 'fold1':
    test_by_class[class1+class2].append(f)

import json

# print('\n\n *** Test data:')
# print(test_by_class.keys(),[len(test_by_class[clas]) for clas in test_by_class.keys()])
#print(json.dumps(test_by_class, indent=2))

"""read text file"""
correct_prediction=0
number_iter=0
for category in test_by_class.keys():
        pos_neg="positive" if "positive" in category else "negative"
        decision="truthful" if "truthful" in category else "deceptive"
        correct_label=decision+' '+pos_neg+' '
        files=test_by_class[category]
        for file in files:
            class_val=invoke_classifier(file)
            #print("class_val=",classifier[class_val]+"  correct_label=",correct_label)
            output_file.write(classifier[class_val]+ file +'\n')
            if classifier[class_val]==correct_label:
                correct_prediction=correct_prediction+1

            number_iter=number_iter+1

total_files=sum([len(test_by_class[clas]) for clas in test_by_class.keys()])
accuracy=(correct_prediction/total_files)*100
print "accuracy is %f" % accuracy
print 'total_files=',total_files)
print('correct_prediction=',correct_prediction)
print('number_iter=',number_iter)

