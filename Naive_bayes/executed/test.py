import sys
import glob
import os
import collections
import re
import math
words_dict = dict()
word_count_category = {'total':0,'PT':0,'PF':0,'NT':0,'NF':0}
probability = dict()
label_dict = dict()
count_category = {'PT':0,'PF':0,'NT':0,'NF':0}

"""stop words in english taken from http://www.ranks.nl/stopwords"""
stop_words = ["a","about","above","across","after","again","against","all","almost","alone","along","already","also","although","always","am","among","an","and","another","any","anybody","anyone","anything","anywhere","are","area","areas","aren't","around","as","ask","asked","asking","asks","at","away","b","back","backed","backing","backs","be","became","because","become","becomes","been","before","began","behind","being","beings","below","best","better","between","big","both","but","by","c","came","can","cannot","can't","case","cases","certain","certainly","clear","clearly","come","could","couldn't","d","did","didn't","differ","different","differently","do","does","doesn't","doing","done","don't","down","downed","downing","downs","during","e","each","early","either","end","ended","ending","ends","enough","even","evenly","ever","every","everybody","everyone","everything","everywhere","f","face","faces","fact","facts","far","felt","few","find","finds","first","for","four","from","full","fully","further","furthered","furthering","furthers","g","gave","general","generally","get","gets","give","given","gives","go","going","good","goods","got","great","greater","greatest","group","grouped","grouping","groups","h","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","her","here","here's","hers","herself","he's","high","higher","highest","him","himself","his","how","however","how's","i","i'd","if","i'll","i'm","important","in","interest","interested","interesting","interests","into","is","isn't","it","its","it's","itself","i've","j","just","k","keep","keeps","kind","knew","know","known","knows","l","large","largely","last","later","latest","least","less","let","lets","let's","like","likely","long","longer","longest","m","made","make","making","man","many","may","me","member","members","men","might","more","most","mostly","mr","mrs","much","must","mustn't","my","myself","n","necessary","need","needed","needing","needs","never","new","newer","newest","next","no","nobody","non","noone","nor","not","nothing","now","nowhere","number","numbers","o","of","off","often","old","older","oldest","on","once","one","only","open","opened","opening","opens","or","order","ordered","ordering","orders","other","others","ought","our","ours","ourselves","out","over","own","p","part","parted","parting","parts","per","perhaps","place","places","point","pointed","pointing","points","possible","present","presented","presenting","presents","problem","problems","put","puts","q","quite","r","rather","really","right","room","rooms","s","said","same","saw","say","says","second","seconds","see","seem","seemed","seeming","seems","sees","several","shall","shan't","she","she'd","she'll","she's","should","shouldn't","show","showed","showing","shows","side","sides","since","small","smaller","smallest","so","some","somebody","someone","something","somewhere","state","states","still","such","sure","t","take","taken","than","that","that's","the","their","theirs","them","themselves","then","there","therefore","there's","these","they","they'd","they'll","they're","they've","thing","things","think","thinks","this","those","though","thought","thoughts","three","through","thus","to","today","together","too","took","toward","turn","turned","turning","turns","two","u","under","until","up","upon","us","use","used","uses","v","very","w","want","wanted","wanting","wants","was","wasn't","way","ways","we","we'd","well","we'll","wells","went","were","we're","weren't","we've","what","what's","when","when's","where","where's","whether","which","while","who","whole","whom","who's","whose","why","why's","will","with","within","without","won't","work","worked","working","works","would","wouldn't","x","y","year","years","yes","yet","you","you'd","you'll","young","younger","youngest","your","you're","yours","yourself","yourselves","you've","z"]

def check_if_valid_word(word):
    if not word:
        return False
    elif word in stop_words:
        return False;
    else:
        return True

def bayes_model(filename,pol,dec):
    fd=open(filename,'r')
    text=fd.read()
    text = re.sub('[^a-zA-Z0-9]', ' ', text.replace("\n", " ").replace("\r", " ").replace("&", " and "))
    truth_flag = True if decision == 'truthful' else False
    positive_flag = True if pos_neg == 'positive' else False
    words_list = text.split(' ')
    class_type = ''
    if positive_flag and truth_flag:
      class_type = 'PT'
      count_category[class_type] += 1
    if positive_flag and not truth_flag:
      class_type = 'PF'
      count_category[class_type] += 1
    if not positive_flag and truth_flag:
      class_type = 'NT'
      count_category[class_type] += 1
    if not positive_flag and not truth_flag:
      class_type = 'NF'
      count_category[class_type] += 1
    for word in words_list:
        if check_if_valid_word(word):
            wordObj =  {'PT':0,'PF':0,'NT':0,'NF':0}
            if not word.isupper():
                word = word.lower()
            if word in words_dict:
                wordObj = words_dict[word]
            else:
                word_count_category['total'] = word_count_category['total'] + 1

            word_count_category[class_type] = word_count_category[class_type] + 1
            wordObj[class_type] = wordObj[class_type] + 1
            words_dict[word] = wordObj

    fd.close()

def do_smoothing():
  for key in words_dict:
    obj = words_dict[key]
    wordObj = dict()
    wordObj['PT'] = math.log(obj['PT'] + 1) - math.log(word_count_category['PT'] + word_count_category['total'])
    wordObj['PF'] = math.log(obj['PF'] + 1) - math.log(word_count_category['PF'] + word_count_category['total'])
    wordObj['NT'] = math.log(obj['NT'] + 1) - math.log(word_count_category['NT'] + word_count_category['total'])
    wordObj['NF'] = math.log(obj['NF'] + 1) - math.log(word_count_category['NF'] + word_count_category['total'])
    probability[key] = wordObj

def get_model_file():
  model_file = open('nbmodel.txt', 'w')
  model_file.write("PRIORS " + str(count_category['PT'] / float(total_reviews)) + ' ' + str(
    count_category['PF'] / float(total_reviews)) + ' ' + str(count_category['NT'] / float(total_reviews)) + ' ' + str(
    count_category['NF'] / float(total_reviews)) + ' \n');
  model_file.write('\n')

  for key in probability:
    strToPrint = key + ' PT ' + str(probability[key]['PT']) + ' PF ' + str(probability[key]['PF'])
    strToPrint = strToPrint + ' NT ' + str(probability[key]['NT']) + ' NF ' + str(probability[key]['NF'])
    model_file.write(strToPrint + '\n')

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
print(all_files)

test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
train_by_class = collections.defaultdict(list)

for f in all_files:

  class1, class2, fold, fname = f.split('/')[-4:]
  train_by_class[class1 + class2].append(f)
  if fold == 'fold1':
    test_by_class[class1+class2].append(f)
  else:
    train_by_class[class1+class2].append(f)



import json

print('\n\n *** Test data:')
# print(test_by_class.keys(),[len(test_by_class[clas]) for clas in test_by_class.keys()])
print(json.dumps(test_by_class, indent=2))
print('\n\n *** Train data:')
# print(train_by_class.keys(),[len(train_by_class[clas]) for clas in train_by_class.keys()])
print(json.dumps(train_by_class, indent=2))

total_reviews=sum([len(train_by_class[clas]) for clas in train_by_class.keys()])

for category in train_by_class.keys():
        pos_neg="positive" if "positive" in category else "negative"
        decision="truthful" if "truthful" in category else "deceptive"
        files=train_by_class[category]
        for file in files:
            bayes_model(file, pos_neg, decision)

do_smoothing()
get_model_file()