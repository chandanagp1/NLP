import sys
import ast
import codecs

def main():
    corpus = readFile()
    transProb, smoothtransProb, emiProb, tagCounts = readProb("hmmmodel.txt")
    viterbiDecode(corpus, smoothtransProb, emiProb, tagCounts)
    print "Done"   
    
def readFile(): 
    corpus = []
    with codecs.open(sys.argv[1], encoding='utf8') as f:
        for line in f:
            corpus.append(line.strip())
    return corpus

def writeFile(outputSentence):
    with codecs.open('hmmoutput.txt','w', encoding='utf8') as opf:
        opf.writelines(outputSentence)

def readProb(modelFile):
    f = open(modelFile)
    content = f.readlines()
    transProb = ast.literal_eval(content[1])
    smoothtransProb = ast.literal_eval(content[3])
    emiProb = ast.literal_eval(content[5])
    tagCounts = ast.literal_eval(content[7])
    return transProb, smoothtransProb, emiProb, tagCounts 


def viterbiDecode(corpus, smoothtransProb, emiProb, tagCounts):
    outputSentence = []
    for sentence in corpus:
        #Compute states in the model
        states = {0 : {"start":1.0}}
        words = sentence.strip().split(" ")
        wordCount = 1
        for word in words:
            #For a seen word
            if word in emiProb:
                states[wordCount] = emiProb[word]
            #For an unseen word
            else:
                states[wordCount] = {key : 1.0 for key in tagCounts.keys()}
            wordCount += 1
        states[wordCount] = {"end":1.0}
        #Compute probabilities of transitions
        priors = {}
        priors["start"] = 1.0
        transitions = {}
        for i in range(len(states)-1):
            currpriors = {}
            for s2, eprob in states[i+1].items():
                currpriors[s2] = 0.0
                currprior = 0.0
                for s1 in states[i].keys():
                    pprob = priors[s1]
                    tprob = smoothtransProb[s1][s2]
                    totprob = pprob * tprob * eprob
                    if totprob > currprior:
                        currprior = totprob
                        if i in transitions:
                            bckptr = transitions[i]
                        else:
                            bckptr = {}
                        bckptr[s2] = [s1, currprior]
                        transitions[i] = bckptr
                #DONOTUNCOMMENTpriors[s2] = currprior
                currpriors[s2] = currprior
            priors = currpriors
        #Call function to resolve tags
        outputSentence.append(resolveTags(transitions, sentence) + "\n")
    writeFile(outputSentence)
        
def resolveTags(transitions, sentence):
    #print transitions
    states = transitions.keys()
    words = sentence.strip().split(" ")
    outputSentence = ""
    #Pick max probability state from end
    l = len(transitions)
    endchoice = transitions[l-1]
    maxp = 0.0
    pick = ""
    s = endchoice["end"][0]
    p = endchoice["end"][1]
    if p > maxp:
        pick = s
        maxp = p
    wtpair = words[l-2] + "/" + pick + " "
    outputSentence = wtpair + outputSentence
    for i in range(l-2, 0, -1):
        tag = transitions[i][pick][0]
        pick = tag
        wtpair = words[i-1] + "/" + pick + " "
        outputSentence = wtpair + outputSentence
    return outputSentence.strip()
    
if __name__ == "__main__":
    main()