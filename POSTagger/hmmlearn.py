import sys
import codecs

def main():
    #Read file
    corpus = readFile()
    transitionTally, emissionTally, tagCounts = parseCorpus(corpus)
    transitionMatrix, smoothedMatrix = computeTProbability(transitionTally, tagCounts)
    emissionMatrix = computeEProbability(emissionTally, tagCounts)
    writeFile(transitionMatrix, smoothedMatrix, emissionMatrix, tagCounts)
    print "Done"
   
def readFile(): 
    corpus = []
    with codecs.open(sys.argv[1], encoding='utf8') as f:
        for line in f:
            corpus.append(line)
    return corpus

def writeFile(transitionMatrix, smoothedMatrix, emissionMatrix, tagCounts):
    f = codecs.open("hmmmodel.txt","w", encoding='utf8')
    f.write("Transition Matrix\n")
    f.write(str(transitionMatrix))
    f.write("\nSmoothed Transition Matrix\n")
    f.write(str(smoothedMatrix))
    f.write("\nEmission Matrix\n")
    f.write(str(emissionMatrix))
    f.write("\nTag Counts\n")
    f.write(str(tagCounts))
    f.close()
    return

def parseCorpus(corpus):
    emissionTally = {}
    transitionTally = {}
    tagCounts = {}
    for eachLine in corpus:
        splitLine = eachLine.strip().split(" ")
        prev = "start"
        for wtPair in splitLine:
            wtPair = wtPair.rsplit('/', 1)
            #COMPUTING TRANSITION TALLY
            #For an existing tag transition
            if prev in transitionTally:
                transitionState = transitionTally[prev]
                if wtPair[1] in transitionState:
                    transitionState[wtPair[1]] += 1
                else:
                    transitionState[wtPair[1]] = 1
            #For a newly found tag transition
            else:
                transitionState = {wtPair[1] : 1}
                transitionTally[prev] = transitionState
            #Newly transitioned state
            prev = wtPair[1]
            #COMPUTING EMISSION TALLY
            #For every existing word encountered
            if wtPair[0] in emissionTally:
                possibleTags = emissionTally[wtPair[0]]
                if wtPair[1] in possibleTags:
                    possibleTags[wtPair[1]] += 1
                else:
                    possibleTags[wtPair[1]] = 1
            #For every new word encountered
            else:
                possibleTags = {wtPair[1] : 1}
                emissionTally[wtPair[0]] = possibleTags
            #Track the tags; Check if this is even needed!
            if wtPair[1] in tagCounts:
                tagCounts[wtPair[1]] += 1
            else:
                tagCounts[wtPair[1]] = 1
        #Add state 'end' to last tag
        if prev in transitionTally:
            transitionState = transitionTally[prev]
            if "end" in transitionState:
                transitionState["end"] += 1
            else:
                transitionState["end"] = 1
        else:
            transitionState = {"end" : 1}
            transitionTally[prev] = transitionState
    return transitionTally, emissionTally, tagCounts

def computeTProbability(transitionTally, tagCounts):
    transitionMatrix = {key: {key_: val_/ float(sum(val.values())) for key_, val_ in val.items()} for key, val in transitionTally.items()}
    #smoothedMatrix = {key: {key_: (val_ + 1)/ (sum(val.values()) + len(tagCounts)) for key_, val_ in val.items()} for key, val in transitionTally.items()}
    smoothedMatrix = {}
    for key, val in transitionTally.items():
        smoothVals = {}
        for key_, val_ in val.items():
            if key == "start":
                smoothVals[key_] = (val_ + 1)/ float(sum(val.values()) + len(tagCounts))
            else:
                smoothVals[key_] = (val_ + 1)/ float(sum(val.values()) + len(tagCounts) + 1)
        #Unseen transitions
        for tag in tagCounts:
            if tag not in smoothVals:
                if key == "start":
                    smoothVals[tag] = 1 / float(sum(val.values()) + len(tagCounts))
                else:
                    smoothVals[tag] = 1 / float(sum(val.values()) + len(tagCounts) + 1)
        if "end" not in smoothVals and key != "start":
            smoothVals["end"] = 1 / float(sum(val.values()) + len(tagCounts) + 1)
        smoothedMatrix[key] = smoothVals        
    return transitionMatrix, smoothedMatrix

def computeEProbability(emissionTally, tagCounts):
    emissionMatrix = {key: {key_: val_/ float(tagCounts[key_]) for key_, val_ in val.items()} for key, val in emissionTally.items()}
    return emissionMatrix
   
if __name__ == "__main__":
    main()