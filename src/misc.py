import nltk

def get_raw_text(txt_file_path):

    with open(txt_file_path) as f:
        rawText = f.read()
    
    return rawText

def get_raw_text_latin(txt_file_path):
    with open(txt_file_path, encoding='latin-1') as f:
        rawText = f.read()
    
    return rawText

def tokenize_raw_text(rawText):

    sents = nltk.sent_tokenize(rawText)

    document = {'sents':[], 'tokens':[]}

    for i, sent in enumerate(sents):
        document['sents'].append({'text':sent,'tokens':[]})

        tokens = nltk.word_tokenize(sent)
        for token in tokens:
            document['sents'][i]['tokens'].append(token)
            document['tokens'].append(token)

    return document



####################################

import pickle

def save_dict(dictName, saveFilePath) -> None:

    with open(saveFilePath, 'wb') as f:
        pickle.dump(dictName, f)

def open_dict(filePath):

    with open(filePath, 'rb') as f:
        dict = pickle.load(f)

    return dict


#####
import glob
def get_file_names(folder, extention = '.p'):
    '''
    get list of file names from a folder, excluding the extention

    params:
        folder - e.g 'data/LitBank/texts/' (FROM ROOT)
        extention - e.g. '.txt'
    '''

    filePaths =  glob.glob( folder + "*" + extention )
    fileNames = []

    for filePath in filePaths:

        fileName = filePath.split('\\')[-1]
        fileName = fileName.split('.')[0]
        fileNames.append(fileName)

    return fileNames

    
# add allenNLP token indexes to the SW token dict
def token_map(tokenDictA, tokenDictSW):
    '''
    tokenDictA is the NEW token dictionary. tokenDictSW is the OLD token dictionary.
    This function adds the NEW token index to the OLD token dictionaey    
    
    '''
    keysA = list(tokenDictA.keys())
    keysSW = list(tokenDictSW.keys())

    repeat = True
    while repeat == True:
        repeat = False

        for index, (keyA,keySW) in enumerate(zip(keysA, keysSW)):
            
            # exact match
            if tokenDictA[keyA]['word'] == tokenDictSW[keySW]['word']:
                tokenDictSW[keySW]['Allen Index'] = keyA
                continue

            # difference in tokenization e.g. " vs ''
            lenKeysA = len(keysA)
            lenKeysSW = len(keysSW)

            # difference in tokenization e.g. " vs ''
            if index < lenKeysA - 1 and index < lenKeysSW - 1:
                if tokenDictSW[keysSW[index + 1]]['word'] == tokenDictA[keysA[index + 1]]['word']:
                    tokenDictSW[keySW]['Allen Index'] = keyA
                    keysA = keysA[index+1:]
                    keysSW = keysSW[index+1:]
                    repeat = True
                    
            if repeat == True:
                break

            # allenNLP has grouped stuff
            skip = 1
            repeatSkipLoop = True
            while skip < 6 and repeatSkipLoop:
                if index < lenKeysSW - skip - 1 and index < lenKeysA - 1:
                    if tokenDictA[keysA[index + 1]]['word'] == tokenDictSW[keysSW[index + 1 + skip]]['word']:
                        
                        for i in range(skip + 1):
                            tokenDictSW[keysSW[index+i]]['Notes'] = 'AllenNLP has grouped tokens'
                            tokenDictSW[keysSW[index+i]]['Allen Index'] = keyA

                        keysA = keysA[index+1:]
                        keysSW = keysSW[index+skip+1:]
                        repeat = True
                        break
                    else:
                        skip += 1

                else:
                    repeatSkipLoop = False
                    break

            if repeat == True:
                break
            
            # Story workbench has grouped stuff
            skip = 1
            repeatSkipLoop = True
            while skip < 6 and repeatSkipLoop:
                if index < lenKeysA - skip - 1 and index < lenKeysSW - 1:
                    if tokenDictSW[keysSW[index + 1]]['word'] == tokenDictA[keysA[index + 1 + skip]]['word']:

                        tokenDictSW[keySW]['Allen Index'] = []
                        tokenDictSW[keySW]['Notes'] = "Story Workbench has grouped tokens"

                        for i in range(skip+1):
                            tokenDictSW[keySW]['Allen Index'].append(keysA[index + i])

                        keysSW = keysSW[index+1:]
                        keysA = keysA[index+skip+1:]
                        repeat = True
                        break
                    else:
                        skip += 1

                else:
                    repeatSkipLoop = False
                    print('Check end of dict manually.')
                    break

            if repeat == True:
                break

    return tokenDictSW


import numpy as np
def get_ref_expressions(labChains):
    '''
    Convert jahan labelled data to list of lists, containing referring expressions only
    And character labels, and animacy labels.
    
    '''

    chainsList = labChains.split("\n")

    for i in range(len(chainsList)-1, -1, -1):
        if chainsList[i] == '':
            del chainsList[i]

    numChains = len(chainsList)

    characterLabels = np.zeros(numChains)
    animateLabels = np.zeros(numChains)
    refExpressions = []

    for i, chain in enumerate(chainsList):

        splitChain = chain.split("|")

        characterLabels[i] = splitChain[0].split("\t")[1]
        animateLabels[i] = splitChain[0].split("\t")[0]
        
        refExps = []
        for j in range(1, len(splitChain) - 1):
            refExps.append(splitChain[j].strip())

        refExpressions.append(refExps)

    return refExpressions, characterLabels, animateLabels


import json
def save_list(listName, jsonFilePath) -> None:
    with open(jsonFilePath, "w") as f:
        json.dump(listName, f)

def open_list(jsonFilePath):
    with open(jsonFilePath, "r") as fp:
        lst = json.load(fp)
    return lst



################
from collections import Counter
def scrape_char_and_animacy_labels(corefsDir, labelledChainsDir, featuresDir, caseDiff = True):
    '''
    This function goes through coreference chains in a corpus, and compares them with Jahan's labelled coreference chains.
    It maps character and animacy labels from Jahan's labelled chains, to the "new" coreference chains for the corpus.
    It does this by: going through mentions in the 'new' coreference chain. If the mention appears in one of Jahan's coreference chain,
    the 'new' coreference chain receives a vote for Jahan's corresponding label. The majority vote is taken at the end to label the 'new' coref chain.

    params:
        corefsDir - directory in which new coref chains can be found
        labelledChainsDir - dir in which Jahan's labelled coref chains can be found
        featuresDir - directory in which to save scraped char labels and animacy labels
        caseDiff- whether the Jahan labelled chains filenames have diff case to usual filename
    '''
    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
                    "hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
                    "they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" , "themselves", "this", "'s"]

    fileNames = get_file_names(corefsDir, '.p')

    for fileName in fileNames:

        if caseDiff:
            fileNameRaw = ('').join([fileName[0].upper(),fileName[1:]])
        else:
            fileNameRaw = fileName

        corefs = open_dict(corefsDir + fileName + '.p')

        # get referring expressions
        labChains = get_raw_text_latin(labelledChainsDir + fileNameRaw + '.txt')
        refExpressions, charLabelsGold, animLabelsGold = get_ref_expressions(labChains)

        # compare canonical name to 'Gold Standard' referring expressions. Get animacy label for matching GS coref chains.
        charLabelsScraped = [] # np.zeros(len(corefs['clusters']))
        animLabelsScraped = [] # np.zeros(len(corefs['clusters']))

        for i, chain in enumerate(corefs['clusters']):

            charLabelsScraped.append([])
            animLabelsScraped.append([])

            for mention in chain['mentions']:

                if mention['text'].lower().strip() in pronouns:
                    continue
            
                for j, refExps in enumerate(refExpressions):

                    for ref in refExps:  
                        if ref.lower().strip() in pronouns:
                            continue
                        
                        if mention['text'].strip() == ref.strip():
                            charLabelsScraped[i].append(charLabelsGold[j])
                            animLabelsScraped[i].append(animLabelsGold[j])

                        # elif len(mention['text'].split()) > 1:                
                        #     if ' '.join(mention['text'].split()[1:]) == refExp.strip() and ' '.join(mention['text'].split()[1:]).lower().strip() not in pronouns:
                        #         charLabelsScraped[i].append(charLabelsGold[j])
                        #         animLabelsScraped[i].append(animLabelsGold[j])

                        #     elif ' '.join(mention['text'].split()[-1:]) == refExp.strip() and ' '.join(mention['text'].split()[-1:]).lower().strip() not in pronouns:
                        #         charLabelsScraped[i].append(charLabelsGold[j])
                        #         animLabelsScraped[i].append(animLabelsGold[j])

        charLabelsScrapedVote = np.zeros(len(charLabelsScraped))
        animLabelsScrapedVote = np.zeros(len(animLabelsScraped))

        for i, (charChainLabels, animChainLabels) in enumerate(zip(charLabelsScraped, animLabelsScraped)):
            charCount = Counter(charChainLabels)
            animCount = Counter(animChainLabels)

            charMost = charCount.most_common(1)
            animMost = animCount.most_common(1)

            if charMost == []:
                continue
            else:
                charLabelsScrapedVote[i] = charMost[0][0]

            if animMost == []:
                continue
            else:
                animLabelsScrapedVote[i] = animMost[0][0]

        
        charLabelsDir = featuresDir + 'character_labels_scrape/'
        animLabelsDir = featuresDir + 'animacy_labels_scraped/'
        
        np.save(charLabelsDir + fileName, charLabelsScrapedVote)
        np.save(animLabelsDir + fileName, animLabelsScrapedVote)


def scrape_char_and_animacy_labels_new(corefsDir, labelledChainsDir, featuresDir, caseDiff = True):
    '''
    This function goes through coreference chains in a corpus, and compares them with Jahan's labelled coreference chains.
    It maps character and animacy labels from Jahan's labelled chains, to the "new" coreference chains for the corpus.
    It does this by: going through mentions in the 'new' coreference chain. If the mention appears in one of Jahan's coreference chain,
    the 'new' coreference chain receives a vote for Jahan's corresponding label. The majority vote is taken at the end to label the 'new' coref chain.

    params:
        corefsDir - directory in which new coref chains can be found
        labelledChainsDir - dir in which Jahan's labelled coref chains can be found
        featuresDir - directory in which to save scraped char labels and animacy labels
        caseDiff- whether the Jahan labelled chains filenames have diff case to usual filename
    '''
    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
                    "hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
                    "they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" , "themselves", "this", "'s"]

    fileNames = get_file_names(corefsDir, '.p')

    for fileName in fileNames:

        if caseDiff:
            fileNameRaw = ('').join([fileName[0].upper(),fileName[1:]])
        else:
            fileNameRaw = fileName

        corefs = open_dict(corefsDir + fileName + '.p')

        # get referring expressions
        labChains = get_raw_text_latin(labelledChainsDir + fileNameRaw + '.txt')
        refExpressions, charLabelsGold, animLabelsGold = get_ref_expressions(labChains)

        # compare canonical name to 'Gold Standard' referring expressions. Get animacy label for matching GS coref chains.
        charLabelsScraped = [] # np.zeros(len(corefs['clusters']))
        animLabelsScraped = [] # np.zeros(len(corefs['clusters']))

        for i, chain in enumerate(corefs['clusters']):
            


            charLabelsScraped.append([])
            animLabelsScraped.append([])

            # if canon name is None or 's mark as not a character
            if chain['name'] == None or chain['name'].strip() == "'s":
                charLabelsScraped[i].append(0)
                animLabelsScraped[i].append(0)
                continue

            canonName = chain['name'].strip()

            # remove " 's" from end of canon name
            if len(canonName) > 3:
                if canonName[-3:] == " 's":
                    canonName = canonName[:-3]


            for j, refExps in enumerate(refExpressions):

                for ref in refExps:  
                    if ref.lower().strip() in pronouns:
                        continue
                    
                    if canonName == ref.strip():
                        charLabelsScraped[i].append(charLabelsGold[j])
                        animLabelsScraped[i].append(animLabelsGold[j])


        charLabelsScrapedVote = np.zeros(len(charLabelsScraped))
        animLabelsScrapedVote = np.zeros(len(animLabelsScraped))

        for i, (charChainLabels, animChainLabels) in enumerate(zip(charLabelsScraped, animLabelsScraped)):
            charCount = Counter(charChainLabels)
            animCount = Counter(animChainLabels)

            charMost = charCount.most_common(1)
            animMost = animCount.most_common(1)

            if charMost == []:
                continue
            else:
                charLabelsScrapedVote[i] = charMost[0][0]

            if animMost == []:
                continue
            else:
                animLabelsScrapedVote[i] = animMost[0][0]

        
        charLabelsDir = featuresDir + 'character_labels_scraped_new/'
        animLabelsDir = featuresDir + 'animacy_labels_scraped_new/'
        
        np.save(charLabelsDir + fileName, charLabelsScrapedVote)
        np.save(animLabelsDir + fileName, animLabelsScrapedVote)