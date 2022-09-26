from turtle import position
import numpy as np
import requests
from nltk.corpus import wordnet as wn
from misc import token_map


# feature 1
def SRL_feat(srlParses, sentences, corefChains):

    arg0Positions = get_arg0_positions(srlParses, sentences)

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):
        for mention in corefChain['mentions']:
            for arg0Position in arg0Positions:
                if mention['position'] == arg0Position:
                    feature[i] += 1
                    break

    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))

def get_arg0_positions(srlParses, sentences):
    positions = []

    tokenOffset = 0

    for i, srlParse in enumerate(srlParses):
        
        for verb in srlParse['verbs']:
            for j, tag in enumerate(verb['tags']):
                if tag == 'B-ARG0':
                    count = 0

                    currIndex = j
    
                    while currIndex < len(verb['tags']) - 2: 
                        currIndex += 1
                        if verb['tags'][currIndex] != 'I-ARG0':
                            break
                        count+= 1

                    positions.append([j+tokenOffset, j+count+tokenOffset])                    

        tokenOffset += len(sentences[i]['tokens'])


    return positions

# feature 2

def dep_feat(depParses, sentences, corefChains):

    nsubjPositions = get_nsubj_positions(depParses, sentences)

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):
        for mention in corefChain['mentions']:
            if mention['position'] == []:
                continue

            for nsubjPosition in nsubjPositions:

                if nsubjPosition == mention['position'][-1]:
                    feature[i] += 1
                    break
    
    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))

def get_nsubj_positions(depParses, sentences):
    positions = []

    tokenOffset = 0

    for i, depParse in enumerate(depParses):

        for j, label in enumerate(depParse['predicted_dependencies']):
            if label == 'nsubj':
                positions.append(j + tokenOffset)

        tokenOffset += len(sentences[i]['tokens'])
        
    return positions

# feature 3

def ner_feat(nerParses, sentences, corefChains):

    perPositions = get_PER_positions(nerParses, sentences)

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):
        for mention in corefChain['mentions']:
            if mention['position'] == []:
                continue

            for perPosition in perPositions:
                if perPosition == mention['position'][-1]:
                    feature[i] += 1
                    break
    
    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))

def get_PER_positions(nerParses, sentences):
    positions = []

    tokenOffset = 0

    for i, nerParse in enumerate(nerParses):

        for j, tag in enumerate(nerParse['tags']):
            if 'PER' in tag:
                positions.append(j + tokenOffset)

        tokenOffset += len(sentences[i]['tokens'])
        
    return positions

# feature 4


def openie_feat(openieParses, sentences, corefChains):
    triplePositions = get_triple_subject_positions(openieParses, sentences)
    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):
        for mention in corefChain['mentions']:
            for position in triplePositions:
                if mention['position'] == position:
                    feature[i] += 1
                    break

    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))

def get_triple_subject_positions(openieParses, sentences):

    positions = []
    offset = 0
    for i, parse in enumerate(openieParses):

        for j, verb in enumerate(parse['verbs']):

            tags = []
            for tag in verb['tags']:
                tags.append(tag.split('-')[-1])
                tags = list(set(tags))

            for j in range(len(tags)-1, -1, -1):
                if tags[j] == 'O':
                    del tags[j]
            
            if len(tags) < 3:
                continue

            for j in range(0,5):
                if 'ARG' + str(j) in tags:
                    subjTag = 'ARG' + str(j)
                    break

            for k, tag in enumerate(verb['tags']):
                if tag == 'B-'+subjTag:

                    count = 1

                    while k + count < len(verb['tags']):
                        if verb['tags'][k + count] != 'I-' + subjTag:
                            break
                        count += 1

                    positions.append([k + offset, k + offset + count -1])
                    break

        offset += len(sentences[i]['tokens'])

    return positions



# feature 5
def len_feat(corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):
        feature[i] = len(corefChain['mentions'])

    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))
    
# feature 6
def CN_feat(corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):

        name = corefChain['name']

        if name == None:
            feature[i] = 0.
            continue

        nameHead = name.split()[-1]

        response = requests.get("https://api.conceptnet.io/c/en/" + nameHead).json()
        edges = response['edges']

        for edge in edges:
            if 'person' in edge['@id']:
                feature[i] = 1.
                break

    return feature


# feature 7
def WN_feat(corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):

        name = corefChain['name']

        if name == None:
            feature[i] = 0.
            break

        word = name.split()[-1]
        synsets = wn.synsets(word, pos=wn.NOUN)

        while synsets != []:
            synset = synsets[0]

            if synset == wn.synset('person.n.01'):
                feature[i] = 1.
                break

            synsets = synset.hypernyms()

    return feature


# feature 8
def disp_feat(corefChains, document, n=20):

    counts = np.zeros((len(corefChains['clusters']), n))

    textLength = len(document['tokens'])


    tokenIndexArrays = np.array_split(list(range(textLength)), n)

    for i, corefChain in enumerate(corefChains['clusters']):

        for mention in corefChain['mentions']:

            for j, chunk in enumerate(tokenIndexArrays):
                
                if len(mention['position']) == 0:
                    continue
                if mention['position'][0] in chunk:

                    counts[i][j] += 1


    means = np.mean(counts, axis=1)

    stds = np.std(counts, axis = 1)


    return  1 - (stds / means) / np.sqrt(n - 1)

# Quotation Attribution 

def QU_feat(corefs, ann, tokenized):
    '''
    feature is notmalized number of quotes which have been attributed to a coreference chain
    '''

    ### get ann token dict, with corresponding allenNLP token indices
    annTokenDict = {}
    offset = 0

    for sent in ann['sentences']:


        for token in sent['tokens']:
            annTokenDict[ token['index'] + offset - 1] = {'word':token['word']}


        offset += len(sent['tokens'])

    allenTokenDict = {}
    for i, token in enumerate(tokenized['tokens']):
        allenTokenDict[i] = {'word':token}

    annTokenDict = token_map(allenTokenDict, annTokenDict)

    ### get indexes of speakers
    speakerIndexes = get_speakers_indexes(ann, annTokenDict)


    ### construct feature by matching speaker indexes with coref chain indexes
    feature = np.zeros(len(corefs['clusters']))
    for i, chain in enumerate(corefs['clusters']):
        for mention in chain['mentions']:

            if len(mention['position']) == 0:
                continue

            for index in speakerIndexes:

                for position in range(mention['position'][0], mention['position'][-1] + 1):

                    if position in list(range(index[0], index[-1] + 1)):
                        feature[i] += 1
                        break

    if np.std(feature) == 0:
        return np.zeros(len(corefs['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))
    # return feature

def get_speakers_indexes(ann, annTokenDict):
    speakersIndexes = []

    for quote in ann['quotes']:
        if 'mention' in quote.keys():

            
            indexStart = quote['mentionBegin']
            indexEnd = quote['mentionEnd']

            allenIndexStart = 0
            allenIndexEnd = 0

            for i in range(5):

                if 'Allen Index' in annTokenDict[indexStart - i].keys():
                    allenIndexStart = annTokenDict[indexStart - i]['Allen Index']
                    if type(allenIndexStart) == list:
                        allenIndexStart = allenIndexStart[0]
                    break
            
            for i in range(5):
                if 'Allen Index' in annTokenDict[indexEnd + i].keys():
                    allenIndexEnd = annTokenDict[indexEnd + i]['Allen Index']

                    if type(allenIndexEnd) == list:
                        allenIndexEnd = allenIndexEnd[-1]
                    break

            

            speakersIndexes.append([allenIndexStart, allenIndexEnd])
    
    return 
    

# constituency parse feature


def const_feat(constParses, sentences, corefChains):

    positions = get_arg0s_const(constParses, sentences)

    feature = np.zeros(len(corefChains['clusters'])) 

    for i, corefChain in enumerate(corefChains['clusters']):
        for mention in corefChain['mentions']:
            for position in positions:
                if mention['position'] == position:
                    feature[i] += 1
                    break

    if np.std(feature) == 0:
        return np.zeros(len(corefChains['clusters']))
    
    return (feature - np.mean(feature)) / (np.std(feature))

def get_arg0s_const(constParses, sentences):
    positions = []

    tokenOffset = 0

    for i, parse in enumerate(constParses):
        

        for verb in parse['verbs']:
            for j, tag in enumerate(verb['tags']):
                if tag == 'B-ARG0':
                    count = 0

                    currIndex = j

                    while currIndex < len(verb['tags']) - 2:
                        currIndex += 1
                        if verb['tags'][currIndex] != 'I-ARG0':
                            break
                        count+= 1

                    positions.append([j+tokenOffset, j+count+tokenOffset])

        tokenOffset += len(sentences[i]['tokens'])

    return positions
