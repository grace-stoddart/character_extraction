import numpy as np


# feat 1
def SRL_feat_original(srlParses, corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    semanticSubjects = get_semantic_subject_words(srlParses)

    for i, corefChain in enumerate(corefChains['clusters']):

        chainHeadOfHead = get_chain_head_of_head(corefChain).lower().strip()

        for semanticSubject in semanticSubjects:

            if semanticSubject.split(' ')[-1].lower().strip() == chainHeadOfHead:
                feature[i] = 1.
                break

    return feature

def get_semantic_subject_words(srlParses):
    '''
    Get list of referring expressions which appear as ARG0 to a verb. List doesn't contain duplicates.
    '''

    semanticSubjects = []

    for parse in srlParses['parses']:

        for verb in parse['verbs']:            
        
            for i, (tag, word) in enumerate(zip(verb['tags'], parse['words'])):

                if tag == 'B-ARG0':
                    refExp = word

                    for j in range(1, 10):
                        if (i+j) >= len(verb['tags']) or verb['tags'][i+j] != 'I-ARG0':
                            break

                        # if parse['words'][i+j][0] != "'" and parse['words'][i+j][0] != ",":
                        #     refExp += " "
                        refExp += " "
                        refExp += parse['words'][i+j]

                    semanticSubjects.append(refExp)

    return list(set(semanticSubjects))


### feature 2
def dep_feat_original(depParses, corefChains):

    feature = np.zeros(len(corefChains['clusters']))
    dependencySubjs = get_dependency_subjs(depParses)

    for i, corefChain in enumerate(corefChains['clusters']):

        chainHead = get_chain_head(corefChain).lower().strip()

        for dependencySubj in dependencySubjs:
            if dependencySubj.strip().lower() in chainHead:
                feature[i] = 1.
                break

    return feature

def get_dependency_subjs(depParses):

    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
				"hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "themselves", "your", "my", "it",
				"they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]
    
    dependencySubjs = []

    for parse in depParses['parses']:

        for word, label in zip(parse['words'], parse['predicted_dependencies']):

            if label == 'nsubj' and word not in pronouns:
                dependencySubjs.append(word)

    return list(set(dependencySubjs))



### feature 3


def ner_feat_original(nerParses, corefChains):

    feature = np.zeros(len(corefChains['clusters']))
    people = get_people(nerParses)   

    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
            "hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "themselves", "your", "my", "it",
            "they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]


    for i, corefChain in enumerate(corefChains['clusters']):

        chainHead = get_chain_head(corefChain).lower().strip()

        for person in people:
            if person.lower().strip() in pronouns:
                break

            if person.lower().strip() in chainHead:
                feature[i] = 1.
                break

    return feature

def get_people(nerParses):

    people = []

    for parse in nerParses['parses']:

        for i, (word, label) in enumerate(zip(parse['words'], parse['tags'])):

            if label == 'U-PER':
                people.append(word)

            
            elif label == 'B-PER':
                person = word

                for j in range(1,10):
                    if (i+j) >= len(parse['tags']) or 'PER' not in parse['tags'][i+j]:
                            break

                    person += ' '
                    person += parse['words'][i+j]

                people.append(person)

    return list(set(people))



### feature 4
def openie_feat_original(openieParses, corefChains):

    feature = np.zeros(len(corefChains['clusters']))
    tripleSubjs = get_triple_subjects(openieParses)

    for i, corefChain in enumerate(corefChains['clusters']):

        chainHeadOfHead = get_chain_head_of_head(corefChain).lower().strip()

        for tripleSubj in tripleSubjs:

            if chainHeadOfHead in tripleSubj.lower():
                feature[i] = 1.
                break

    return feature


def get_triple_subjects(openieParses):

    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
    "hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "themselves", "your", "my", "it",
    "they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]

    tripleSubjs = []

    for parse in openieParses['parses']:
        for verb in parse['verbs']:
            tags = []
            for tag in verb['tags']:
                tags.append(tag.split('-')[-1])
            tags = list(set(tags))

            for i in range(len(tags)-1, -1, -1):
                if tags[i] == 'O':
                    del tags[i]

            if len(tags) < 3:
                continue

            for i in range(0,5):
                if 'ARG' + str(i) in tags:
                    subjTag = 'ARG' + str(i)
                    break

            for j, (tag, word) in enumerate(zip(verb['tags'], parse['words'])):

                if tag == 'B-'+subjTag:
                    refExp = word

                    for k in range(1, 10):
                        if (j+k) >= len(verb['tags']):
                            break
                        
                        if verb['tags'][j+k] != 'I-' + subjTag:
                            break

                        # if parse['words'][i+j][0] != "'" and parse['words'][i+j][0] != ",":
                        #     refExp += " "
                        refExp += " "
                        refExp += parse['words'][j+k]
                    
                    if refExp.lower() not in pronouns:
                        tripleSubjs.append(refExp)
    
    return list(set(tripleSubjs))



### feature 5
import requests

def CN_feat_original(corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):

        chainHeadOfHead = get_chain_head_of_head(corefChain).lower().strip()

        response = requests.get("https://api.conceptnet.io/c/en/" + chainHeadOfHead).json()
        edges = response['edges']

        for j in range(len(edges)):
            if 'person' in edges[j]['@id']:
                feature[i] = 1.
                break

    return feature


### feature 6
from nltk.corpus import wordnet as wn

def WN_feat_original(corefChains):

    feature = np.zeros(len(corefChains['clusters']))

    for i, corefChain in enumerate(corefChains['clusters']):

        chainHeadOfHead = get_chain_head_of_head(corefChain).lower().strip()

        synsets = wn.synsets(chainHeadOfHead, pos=wn.NOUN)

        while synsets != []:

            synset = synsets[0]

            if synset == wn.synset('person.n.01'):
                feature[i] = 1
                break  

            synsets = synset.hypernyms() 

    return feature


############### features using old, corenlp parses

def dep_feat_coreNLP(ann, corefs):
    '''
    dependencies - list of words which appear as verb subjects according to enhanced ++ dependencies, foudn using CoreNLP
    corefs - usual corefs dict
    '''

    dependencies = get_dependencies_list_core(ann)

    feature = np.zeros(len(corefs['clusters']))

    for i, corefChain in enumerate(corefs['clusters']):

        chainHead = get_chain_head(corefChain).lower().strip()

        for dependencySubj in dependencies:
            if dependencySubj.strip().lower() in chainHead:
                feature[i] = 1.
                break

    return feature

def get_dependencies_list_core(ann):

    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
				"hers",  "themselves", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
				"they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]
    
    dependencies = []

    for sentence in ann["sentences"]:
        # get enhanced ++ dependencies
        eppdp = sentence["enhancedPlusPlusDependencies"]

        for edge in eppdp:
            if edge['dep'] == 'nsubj':
                isPronoun = False
                for pronoun in pronouns:
                    if edge['dependentGloss'].strip().lower() == pronoun:
                        isPronoun = True
                        break
                
                if isPronoun == False:
                    dependencies.append(edge['dependentGloss'])
                    
    dependenciesUnique = list(set(list(dependencies)))

    return dependenciesUnique

### NE Feature from CoreNLP parse
def ner_feat_coreNLP(ann, corefs):
    '''
    corefs - usual corefs dict
    ann - CoreNLP parse, with NER annotations
    '''

    feature = np.zeros(len(corefs['clusters']))

    perList = get_PER_list_core(ann)

    for i, corefChain in enumerate(corefs['clusters']):

        chainHead = get_chain_head(corefChain).lower().strip()

        for per in perList:
            if per.strip().lower() in chainHead:
                feature[i] = 1.
                break

    return feature

def get_PER_list_core(ann):
    perList = []

    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
				"hers", "ours", "myself", "themselves", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
				"they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]

    for sentence in ann['sentences']:
        for entityMention in sentence['entitymentions']:
            if 'PERSON' in entityMention['ner']:
                if entityMention['text'].lower().strip() not in pronouns:
                    perList.append(entityMention['text'])
    
    return list(set(perList))



### Triple feature from CoreNLP Parse
def openie_feat_coreNLP(ann, corefs):
    '''
    triples - list of text appearing as subject in a triple, according to CoreNLP OpenIE parse
    corefs - usual coref dict
    '''
    tripleSubjs = get_triple_subjects_core(ann)

    feature = np.zeros(len(corefs['clusters']))

    for i, corefChain in enumerate(corefs['clusters']):

        chainHeadOfHead = get_chain_head_of_head(corefChain).lower().strip()

        for tripleSubj in tripleSubjs:

            if chainHeadOfHead in tripleSubj.lower():
                feature[i] = 1.
                break

    return feature


def get_triple_subjects_core(ann):
    tripleSubjects = []
        
    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
        "hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "themselves", "your", "my", "it",
        "they", "them", "that", "these", "those", "who", "whom", "what", "which", "this" ]

    for sentence in ann["sentences"]:
        for triple in sentence["openie"]:

            if triple["subject"].lower() not in pronouns:
                tripleSubjects.append(triple["subject"])

    return list(set(tripleSubjects))


#### helper 

# # get list of chain heads
# def chain_heads(CRChains):
#     chainHeads = []
#     for chain in CRChains:
#         chainHeads.append(chain[0])

#     return chainHeads

# # get list of chain head of heads
# def chain_head_of_heads(chainHeads):
#     chainHeadOfHeads = []
#     for head in chainHeads:
#         split = head.split(" ")
#         chainHeadOfHeads.append(split[-1])

#     return chainHeadOfHeads

def get_chain_head(corefChain):

    # try:
    #     head = corefChain['mentions'][0]['text'].strip()
    
    # except:
    #     print(corefChain)
    #     head = ''


    return corefChain['mentions'][0]['text'].strip()


def get_chain_head_of_head(corefChain):

    chainHead = get_chain_head(corefChain)

    return chainHead.split(' ')[-1]




