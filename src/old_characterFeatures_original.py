import numpy as np
# for WN feature
from nltk.corpus import wordnet as wn
# to make API requests to conceptnet, for CN feature
import requests
# imports/installs needed to use Stanza (NLP software including python wrapper for Stanford CoreNLP)
from stanza.server import CoreNLPClient
## Only need to do the install below once:
# stanza.install_corenlp()

# Feature 1, relative CR chain length
def CL(CRChains):
    """
    Returns the CR chain length feature for a single story
    Parameter:
        Annotated CR chains for a story
    Returns:
        Array of decimals representing the normalized length of each CR chain
    """
    n = len(CRChains)
    lengths = np.zeros(n)

    for i in range(n):
        lengths[i] = len(CRChains[i])
    
    return (lengths - np.mean(lengths))/np.std(lengths)

# Feature 2, whether CR chain head appears in text as a semantic subject
def SS(chainHeadOfHeads, semanticSubjects):

    n = len(chainHeadOfHeads)
    SSFeature = np.zeros(n)

    for i in range(n):
        for ss in semanticSubjects:
            ssSplit = ss.split(' ')
            ssLast = ssSplit[len(ssSplit) - 1]

            if chainHeadOfHeads[i].lower().strip() == ssLast.lower().strip():
                SSFeature[i] = 1.
                break

    return SSFeature

# Feature 3, whether chain head contains NER type 'PERSON'
def NE(chainHeadsNER):
    """
    Returns Named Entity feature for a single story. 
    Parameter:
        List containing NER labels for each CR chain head, constructed using standard API of Standorf Dependency Parse.
    Returns:
        Array of binary values corresponding to whether the chain head is a named entity of type PERSON
    """
    NEFeature = np.zeros(len(chainHeadsNER))

    for i, label in enumerate(chainHeadsNER):
        if 'PERSON' in label:
            NEFeature[i] = 1.    

    return NEFeature

# Feature 4, whether chain head of head is descendent of 'PERSON' in WordNet
def WN(chainHeadOfHeads):
    """
    Returns Wordnet feature for a single story. 
    Parameter:
        Array of chain head of heads for a particular story
    Returns:
        Array of binary values depending on whether CR chain head of head is a descendent of PERSON in WordNet
    """ 
    n = len(chainHeadOfHeads)
    WNFeature = np.zeros(n)

    for i in range(n):
        word = chainHeadOfHeads[i]
        synsets = wn.synsets(word, pos=wn.NOUN)

        while synsets != []:

            synset = synsets[0]

            if synset == wn.synset('person.n.01'):
                WNFeature[i] = 1
                break  

            synsets = synset.hypernyms() 

    return WNFeature

# Feature 5, whether chain head fo head appears in text as a evrb subject
def DP(chainHeadOfHeads, dependencies):
    """
    Returns Dependency Link feature for a story
    Parameters:
        Array containing head of heads for each CR chain
        Dependencies list of t argets in enhanced++ nsubj dependences
    Returns: 
        Array of binary values depending on whether CR chain head is a dependent of an nsubj dependency link.
    """

    DPFeature = np.zeros(len(chainHeadOfHeads))

    for i, headOfHead in enumerate(chainHeadOfHeads):
        for dependent in dependencies:
            depLast = dependent.split(" ")[-1]
            if headOfHead.strip().lower() == depLast.strip().lower():
                DPFeature[i] = 1.

    return DPFeature

# Feature 6, whether chain head appears in text as subject of a triple
def TP(chainHeadOfHeads, tripleSubjects):
    """
    Returns Truple feature for a story
    Parameters: 
        Array containing the head of each CR chain in a story
        n x 1 array containing all the subjecs for each triple in a story. (n = num of triples)
    Returns:
        Array of binary values depending on whther head of CR chain is in the subject position fo a triple, excl. pronouns
    """
    n = len(chainHeadOfHeads)
    TPFeature = np.zeros(n)

    pronouns = ["he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
				"hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
				"they", "them", "that", "these", "those", "who", "whom", "what", "which", "this", "themselves" ]
                
    for i in range(n):
        head = chainHeadOfHeads[i]
        isPronoun = False
        v=0

        for subj in tripleSubjects:
            if head.lower().strip() in subj.lower().strip():
                v=1
                break

        for pronoun in pronouns:
            if pronoun == head.lower().strip():
                isPronoun == True
                break
        
        if v==1 and isPronoun==False:
            TPFeature[i] = 1

    return TPFeature

# Feature 7, whether chain head of head has edge related to 'PERSON' in ConceptNet
def CN(chainHeadOfHeads):
    """
    Returns ConceptNet feature for a story
    Parameters:
        Array of head of heads for each CR chain in story
    Returns:
        Array of binary values depending on whether head has any edge related to PERSON in ConceptNet
    """

    n = len(chainHeadOfHeads)
    CNFeature = np.zeros(n)

    for i in range(n):
        word = chainHeadOfHeads[i]
        response = requests.get("https://api.conceptnet.io/c/en/"+word).json()
        edges = response['edges']

        for j in range(len(edges)):
            if 'person' in edges[j]['@id']:
                CNFeature[i] = 1
                break

    return CNFeature


# retrive triple subjects from CoreNLP parse
def get_triple_subjects(ann):
    tripleSubjects = []

    for sentence in ann["sentences"]:
        for triple in sentence["openie"]:
            tripleSubjects.append(triple["subject"])

    return tripleSubjects

# retrieve list of dependents for nsubj type depency links. Enhanced++ dependencies are retreived from CoreNLP parse.
def get_dependencies(ann):
    """
    Returns all dependents of nsubj dependency links, among enhanced++ dependencies of each sentence, for use in DP feature
    Parameters:
        Raw text for a story
    Returns:
        Array containing the dependents of nsubj dependency links, constructed using the standard API of the Stanford Dependency Parse.
    """
    pronouns = [ "he", "she", "his", "her", "i", "you", "we", "me", "him", "us", "mine", "our", "yours",
				"hers", "ours", "myself", "yourself", "himself", "herself", "oneself", "ourselves", "your", "my", "it",
				"they", "them", "that", "these", "those", "who", "whom", "what", "which", "this", "themselves" ]
    
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


# get list of NER categories for the chain heads
def chain_head_NER(chainHeads, endpoint = 'http://localhost:9020'):
    """
    Returns NER labels for each CR chain head
    Parameters:
        array of heads for each CR chain in story
    Returns:
        Array containing the NER label for each CR chain head
    """
    chainHeadsNER = []

    for head in chainHeads:
        NER = ""
        with CoreNLPClient(
            annotators=['ner'],
            timeout=30000,
            endpoint=endpoint,
            memory='6G',
            be_quiet=True) as client:
            ann = client.annotate(head)
        
        for sent in ann.sentence:
            for token in sent.token:
                NER += token.ner

        chainHeadsNER.append(NER)
        
    return chainHeadsNER

# get list of chain heads
def chain_heads(CRChains):
    chainHeads = []
    for chain in CRChains:
        chainHeads.append(chain[0])

    return chainHeads

# get list of chain head of heads
def chain_head_of_heads(chainHeads):
    chainHeadOfHeads = []
    for head in chainHeads:
        split = head.split(" ")
        chainHeadOfHeads.append(split[-1])

    return chainHeadOfHeads