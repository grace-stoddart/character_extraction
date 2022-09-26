# for WN feature
from nltk.corpus import wordnet as wn


# imports/installs needed to use Stanza (NLP software including python wrapper for Stanford CoreNLP)
from stanza.server import CoreNLPClient
## Only need to do the install below once:
# stanza.install_corenlp()


def ruleBasedClassifier(ref, semanticSubjects, endpoint = 'http://localhost:9020'):

    refLast = getLastWord(ref)

    # run first rule
    result = pronoun(refLast)
    if result != 2.:
        print("rule 1")
        return result
    
    # run second rule
    result = rule_2(ref, semanticSubjects)
    if result != 2.:
        print("rule 2")
        return result

    # run third rule
    result =  rule_3(ref, endpoint)
    if result != 2.:
        print("rule 3")
        return result

    # run third & fourth rules
    result = rule_4_5(refLast)
    print("rule 4/5")

    # should change beolw to just return result. This is just so 2. never gets returned (temporary hack!). Should run svm classifier if unclassified.
    return min(result, 1.)


def getLastWord(ref):
    '''
    Parameters:
        ref - referring expression (String)
    Returns:
        refLast - last word of referring expression
    '''
    words = ref.split(' ')
    refLast = words[-1]
    
    return refLast


# Rule 1
# If the last word of a referring expression is a gendered personal, reflexive, or possessive pronoun (i.e., excluding it, its, itself, etc.), we marked it animate
def pronoun(refLast):
    '''
    parameters:
        refLast - last word of reffering expression (String)
    return:
        1 if ref corresponds to animate entity
        0 if ref correspoonds to inanimate entity
        2 if undetermined
    '''

    pronouns = ["he","she","his","her","i","you","we","me","him","us","mine","our","yours","hers",
			"ours","myself","yourself","himself","herself","oneself","ourselves","your","my", "themselves"]
    other = ["it","its","itself"]
    
    for p in pronouns:
        if p.strip().lower() == refLast.strip().lower():
            return 1.
    
    for o in other:
        if o.strip().lower() == refLast.strip().lower():
            return 0.
    
    return 2.


# rule 2, semantic subject rule
def rule_2(ref, semanticSubjectsUnique):
    '''
    parameters:
        refLast - last word of referring expression (string)
        semanticSubjects - list of semantic subjects from corresponding story
    Returns:
        1 if animate
        0 if inanimate
        2 if undetermined
    '''
    for ss in semanticSubjectsUnique:
        if ss.lower().strip() == ref.lower().strip():
            return 1.
    
    return 2.

# rule 3, NER rule
def rule_3(ref, endpoint = 'http://localhost:9020'):
    '''
    Performds NER on the refferring expression with Stanford Core NLP.
    Parameters:
        ref - referring expression
        endpoint - location of free port on local device
    Returns:
        1 is NER type person
        0 if NER type LOC, ORG, MONEY etc
        2 if undetermined
    '''
    NER = ""

    with CoreNLPClient(
        annotators=['ner'],
        timeout=30000,
        endpoint=endpoint,
        memory='6G',
        be_quiet=True) as client:
        ann = client.annotate(ref)
    
    for sent in ann.sentence:
        for token in sent.token:
            NER += token.ner

    if "PERSON" in NER:
        return 1.
    
    if any(i in NER for i in ("LOCATION", "ORGANIZATION", "MONEY", "DATE", "NUMBER", "TIME", "DURATION")):
        return 0.
    
    return 2.

# Rules 4 and 5, WordNet
def rule_4_5(refLast):
    # doesn#t seem great, says that "dragon" is not animate.
    '''
    Parameters:
        refLast - last word of referring expression
    Returns:
        1 if refLast descendent of living_thing in wordnet
        0 if refLast descendant of entity
        2 if neither
    '''
    synsets = wn.synsets(refLast, pos=wn.NOUN)

    while synsets != []:
        synset = synsets[0]

        if synset == wn.synset('living_thing.n.01'):
            return 1.
        
        if synset == wn.synset('entity.n.01'):
            return 0.  
        
        synsets = synset.hypernyms() 

    return 2.

