import numpy as np
from nltk.corpus import wordnet as wn

# classify chains in a story
def categorise_coref_chains(corefChains, nerParses, srlParses, sentences):
    '''
    corefChains - coreference chains (clusters) for a story
    srl_ARG0_positions - indexes corresponding to ARG0 positions in story
    ner_PER_positions - 
    ner_LOC_ORG_positions -    
    '''
    srl_ARG0_positions = get_srl_ARG0_positions(srlParses, sentences)
    ner_PER_positions = get_PER_positions(nerParses, sentences)
    ner_LOC_ORG_positions = get_LOC_ORG_positions(nerParses, sentences)

    n = len(corefChains)
    animacyLabels = np.zeros(n)

    for i, chain in enumerate(corefChains):
        voteCounts = np.zeros(3)

        for j, mention in enumerate(chain['mentions']):
            result = int(categorise_mention(mention, srl_ARG0_positions, ner_PER_positions, ner_LOC_ORG_positions))
            voteCounts[result] += 1

        animacyLabels[i] = np.argmax(voteCounts)

    return animacyLabels


# classify a single mention
def categorise_mention(mention, srl_ARG0_positions, ner_PER_positions, ner_LOC_ORG_positions):
    ref = mention['text']
    refLast = get_last_word(ref)

    # run first rule
    result = rule_one(refLast)
    if result != 2.:
        return result
    
    # run second rule
    result = rule_two(mention, srl_ARG0_positions)
    if result != 2.:
        return result

    # run third rule
    result =  rule_three(mention, ner_PER_positions, ner_LOC_ORG_positions)
    if result != 2.:
        return result

    # run third & fourth rules
    result = rule_four_five(refLast)

    # should change beolw to just return result. This is just so 2 never gets returned (temporary hack!). Should run svm classifier if unclassified.
    # return min(result, 1.)
    return result

# helper
def get_last_word(ref):
    '''
    Parameters:
        ref - referring expression (String)
    Returns:
        refLast - last word of referring expression
    '''
    words = ref.split(' ')
    refLast = words[-1]
    
    return refLast

# rule 1
def rule_one(refLast):
    '''
    parameters:
        refLast - last word of reffering expression (String)
    return:
        1 if ref corresponds to animate entity
        0 if ref correspoonds to inanimate entity
        2 if undetermined
    '''
    pronouns = ["he","she","his","her","i","you","we","me","him","us","mine","our","yours","hers",
			"ours","myself","yourself","himself","herself","oneself","ourselves","your","my", 'their', "themselves"]
    other = ["it","its","itself",'this']
    
    for p in pronouns:
        if p.strip().lower() == refLast.strip().lower():
            return 1.
    
    for o in other:
        if o.strip().lower() == refLast.strip().lower():
            return 0.
    
    return 2.

# rule 2
### does referring expression appear as subject to a verb (ARG0)? If YES, mark as animate.
def rule_two(mention, srl_ARG0_positions):
    '''
    parameters:
        mention - coreference mention (text and positions)
        srl_ARG0_positions - positions of ARG0 tokens from SRL parse
    Returns:
        1 if ref is ARG0
        2 itherwwise (undetermined)
    '''
    indexRange = mention['position']
    
    if indexRange == []:
        return 2.
        
    allIndexes = list(range(indexRange[0], indexRange[-1] + 1))

    for index in allIndexes:
        if index in srl_ARG0_positions:
            return 1.

    return 2.

def get_srl_ARG0_positions(srlParses, sentences):

    positions = []
    tokenOffset = 0


    for i, srlParse in enumerate(srlParses):

        for verb in srlParse['verbs']:

            for j, tag in enumerate(verb['tags']):
                if 'ARG0' in tag:
                    positions.append(j + tokenOffset)

        tokenOffset += len(sentences[i]['tokens'])


    return positions

# Rule 3

def rule_three(mention, PER_positions, LOC_ORG_positions):
    '''
    Parameters:
        mention - dict, included text and position
        PER_positions - list of PER indexes
        LOC_ORG_positions - list of LOC/ ORG indexes

    Returns:
        1 is NER type person
        0 if NER type LOC, ORG
        2 if undetermined
    '''
    indexRange = mention['position']

    if indexRange == []:
        return 2.

    allIndexes = list(range(indexRange[0], indexRange[-1] + 1))

    for index in allIndexes:
        if index in PER_positions:
            return 1.

        if index in LOC_ORG_positions:
            return 0.

    return 2.
def get_PER_positions(nerParses, sentences):
    positions = []

    tokenOffset = 0

    for i, nerParse in enumerate(nerParses):

        for j, tag in enumerate(nerParse['tags']):
            if 'PER' in tag:
                positions.append(j + tokenOffset)

        tokenOffset += len(sentences[i]['tokens'])
        
    return positions

def get_LOC_ORG_positions(nerParses, sentences):
    positions = []

    tokenOffset = 0

    for i, nerParse in enumerate(nerParses):

        for j, tag in enumerate(nerParse['tags']):
            if 'LOC' in tag or 'ORG' in tag:
                positions.append(j + tokenOffset)

        tokenOffset += len(sentences[i]['tokens'])
        
    return positions

# Rules 4 and 5, WordNet
def rule_four_five(refLast):
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

