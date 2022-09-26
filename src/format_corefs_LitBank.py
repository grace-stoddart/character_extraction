import re

#### Parse brat coref entities into dictionary, sorted by entity name
def get_coreference_chains_from_brat_annotation_file(brat):
    # get list of entity labels and make keys of dict
    annotations_by_label = {}

    labels = []

    for annotation in brat:
        labelPart = annotation[1]
        label = labelPart.split(" ")[0]
        labels.append(label)

    labels = set(labels)

    for label in labels:
        annotations_by_label[label] = []

    # add annotations to dict
    for annotation in brat:
        if len(annotation) != 3:
            continue

        labelPart = annotation[1]
        label = labelPart.split(" ")[0]


        indices = [int(labelPart.split(" ")[1]), int(labelPart.split(" ")[2])]

        text = (annotation[2])

        annotations_by_label[label].append({'text':text,'position_LitBank':indices})

    # remove labels which aren't to do with CR
    removed = []
    keys = annotations_by_label.keys()
    keys = list(keys)

    for key in keys:
        if key.upper() == key:
            del annotations_by_label[key]
            removed.append(key)

    return annotations_by_label

### convert indices from litbank character indices to allenNLP token indices
# def litbank_to_allen_indices_map(tokensAllen, bratInput):
#     # create fictionaries for LitBank character tokenization and AllenNLP tokenization
#     charDict = {}
#     for i, char in enumerate(bratInput):
#         charDict[i] = char

#     tokenDict = {}
#     for j, token in enumerate(tokensAllen['tokens']):
#         tokenDict[j] = {'word': token}

#     tokenNum = 0
#     word = ''
#     indices = []
#     append = False

#     for i, char in charDict.items():

#         token = tokenDict[tokenNum]['word']

#         if token == "``":
#             tokenNum += 1
#             continue

#         if token == "''":
#             tokenNum += 1
#             print('here')
#             continue

#         if char == token[0]:
#             append = True

#         if append == True:
#             word += char
#             indices.append(i)

#             if word == token:

#                 tokenDict[tokenNum]['char_indices'] = (indices[0], indices[-1] + 1)

#                 word = ''
#                 indices = []
#                 tokenNum += 1

#                 append = False

#                 continue

#             if char == ' ':
#                 if token[:len(indices)-1] == word[:len(indices)-1]:
#                     word = word[:-1]

#     keys = list(tokenDict.keys())

#     # for key in keys:
#     #     if key >= tokenNum:
#     #         del tokenDict[key]

#     return tokenDict, tokenNum

### convert indices from litbank character indices to allenNLP token indices

def litbank_to_allen_indices_map(tokensAllen, bratInput):

    charDict = {}
    for i, char in enumerate(bratInput):
        charDict[i] = char

    tokenDict = {}
    for j, token in enumerate(tokensAllen['tokens']):
        tokenDict[j] = {'word': token}


    charDictKeys = list(charDict.keys())

    tokenNum = 0
    beginWord = False
    word = ''
    charIndexes = []
    repeatLoop = True

    while repeatLoop == True:
        for i, key in enumerate(charDictKeys):
            repeatLoop = False

            token = tokenDict[tokenNum]['word']
            character = charDict[key]

            if token == '"':
                token = '``'

            if token == "''":
                token = '``'

            if character == '"':
                character = '``'
                if character[0] == token[0]:
                    beginWord = True
            

            if character == token[0]:
                beginWord = True

            

            if beginWord == True:
                word += character
                charIndexes.append(key)

                if word == token:
                    beginWord = False
                    word = ''
                    tokenDict[tokenNum]['char_indices'] = [charIndexes[0], charIndexes[-1] + 1]
                    charIndexes = []
                    tokenNum += 1

                elif character == ' ' and token[:len(charIndexes)] != word:
                    word = word[:-1]

                elif character == '\n' and token[:len(charIndexes)] != word:
                    word = word.replace('\n', '')

                elif len(word) > len(token) + 5:
                    tokenNum += 1
                    charDictKeys = charDictKeys[i - len(word):]
                    beginWord = False
                    repeatLoop = True

            if repeatLoop == True:
                break

    keys = list(tokenDict.keys())
    for key in keys:
        if key >= tokenNum:
            del tokenDict[key]

    return tokenDict, tokenNum

### cut allen NLP tokenized document based on length of brat tokenized doc
# def cut_tokenized_document(tokenNum, tokensAllen):

#     for i in range(len(tokensAllen['tokens']) - 1, tokenNum -1, -1):
#         del tokensAllen['tokens'][i]

#     last5Tokens = tokensAllen['tokens'][-6:-1]
#     tokensJoined = "".join(last5Tokens).replace('"','``').replace("''",'``')

#     lastSentNum = []
#     for i, sent in enumerate(tokensAllen['sents']):

#         sentJoined = "".join(sent['tokens']).replace('"','``')

#         if tokensJoined in sentJoined:
#             lastSentNum.append(i)

#     if len(lastSentNum) == 1:
#         for i in range(len(tokensAllen['sents']) -1, lastSentNum[0], -1):
#             del tokensAllen['sents'][i]

#         return tokensAllen

#     elif len(lastSentNum) == 0:
#         print("Couldn't find last sentence.")
#         return

#     else:
#         print("Foudn multiple possible last sentences.")
#         return

def cut_tokenized_document(tokenNum, tokensAllen):

    for i in range(len(tokensAllen['tokens']) - 1, tokenNum -1, -1):
        del tokensAllen['tokens'][i]

    tokenCount = 0
    for j, sent in enumerate(tokensAllen['sents']):

        for token in sent['tokens']:
            tokenCount += 1

        if tokenCount >= tokenNum:
            lastSentNum = j
            break

    for k in range(len(tokensAllen['sents']) - 1, lastSentNum, -1):
        del tokensAllen['sents'][k]

    return tokensAllen

### add allenNLP indices to the coref annotations dict
def add_allenNLP_indices_to_coref_dict(annotations_by_label, tokenDict):

    for cluster in annotations_by_label.values():
        for mention in cluster:

            tokenIndices = []

            charIndexStart_mention = mention['position_LitBank'][0]
            charIndexEnd_mention = mention['position_LitBank'][1]

            for tokenIndex, token in tokenDict.items():

                if 'char_indices' not in token.keys():
                    continue

                charIndexStart_token = token['char_indices'][0]
                charIndexEnd_token = token['char_indices'][1]

                if charIndexStart_mention == charIndexStart_token and charIndexEnd_mention == charIndexEnd_token:
                    tokenIndices.append(tokenIndex)
                    tokenIndices.append(tokenIndex)
                    break

                elif charIndexStart_mention == charIndexStart_token:
                    tokenIndices.append(tokenIndex)
                    
                elif charIndexEnd_mention == charIndexEnd_token:
                    tokenIndices.append(tokenIndex)
                    break

                else:
                    continue

            
            mention['position'] = tokenIndices  

    return annotations_by_label

### get into same format as other coref dicts
from coreference_resolution import get_canonical_character_name
def format_coref_dict(annotations_by_label, tokenizedCut):
    coref_dict_final = {'clusters':[], 'tokenizedDocument':tokenizedCut}

    skip = 0

    for i, (key, cluster) in enumerate(annotations_by_label.items()):

        name = key.replace('_',' ')
        name = re.sub(r'[0-9]+', '', name)
        name = name.strip('-')

        if name == 'appos' or name == 'cop':
            skip += 1
            continue

        coref_dict_final['clusters'].append({})

        coref_dict_final['clusters'][i - skip]['mentions'] = []


        for mention in cluster:
            coref_dict_final['clusters'][i - skip]['mentions'].append(mention)

        coref_dict_final['clusters'][i - skip]['name'] = get_canonical_character_name(coref_dict_final['clusters'][i - skip]['mentions'])




    return coref_dict_final

def get_id_and_title(filePath):
    '''
    Returns predicted Litbank story ID and story title, from the file path.
    Story title has format like: the-wind-in-the-willows
    '''
    storyName = filePath.split("/")[-1]
    storyName = storyName.split(".txt")[0]

    storyID = int(re.search(r'\d+', storyName).group())

    storyName = re.sub(r'[0-9]+', '', storyName)
    storyName  = storyName.strip("_")
    storyName = storyName.replace('_','-')
    storyName = storyName.replace("'",'')

    return storyID, storyName


# match coreference chains with characters 
import numpy as np
def get_character_labels(corefDict, characterList):
    charLabels = np.zeros(len(corefDict['clusters']))

    for i, cluster in enumerate(corefDict['clusters']):

        for mention in cluster['mentions']:

            text = mention['text'].lower().strip()
            text = text.replace('\n', ' ')
            text = text.replace('  ', ' ')

            textOptions = [text, text.replace('.',''), text.replace('the ',''), text.replace('the ','').replace('a ','')]

            for character in characterList:
                character = character.lower().strip().replace('\n', ' ')

                charOptions = [character, re.sub("[\(\[].*?[\)\]]", "", character).strip().replace('  ',' '), get_text_from_within_brackets(character),get_text_from_within_brackets(character).replace('"',''), remove_title(character), character.replace('.',''), remove_mr(character), character.replace("'",' ')]

                
                charactersSep = separate_ands(character)
                if len(charactersSep) > 1:
                    for c in charactersSep:
                        charOptions.append(c)
                
                if len(character.split(',')) > 1:
                    commasSplit = True
                else:
                    commasSplit = False

                splits = character.split(' ')
                
                if not commasSplit:
                    if len(splits) > 2:
                        charOptions.append(' '.join(character.split(' ')[-2:]).lower())

                    if len(splits) == 3:
                        charOptions.append(' '.join([character.split(' ')[0], character.lower().split(' ')[-1]]))

                charsSepComma = separated_by_comma(character)
                if len(charsSepComma) > 1:
                    for c in charsSepComma:
                        charOptions.append(c)

                    for c in charsSepComma:
                        if c != remove_mr(c):
                            charOptions.append(remove_mr(c))


                for ch in charOptions:
                    for txt in textOptions:
                        if ch == txt:
                            charLabels[i] = 1.
                            break
                    
                    if charLabels[i] == 1.:
                        break
                
                if charLabels[i] == 1.:
                        break
            if charLabels[i] == 1.:
                        break

    return charLabels


def remove_title(character):
    '''
    assumes form 'Mr John Clarke'
    returns 'John Clarke'
    returns character with honorific removed
    '''
    titles = ['mr', 'mrs', 'mr.', 'mrs.', 'miss', 'sir', 'aunt', 'uncle']

    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character


    if splits[0].lower in titles and len(splits) > 2:
        return ' '.join(splits[1:])

    else:
        return character
    
   

def get_text_from_within_brackets(character):
    '''
    assume no ands in character.
    returns text from within brackets. If not brackets are present, returns original character text.
    '''
    res = re.findall(r'\(.*?\)', character)

    if len(res) == 0:
        return character

    return res[0].replace('(','').replace(')','')



def get_first_name(character):
    '''
    assumes character is made up of first name + last name
    Removes surname
    '''
    titles = ['mr', 'mrs', 'mr.', 'mrs.', 'miss', 'sir', 'aunt', 'uncle']

    if character.split(' ')[0].lower() in titles or len(character.split(' ')) !=  2:
        return character

    return character.split(' ')[0].strip()


def separate_ands(character):
    '''
    Assumes input of form 'Mr and Mrs Darcy' or 'James and Joan Clarke'
    Returns [Mr Darcy, Mrs Darcy], [James Clarke, Joan Clark]
    If not in assumed form, returns character
    '''
    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if not containsAnd:
        return [character]

    names = []
    text = ''
    for split in splits:
        if split == 'and':
            names.append(text)
            text = ''
            continue
        text += split + ' '

    names.append(text)

    for i, name in enumerate(names):
        names[i] = name.strip()

    for i, name in enumerate(names):
        if len(name.split(' ')) == 1:
            names[i] = name + ' ' + splits[-1]

    return names

def remove_mr(character):
    '''
    assumes form 'Mr John Clarke or Mr Clarke'
    returns 'Clarke'
    returns character with honorific removed
    '''
    titles = ['mr', 'mr.', 'sir', 'doctor.', 'doctor', 'sir.']

    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character


    if splits[0].lower() in titles and "'" not in splits[1]:
        return splits[-1]

    else:
        return character
    

   
def separated_by_comma(character):
    splits = character.split(',')

    if len(splits) < 3:
        return [character]

    chars=[]
    for split in splits:

        split = split.strip().lstrip('and ')

        chars.append(split.strip())

    return chars

