from calendar import c
from misc import open_dict
import numpy as np
import re

titles = ['mr', 'mrs', 'mr.', 'mrs.', 'miss', 'sir', 'aunt', 'uncle', 'lady', 'ms.', 'doctor.', 'doctor', 'dr.', 'general', 'rev.', 'reverend']
maleTitles = ['mr', 'mr.', 'sir', 'uncle', 'doctor.', 'doctor', 'dr.', 'general', 'rev.', 'reverend']

# from self import remove_title, get_first_name, remove_mr, remove_first_name, get_last_three_words, remove_middle_name

def score_output(charChains, y_pred, outputChains):  
    
    # y_pred = process_predictions(y_pred, outputChains)

    predictions = np.zeros((y_pred.shape[0], len(charChains)))

    TP = 0
    FP = 0
    FN = 0

    for i, (prediction, outputChain) in enumerate(zip(y_pred, outputChains)):
        if prediction == 1:

            found = False

            canonicalCharName = str(outputChain[0])

            if len(canonicalCharName) > 3:
                if canonicalCharName[-3:] == " 's":
                    canonicalCharName = canonicalCharName[:-3]

            for j, groundTruthChar in enumerate(charChains):
                for charVariation in groundTruthChar:
                    if charVariation.lower().strip() == canonicalCharName.lower().strip():
                        predictions[i,j] += 1
                        found = True
                        break

                if found:
                    break

        votes = np.sum(predictions, axis = 0)
        finds = np.sum(predictions, axis = 1)

    for k, vote in enumerate(votes):
        if vote == 0:
            FN += 1
        elif vote > 0:
            TP += 1
            FP += (vote - 1)

    for k, find in enumerate(finds):
        if y_pred[k] == 1:
            if find == 0:
                FP += 1

    ## check for errors
    if FP + TP != sum(y_pred):
        print('error 1')

    if FN + TP != len(charChains):
        print('error 2')

    ##
    if np.sum(y_pred) == 0:
        accuracy = 0
    else:
        accuracy = TP / np.sum(y_pred)

    if TP == 0:
        recall = 0
        precision = 0
        f1 = 0
    else:
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1 = 2 * ((precision * recall) / (precision + recall))


    return accuracy, recall, precision, f1, finds, votes
    

def process_predictions(y_pred, outputChains):
    for i, (prediction, outputChain) in enumerate(zip(y_pred, outputChains)):
        if prediction == 1:
            if outputChain[0] == None or outputChain[0].strip() == "'s":
                y_pred[i] = 0.

    return y_pred


def combine_corefs(corefsDir, fileNames):
    '''
    Returns list of all the coref chains for a set of stories.
    '''
    fileName = fileNames[0]
    corefs = open_dict(corefsDir + fileName + '.p')
    refsAll = get_ref_exps_from_coref_dict(corefs)

    for i in range(1, len(fileNames)):
        fileName = fileNames[i]
        corefs = open_dict(corefsDir + fileName + '.p')
        refs = get_ref_exps_from_coref_dict(corefs)

        for ref in refs:
            refsAll.append(ref)

    return refsAll

def get_ref_exps_from_coref_dict(corefs):
    '''
    Takes coref dict. 
    Returns lists of referring expressions, corresponsing to each CR chain. 
    The first item in each list corresponds to the coref chain 'title'
    '''
    refExps = []

    for chain in corefs['clusters']:
        refs = []
        refs.append(chain['name'])

        for mention in chain['mentions']:
            refs.append(mention['text'])

        refExps.append(refs)
        
    return refExps


def get_all_variations_catchall(characterList):

    variationsList = []

    for character in characterList:

        l = [character]

        characters = separate_character(character)

        l = [character]

        for char in characters:
            l += get_variations(char)

        variationsList.append(l)

    return variationsList


def get_all_variations(characterList):

    variationsList = []

    for character in characterList:

        characters = separate_character(character)

        for char in characters:
            variationsList.append(get_variations(char))

    return variationsList

def get_variations(character):

    character = character.strip()

    variations = [character]

    # remove brackets / quoted from character. Add what#s within brackets / quotes to variations
    if remove_brackets(character) != character:
        variations.append(get_text_from_within_brackets(character).replace('"','').replace('“','').replace('”',''))
        variations = iterate_through_var_functions(get_text_from_within_brackets(character), variations)
        character = remove_brackets(character).strip()
        if character == '':
            return variations
        variations.append(character)

    if remove_quoted(character) != character:
        variations.append(get_text_within_quoted(character))
        variations = iterate_through_var_functions(get_text_within_quoted(character), variations)
        character = remove_quoted(character).strip()
        if character == '':
            return variations
        variations.append(character)


    variations = iterate_through_var_functions(character, variations)

    for varNum, var in enumerate(variations):
        variations[varNum] = remove_lonely_brackets(var).strip()

    for i in range(len(variations) -1, -1, -1):
        if variations[i].replace('\n','') != variations[i]:
            variations.append(variations[i].replace('\n',''))
 
    return variations


def separate_character(character):
    '''
    takes in character ( "Mr and Mrs Dark", "Pip", "Billy, Andy and John")
    returns list of all the characters there. might be length 1
    '''

    if not ', ' in character:
        return (separate_ands(character))

    return separated_by_comma(character)
    

def remove_letters(character, letters):
    return character.replace(letters, '')
    

def remove_title(character):
    '''
    assumes form 'Mr John Clarke'
    returns 'John Clarke'
    returns character with honorific removed
    '''

    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character


    if splits[0].lower() in titles and len(splits) > 2:
        return ' '.join(splits[1:])

    else:
        return character
    
def remove_title_keep_first_name(character):
    '''
    assumes form 'Mr John Clarke'
    returns 'John Clarke'
    returns character with honorific removed
    '''

    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character


    if splits[0].lower() in titles and len(splits) > 2:
        return ' '.join([splits[0], splits[1]])

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
    assumes character is made up of first name + last
    Removes surname
    '''

    if character.split(' ')[0].lower() in titles or len(character.split(' ')) !=  2:
        return character

    if character.split(' ')[0].lower() == 'the':
        return character

    return character.split(' ')[0].strip()




def separate_ands(character):
    '''
    Assumes input of form 'Mr and Mrs Darcy' or 'James and Joan Clarke'
    Returns [Mr Darcy, Mrs Darcy], [James Clarke, Joan Clark]
    If not in assumed form, returns character
    '''
    splitsComma = character.split(', ')
    if len(splitsComma) > 1:
        return([character])


    splitsAnd = character.split(' and ')
    if len(splitsAnd) != 2:
        return([character])

    # Mr and Mrs James Brown -> Mr James Brown, Mrs James Brown

    if len(splitsAnd[0].split(' ')) == 1 and len(splitsAnd[1].split(' ')) == 1:
        return [splitsAnd[0], splitsAnd[0]]

    if len(splitsAnd[0].split(' ')) == 1 and len(splitsAnd[1].split(' ')) > 1:
        return [ ' '.join([splitsAnd[0], ' '.join(splitsAnd[1].split(' ')[1:])])  , splitsAnd[1] ]

    else:
        return [splitsAnd[0], splitsAnd[0]]

def remove_mr(character):
    '''
    assumes form 'Mr John Clarke or Mr Clarke'
    returns 'Clarke'
    returns character with honorific removed
    '''

    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character


    if splits[0].lower() in maleTitles and "'" not in splits[1]:
        return splits[-1]

    else:
        return character

def separated_by_comma(character):
    '''
    assumes form: 'X, Y and Z' or 'X, Y, and Z'
    
    
    '''
    splits = character.split(',')
    chars = []


    if len(splits) == 1:
        return [character]

    if len(splits) == 2:
        containsAnd = False
        for split in splits:
            if ' and ' in split:
                containsAnd = True

        if not containsAnd:
            return [character]

    if ' and ' in character:
        splitsNew = []
        for s in splits:
            ss = s.split(' and ')
            for a in ss:
                if a != '':
                    splitsNew.append(a.strip())

        splits = splitsNew

    addSurname = True
    for i, s in enumerate(splits):
        
        if i < len(splits) - 1:
            if len(s.split(' ')) != 1:
                addSurname = False
                break
        
        if i == len(splits) - 1:
            if len(s.split(' ')) != 2:
                addSurname = False
                break

    for i, s in enumerate(splits):
        if i < len(splits) - 1:
            if not addSurname:
                chars.append(s)
            else:
                chars.append(' '.join([s, splits[len(splits) - 1].split(' ')[-1]]))
        else:
            chars.append(s)

    return chars


def remove_first_name(character):
    '''
    Mrs. John Sedley -> Mrs. Sedley
    Mr. John Sedley -> Mr. Sedley
    '''
    splits = character.split(' ')

    containsAnd = False
    for split in splits:
        if split.lower() == 'and':
            containsAnd = True
            break

    if containsAnd:
        return character

    if splits[0].lower() in titles and len(splits) == 3:
            return ' '.join([splits[0], splits[-1]])

    else:
        return character

def remove_brackets(character):

    return re.sub("[\(\[].*?[\)\]]", "", character).strip().replace('  ',' ')


def remove_quoted(character):

    if '“' in character: 
        return re.sub(r'“.*”',"",character).strip().replace('  ',' ')
    
    if '"' in character: 
        return re.sub(r'".*"',"",character).strip().replace('  ',' ')

    else:
        return character


def get_text_within_quoted(character):

    if '“' in character:
        res = re.findall(r'\“.*?\”', character)

        if len(res) == 0:
            return character

        return res[0].replace('“','').replace('”','')

    if '"' in character:
        res = re.findall(r'\".*?\"', character)

        if len(res) == 0:
            return character

        return res[0].replace('"','')
    else:
        return character

def remove_middle_name(character):

    splits = character.split(' ')

    if len(splits) == 3:
        return ' '.join([character.split(' ')[0], character.split(' ')[-1]])

    return character


def get_last_three_words(character):

    splits = character.split(' ')
    
    if len(splits) > 2:
        return ' '.join(character.split(' ')[-2:])

    return character

def desc_comma(character):

    splits = character.split(', ')

    if len(splits) != 2:
        return character

    return splits[0], splits[-1]

def remove_lonely_brackets(variation):
    if '(' in variation and ')' not in variation:
        variation = variation.replace('(','')

    elif ')' in variation and '(' not in variation:
        variation = variation.replace(')','')

    return variation


def iterate_through_var_functions(character, variations, varFunctions = [remove_title, get_first_name, remove_mr, remove_first_name, get_last_three_words, remove_middle_name, remove_title_keep_first_name]):
    for function in varFunctions:

        if character != function(character):
            variations.append(function(character))


    if character != remove_title(character):
        for function in varFunctions:
            if remove_title(character) != function(remove_title(character)):
                variations.append(function(remove_title(character)))

    if character != desc_comma(character):
        variations += desc_comma(character)

    # get rid of commas / apostrophes; add to variations
    if character != character.replace("'",' '):
        variations.append(character.replace("'",' '))

    if character != character.replace('.',''):
        variations.append(character.replace('.',''))

    return variations


# def get_canon_name_variations(character, varFunctions = [remove_apostrophe, ]):
#     character = character.strip()
#     variations = [character]

#     if remove_brackets(character) != character:
#         variations.append(get_text_from_within_brackets(character).replace('"','').replace('“','').replace('”',''))
#         variations = iterate_through_var_functions(get_text_from_within_brackets(character), variations)
#         character = remove_brackets(character).strip()
#         if character == '':
#             return variations
#         variations.append(character)

