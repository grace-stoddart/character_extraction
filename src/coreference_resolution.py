from collections import Counter

def get_canonical_character_name(cluster):
    '''
    Returns Canonical name for a coreference cluster.
    Input:
    cluster - list of referring expressions from coref resolution output Dict i.e. outputDict['clusters'][i]['mentions']
    '''

    stops = ['he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'they', 'them', 'their', 'theirs',
            'i', 'me', 'my', 'mine',
            'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'yourself',
            'mr.', 'mr', 'ms', 'ms.', 'miss', 'miss.', 'mrs', 'mrs.',
            'sir', 'it', 'themselves', 'myself', 'yourselves', 'yourself', 'this'
            ]

    count = Counter()    

    for ref in cluster:
        if ref['text'].lower() not in stops:
            count[ref['text']] += 1

    ranked = count.most_common(1)

    if len(ranked) == 0:
        return None
    else:
        return count.most_common(1)[0][0]


def get_canonical_character_name_from_list(list):

    stops = ['he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'they', 'them', 'their', 'theirs',
            'i', 'me', 'my', 'mine',
            'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'yourself',
            'mr.', 'mr', 'ms', 'ms.', 'miss', 'miss.', 'mrs', 'mrs.',
            'sir', 'it', 'themselves', 'myself', 'yourselves', 'yourself', 'this'
            ]
    
    count = Counter()    
    
    for ref in list:
        if ref.lower() not in stops:
            count[ref] += 1

    ranked = count.most_common(1)

    if len(ranked) == 0:
        return None
    else:
        return count.most_common(1)[0][0]


def get_coref_dict(document, predictor):

    output = predictor.predict_tokenized(tokenized_document = document['tokens'])

    outputDict = {'tokenizedDocument': output['document'],'clusters':[]}
    
    for cluster in output['clusters']:
        mentions = []
        for indexes in cluster:
            NP = ''
            for i in range(indexes[1] - indexes[0] + 1):
                word = output['document'][indexes[0] + i]
                NP += word + ' '
            NP = NP.strip()
            mentions.append({'position': indexes, 'text':NP})

        name = get_canonical_character_name(mentions)
        outputDict['clusters'].append({'mentions':mentions, 'name':name})
    
    return outputDict