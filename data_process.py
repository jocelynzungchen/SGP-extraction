import spacy

nlp = spacy.load('en_core_web_lg')
merge_nps = nlp.create_pipe("merge_noun_chunks")
nlp.add_pipe(merge_nps)

cobuild_prep = {'about', 'across', 'after', 'against', 'among', 'around', 'as', 'at', 'between', 'by', 'for', 'from', 'in', 'into', 'like', 'not', 'of',
 'off', 'on', 'onto', 'out', 'over', 'round', 'so', 'though', 'through', 'to', 'together', 'toward', 'towards', 'under', 'with'}


# for crf model
def sent2features(sent):
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        label = sent[i][2]
        
        features = {
            'word': word,
            'postag': postag,
        }
        if label == 'V':
            features['target'] = True
        if i == 0:
            features['BOS'] = True
        else:
            word_1 = sent[i-1][0]
            features.update({
                'passive': word_1 in 'be/is/are/was/were'.split('/') and postag == 'VERB' and word[-3:] != 'ing',
                '-1:word == \'and\'': word_1 == 'and',
            })
        if i == len(sent)-1:
            features['EOS'] = True
                    
        return features

    return [word2features(sent, i) for i in range(len(sent))]


# Transforming Raw text to CRF model format
def to_crf_format(line):
    sent = []
    V_flag = False
    for token in nlp(line):
        if token.pos_ == 'VERB':
            sent += [(token.text, token.pos_, 'V')]
            V_flag = True
        else:
            for t in token.text.split():
                sent += [(t, token.pos_, 'O')]
    if not V_flag: # if the sentence contains no 'V' -> ignore
        return []
    return sent


# Transforming pattern passive form to simple form
# example: 'be Ved n' -> 'V n n', 'be Ved about n' -> 'V n about n'
def revert_to_simple_form(pat):
    def contains(pat, elements):
        for e in elements:
            if e in pat:
                return True
        return False

    if 'Ved' not in pat.split() or 'be' not in pat.split(): # excluded patterns without 'be' or 'Ved' -> regard as 'simple form'
        return pat
    pat = pat.lstrip('n').strip().strip('be').replace('Ved', 'V').strip()
    p_index = -1
    for i, p in enumerate(pat.split()):
        if p in cobuild_prep:
            p_index = i # index of the last prep
    index = pat.index('V')
    pat = pat.split()
    if p_index > 0: # patterns containing prep
        if contains(pat, 'n/pl-n/adj/adv/ving/ved/vp/together/wh'.split('/')): # double objects, eg. 'V n prep n', 'V n as adj' -> add 'n' after 'V'
            pat.insert(index+1, 'n')
        else: # 'V p n', eg. 'be Ved about' -> add 'n' after prep
            pat.insert(p_index+1, 'n')
    else: # eg. 'V n adj' -> add 'n' after 'V'
        pat.insert(index+1, 'n')
    return ' '.join(pat)


# Transforming crf results to pattern format
# eg. sim_pattern_form(['V', 'V', 'onto', 'B-amount', 'I-amount', 'I-amount']) -> 'V onto amount'
def sim_pattern_form(pat_result):
    pattern = [pat_result[0]]
    for element in pat_result[1:]:
        if pattern[-1] != element and element not in ['I-n', 'I-pl-n', 'I-amount']:
            pattern += [ element.strip('B-') ]
    return ' '.join(pattern)


# Extracting pattern instance for a given sentence
# eg. 
def get_pattern_instance(training_sent, crf_pred):
    pats = [] # could be more than one pattern
    pat = []
    V_flag = False
    for i, label in enumerate(crf_pred):
        if label == 'O':
            pass
        else:
            if 'V' in label:
                if V_flag and 'V' not in crf_pred[i-1]: # 'V' already in the pattern, and is not continuous 'V' -> another pattern
                    pats += [pat.copy()]
                    pat.clear()
                V_flag = True
            pat += [ (label, training_sent[i][0], i) ]
    if pat:
        pats += [pat]
    return pats