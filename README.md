# SGP-extraction

### Overview
This is a project to extract Synchronous Grammar Patterns (SGP) pairs of English and Chinese\
eg. talk: V to n | 說話: 跟 n V, example pattern instance: talk to people | 跟 人 說話

### File Description
data/:
> **alignment_table_all_final.json**: phrase table\
  Format：dict[en_word][ch_word] = count
  
> **crf_model_passive.joblib**: crf model (to automatically identify English grammar patterns for a given sentence)

> **ch_pat_count.json**: manual annotated data\
  Format：dict[ch_pattern] = count
  
> **cobuild_all_patterns.txt**: all grammar patterns listed in GRAMMAR PATTERN 1: VERBS

> **annotation_prep.txt**: Chinese prepositions that appears in annotated data(ch_pat_count.json)

> **common_words.txt**: English common words. For calculating GDEX socre.

code:
> **build_model.ipynb**: main program to extract SGP pairs\
> **build_phrase_table.ipynb**: build the phrase table\
> **data_format_transform.py**: some data process for build_model.ipynb\
> **pattern_recognition.ipynb**: build the crf model for automatically identifying English grammar patterns

### Method Flow
1. Train a crf model to identify English grammar patterns for a given sentence
<img src="https://github.com/jocelynzungchen/SGP-extraction/blob/master/images/method_part1.png" width="50%" height="50%">
2. Extract SGPs from parallel corpus (Discover the counterpart of the identified English grammar patterns)
<img src="https://github.com/jocelynzungchen/SGP-extraction/blob/master/images/method_part2.png" width="50%" height="50%">
- obtain aligned Chinese tokens by using fast_align, and filter out those that are not in the phrase table
- obtain Chinese pos tags by using CKIP, and simplify into Chinese grammar patterns
- filter out Chinese grammar patterns that are not in the annotated data
3. filter out SGP pairs with low frequency and select good example sentences using GDEX

### Execution
