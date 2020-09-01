# SGP-extraction

### Overview
This is a project to extract SGP pairs of English and Chinese
### File Description
data/:
> **alignment_table_all_final.json**: phrase table \
  Format：dict[en_word][ch_word] = count\
  
> **crf_model_passive.joblib**: crf model (to automatically identify English grammar pattern for a given sentence)\

> **ch_pat_count.json**: manual annotated data\
  Format：dict[ch_pattern] = count\
  
> **cobuild_all_patterns.txt**: all grammar patterns listed in GRAMMAR PATTERN 1: VERBS\

> **annotation_prep.txt**: Chinese prepositions that appears in annotated data(ch_pat_count.json)\

> **common_words.txt**: English common words. For calculating GDEX socre.\

code:
> **build_model.ipynb**: main program to extract SGP pairs\
> **build_phrase_table.ipynb**: build phrase table\
> **data_format_transform.py**: some data process for build_model.ipynb\
> **pattern_recognition.ipynb**: build crf model for automatically identifying English grammar pattern\

### Method Flow

### Execution
