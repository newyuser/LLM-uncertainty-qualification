# self-uncertainty experiments

## PerplexityAI.jsonl

The original data file from the open source code files.

## simplify_PerplexityAI.jsonl

Get all of the atomic fact with label in PerplexityAI.jsonl.

Include 5568 facts(5568 lines) of 183 people.

Every line is a dict, keys = ["person", "text", "label"].

E.g. {"person": "Shahnaz Pahlavi", "text": "Shahnaz Pahlavi was born in Tehran.", "label": "S"}

PS: I lost the code that inputs PerplexityAI.jsonl and outputs simplify_PerplexityAI.jsonl.

## temp.py(training code)

The mainly training code.

Read the simplify_PerplexityAI.jsonl, run models to generate response, get judgement and token_allocation from response by regular expression, and save these data as csv file(columns=['person', 'text', 'label', 'model', 'judgement', 'token_allocation', 'res']).

## prompt_C.jsonl

Because regular expressions don't recognise all responses, the judgement and token_allocation are manually complemented in the csv file and then stored in prompt_C.jsonl.(Include 2056 facts of 64 people now, more than yesterday's report. And I'll finish the rest later.)


## test.ipynb(analysis code)

Run these cells directly for data analysis and visualisation of prompt_C.jsonl. 

