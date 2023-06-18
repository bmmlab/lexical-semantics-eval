# lexical-semantics-eval
This code accmponaies the paper 'The Importance of Context in the Evaluation of Word Embeddings: The Effects of Antonymy and Polysemy'.

A guide to the included files follows.
- 'Expr_Datasets': Contains all experimental datasets of human similarity or relatedness judgements investigated in this study. 
Files marked 'mod' have antonyms removed. Also included are several combined vocabulary lists which were used for construction of embeddings files.
- 'Make_Plots': Contains figures from the paper and the code for generating them.
- 'Save_Embeddings': Contains code for computing and saving different types of word embeddings for a specified vocabulary of words. 
These notebooks all output text files, with each line continaing a single word and its corresponding embedding. The code requires the appropriate set of static embeddings or transformer model in order to run successfully.
- 'lexical-semantics-eval-corpus': Includes the corpuses used for the different types of contextualised ERNIE embeddings, as explained in the paper.
- 'lexical-semantics-eval-sims': Includes the computed similarities between word pairs for different experimental datasets and different word embedding models. Also includes sense embeddings for the ERNIE sense embeddings. 

The files in the main directory are the most important.
- Eval_embeddings_main: The main notebook used for evaluation and comparison of word embeddings models. This is the main one to start with.
- similarity_analysis.py: Python module including functions necessary for running Eval_embeddings_main.
- all_model_dataset_cosine_sims.json: Data file continaing pre-computed cosine similarities for all datasets and embedding models. Needed for running Eval_embeddings_main unless you want to recompute them yourself, which takes about 20 mins.
