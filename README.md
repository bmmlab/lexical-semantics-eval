# Introduction
This code accompanies the paper 'The Importance of Context in the Evaluation of Word Embeddings: The Effects of Antonymy and Polysemy'.

# Word embeddings
Note that even these highly compressed files are very large, and will require a request for access to download.
- Static word embeddings (10gb) are available [here](https://drive.google.com/file/d/1O1FuDXSqJITMz5PodW8gd_O4Xd84txaY/view?usp=sharing).
- Contextualised word embeddings (5gb) extracted from the ERNIE transformer are available [here](https://drive.google.com/file/d/1iGhSK1yk8Vn80wMMlBJWNTUgOU0sMfla/view?usp=sharing).

# Overview of subfolders
- 'Corpus_Data': Includes the corpuses used for the different types of contextualised ERNIE embeddings, as explained in the paper.
- 'Expr_Datasets': Contains all experimental datasets of human similarity or relatedness judgements investigated in this study. 
Files marked 'mod' have antonyms removed. Also included are several combined vocabulary lists which were used for construction of embeddings files.
- 'Make_Plots': Contains figures from the paper and the code for generating them.
- 'Save_Embeddings': Contains code for computing and saving different types of word embeddings for a specified vocabulary of words. These notebooks all output text files, with each line continaing a single word and its corresponding embedding. The code requires the appropriate set of static embeddings or transformer model in order to run successfully.
- 'Word_Similarities': Includes the computed similarities between word pairs for different experimental datasets and different word embedding models. Also includes sense embeddings for the ERNIE Sense Embeddings, as outlined in the paper. 

# Overview of files in main directory
The files in the main directory are the most important in the project. Most key analyses can be run with only these three files.
- Eval_embeddings_main: The main notebook used for evaluation and comparison of word embeddings models. This is the one to start with. 
- similarity_analysis.py: Python module including functions necessary for running Eval_embeddings_main.
- all_model_dataset_cosine_sims.json: Data file containing pre-computed cosine similarities for all datasets and embedding models. Needed for running Eval_embeddings_main unless you want to recompute them yourself, which takes about 20 mins.

# Workflow
The workflow to generate all results from scratch is as follows:
1. Download required word embedding models.
2. Download experimental datasets from Expr_Datasets folder.
3. Use the code in the 'Save_Embeddings' folder to compute and save word embeddings for all words in the experimental datasets from (2), using the models downloaded in (1).
4. Use the corpuses in 'lexical-semantics-eval-corpus' with the code from 'Calc_sense_sims' to compute and save sense embeddings.
5. Use the notebook 'Eval_embeddings_main' to compute and save a .json file containing pre-computed cosine similarities between all word embeddings with vocab containing in all datasets.
6. Use the notebook 'Eval_embeddings_main' and the .json file generated in (5) to perform further analysis of word embeddings.
