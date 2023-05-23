import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage

## A class of functions and variables to use for computing similarites between word embedding models and experimental data
class similarity_analysis(object):
    
    def __init__(self, folder_loc, model_loc, dataset_loc):
        # define file locatoins
        self.folder_loc = folder_loc
        self.model_loc = model_loc
        self.dataset_loc = dataset_loc
        
        # load list of models and databases
        self.models = list(self.model_files.keys())
        self.datasets = list(self.dataset_files.keys())
        
        # generate storage lists and dictionaries for models and databases
        for model in self.models:
            self.model_embed_storage[model] = []
            self.model_sim_storage[model] = {}

        for dataset in self.datasets:
            self.dataset_storage[dataset] = [] 
            self.dataset_sim_storage[dataset] = {} 
            self.missing_vocab_storage[dataset] = {}

        # get dictionary from models to indicies
        i=0
        self.model_to_index = {}
        for model in self.models:
            self.model_to_index[model] = i
            i=i+1
            
        # generate vocab set
        self.generate_vocab_set()
        self.import_all_datasets()
        print('Available models:')
        print(self.models)
        print('Available datasets:')
        print(self.datasets)
        
        # Rare word substitutes (used when the embedding for a target word isn't available for the RW dataset)
        rare_word_loc = self.folder_loc+'Corpus Data/Key vocab/Rare_Words_substitutions.csv'
        RW_mod_pd = pd.read_csv(rare_word_loc, index_col=0, header=None, on_bad_lines='skip')
        self.RW_mod_dict = RW_mod_pd.to_dict()[1] 
        return(None)
    
    # All the word embedding models analysed
    model_files = {
                   'CW_vectors':'Collobert and Weston Vectors/embeddings.txt',
                   'dm_vectors':'Distributional Memory Embeddings/dm_vectors_mini.txt',
                   'dissect_ppmi':'Dissect PPMI Embeddings/ppmi.svd.500_mini.txt',
                   'word2vec_skip':'Word2vec Skipgram CoNLL17/model_mini.txt',
                   'gensim_skip':'Gensim Skipgram wiki+giga/model_mini.txt',
                   'gensim_BNC':'Gensim Skipgram BNC/model_mini.txt',
                   'gensim_cbow':'Gensim CBoW giga/2010_mini.txt',
                   'glove':'Glove Word Embeddings/glove.840B.300d.mini.txt',
                   'lexvec':'LexVec Embeddings/lexvec_wiki+newscrawl_300d_mini.txt',
                   'fasttext':'FastText Skipgram wiki+giga/model_mini.txt',
                   'elmo':'Elmo Embeddings/elmo_mini.txt',
                   'conceptnet':'ConceptNet Embeddings/numberbatch-en-mini.txt',
                   'conceptnet_normalised':'ConceptNet Embeddings/numberbatch-en-mini-normalised.txt',
                   'wordnet':'WordNet Word Embeddings/wn2vec_mini.txt',
                   'bert_large':'bert_large_uncased_mini.txt',
                   'gpt2_large':'gpt2_large_mini.txt',
                   'electra_large':'electra_large_mini.txt',
                   'albert_xxlarge':'albert-xxlarge-v2_mini.txt',
                   'xlnet_large':'xlnet_large_cased_mini.txt',
                   'xlm_roberta':'xlm_roberta_large_mini.txt',
                   't5_large':'t5_large_mini.txt',
                   'comet-atomic':'comet-atomic_mini.txt',
                   'sembert':'sembert_mini.txt',
                   'libert_2m':'libert-2m_mini.txt',
                   'ernie_base_0':'Ernie Base Embeddings/ernie-2.0-en-layer-0.txt',
                   'ernie_base_1':'Ernie Base Embeddings/ernie-2.0-en-layer-1.txt',
                   'ernie_base_2':'Ernie Base Embeddings/ernie-2.0-en-layer-2.txt',
                   'ernie_base_3':'Ernie Base Embeddings/ernie-2.0-en-layer-3.txt',
                   'ernie_base_4':'Ernie Base Embeddings/ernie-2.0-en-layer-4.txt',
                   'ernie_base_5':'Ernie Base Embeddings/ernie-2.0-en-layer-5.txt',
                   'ernie_base_6':'Ernie Base Embeddings/ernie-2.0-en-layer-6.txt',
                   'ernie_base_7':'Ernie Base Embeddings/ernie-2.0-en-layer-7.txt',
                   'ernie_base_8':'Ernie Base Embeddings/ernie-2.0-en-layer-8.txt',
                   'ernie_base_9':'Ernie Base Embeddings/ernie-2.0-en-layer-9.txt',
                   'ernie_base_10':'Ernie Base Embeddings/ernie-2.0-en-layer-10.txt',
                   'ernie_base_11':'Ernie Base Embeddings/ernie-2.0-en-layer-11.txt',
                   'ernie_base_12':'Ernie Base Embeddings/ernie-2.0-en-layer-12.txt',
                   'ernie_context_1':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_1.txt',
                   'ernie_context_2':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_2.txt',
                   'ernie_context_3':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_3.txt',
                   'ernie_context_4':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_4.txt',
                   'ernie_context_5':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_5.txt',
                   'ernie_context_6':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_6.txt',
                   'ernie_context_7':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_7.txt',
                   'ernie_context_8':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_8.txt',
                   'ernie_context_9':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_9.txt',
                   'ernie_context_10':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_10.txt',
                   'ernie_context_11':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_11.txt',
                   'ernie_context_12':'Ernie Wikipedia Embeddings/Combined Embeddings/Generic Embeddings/contextual_embeddings_layer_normalised_12.txt',
                   'ernie_context_1_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_1.txt',
                   'ernie_context_2_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_2.txt',
                   'ernie_context_3_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_3.txt',
                   'ernie_context_4_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_4.txt',
                   'ernie_context_5_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_5.txt',
                   'ernie_context_6_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_6.txt',
                   'ernie_context_7_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_7.txt',
                   'ernie_context_8_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_8.txt',
                   'ernie_context_9_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_9.txt',
                   'ernie_context_10_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_10.txt',
                   'ernie_context_11_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_11.txt',
                   'ernie_context_12_v':'Ernie Wikipedia Embeddings/Combined Embeddings/Verb Embeddings/contextual_embeddings_layer_normalised_12.txt',
                   'ernie_context_1_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_1.txt',
                   'ernie_context_2_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_2.txt',
                   'ernie_context_3_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_3.txt',
                   'ernie_context_4_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_4.txt',
                   'ernie_context_5_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_5.txt',
                   'ernie_context_6_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_6.txt',
                   'ernie_context_7_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_7.txt',
                   'ernie_context_8_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_8.txt',
                   'ernie_context_9_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_9.txt',
                   'ernie_context_10_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_10.txt',
                   'ernie_context_11_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_11.txt',
                   'ernie_context_12_n':'Ernie Wikipedia Embeddings/Combined Embeddings/Noun Embeddings/contextual_embeddings_layer_normalised_12.txt',
                   'ernie_oxford_1':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_1.txt',
                   'ernie_oxford_2':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_2.txt',
                   'ernie_oxford_3':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_3.txt',
                   'ernie_oxford_4':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_4.txt',
                   'ernie_oxford_5':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_5.txt',
                   'ernie_oxford_6':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_6.txt',
                   'ernie_oxford_7':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_7.txt',
                   'ernie_oxford_8':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_8.txt',
                   'ernie_oxford_9':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_9.txt',
                   'ernie_oxford_10':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_10.txt',
                   'ernie_oxford_11':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_11.txt',
                   'ernie_oxford_12':'Ernie Dictionary Embeddings/Combined Embeddings/normalised_12.txt',
                   'bert_large_0':'Bert Large Embeddings/bert-large-uncased-layer-0.txt',
                   'bert_large_1':'Bert Large Embeddings/bert-large-uncased-layer-1.txt',
                   'bert_large_2':'Bert Large Embeddings/bert-large-uncased-layer-2.txt',
                   'bert_large_3':'Bert Large Embeddings/bert-large-uncased-layer-3.txt',
                   'bert_large_4':'Bert Large Embeddings/bert-large-uncased-layer-4.txt',
                   'bert_large_5':'Bert Large Embeddings/bert-large-uncased-layer-5.txt',
                   'bert_large_6':'Bert Large Embeddings/bert-large-uncased-layer-6.txt',
                   'bert_large_7':'Bert Large Embeddings/bert-large-uncased-layer-7.txt',
                   'bert_large_8':'Bert Large Embeddings/bert-large-uncased-layer-8.txt',
                   'bert_large_9':'Bert Large Embeddings/bert-large-uncased-layer-9.txt',
                   'bert_large_10':'Bert Large Embeddings/bert-large-uncased-layer-10.txt',
                   'bert_large_11':'Bert Large Embeddings/bert-large-uncased-layer-11.txt',
                   'bert_large_12':'Bert Large Embeddings/bert-large-uncased-layer-12.txt'
                  }
    
    # All the experimental datasets analysed
    dataset_files = {
                     'RG65':'EN-RG-65.txt',
                     'YP130':'EN-YP-130.txt',
                     'Verb143':'Verb143-sim-final.txt',
                     'MTurk287':'EN-MTurk-287.txt',
                     'MTurk213':'EN-MTurk-213-mod.txt',
                     'MTurk771':'EN-MTurk-771.txt',
                     'WS353':'EN-WS-353-ALL.txt',
                     'WS198':'EN-WS-198-SIM-mod.txt',
                     'RW':'EN-RW-STANFORD.txt',
                     'RW_mod':'EN-RW-STANFORD-mod.txt',
                     'MEN':'EN-MEN-3k.txt',
                     'MEN_animals':'MEN by Category/animals.txt',
                     'MEN_artefacts':'MEN by Category/artefacts.txt',
                     'MEN_colours':'MEN by Category/colours.txt',
                     'MEN_foods':'MEN by Category/foods.txt',
                     'MEN_locations':'MEN by Category/locations.txt',
                     'MEN_nature':'MEN by Category/nature.txt',
                     'MEN_plants':'MEN by Category/plants.txt',
                     'MEN_social':'MEN by Category/social.txt',
                     'SimVerb':'EN-SimVerb-3500.txt',
                     'SimVerb_mod':'EN-SimVerb-3200-mod.txt',  
                     'SimVerb_noant':'EN-SimVerb-3260-noant.txt',  
                     'SimLex':'EN-SIMLEX-999.txt',
                     'SimLex_mod':'EN-SIMLEX-893-mod.txt',
                     'SimLexN':'EN-SIMLEX-999-NOUN.txt',
                     'SimLexN_mod':'EN-SIMLEX-999-NOUN-mod.txt',
                     'SimLexV':'EN-SIMLEX-999-VERB.txt',
                     'SimLexV_mod':'EN-SIMLEX-999-VERB-mod.txt',
                     'SimLexA':'EN-SIMLEX-999-ADJ.txt',
                     'MultiSim':'EN-MultiSim.txt',
                     'MultiSimV':'EN-MultiSim-VERB.txt',
                     'MultiSimV_mod':'EN-MultiSim-VERB-mod.txt',
                     'MultiSimN':'EN-MultiSim-NOUN.txt',
                     'MultiSimN_mod':'EN-MultiSim-NOUN-mod.txt',
                    #  'Binder14k':'Binder14k.txt',
                     'SemEval2017':'SemEval17.txt',
                     'TR1058':'TR-1058.txt',
                     'combined_nouns':'combined_dataset_nouns.txt',
                     'combined_verbs':'combined_dataset_verbs.txt',
                     'combined_full':'combined_dataset_full.txt',
                     'LNCD_5k':'LNCD/LNCD-5k.txt',
                     'Lee2300':'LBDR/LBDR-2300.txt',
                     'Alternatives_it':'AlternativesIT-1200.txt',
                     'Alternatives_there':'AlternativesTHERE-1200.txt',
                    }
    
    # Define storage dictionaries
    model_embed_storage = {} # for storing the embeddings directly as numpy arrays
    model_sim_storage = {} # for storing pre-computed similarities between word pairs as a dictionary
    dataset_storage = {} # for storing the datasets directly as numpy arrays
    dataset_sim_storage = {} # for storing experimental similarities between word pairs as a dictionary
    
    unique_vocab_set = [] # storage for vocab set
    missing_vocab_storage = {} # storage for missing vocab for each model_dataset pair
    
    
    #### Functions to import vocab, models, and datasets
    
    # Generate full set of unique vocabulary needed to run all the models
    def generate_vocab_set(self):
        """ None -> None
        Collates a full list of all words appearing in any of the datasets loaded.
        """
        full_vocab_set = []
        for dataset in self.datasets:
            if len(self.dataset_storage[dataset])==0: # load dataset if needed
                self.import_dataset(dataset)
            wordset = self.dataset_storage[dataset][0] # get set of words in that dataset
            for wordpair in wordset:
                full_vocab_set.append(wordpair[0])
                full_vocab_set.append(wordpair[1])
        unique_vocab_set = list(set(full_vocab_set)) # unique words only
        unique_vocab_set.sort()
        self.unique_vocab_set = unique_vocab_set
        print('Total tested words:',len(full_vocab_set))
        print('Unique tested words:',len(unique_vocab_set))
        
            
    # Find the combined set of missing vocab over all models for a given dataset
    def missing_vocab_set(self, dataset_name, excluded_models):
        """ string, list_str -> list_str
        Constructs a single consolidated set of missing words for a given dataset, over all the models
        but those in the excluded set. This will be used to conduct a comparison of the models on
        a shared set of vocabulary.
        """
        all_missing = np.array([])
        for model in self.missing_vocab_storage[dataset_name]:
            if model in excluded_models: # skip models to be excluded
                continue
            else:
                model_np = self.missing_vocab_storage[dataset_name][model] # store missing vocab
                all_missing = np.append(all_missing, model_np)
        all_missing = list(set(list(all_missing))) # consolidated set of missing vocab
        return(all_missing)


    # Simple function to print a single model
    def print_model(self, model_name):
        """ string -> None
        Print a summary of the embeddings for a given model.
        """
        if self.model_embed_storage[model_name]==0:
            print('not loaded')
        else:
            print(self.model_embed_storage[model_name][0])
            
            
    # Function to load a specific word embedding model
    def import_model(self, model_name, full_import=False):
        """ string -> None
        Imports an embedding model, storing it in the model_embed_storage dictionary.
        """
        if len(self.model_embed_storage[model_name])==0 or full_import==True: #only if its not already loaded
            
            # open relevant file
            file_loc = self.model_files[model_name]
            filename = self.folder_loc+self.model_loc+file_loc
            with open(filename) as file:
                lines = [line.rstrip('\n') for line in file]

            model_dict = {} # create word dictionary for specific model
            for line in lines:
                word_list = line.split()
                word = word_list[0]
                if full_import==False and word in self.unique_vocab_set: # only  words for testing if full_import==False
                    embedding_list = [float(x) for x in word_list[1:-1]] # store embeddings
                    embedding_np = np.array(embedding_list)
                    model_dict[word] = embedding_np
                elif full_import==True: # this will import all words in the vocab set, not just those for testing
                    embedding_list = [float(x) for x in word_list[1:-1]] # store embeddings
                    embedding_np = np.array(embedding_list)
                    model_dict[word] = embedding_np
                else:
                    continue

            self.model_embed_storage[model_name] = model_dict # store model dictionary in the models dictionary
            print(model_name+' loaded')
            
            
    # Function to import all models into the storage dictionary
    def import_all_models(self, full_import):
        """ None -> None
        Imports all models into the model_embed_storage dictionary.
        """
        for model_name in self.model_embed_storage:
            self.import_model(model_name, full_import)
    
    
    # Function to load word similarity data for specified dataset
    def import_dataset(self, dataset_name):
        """ string -> None
        Imports a dataset, storing a value of the form (list, numpy_array) in the dataset_storage dictionary.
        """
        if len(self.dataset_storage[dataset_name])==0: # if dataset not yet loaded
            file_loc = self.dataset_files[dataset_name]
            filename = self.folder_loc+self.dataset_loc+file_loc
            with open(filename, encoding='utf-8') as file:
                lines = file.readlines()

            wordpairs = [None]*len(lines) # initialise storage
            ratings = [None]*len(lines)
            i=0
            for line in lines:
                line = line.strip() # remove new line chars
                wordpairs[i] = line.split() # split at any whitespace chars
                ratings[i] = float(wordpairs[i][2])
                wordpair_str = wordpairs[i][0]+' '+wordpairs[i][1]
                self.dataset_sim_storage[dataset_name][wordpair_str] = ratings[i] # store experimental data
                i=i+1
            ratings = np.array(ratings)

            self.dataset_storage[dataset_name] = (wordpairs, ratings)

            
    # Function to import all datasets into the storage dictionary
    def import_all_datasets(self):
        """ None -> None
        Imports all datasets into the dataset_embed_storage dictionary.
        """
        for dataset_name in self.dataset_storage.keys():
            self.import_dataset(dataset_name)
    
    
    # Load in sense similarity data (different format than word similarity data)
    def import_sense_sim_data(self, sense_similarity_list, sense_embed_loc):
        for model in sense_similarity_list:
            for transformer_layer in np.arange(1,13):
                
                # load data for a certain model
                filename = model+'_'+str(transformer_layer)+'_'+'combined_verb_results.txt'
                full_address = self.folder_loc+sense_embed_loc+filename
                sim_array = np.loadtxt(full_address,  delimiter=',', dtype='str', encoding='utf-8')
                
                # put data into dictionary
                sim_dict = {}
                for word_pair in sim_array:
                    sim_dict[word_pair[0]] = float(word_pair[1])
                
                # store in main storage dictionary
                model_name = filename[0:-4]
                self.model_sim_storage[model+'_'+str(transformer_layer)] = sim_dict
                self.models.append(model+'_'+str(transformer_layer))

    
    #### Functions to get word embeddings and compute cosine similarities
    
    # Function to get the embedding for a specific word, given a model
    def get_word_embed(self, model_name, word, full_import):
        """ string, string -> numpy_array, boolean
        Gets the word embedding for a specied model and word.
        """
        self.import_model(model_name, full_import)
        model = self.model_embed_storage[model_name]
        missed = False
                       
        embed_dim = len(list(model.keys())[0]) # get embedding length
        try:
            alt_word = self.RW_mod_dict[word] # alterantive word for RW dataset (similar meaning)
        except:
            alt_word = 0 # no alt word available

        word_list = model.keys() # list of all words in the current model
        if word in word_list:
            word_embed = model[word] # get embedding from array
        elif alt_word!=0 and alt_word in word_list: # try alt word if one is defined
            word_embed = model[alt_word]
        elif word.capitalize() in word_list: # see if capitalised is in there
            word_embed = model[word.capitalize()]
        elif word.lower() in word_list: # see if lower case is in there
            word_embed = model[word.lower()]
        elif word[0:-1] in word_list: # see if non-plural is there
            word_embed = model[word[0:-1]]
        elif word[-2:]=='ed' and word[0:-2] in word_list: # see if non-past tense is there
            word_embed = model[word[0:-2]]
        elif word[-3:]=='ing' and word[0:-3] in word_list: # see if non-infinitive form is there
            word_embed = model[word[0:-3]]
        else: # if the word can't be found in the model
            word_embed = np.random.rand(1,embed_dim)[0] # random embedding
            missed = True

        word_embedding = np.array(word_embed)
        return(word_embedding, missed)
    
    
    # Function to calculate cosine similarity between two embeddings
    def cosine_sim(self, embed_1, embed_2):
        """ numpy_array, numpy_array -> float
        Returns the cosine similarity (-1 to 1) between two embeddings, inputted as vectors.
        """
        if np.dot(embed_1,embed_2) == 0:
            similarity = 0 # don't normalise if similarity is zero
        else:
            similarity = np.dot(embed_1,embed_2)/(np.linalg.norm(embed_1)*np.linalg.norm(embed_2))
            #similarity, _ = spearmanr(embed_1, embed_2)
        return(similarity)
    
    
    # Function to calculate cosine similarity between two words
    def cosine_sim_words(self, word_1, word_2, model_name):
        """ string, string -> float
        Returns the cosine similarity (-1 to 1) between two words, inputted as strings.
        """
        embed_1 = self.get_word_embed(model_name, word_1, False)[0]
        embed_2 = self.get_word_embed(model_name, word_2, False)[0]
        similarity = self.cosine_sim(embed_1, embed_2)
        return(similarity)
      
    
    # Function to compute the similarities of all word pairs from a given database and for a given model
    def compute_model_sims(self, model_name, dataset_name):
        """ string, string -> list_flt
        Returns a list of similarities for all word pairs in a specified dataset, using a specified model.
        """
        self.import_dataset(dataset_name) # load dataset if needed
        dataset_words = self.dataset_storage[dataset_name][0] # word pairs are stored in [0]
        embed_sims = [None]*len(dataset_words)
        missing_vocab = []
        i=0
        for word_pair in dataset_words:
            # get embeddings for both words in word pair
            word_embed_1,miss_1 = self.get_word_embed(model_name, word_pair[0], full_import=False)
            word_embed_2,miss_2 = self.get_word_embed(model_name, word_pair[1], full_import=False)
            if miss_1==True or miss_2==True:
                embed_sims[i] = math.nan # return NaN if either word is missing
            else:
                embed_sims[i] = self.cosine_sim(word_embed_1, word_embed_2)
            
            # keep track of which words are missing
            if miss_1==True:
                missing_vocab.append(word_pair[0])
            if miss_2==True:
                missing_vocab.append(word_pair[1])
            
            # store similarities in dictionary
            word_pair_str = word_pair[0]+' '+word_pair[1] # word pair as string for dictionary storage
            self.model_sim_storage[model_name][word_pair_str] = embed_sims[i] # store model-based similarity             
            i=i+1
                
        dict_key = model_name+'_'+dataset_name # store missing vocab by model and dataset
        self.missing_vocab_storage[dataset_name][model_name] = missing_vocab # add missing words to storage dictionary
        return(embed_sims)
    
    
    # Computes and stores the similarities of all word pairs for all dataset and model combinations
    def compute_all_model_sims(self):
        """ None -> None
        Stores for later use the set of pairwise word similarities for all dataset and model combinations.
        """
        for model_name in self.model_files:
            for dataset_name in self.dataset_files:
                self.compute_model_sims(model_name, dataset_name)
            print(model_name+' similarities computed')
    
    # Function to compute and store all model vs dataset similarities
    def store_model_dataset_sims(self, model_name, dataset_name, words_to_omit, full_import, printing=False):
        """ string, string, list_str, boolean, boolean -> (list_flt, list_flt, float, list_str, list_str, list_str)
        Constructs a set of words present in both the given model and dataset, returning their cosine similarities for
        the model and dataset, as well as the excluded words, words missing, words included, and wordpairs included.
        """
        # ensure similarities for relevant model and dataset are loaded and computed
        self.import_dataset(dataset_name) # load dataset if needed
        dataset_sims = self.dataset_sim_storage[dataset_name]
        self.import_model(model_name, full_import) # load model if needed
        if self.model_sim_storage[model_name] == {}: # compute model similarities if needed
            self.compute_model_sims(model_name, dataset_name)

        # initialise storage lists and some key variables
        scale = max(list(self.dataset_sim_storage[dataset_name].values())) # get scale of experimental data
        excluded_words = []
        missing_words = []
        included_words = []
        included_model_sims = []
        included_dataset_sims = []
        included_wordpairs = []
        unique_words = []
        
        # loop over all wordpairs in the relevant dataset storage
        for word_pair in self.dataset_sim_storage[dataset_name].keys():
            word_1, word_2 = word_pair.split()
            dataset_similarity = self.dataset_sim_storage[dataset_name][word_pair]
            model_similarity = self.model_sim_storage[model_name][word_pair]
            unique_words.append(word_1)
            unique_words.append(word_2)
            
            if word_1 in words_to_omit or word_2 in words_to_omit: # check for words to omit from analysis
                if word_1 in words_to_omit: 
                    excluded_words.append(word_1)
                if word_2 in words_to_omit:
                    excluded_words.append(word_2)
                
            elif np.isnan(model_similarity): # check for missing words (excluding those omitted above)
                if word_1 not in self.model_embed_storage[model_name]:
                    missing_words.append(word_1)
                elif word_2 not in self.model_embed_storage[model_name]:
                    missing_words.append(word_2)
                    
            else: # only append the word pairs we want and are actually present
                included_words.append(word_1)
                included_words.append(word_2)
                included_dataset_sims.append(dataset_similarity)
                included_model_sims.append(model_similarity)
                included_wordpairs.append(word_pair)
        
        if printing==True: # printing results
            num_unique_words = len(list(set(unique_words)))
            num_included_words = len(set(included_words))
            num_excluded_words = len(set(excluded_words))
            num_missing_words = len(set(missing_words))
            print('evaluating '+model_name+' against '+dataset_name)
            print('included words: '+str(num_included_words)+' out of '+str(num_unique_words))
            print('excluded words: '+str(num_excluded_words)+' out of '+str(num_unique_words))
            print('missing words: '+str(num_missing_words)+' out of '+str(num_unique_words))
        return(included_dataset_sims, included_model_sims, scale, excluded_words, missing_words, included_words, included_wordpairs)
    
    
    # Function to compute the correlation between a given set of model and dataset embedding similarities
    def compute_embed_correls(self, dataset_similarities, model_similarities, scale, printing=False):
        """ list_flt, list_flt, int, boolean -> (list_flt, list_flt, list_flt)
        Computes the pearson_r and spearman_r between word similarities for a dataset and model.
        """
        pearson_r = np.corrcoef(dataset_similarities, model_similarities)[0,1]
        spearman_r, p = spearmanr(dataset_similarities, model_similarities)
        differences = np.array(model_similarities)-np.array(dataset_similarities)/scale # model minus dataset
              
        if printing==True: # printing results
            print('pearson: {:.3f}'.format(pearson_r), '\nspearman: {:.3f}\n'.format(spearman_r))
        return(pearson_r, spearman_r, differences)
    
    
    # Saves a full set of model comparison results to a text file 
    def save_results(self, model_name, dataset_name, word_pairs, differences, data, model):
        """ str, str, list_str, list_flt, list_flt, list_flt -> None
        Saves a list of word pair model similarities, experimental similarities, differences between them,
        and the word pairs themselves in a simple text file.
        """
        filename = model_name+'_'+dataset_name+'_results.txt'
        save_file = open(filename, 'a', encoding='utf-8')
        i=0
        for word_pair in word_pairs:
            save_file.writelines(word_pair+','+str(differences[i])+','+str(data[i])+','+str(model[i]))
            save_file.write('\n')
            i=i+1
        save_file.close()
        print('Results saved')

        
    # Function to compute the correlation between all the models
    def compute_models_corr_matrix(self, included_models, vocab_set, excluded_vocab):
        """ list_str, list_str, list_str -> list_flt
        Takes a list of embedding models, a set of word pairs, and a set of words to exlude,
        and computes the correlation between experimental similarites and model embedding similarities
        for all the models in the given list over all word pairs in vocab_set, less the excluded_vocab.
        """
        n = len(included_models) # get number of models
        corr_storage = np.zeros((n,n))
        i=0
        j=0

        for model_1_name in list(included_models): # loop over all model pairs
            j=0
            for model_2_name in list(included_models):

                # initialise storage lists and some key variables
                excluded_words = []
                missing_words_model_1 = []
                missing_words_model_2 = []
                included_words = []
                included_model_1_sims = []
                included_model_2_sims = []
                included_wordpairs = []
                unique_words = []

                # loop over all wordpairs in the relevant dataset storage
                for word_pair in self.dataset_sim_storage[vocab_set].keys():
                    word_1, word_2 = word_pair.split()
                    model_1_similarity = self.model_sim_storage[model_1_name][word_pair]
                    model_2_similarity = self.model_sim_storage[model_2_name][word_pair]
                    unique_words.append(word_1)
                    unique_words.append(word_2)

                    if word_1 in excluded_vocab or word_2 in excluded_vocab: # check for words to omit from analysis
                        if word_1 in excluded_vocab: 
                            excluded_words.append(word_1)
                        if word_2 in excluded_vocab:
                            excluded_words.append(word_2)

                    elif np.isnan(model_1_similarity): # check for missing words (excluding those omitted above)
                        if word_1 not in self.model_embed_storage[model_1_name]:
                            missing_words_model_1.append(word_1)
                        elif word_2 not in self.model_embed_storage[model_1_name]:
                            missing_words_model_1.append(word_2)
                    elif np.isnan(model_2_similarity):
                        if word_1 not in self.model_embed_storage[model_2_name]:
                            missing_words_model_2.append(word_1)
                        elif word_2 not in self.model_embed_storage[model_2_name]:
                            missing_words_model_2.append(word_2)

                    else: # only append the word pairs we want and are actually present
                        included_words.append(word_1)
                        included_words.append(word_2)
                        included_model_1_sims.append(model_1_similarity)
                        included_model_2_sims.append(model_2_similarity)
                        included_wordpairs.append(word_pair)
                
                # get counts for included, missing, and excluded words
                num_unique_words = len(set(unique_words))
                num_included_words = len(set(included_words))
                num_excluded_words = len(set(excluded_words))
                num_missing_words = len(set(missing_words_model_1+missing_words_model_2))
                print('evaluating '+model_1_name+' against '+model_2_name)
                print('included words: '+str(num_included_words)+' out of '+str(num_unique_words))
                print('excluded words: '+str(num_excluded_words)+' out of '+str(num_unique_words))
                print('missing words: '+str(num_missing_words)+' out of '+str(num_unique_words))
                
                # calculate correlation between model_1 and model_2
                pearson_r = np.corrcoef(np.array(included_model_1_sims), np.array(included_model_2_sims))[0,1]
                spearman_r, p = spearmanr(np.array(included_model_1_sims), np.array(included_model_2_sims))
                print('pearson: {:.3f}'.format(pearson_r), '\nspearman: {:.3f}\n'.format(spearman_r))
                
                corr_storage[i][j] = spearman_r
                j=j+1
            i=i+1
        return(corr_storage)
       
        
    #### Bootstrapping functions for computing error bars
        
    # Bootstrap a set of draws for a given dataset and model
    def single_bootstrap(self, dataset_similarities, model_similarities, scale):
        """ list_flt, list_flt, int -> float, float, list_flt
        Take a set of draws with replacement from word pairs in dataset_similarities, computing the correlation
        of selected pairs with corresponding word pairsi n model_similarities. The experimental similarities
        are adjusted by scale to range from 0-1. Returns a the pearson and spearman correlations for the
        bootstrap, and a list of pairwise differences in similarity from model to experimental.
        """
        n = len(dataset_similarities) # number of word pairs
        rd_vocab_model_sims = np.zeros(n)
        rd_vocab_dataset_sims = np.zeros(n)
        
        # Generate bootstrap sample
        index_list = np.arange(0,n)
        rd_indices = np.random.choice(index_list,n) # get random indices for word pairs in bootstrap
        i=0
        for index in rd_indices:
            rd_vocab = rd_indices[index] # get random vocab pair
            rd_vocab_model_sims[i] = model_similarities[index] # add model similarity
            rd_vocab_dataset_sims[i] = dataset_similarities[index] # add experimental similarity
            i=i+1

        # Calculate correlation and pairwise differences
        pearson_r = np.corrcoef(rd_vocab_dataset_sims, rd_vocab_model_sims)[0,1] # correlation with masks
        spearman_r, p = spearmanr(rd_vocab_dataset_sims, rd_vocab_model_sims)
        differences = (rd_vocab_dataset_sims/scale)-rd_vocab_model_sims
        return(pearson_r, spearman_r, differences)
    
    
    # Compute a series of bootstrap samples for a given model and dataset
    def set_of_bootstraps(self, model_name, dataset_name, words_to_omit, num_samples):
        """ str, str, list_str, int -> list_flt, list_flt
        Computes a number (num_samples) of boostraps comparing the specified model and datasets.
        Ignores any words in words_to_omit. Returns lists of pearson and spearman correlations.
        """
        corr_samples_pearson = []
        corr_samples_spearman = []
        for run in np.arange(0,num_samples):
            data_sims, model_sims, scale, _, _, _, _ = self.store_model_dataset_sims(model_name, dataset_name, words_to_omit, full_import=False, printing=False)
            correlations = self.single_bootstrap(data_sims, model_sims, scale)
            corr_samples_pearson.append(correlations[0])
            corr_samples_spearman.append(correlations[1])
        return(corr_samples_pearson, corr_samples_spearman)
    
       
    # Plot bootstrap results for a given dataset
    def plot_model_bootstraps(self, dataset_name, words_to_omit, excluded_models, num_samples, printing=True):
        """ str, list_str, list_str, int, bool -> list_flt, list_flt, list_flt
        Plot the boostrapped correlations between similarities of all models apart from those 
        in excluded_models and the specified experimental dataset. All word pairs are used 
        apart from those in words_to_omit. Returns lists of pearson and spearman correlations.
        """
        mean_correlations = np.array([])
        std_correlations = np.array([])
        CI_95_percents = np.array([])
        models_to_plot = [model for model in self.models if model not in list(excluded_models)]
        
        # compute mean and std dev of correlations between all models and specified dataset
        for model_name in models_to_plot:
            pearson_corrs, spearman_corrs = self.set_of_bootstraps(model_name, dataset_name, words_to_omit, num_samples)
            mean_correlations = np.append(mean_correlations, np.mean(spearman_corrs))
            std_correlations = np.append(std_correlations, np.std(spearman_corrs))
            pt_25, pt_975 = np.percentile(spearman_corrs, [2.5,97.5]) # get percentiles for 95% CI
            CI_95_percents = np.append(CI_95_percents, (pt_975-pt_25)/2)
        
        # plot barplot
        if printing==True:
            fig, ax = plt.subplots(figsize=(30,4))
            ax.bar(x=models_to_plot, #x-coordinates of bars
                height=mean_correlations, #height of bars
                yerr=CI_95_percents, #error bar width
                capsize=6) #length of error bar caps
            ax.set_ylabel('Correlation with '+dataset_name)
            plt.show()
        return(mean_correlations, std_correlations, CI_95_percents)


    # Code to call the bootstraps function for a given dataset
    def plot_multi_models_bootstraps(self, dataset_name, models_to_plot, num_samples=100, printing=True):
        """ str, list_str, int, bool -> list_flt, list_flt, list_flt
        Performs a boostrap calculation of mean correlations for specified models against specified datset.
        Num_samples specifies the number of bootstraps to take. Returns the mean, standard deviation, 
        and list of errors.
        """
        model_set = np.arange(0,len(np.array(self.models)))
        excluded_indices = np.array([idx for idx in model_set if idx not in models_to_plot]) # all models less those to plot
        excluded_models = [self.models[idx] for idx in excluded_indices]
        excluded_vocab = self.missing_vocab_set(dataset_name, excluded_models) # computed automatically
        means, stds, errors = self.plot_model_bootstraps(dataset_name, excluded_vocab, excluded_models, num_samples, printing)
        if printing==True:
            print('missing words: '+str(len(excluded_vocab)))
            print('means:', means)
            print('stds:', stds)
            print('errors:', errors)
        return(means, stds, errors)
    
    
    #### Functions for plotting scatterplots and regression lines
    
    # Polynomial regression
    def poly_reg(self, data):
        """ list_flt -> reg_object
        Returns a polynomial regression function for plotting.
        """
        xs = data[:,0]
        ys = data[:,1]
        xs_sq = PolynomialFeatures(degree=2).fit_transform(xs.reshape(-1, 1))
        regression = LinearRegression()
        regression.fit(xs_sq, ys)
        return(regression)


    # Linear regression
    def lin_reg(self, data):
        """ list_flt -> reg_object
        Returns a linear regression function for plotting.
        """
        xs = data[:,0]
        ys = data[:,1]
        regression = LinearRegression()
        regression.fit(xs.reshape(-1, 1), ys)
        return(regression)


    # Function to add subplot to main plot with regression lines
    def gen_sub_plot(self, data, fig, plot_num, y_label, x_label, linear_reg='True', poly_reg='False', label_size=14):
        """ list_flt, fig_obj, int, str, str, bool, bool, int -> None
        Produces a subplot to use for plotting with regression lines. Requires data to plot, the figure object,
        the subplot number, x and y axis labels, and booleans specifying which regressions to include.
        """
        
        # generate figure
        xs = data[:,0]
        ys = data[:,1]
        fig_axis = fig.add_subplot(*plot_num) #create figure subplot
        fig_axis.scatter(xs, ys, s=2) # 's' gives dot size
        fig_axis.set_xlim(0,1)
        fig_axis.set_ylim(-0.1,1)
        fig_axis.tick_params(axis='both', which='major', labelsize=label_size)
        
        # calculate values needed for regressions and plotting
        x_range = np.arange(0, 1, step=0.01)
        x_range_sq = PolynomialFeatures(degree=2).fit_transform(x_range.reshape(-1, 1))
        xs_sq = PolynomialFeatures(degree=2).fit_transform(xs.reshape(-1, 1))
        
        # add linear reg line
        if linear_reg==True:
            lin_regression = self.lin_reg(data) # linear regression
            fig_axis.plot(x_range, lin_regression.predict(x_range.reshape(-1, 1)), color='blue', linewidth=1) 
            r_square_lin = r2_score(ys, lin_regression.predict(xs.reshape(-1, 1)))
            fig_axis.text(0.05, 0.85, 'R-sq: {:.3f}'.format(r_square_lin))
        
        # add poly reg line
        if poly_reg==True:
            poly_regression = self.poly_reg(data) # polynomial regression
            fig_axis.plot(x_range, poly_regression.predict(x_range_sq), color='red', linewidth=1) 
            r_square_poly = r2_score(ys, poly_regression.predict(xs_sq))
            fig_axis.text(0.05, 0.85, 'R-sq: {:.3f}'.format(r_square_poly))
        
        # add labels
        # fig_axis.title.set_text(subheading) # subfigure figure title
        fig_axis.set(ylabel=y_label, xlabel=x_label)
        fig_axis.set_ylabel('model_name', fontsize=14)
        
        
    #### Assorted other functions
    
    # Plot dendrogram for a given model and set of words
    def plot_dendrogram(self, model, word_file):
        # load the list of words to plot a dendrogram for
        file_path = self.folder_loc+'Corpus Data\Key vocab\\'+word_file+'.txt'
        with open(file_path) as file:
             word_list = [line.rstrip('\n').lower() for line in file]
        
        # get the set of embeddings for the word list
        self.import_model(model, full_import=True)
        embeddings_list = []
        for word in word_list:
            word_embed = self.get_word_embed(model, word, full_import=False)[0]
            embeddings_list.append(word_embed)
            if len(word_embed)<10: # if a word doesn't load properly
                return('Problem with',word)                
        embeddings_list = np.array(embeddings_list)    

        # for linkage, complete, weighted, average are the main methods
        Z = linkage(embeddings_list, method='average', metric='cosine', optimal_ordering=True) 
        fig = plt.figure(figsize=(10,30))
        dn = dendrogram(Z, labels=word_list, orientation='right', distance_sort=True, leaf_font_size=12)
        plt.title(word_file+' dendrogram', fontsize=20)
        plt.show()


    # Run multiple regression over several models
    def multi_regression(self, model_sims, dataset_sims, model_indices, model_names):
        x = np.array(model_sims).transpose()
        y = np.array(dataset_sims)
        reg_model = LinearRegression(fit_intercept=True)
        reg_model.fit(x[:,model_indices], y)
        R_square = reg_model.score(x[:,model_indices], y)
        effective_corr = R_square**0.5 # get the effect correlation of the reg model
        
        print('Regression coefficients:')
        print([name[0:4] for name in model_names])
        print(reg_model.coef_)
        print('rho: {:.4f}'.format(effective_corr))
        return(effective_corr)
    