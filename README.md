# Deep Averaging Networks (DAN)
code for model described in
<http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf> along with negation dataset (negation_dataset.txt). 
feel free to email me at miyyer@umd.edu with any comments/problems/questions/suggestions.

### dependencies: 
- python 2.7.9, numpy 1.9.2 (might work w/ other versions but not tested), nltk

### commands to run DAN on Stanford Sentiment Treebank:
- bash run.sh (downloads word embeddings and dataset, preprocesses PTB trees into DAN format)
- python dan_sentiment.py (can tweak hyperparameters via command-line arguments, currently this runs the fine-grained experiment on only root-level labels and should take a few minutes to finish training)

### QA DAN code available in the repository for our full quiz bowl system 
- https://github.com/Pinafore/qb/blob/master/qanta/guesser/dan.py

### DAN input format (for your own data!):
- each training/test instance must be a tuple with the following format: ([list of word embedding lookup indices associated with text], label)
- if you want to use pretrained word embeddings, you should also pass a pickled matrix using the --We argument, where the matrix is of size d x V (each column stores the embedding for the corresponding word lookup index)

### important hyperparameters:
- batch size (the smaller the better, but also slower)
- adagrad initial learning rate (should be decreased as the batch size is decreased)
- word dropout probability (30% is the default but might be too high for some tasks)
- number of epochs (increase when using random initialization)

if you use this code, please cite:

    @InProceedings{Iyyer:Manjunatha:Boyd-Graber:III}-2015,
        Title = {Deep Unordered Composition Rivals Syntactic Methods for Text Classification},
        Booktitle = {Association for Computational Linguistics},
        Author = {Mohit Iyyer and Varun Manjunatha and Jordan Boyd-Graber and Hal {Daum\'{e} III}},
        Year = {2015},
        Location = {Beijing, China}
    }
