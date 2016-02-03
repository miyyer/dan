#!/bin/bash

# Create directory for data and models
mkdir -p data
mkdir -p models

# Get trees
loc=trainDevTestTrees_PTB.zip
curl -O http://nlp.stanford.edu/sentiment/$loc
unzip $loc -d data
mv data/trees data/sentiment
rm -f $loc

# Get Glove 300d embeddings
curl -O http://www-nlp.stanford.edu/data/glove.840B.300d.zip
mv glove.840B.300d.zip data/

# Convert tree format
cd preprocess
python sentiment_trees.py
python load_embeddings.py
