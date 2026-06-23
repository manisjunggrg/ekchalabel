#!/bin/bash
# Download spaCy English model if not already present
python -m spacy download en_core_web_sm --quiet
