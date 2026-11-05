# Corpus Processing for MIRA User Study

This repository contains the code to load, pre-process, and upload to Hugging Face the corpus used in the evaluation of MIRA.

Check out our work here:

```bibtext
TBA
```

Check out the MIRA codebase here: [LCorti/mira_codebase/](https://github.com/LCorti/mira_codebase/).

## Corpus details

The corpus we used for our project is based on CL-SciSumm: a collection of *CL papers from the ACL Anthology as well as their human-curated summaries.

- [Original repository](https://github.com/WING-NUS/scisumm-corpus/)
- [LICENSE](raw_data/LICENCE_CL-SciSumm)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Pre-processing

**Notes**:
- We only considered data related to Task 2.
- Several folders were deleted (including PDFs).
- Since we are not interested in training or benchmarking any model in particular, the train and test splits have been merged into one.
- Because of licensing, LaySumm data is completely excluded.

**Steps**:
1. Copied the gold summaries from ```Test-Set-2018-Gold/Task 2``` to the respective directories in ```Test-Set-2018/[AA_paper_ID]/summary```; Task 1 was ignored.
2. Removed ```Test-Set-2018-Gold``` since it is no longer useful in this case.
3. Things related to LaySumm were ignored and removed.
4. Removed ```publications```.
5. Removed PDFs from ```Development-Set-Apr8```, ```Test-Set-2016```, ```Test-Set-2017```.
6. Removed TXTs from ```Development-Set-Apr8```, ```Test-Set-2016```, ```Test-Set-2017```.
7. Several paper directories for ```Test-Set-2016```, ```Test-Set-2017```, ```Test-Set-2018```, ```Training-Set-2017```, and ```Training-Set-2018``` contain a folder named with the paper ID from ACL Anthology. These folders have the same contents as the respective Reference_XML and Citance_XML. I removed them too.
8. Renamed the sub-directories in ```TRAINING-SET-2016``` to remove ```_TRAIN``` the names.
9. Removed ```README.md``` files from other years. They can all be found in the original repository.
10. Removed ```Task1``` from ```TRAINING-SET-2019``` and moved the contents of ```Task2``` into the main directory.

## Scenario preparation

### Text Summarisation Scenario

-> [**Directory**](./summarisation/)

We directly used the human-curated summaries available in the datasets above.

We also use `gpt-oss 120b`, hosted through Ollama ([link](https://ollama.com/library/gpt-oss:120b-cloud)), to experiment and create synthetic summarisations.

### Text Simplification Scenario

-> [**Directory**](./simplification/)

We generated synthetic simplifications of the documents using `gpt-oss 120b`, hosted through Ollama ([link](https://ollama.com/library/gpt-oss:120b-cloud)).

### Retrieval/Reranking Scenario

-> [**Directory**](./retrieval/)

Here, we used `gpt-oss 120b`, hosted through Ollama ([link](https://ollama.com/library/gpt-oss:120b-cloud)), self-hosted `Qwen/Qwen3-Embedding-0.6B` to produce queries over the corpus as well as relevance judgements.

The procedure is detailed in [this file](./retrieval/README.md).
