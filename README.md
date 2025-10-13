# Scientific Papers from ACL Anthology

> The two subsets might have overlaps.

## CL-SciSumm

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

- [README](data_scisumm/README_CL-SciSumm.md)
- [LICENSE](data_scisumm/LICENCE_CL-SciSumm)
- Original repository: [link](https://github.com/WING-NUS/scisumm-corpus/)

### LLM Summarisation and Simplification

We create example machine summarisations and simplifications of the papers in this subset using `gpt-oss 120b` through Ollama.

Before running either one of `summarise_gpt.py` or `simplify_gpt.py` make sure Ollama is installed on your system and run the following command to prepare Ollama to use `gpt-oss 120b` as a cloud model instead of downloading it.

~~~~bash
ollama pull gpt-oss:120b-cloud
~~~~

If you want to change model to either a local one or a cloud one, make sure to update the scripts.

## ACL-OCL

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- [README](data_acl-ocl/README_ACL-OCL.md)
- [LICENSE](data_acl-ocl/LICENCE_ACL-OCL)
- Original repository: [link](https://github.com/shauryr/ACL-anthology-corpus/)