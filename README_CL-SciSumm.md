# README CL-SciSumm

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Here you can find most of the data from [CL-SciSumm](https://github.com/WING-NUS/scisumm-corpus).

The goal here is to have a simple collection of scientific papers in plain text for testing machine summarisation with LLMs.

Note that
- Only the contents related to Task 2 are considered.
- I deleted several folders and files (especially PDFs) to reduce the size of the repository.
- I merged the training and test splits into a single corpus. If this is not what your are looking for, you should easily exclude those datasets using the ```parse_scisumm.ipynb``` notebook.
- LaySumm is completely excluded.

Please refer to the original repository ([https://github.com/WING-NUS/scisumm-corpus/](https://github.com/WING-NUS/scisumm-corpus/)) for all the details and the original contents (incl. revisions).

## Data Processing Steps
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

## Attribution

Please, follow the indications of the authors and (1) let them know and (2) use their CL-SciSumm 2019 task overview paper for citation/attribution:

```
@inproceedings{,
    title={Overview and Results: CL-SciSumm Shared Task 2019},
    author={Chandrasekaran, Muthu Kumar and Yasunaga, Michihiro and Radev, Dragomir and Freitag, Dayne and Kan, Min-Yen},
    booktitle={In Proceedings of Joint Workshop on Bibliometric-enhanced Information Retrieval and NLP for Digital Libraries (BIRNDL 2019)},
    year={2019}
}
```

## CL-SciSumm Organisers' Contacts

<a href="https://www.comp.nus.edu.sg/~kanmy/">Min-Yen Kan</a> National University of Singapore, [kanmy@comp.nus.edu.sg](mailto:kanmy@comp.nus.edu.sg)
