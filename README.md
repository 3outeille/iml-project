# IML project

- The goal of this project is to develop one of the building blocks of a larger pipeline, which aims at monitoring Dobble games. Students have to build the feature extraction and classification part.

## Setup

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Pipeline
- To build a docker container: `make build`
- To train the model: `make train`
- To run a prediction in a docker container: `make predict`
- To run the evaluation: `make submission`