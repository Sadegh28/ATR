## Python-Based Information Theoretic Multi-Label Feature Selection (PyIT-MLFS) Library 

### Dependencies and Installation
* python >= 3.8
* numpy
* pyitlib
* sklearn
* skmultilearn
* tqdm

1. Clone Repo
```
git clone https://github.com/Sadegh28/PyIT-MLFS.git
```

2. Create Conda Environment
```
conda create --name PyIT_MLFS python=3.8
conda activate PyIT_MLFS
```

3. Install Dependencies
```
pip install pyitlib 
conda install -c conda-forge scikit-learn
pip install scikit-multilearn
conda install -c conda-forge numpy
pip install tqdm
```

### Get Started

#### Mulan Datasets
Use the following command to rank features of a dataset from the mulan repository:
```
python PyIT-MLFS.py   --datasets   d1, d2, ..., dn   --fs-methods a1, a2, ..., am


```
Each di must be a mulan dataset: 
        ```
        {'Corel5k', 'bibtex', 'birds', 'delicious', 'emotions',
        'enron', 'genbase', 'mediamill', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3',
        'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast'}
        ```
and each ai must be a multi-label feature selection method supportd by PyIT-MLFS library: 
        ```
        {'LRFS', 'PPT_MI', 'IGMF', 'PMU', 'D2F', 'SCLS'
              'MDMR', 'LSMFS', 'MLSMFS' }
        ```
