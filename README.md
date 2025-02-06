# <div align="center"> PU Learning for DR Gene Discovery </div>

### <div align="center"> Jorge Paz-Ruza*, Alex A. Freitas, Amparo Alonso-Betanzos, Bertha Guijarro-Berdiñas <br> <br> [Positive-Unlabelled learning for identifying new candidate <br> Dietary Restriction-related genes among ageing-related genes](https://doi.org/10.1016/j.compbiomed.2024.108999) </div>

##### <div align="center"> Published on <b>Computers in Biology and Medicine</b>, Vol. 180, 2024</div>


<br>


### 1. Abstract

<p align="justify"> Dietary Restriction (DR) is one of the most popular anti-ageing interventions; recently, Machine Learning (ML) has been explored to identify potential DR-related genes among ageing-related genes, aiming to minimize costly wet lab experiments needed to expand our knowledge on DR. However, to train a model from positive (DR-related) and negative (non-DR-related) examples, the existing ML approach naively labels genes without known DR relation as negative examples, assuming that lack of DR-related annotation for a gene represents evidence of absence of DR-relatedness, rather than absence of evidence. This hinders the reliability of the negative examples (non-DR-related genes) and the method’s ability to identify novel DR-related genes. This work introduces a novel gene prioritization method based on the two-step Positive-Unlabelled (PU) Learning paradigm: using a similarity-based, KNN-inspired approach, our method first selects reliable negative examples among the genes without known DR associations. Then, these reliable negatives and all known positives are used to train a classifier that effectively differentiates DR-related and non-DR-related genes, which is finally employed to generate a more reliable ranking of promising genes for novel DR-relatedness. Our method significantly outperforms (p &lt;0.05) the existing state-of-the-art approach in three predictive accuracy metrics with up to ~40% lower computational cost in the best case, and we identify 4 new promising DR-related genes (PRKAB1, PRKAB2, IRS1, PRKAG1), all with evidence from the existing literature supporting their potential DR-related role.</p>

### 2. Setup

#### 2.1. Environment
- The code in this repository has been tested with Python
- You can install all required packages with `pip install -r requirements.txt`
- This framework was executed in a dedicated Windows 10 Pro machine with an Intel Core i7-10700K CPU @ 3.80GHz, 16GB RAM, and an NVIDIA GeForce 2060 GPU Super.

### 3. Usage

- To train an experiment with a specific hyperparameter configuration, run `python -m code.main --dataset DATASET --classifier CLASSIFIER [--random_state RANDOM_STATE] [--pu_learning] [--pu_k K] [--pu_t T]. This will run a a single outer cross-validation experiment with the specified hyperparameters, for 10 different random seeds. 
 - For instance, running `python -m code.main --dataset "GO" --classifier "BRF" --pu_learning "similarity" --pu_k 8 --pu_t 0.875`
- To train an experiment with multiple hyperparameter configurations (k and t values), run `python -m code.main --dataset DATASET --classifier CLASSIFIER [--random_state RANDOM_STATE] [--pu_learning "similarity"|"threshold"|"False"] [--pu_k K1 K2 ...] [--pu_t T1 T2 ...]`. This will run an inner cross-validation for each hyperparameter configuration (essentially, a standard 10x5 CV), for 10 different random seeds. 
 - For instance, running `python -m code.main --dataset "GO" --classifier "BRF" --pu_learning "similarity" --pu_k 8 16 32 --pu_t 0.875 0.9 0.925`   

### 3.1 Neptune logging

- By default, the code logs experiment results through on-screen printing and CSV files.
- To log the results of the experiments to Neptune, you need to set up a Neptune account and create a project. You can find more information on how to set up Neptune [here](https://docs.neptune.ai/getting-started/installation).
- In the `main.py` file, you need to set the neptune api token and the project name in main.py.       

### 5. Citation

- If you use this code or reference this work, we encourage citing the journal paper:

  - APA:
    ```
    Paz-Ruza, J., Freitas, A. A., Alonso-Betanzos, A., & Guijarro-Berdiñas, B. (2024). Positive-Unlabelled Learning for Identifying New Candidate Dietary Restriction-related Genes among Ageing-related Genes. Computers in Biology and Medicine, Vol. 180, 108999, https://doi.org/10.1016/j.compbiomed.2024.108999
    ```

  - Bibtex:
    ```
    @article{PAZRUZA2024108999,
    title = {Positive-Unlabelled learning for identifying new candidate Dietary Restriction-related genes among ageing-related genes},
    journal = {Computers in Biology and Medicine},
    volume = {180},
    pages = {108999},
    year = {2024},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2024.108999},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482524010849},
    author = {Jorge Paz-Ruza and Alex A. Freitas and Amparo Alonso-Betanzos and Bertha Guijarro-Berdiñas},
    keywords = {Machine Learning, Positive-Unlabelled learning, Bioinformatics, Ageing, Dietary Restriction},
}
    ```