# EBERT
This repository serves as the official code release of the paper [EBERT: Efficient BERT Inference with Dynamic Structured Pruning](https://aclanthology.org/2021.findings-acl.425/) (pubilished at Findings of ACL 2021).

<div align=center>
<img src=EBERT.png>
</div>

EBERT is a dynamic structured pruning algorithm for efficient BERT inference. Unlike previous methods that randomly prune the model weights for static inference, EBERT dynamically determines and prunes the unimportant heads in multi-head self-attention layers and the unimportant structured computations in feed-forward network for each input sample at run-time.

## Citation
If you found the library useful for your work, please kindly cite our work:
```
@inproceedings{liu-etal-2021-ebert,
    title = "{EBERT}: Efficient {BERT} Inference with Dynamic Structured Pruning",
    author = "Liu, Zejian  and
              Li, Fanrong  and
              Li, Gang  and
              Cheng, Jian",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.425",
    doi = "10.18653/v1/2021.findings-acl.425",
    pages = "4814--4823",
}
```

## Prerequisites

The code has the following dependencies:
* python >= 3.8.5
* pytorch >= 1.4.0
* transformers = 3.3.1


## Only key codes are provided now, and completed codes will be released soon.