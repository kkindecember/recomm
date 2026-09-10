# RaSeRec
The code of our paper "RaSeRec: Retrieval-Augmented Sequential Recommendation" [[pdf]](https://arxiv.org/abs/2412.18378).

# Ckpts
We have provided checkpoints trained on the Beauty datasets under the log directory.

# Framework
We propose a new SeRec learning paradigm, RaSeRec, which explores RAG in sequential recommendation (SeRec) to solve issues existing in previous paradigms, i.e., preference drift and implicit memory.
![image](https://github.com/user-attachments/assets/2563e6ad-76a0-4181-85ef-7ac0aed7a3a9)

# Main Results
## Comparison with Baselines
![image](https://github.com/user-attachments/assets/a666af15-cd31-45fa-be3e-6ac41e996620)
## Improving Base Backbones
![image](https://github.com/user-attachments/assets/f98dde00-4071-4eef-b0e9-d0b68c4e0c69)
## Parameter Sensitivity
![image](https://github.com/user-attachments/assets/ef153798-8d84-4276-a130-0d9ddfc82504)


# Usage
We have provided the Beatuty dataset. More datasets can be downloaded from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.
```
$ tree
├── Amazon_Beauty
    ├── Amazon_Beauty.inter
    └── Amazon_Beauty.item
```

Run `raserec.sh`.


# Cite

If you find this repo useful, please cite
```
@misc{zhao2024raserec,
    title={RaSeRec: Retrieval-Augmented Sequential Recommendation},
    author={Xinping Zhao and Baotian Hu and Yan Zhong and Shouzheng Huang and Zihao Zheng and Meng Wang and Haofen Wang and Min Zhang},
    year={2024},
    eprint={2412.18378},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```

# Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DuoRec](https://github.com/RuihongQiu/DuoRec).
