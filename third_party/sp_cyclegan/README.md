# Structure-preserving image translation for depth estimation in colonoscopy
We provide our PyTorch implementation of structure-preserving image translation using a mutual information loss to enforce depth consistency. This code is adapted from the implementation of [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) with the addition of mutual information loss based on the PyTorch Lightning implementation of [mutual information score](https://lightning.ai/docs/torchmetrics/stable/clustering/mutual_info_score.html). 

Our sample scripts 'train.sh' and 'predict.sh' to demonstrate how to run the training/inference phases. Please see the original CUT repository for more in-depth instructions.

### Citation
If you use our code for your research, please cite our paper.
```
@inproceedings{wang2024structpres,
  title={Structure-preserving Image Translation for Depth Estimation in Colonoscopy},
  author={Shuxian Wang and Akshay Paruchuri and Zhaoxi Zhang and Sarah McGill and Roni Sengupta},
  booktitle={Medical Image Computing and Computer Assisted Intervention},
  year={2024}
}
```

### Acknowledgments
Our code is adapted from [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation).
