# Vision Transformers for Few-Shot Medical Image Classification

Code for the paper:

> M. Nurgazin, N. A. Tu. **A Comparative Study of Vision Transformer Encoders and Few-Shot Learning for Medical Image Classification.** IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), CVAMD, 2023, pp. 2513–2521. [doi:10.1109/ICCVW60793.2023.00265](https://doi.org/10.1109/ICCVW60793.2023.00265) · [open access](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Nurgazin_A_Comparative_Study_of_Vision_Transformer_Encoders_and_Few-Shot_Learning_ICCVW_2023_paper.html)

Medical datasets are small and expensive to label, which makes few-shot learning attractive. This repository compares ImageNet-pretrained Vision Transformer encoders (ViT, DeiT, Swin, MobileViT) with CNN baselines (ResNet-50, VGG-16) under three few-shot algorithms: Prototypical Networks, Matching Networks and Reptile. The comparison runs on three medical datasets, with the test classes never seen during training.

**Main finding:** ViT encoders combined with ProtoNet consistently outperform similarly sized CNNs and are competitive with specialised few-shot methods.

## Key results

<!-- TODO: replace with the final table from the paper (Table N) so the README matches the publication exactly. -->

ProtoNet on ISIC 2018 (test classes: AKIEC, VASC, DF), mean accuracy (%) over 400 test episodes:

| Backbone | 2-way 3-shot | 2-way 5-shot | 2-way 10-shot | 3-way 3-shot | 3-way 5-shot | 3-way 10-shot |
|---|---|---|---|---|---|---|
| ResNet-50 | 67.99 | 71.69 | 75.66 | 52.93 | 56.53 | 61.60 |
| MobileViTv2-0.5 | 72.41 | 75.70 | 78.59 | 58.79 | 62.00 | 65.62 |
| Swin-B | 72.84 | 75.60 | 80.12 | 58.66 | 63.09 | 68.34 |
| DeiT-B | 73.67 | 77.11 | 81.60 | 58.23 | 62.54 | 70.47 |
| ViT-Ti | 76.66 | 79.88 | 83.41 | 63.42 | 66.62 | 71.95 |
| ViT-S | 78.40 | 81.61 | 86.34 | 65.50 | 70.00 | 76.47 |
| ViT-B | **79.81** | **84.19** | **88.21** | **67.70** | **73.30** | **79.17** |

All logged runs (ProtoNet and Reptile on every dataset) are in [`results/results_from_logs.csv`](results/results_from_logs.csv).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python <!-- TODO --> and a single CUDA GPU. The main dependencies are PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) for the pretrained backbones, and [easyfsl](https://github.com/sicara/easy-few-shot-learning) for episodic sampling and the ProtoNet/MatchingNet implementations.

## Data

The datasets are not redistributed here. Download them from their official sources and place them under one root folder:

| Dataset | Source | Train classes | Test classes |
|---|---|---|---|
| ISIC 2018 (Task 3) | <!-- TODO link --> | MEL, NV, BCC, BKL | AKIEC, VASC, DF |
| BreakHis (40×, 100×, 200×, 400×) | <!-- TODO link --> | 5 remaining classes | adenosis, tubular adenoma, phyllodes tumor |
| Pap smear (Herlev) | <!-- TODO link --> | `train/` folder | `test/` folder |

<!-- TODO: show the expected folder structure, including gnd.csv for BreakHis and gnd/train_gnd.csv for ISIC 2018. -->

## Reproduce

<!-- TODO: replace with the final commands once the scripts accept arguments. -->

```bash
# ProtoNet / MatchingNet: dataset 0 = ISIC2018, 1 = BreakHis, 2 = Papsmear
python -c "from protonet import protonet; protonet(['vit_small_patch16_224'], dataset=0, algo='protonet')"

# Reptile
python -c "from reptile_runner import reptile; reptile(['vit_small_patch16_224'], dataset=0, n_way=2)"
```

Seeds are fixed (`0`) and cuDNN is set to deterministic mode.

## Evaluation protocol

Episodes are sampled with easyfsl's `TaskSampler` (10 query images per class). Training uses only the training classes, and evaluation uses the held-out test classes. In the published ProtoNet experiments, the best epoch was chosen by accuracy on the same held-out classes that were later used for evaluation; there was no separate validation split.

## Repository structure

```
protonet.py          ProtoNet and MatchingNet training + evaluation
reptile_runner.py    Reptile meta-training + evaluation
pytorch_datasets/    Dataset loaders (ISIC 2018, BreakHis, Pap smear) and augmentations
notebooks/           Exploratory notebooks used during the study
results/             Accuracy logs from the experiments
```

## Citation

```bibtex
@InProceedings{Nurgazin_2023_ICCV,
    author    = {Nurgazin, Maxat and Tu, Nguyen Anh},
    title     = {A Comparative Study of Vision Transformer Encoders and Few-Shot Learning for Medical Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {2513-2521}
}
```

## Acknowledgements

Built on [easyfsl](https://github.com/sicara/easy-few-shot-learning) (MIT License) and [timm](https://github.com/huggingface/pytorch-image-models).
