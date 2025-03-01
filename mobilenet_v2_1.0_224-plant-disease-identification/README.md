---
license: other
tags:
- generated_from_trainer
datasets:
- image_folder
metrics:
- accuracy
model-index:
- name: mobilenet_v2_1.0_224-plant-disease-identification
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: New Plant Diseases Dataset
      type: image_folder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9541
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mobilenet_v2_1.0_224-plant-disease-identification

This model is a fine-tuned version of [google/mobilenet_v2_1.0_224](https://huggingface.co/google/mobilenet_v2_1.0_224) on the [Kaggle version](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) of the [Plant Village dataset](https://github.com/spMohanty/PlantVillage-Dataset).
It achieves the following results on the evaluation set:
- Cross Entropy Loss: 0.15
- Accuracy: 0.9541

## Intended uses & limitations

For identifying common diseases in crops and assessing plant health. Not to be used as a replacement for an actual diagnosis from experts.

## Training and evaluation data

The plant village dataset consists of 38 classes of diseases in common crops (including healthy/normal crops).

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-5
- train_batch_size: 256
- eval_batch_size: 256
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.2
- num_epochs: 6

### Framework versions

- Transformers 4.27.3
- Pytorch 1.13.0
- Datasets 2.1.0
- Tokenizers 0.13.2
