---
library_name: transformers
license: other
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: ln=2e5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ln=2e5

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on the wiki_bio_decompose dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1081

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.2271        | 0.0809 | 25   | 0.1248          |
| 0.1127        | 0.1618 | 50   | 0.1092          |
| 0.1075        | 0.2427 | 75   | 0.1046          |
| 0.1107        | 0.3236 | 100  | 0.1167          |
| 0.1764        | 0.4045 | 125  | 0.1616          |
| 0.1073        | 0.4854 | 150  | 0.1133          |
| 0.1013        | 0.5663 | 175  | 0.1187          |
| 0.1132        | 0.6472 | 200  | 0.1084          |
| 0.1076        | 0.7282 | 225  | 0.1072          |
| 0.1081        | 0.8091 | 250  | 0.1066          |
| 0.1048        | 0.8900 | 275  | 0.1041          |
| 0.0944        | 0.9709 | 300  | 0.1059          |
| 0.0647        | 1.0518 | 325  | 0.1087          |
| 0.0602        | 1.1327 | 350  | 0.1043          |
| 0.0585        | 1.2136 | 375  | 0.1076          |
| 0.057         | 1.2945 | 400  | 0.1062          |
| 0.0634        | 1.3754 | 425  | 0.1000          |
| 0.0619        | 1.4563 | 450  | 0.1026          |
| 0.055         | 1.5372 | 475  | 0.1032          |
| 0.0612        | 1.6181 | 500  | 0.1002          |
| 0.0584        | 1.6990 | 525  | 0.0997          |
| 0.0553        | 1.7799 | 550  | 0.0965          |
| 0.0547        | 1.8608 | 575  | 0.0992          |
| 0.0525        | 1.9417 | 600  | 0.0963          |
| 0.0491        | 2.0227 | 625  | 0.0967          |
| 0.026         | 2.1036 | 650  | 0.1098          |
| 0.0281        | 2.1845 | 675  | 0.1095          |
| 0.0254        | 2.2654 | 700  | 0.1101          |
| 0.0241        | 2.3463 | 725  | 0.1098          |
| 0.0229        | 2.4272 | 750  | 0.1080          |
| 0.0242        | 2.5081 | 775  | 0.1099          |
| 0.026         | 2.5890 | 800  | 0.1091          |
| 0.0262        | 2.6699 | 825  | 0.1089          |
| 0.0236        | 2.7508 | 850  | 0.1089          |
| 0.0236        | 2.8317 | 875  | 0.1085          |
| 0.025         | 2.9126 | 900  | 0.1081          |
| 0.0232        | 2.9935 | 925  | 0.1081          |


### Framework versions

- Transformers 4.46.1
- Pytorch 2.1.2+cu118
- Datasets 3.1.0
- Tokenizers 0.20.3
