# Adversarial Flow Networks (AFN)

[[Paper](https://arxiv.org/pdf/2310.02779.pdf)]

PyTorch implementation of Adversarial Flow Networks (AFN). Currently supports self-play training on tic-tact-toe and Connect-4

## Setup
Tested using Python `3.10.12`.
```bash
pip install -r requirements.txt
```


## Quick start
```bash
# For tic-tac-toe
python train.py -cn=ttt
```

```shell
❯ python train.py -cn=ttt
Generating initial buffer
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:04<00:00,  1.34it/s]
Train:  10%|████████▊                                                                               | 100/1000 [00:03<00:24, 37.17it/s]
----------  Eval  ----------

{'WINS ratio': 0.655, 'DRAWS ratio': 0.095, 'LOSSES ratio': 0.25, '% Optimal Moves': 0.6798672566371682, 'step': 100}
Saved checkpoint at checkpoints-TTT/z98hbwmw/ckpt-00000100.pt
----------  Regen  ----------
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.43it/s]
Train:  20%|█████████████████▍                                                                      | 198/1000 [00:08<00:21, 37.93it/s]
----------  Eval  ----------

{'WINS ratio': 0.695, 'DRAWS ratio': 0.065, 'LOSSES ratio': 0.24, '% Optimal Moves': 0.6884955752212389, 'step': 200}
Saved checkpoint at checkpoints-TTT/z98hbwmw/ckpt-00000200.pt
----------  Regen  ----------
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.43it/s]
Train:  30%|██████████████████████████▍                                                             | 300/1000 [00:13<00:17, 39.35it/s]
----------  Eval  ----------

{'WINS ratio': 0.89, 'DRAWS ratio': 0.06, 'LOSSES ratio': 0.05, '% Optimal Moves': 0.9165929203539823, 'step': 300}
Saved checkpoint at checkpoints-TTT/z98hbwmw/ckpt-00000300.pt
...
```

```bash
# For Connect-4
python train.py -cn=connect4
```

```shell
Generating initial buffer
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 245/245 [08:56<00:00,  2.19s/it]
Train:   2%|█▋                                                                                  | 500/25000 [14:05<11:12:13,  1.65s/it]
----------  Eval  ----------

{'WINS ratio': 0.9, 'DRAWS ratio': 0.0, 'LOSSES ratio': 0.1, '% Optimal Moves': 0.4146972596645355, 'step': 500}
Saved checkpoint at checkpoints-Connect4/cyofxiu8/ckpt-00000500.pt
----------  Regen  ----------
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:22<00:00,  2.21s/it]
Train:   4%|███▎                                                                               | 1000/25000 [28:43<11:32:20,  1.73s/it]
----------  Eval  ----------

{'WINS ratio': 0.97, 'DRAWS ratio': 0.0, 'LOSSES ratio': 0.03, '% Optimal Moves': 0.53759765625, 'step': 1000}
Saved checkpoint at checkpoints-Connect4/cyofxiu8/ckpt-00001000.pt
----------  Regen  ----------
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:21<00:00,  2.16s/it]
Train:   6%|████▊                                                                              | 1460/25000 [42:08<10:55:04,  1.67s/it]
```

## Config
Training configurations can be found in `configs/`. See `ttt.yaml` for an annotated version of all the options.
- To enable `wandb` tracking, set `mode: enabled` under `wandb`.


## Playing against trained agents
After training an agent, it can be played against as follows:
```bash
python play_against_ai.py /path/to/ckpt.pt
python play_against_ai.py /path/to/ckpt.pt --human-first # To play first
```

## Citing
If you find this repository useful, please consider citing it:
```
@misc{jiralerspong2023expected,
      title={Expected flow networks in stochastic environments and two-player zero-sum games},
      author={Marco Jiralerspong and Bilun Sun and Danilo Vucetic and Tianyu Zhang and Yoshua Bengio and Gauthier Gidel and Nikolay Malkin},
      year={2023},
      eprint={2310.02779},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
