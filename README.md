# Novel Policy Seeking with Constrained Optimization

In problem-solving, we humans can come up with multiple novel solutions to the same problem. However, reinforcement learning algorithms can only produce a set of monotonous policies that maximize the cumulative reward but lack diversity and novelty. In this work, we address the problem of generating novel policies in reinforcement learning tasks. Instead of following the multi-objective framework used in existing methods, we propose to rethink the problem under a novel perspective of constrained optimization. We first introduce a new metric to evaluate the difference between policies and then design two practical novel policy generation methods following the new perspective. The two proposed methods, namely the Constrained Task Novel Bisector (CTNB) and the Interior Policy Differentiation (IPD), are derived from the feasible direction method and the interior point method commonly known in the constrained optimization literature. Experimental comparisons on the MuJoCo control suite show our methods can achieve substantial improvement over previous novelty-seeking methods in terms of both the novelty of policies and their performances in the primal task.

## Quick Start

![image](./docs/cover_nps.png)
This repo provides the code for the work *Novel Policy Seeking with Constrained Optimization*. The only dependencies are pytorch, gym and mujoco. 

To reproduce the results reported in the paper, please run the follows:

1. Train 10 policies with different random seeds with PPO:

```
python ppo_gpu.py --env_name Hopper-v3 --hid_num 10 --num_episode 800 --use_gpu 6 --num_repeat 10
```

we provide 10 pre-trained policies for each environment used in the paper.

2. Train novel policies with different Novelty Seeking methods, e.g., WSR, TNB, CTNB, IPD, etc.

```bash
python CTNB.py --env_name Hopper-v3 --hid_num 10 --num_episode 800 --use_gpu 7 --thres 0.6 --file_num CTNB --num_repeat 10

python TNB.py --env_name Hopper-v3 --hid_num 10 --num_episode 800 --use_gpu 6 --thres 0.0 --file_num TNB --num_repeat 10

python WSR.py --env_name Hopper-v3 --hid_num 10 --num_episode 800 --use_gpu 5 --thres 0.0 --file_num WSR --num_repeat 10 --weight 0.5

python IPD.py --env_name Hopper-v3 --hid_num 10 --num_episode 800 --use_gpu 4 --thres 0.6 --file_num IPD --num_repeat 10
```

## Bibtex

```
@article{sun2020novel,
  title={Novel Policy Seeking with Constrained Optimization},
  author={Sun, Hao and Peng, Zhenghao and Dai, Bo and Guo, Jian and Lin, Dahua and Zhou, Bolei},
  journal={arXiv preprint arXiv:2005.10696},
  year={2020}
}
```
