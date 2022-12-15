
# Final Project

Setup Instruction: 
1. Install conda by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Create a conda environment that will contain python 3:
	```
	conda create -n cs285 python=3.7
	```
3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate cs285
	```
4. Install the requirements into this conda environment
	```
	pip install -r requirements.txt
	```
5. Allow your code to be able to see 'cs285'
	```
	cd <path_to_project>
	$ pip install -e .
    ```

Run the following commands in hw3 directory.

## To plot the average return for all questions and see saved plots in graph directory, run the following:
```
python cs285/scripts/read_results.py
```

## To visualize the runs using tensorboard:
```
tensorboard --logdir data/run1
```

## Milestone Report: 
Commands to run experiments for Milestone Report
```
python cs285/scripts/run_sac.py \
    --env_name Walker2d-v4 --ep_len 150 \
    --discount 0.99 --scalar_log_freq 1500 \
    -n 200000 -l 2 -s 256 -b 1500 -eb 1500 \
    -lr 0.001 --init_temperature 0.1 --exp_name sac_Walker2d_v4 \
    --seed 2 \
    --transfer_learning True

python cs285/scripts/run_sac.py \
    --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.99 --scalar_log_freq 1500 \
    -n 200000 -l 2 -s 256 -b 1500 -eb 1500 \
    -lr 0.001 --init_temperature 0.1 --exp_name sac_HalfCheetah_v4 \
    --seed 2
```

## Project Experiment

### Baseline Experiments

Command to run HalfCheetah-v4 with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah-v4_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_A with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_A --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_A_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_B with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_B --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_B_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_C with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_C --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_C_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_D with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_D --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_D_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_E with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_E --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_E_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

Command to run HalfCheetah_F with Actor Critic:
```
python cs285/scripts/run_actor_critic.py --env_name HalfCheetah_F --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name HalfCheetah_F_ntu1_ngsptu100 -ntu 1 -ngsptu 100
```

### Sanity Check on HalfCheetah-v4

Command to run HalfCheetah-v4 with Multitask Actor Critic:
```
python cs285/scripts/run_multitask_actor_critic.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_HalfCheetah-v4_1_100 -ntu 1 -ngsptu 100 \
--multitask_learning True \
--tasks HalfCheetah-v4
```

### Multitask Learning Architecture 1 - all 7 tasks
```
python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_all_7_tasks -ntu 1 -ngsptu 100 \
--num_tasks 7 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A HalfCheetah_B HalfCheetah_C HalfCheetah_D HalfCheetah_E HalfCheetah_F
```
### Multitask Learning Architecture 2
```
python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_2_HalfCheetah_v4_A \
-ntu 1 -ngsptu 100 \
--num_tasks 2 \
--multitask_learning True --share_policy True \
--tasks HalfCheetah-v4 HalfCheetah_A
```

### Multitask Learning Architecture 3
Sharing critic, shared layer + task-specific critic
Hyperparameter Sweep for beta = advantage ratio

```
python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.5 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.5 --num_tasks 2 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A

python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.6 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.6 --num_tasks 2 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A

python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.7 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.7 --num_tasks 2 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A

python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.8 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.8 --num_tasks 2 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A

python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_3_HalfCheetah_v4_A_beta_0.9 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.9 --num_tasks 2 \
--multitask_learning True \
--tasks HalfCheetah-v4 HalfCheetah_A
```

### Multitask Learning Architecture 4

Command to run experiment with separated task specific critic and shared critic, different beta values 
```
python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.5 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.5 --num_tasks 2 \
--multitask_learning True --separate_shared_and_specific_critic True \
--tasks HalfCheetah-v4 HalfCheetah_A

python cs285/scripts/run_multitask_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Multitask_Architecture_4_HalfCheetah_v4_A_beta_0.7 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.7 --num_tasks 2 \
--multitask_learning True --separate_shared_and_specific_critic True \
--tasks HalfCheetah-v4 HalfCheetah_A
```

### Continual Learning
Command to run continual learning with new task encountered at different time steps
```
python cs285/scripts/run_continual_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Continual_Multitask_HalfCheetah_v4_A_beta_0.8 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.8 \
--num_tasks 1 --n_iter_tasks 20 \
--num_new_tasks 1 --n_iter_new_tasks 150 \
--multitask_learning True --continual_learning True \
--tasks HalfCheetah-v4 \
--new_tasks HalfCheetah_A

python cs285/scripts/run_continual_actor_critic.py \
--ep_len 150 --discount 0.90 --scalar_log_freq 1 \
-n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 \
--exp_name Continual_Multitask_HalfCheetah_v4_A_beta_0.8 \
-ntu 1 -ngsptu 100 \
--adv_balance 0.8 \
--num_tasks 1 --n_iter_tasks 50 \
--num_new_tasks 1 --n_iter_new_tasks 150 \
--multitask_learning True --continual_learning True \
--tasks HalfCheetah-v4 \
--new_tasks HalfCheetah_A
```


## To compress to a zip file:
```
zip -vr submit.zip submit README.md -x "*.DS Store"
```