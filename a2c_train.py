from utils import evaluate_policy, tuple_to_one_hot
from datetime import datetime
from A2C import A2C_discrete
import gymnasium as gym
import os, shutil
import argparse
import torch

class Hyperparameter:
    pass

# Hyperparameter Setting
opt = Hyperparameter()
opt.dvc = 'cpu'
opt.write = True
opt.render = False
opt.Loadmodel = False
opt.ModelIdex = 300000

opt.seed = 209
opt.T_horizon = 2046
opt.Max_train_steps = 5e7
opt.save_interval = 1e5 # Model saving interval, in steps.
opt.eval_interval = 5e3 # Model evaluating interval, in steps.

opt.gamma = 0.99 # Discounted Factor
opt.lambd = 0.95 # GAE Factor
opt.K_epochs = 10 # A2C update times
opt.net_width = 64 # Hidden net width
opt.lr = 1e-4 # Learning rate
opt.batch_size = 64 # lenth of sliced trajectory
opt.adv_normalization = False # Advantage normalization

def main():
    # Build Training Env and Evaluation Env
    
    EnvName = 'Blackjack-v1'
    Alg_name = 'A2C'
    env = gym.make(EnvName, render_mode = "human" if opt.render else None, natural=False, sab=False)

    eval_env = gym.make(EnvName)
    # opt.state_dim = env.observation_space.shape[0]
    opt.state_dim = 0
    for i in range(len(env.observation_space)):
        opt.state_dim += env.observation_space[i].n

    opt.action_dim = env.action_space.n
    # opt.max_e_steps = env._max_episode_steps

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # print('Env:',EnvName,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
    print('Env:',EnvName,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed)

    print('\n')

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}/{}'.format(Alg_name, EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    agent = A2C_discrete(**vars(opt))
    if opt.Loadmodel: agent.load(opt.ModelIdex)

    if opt.render: # play mode
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
            print(f'Env:{EnvName}, Episode Reward:{ep_r}')
    else: # train mode
        traj_lenth, total_steps = 0, 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            s = tuple_to_one_hot(s, env.observation_space)            
            env_seed += 1
            done = False

            # Interact & trian
            while not done:
                # Interact with Env 
                a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                s_next = tuple_to_one_hot(s_next, env.observation_space)

                done = (dw or tr)

                # Store the current transition
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                s = s_next

                traj_lenth += 1
                total_steps += 1

                # Update if its time
                if traj_lenth % opt.T_horizon == 0:
                    surrogate_loss, value_function_loss, entropy, learning_rate = agent.train()
                    if opt.write: 
                        writer.add_scalar('Loss/surrogate_loss', surrogate_loss, global_step=total_steps)
                        writer.add_scalar('Loss/value_function_loss', value_function_loss, global_step=total_steps)
                        writer.add_scalar('Loss/learning_rate', value_function_loss, global_step=total_steps)
                        writer.add_scalar('Loss/entropy', entropy, global_step=total_steps)

                    traj_lenth = 0

                # Record & log
                if total_steps % opt.eval_interval == 0:
                    score, win_rate = evaluate_policy(eval_env, agent, turns = 1000) # evaluate the policy for 10 times, and get averaged result
                    if opt.write: 
                        writer.add_scalar('Episode/mean_reward', score, global_step=total_steps)
                        writer.add_scalar('Episode/win_rate', win_rate, global_step=total_steps)
                    print('EnvName:',EnvName,'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                # Save model
                if total_steps % opt.save_interval==0:
                    agent.save(total_steps)

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
