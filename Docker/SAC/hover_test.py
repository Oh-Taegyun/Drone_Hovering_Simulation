import argparse
import os
import pickle

import torch
from hover_env import HoverEnv
from hover_env import HoverEnv  
from model.sac import SACAgent  
from model.replay_buffer import ReplayBuffer  

import genesis as gs


def main():
    exp_name = "ex1) drone"       
    path = f"/app/actor_final.pth" 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  

    state_dict = torch.load(path, map_location=torch.device(device))  # 또는 'cuda:0'
    
    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open("/app/cfgs_final.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )


    state_dim = obs_cfg["num_obs"]        
    action_dim = env_cfg["num_actions"]      


    # SAC 관련 하이퍼파라미터 정의
    critic_cfg = {'hidden_dim': 256, 'hidden_depth': 3}  
    actor_cfg = {'hidden_dim': 256, 'hidden_depth': 3, 'log_std_bounds': [-20, 2]}  
    action_range = [-1, 1]      
    discount = 0.99             
    init_temperature = 0.1     
    alpha_lr = 1e-4             
    alpha_betas = (0.9, 0.999)   
    actor_lr = 1e-4             
    actor_betas = (0.9, 0.999)   
    actor_update_frequency = 2  
    critic_lr = 1e-4            
    critic_betas = (0.9, 0.999)  
    critic_tau = 0.005          
    critic_target_update_frequency = 2  
    batch_size = 1024          
    learnable_temperature = False 

    # SAC 에이전트 초기화: 관측, 행동 차원 및 하이퍼파라미터를 전달하여 SAC 에이전트 생성
    agent = SACAgent(
        obs_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        device=device,
        critic_cfg=critic_cfg,
        actor_cfg=actor_cfg,
        discount=discount,
        init_temperature=init_temperature,
        alpha_lr=alpha_lr,
        alpha_betas=alpha_betas,
        actor_lr=actor_lr,
        actor_betas=actor_betas,
        actor_update_frequency=actor_update_frequency,
        critic_lr=critic_lr,
        critic_betas=critic_betas,
        critic_tau=critic_tau,
        critic_target_update_frequency=critic_target_update_frequency,
        batch_size=batch_size,
        learnable_temperature=learnable_temperature
    )
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        for _ in range(max_sim_step):
            action_tensor, _ = agent.act(obs, True)
            obs, _, rews, dones, infos = env.step(torch.tensor(action_tensor))


if __name__ == "__main__":
    main()

