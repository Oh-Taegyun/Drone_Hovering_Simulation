import os        
import pickle    
import shutil   

import numpy as np    

import torch     
import torch.nn as nn      
import torch.optim as optim  
import tqdm

import genesis as gs 
from SAC.model.hover_env import HoverEnv 
from model.sac import SACAgent  
from model.replay_buffer import ReplayBuffer 


# 환경 설정 함수: hover_train.py의 get_cfgs()와 동일한 설정값 반환
def get_cfgs():
    env_cfg = {
        "num_actions": 4,  # 드론이 제어할 액션의 수 (예: 4개의 프로펠러 제어)

        # 종료(termination) 조건들:
        "termination_if_roll_greater_than": 180,  # 롤(roll)이 180도 초과하면 종료
        "termination_if_pitch_greater_than": 180,  # 피치(pitch)가 180도 초과하면 종료
        "termination_if_close_to_ground": 0.1,  # 고도가 0.1 미만이면 종료 (바닥에 너무 가까움)
        "termination_if_x_greater_than": 3.0,  # x방향 상대 위치가 3.0을 초과하면 종료
        "termination_if_y_greater_than": 3.0,  # y방향 상대 위치가 3.0을 초과하면 종료
        "termination_if_z_greater_than": 2.0,  # z방향 상대 위치가 2.0을 초과하면 종료

        # 초기 드론의 위치 및 회전(quaternion) 설정
        "base_init_pos": [0.0, 0.0, 1.0],  # 초기 위치 (x, y, z)
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # 초기 회전 (쿼터니언)
        "episode_length_s": 15.0,  # 에피소드의 최대 시간 (초 단위)
        "at_target_threshold": 0.1,  # 드론이 목표에 도달했다고 판단할 임계값 (거리)
        "resampling_time_s": 3.0,  # 목표 재샘플링 주기 (초 단위)
        "simulate_action_latency": True,  # 행동 지연(simulated latency) 여부
        "clip_actions": 1.0,  # 액션 값의 최대 절댓값 제한

        # 시각화 옵션
        "visualize_target": False,  # 목표(타겟) 시각화 여부
        "visualize_camera": False,  # 카메라 시각화 여부
        "max_visualize_FPS": 60,      # 시각화 시 최대 FPS
    }

    # obs_cfg: 관측(observation) 관련 설정 값들
    obs_cfg = {
        "num_obs": 17,  # 관측 벡터의 차원 (예: 상대 위치, 속도, 회전 등 총 17 차원)
        "obs_scales": {  # 관측 값 스케일링 (정규화 및 클리핑에 사용)
            "rel_pos": 1 / 3.0,   # 상대 위치 스케일
            "lin_vel": 1 / 3.0,   # 선형 속도 스케일
            "ang_vel": 1 / 3.14159,  # 각속도 스케일 (파이로 나누어 정규화)
        },
    }

    # reward_cfg: 보상(reward) 관련 설정 값들
    reward_cfg = {
        "yaw_lambda": -10.0,  # yaw 보상 계산에 사용될 람다 값
        "reward_scales": {  # 각 보상 항목에 곱해질 스케일 값
            "target": 10.0,    # 목표에 대한 보상 스케일
            "smooth": -1e-4,   # 행동 변화(스무스) 보상 스케일 (패널티)
            "yaw": 0.01,       # yaw 보상 스케일
            "angular": -2e-4,  # 각속도(angular) 보상 스케일 (패널티)
            "crash": -10.0,    # 충돌시 패널티 보상 스케일
        },
    }

    # command_cfg: 목표(command) 관련 설정 값들
    command_cfg = {
        "num_commands": 3,  # 목표 명령(command)의 차원 (예: x, y, z)
        "pos_x_range": [-1.0, 1.0],  # x좌표 목표 범위
        "pos_y_range": [-1.0, 1.0],  # y좌표 목표 범위
        "pos_z_range": [1.0, 1.0],   # z좌표 목표 고정 (여기서는 항상 1.0)
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


# ------------------------- 학습 루프 (SAC 기반) -------------------------

def main():
    exp_name = "drone-hovering"      
    vis = False                       # 시각화 여부
    num_envs = 8192                   # 동시 실행 환경 수
    max_steps = 1000000               # 최대 학습 스텝 수

    # genesis 초기화
    gs.init(logging_level="error")

    # 로그 저장 경로 설정 및 이전 로그 삭제 후 새로운 디렉토리 생성
    log_dir = f"Docker_src/logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 시각화 옵션이 활성화된 경우, 환경 설정에서 목표 시각화(True)로 변경
    if vis:
        env_cfg["visualize_target"] = False

    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=vis,
    )

    state_dim = obs_cfg["num_obs"]         
    action_dim = env_cfg["num_actions"]      
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  


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
    learnable_temperature = True  

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

    # ReplayBuffer 초기화
    replay_buffer = ReplayBuffer(
        obs_shape=(state_dim,),
        action_shape=(action_dim,),
        capacity=1000000,  
        device=device
    )

    # 초반 일정 스텝동안은 무작위 행동으로 리플레이 버퍼를 채움
    # 초기 무작위 스텝 수: 에이전트가 학습을 시작하기 전에 무작위 행동을 통해 경험을 쌓는 단계
    initial_random_steps = 10000  # 초기 무작위 스텝 수
    obs, _ = env.reset()  
    obs = obs.cpu().numpy()  

    for step in tqdm.tqdm(range(max_steps)):
        state_tensor = torch.FloatTensor(obs).to(device)
        if step < initial_random_steps:
            action = np.random.uniform(-1, 1, size=(num_envs, action_dim))
        else:
            with torch.no_grad():
                action, _ = agent.act(state_tensor, True)

        next_obs, _, reward, done, extras = env.step(torch.FloatTensor(action).to(device))
        next_obs = next_obs.cpu().numpy()
        reward = reward.cpu().numpy()
        done = done.cpu().numpy()

        for i in range(num_envs):
            done_no_max = done[i]
            if isinstance(extras, dict) and "done_no_max" in extras:
                done_no_max = extras["done_no_max"][i]
            replay_buffer.add(obs[i], action[i], reward[i], next_obs[i], done[i], done_no_max)
        obs = next_obs

        if step >= initial_random_steps and len(replay_buffer) >= batch_size:
            agent.update(replay_buffer, step)

        if step % 1000 == 0:
            print(f"Step: {step}")

        if step % 10000 == 0: # 10000 스텝마다 모델 저장
            torch.save(agent.actor.state_dict(), os.path.join(log_dir, f"actor_{step}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(log_dir, f"critic_{step}.pth"))
            
    torch.save(agent.actor.state_dict(), os.path.join(log_dir, "actor.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(log_dir, "critic.pth"))
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], open(os.path.join(log_dir, "cfgs.pkl"), "wb"))

if __name__ == "__main__":
    main()
