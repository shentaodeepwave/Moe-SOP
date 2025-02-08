import numpy as np
import SWMM_ENV as SWMM_ENV
import torch
from joblib import Parallel, delayed
from MOE_textgraphimage import DQN, sample_action, train_value
from datetime import datetime
from MOE_Buffer import DQNBuffer
from transformers import AutoTokenizer
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(main_device)

env_params = {
    'orf': 'chaohu_test',
    'parm': './states_yaml/chaohu',
    'advance_seconds': 300,
}
train_raindata = np.load('./training_rainfall/training_raindata.npy').tolist()
test_raindata = np.load('./test_rainfall/RSH/test_raindata.npy').tolist()
agent_params = {
    'state_dim':1024,
    'action_dim': 2**7,
    'q_layer': [512, 256],
    'num_rain': 24,
    'num_test': 6,
    'train_iterations': 1,
    'training_step': 2000,
    'gamma': 0.1,
    'epsilon': 0.1,
    'ep_min': 1e-100,
    'ep_decay': 0.999,
    'learning_rate': 0.01,
    'LLM_path': '/data/coding/gpt2-medium/',
    'LLM': 'GPT2',
}
"""
/data/coding/Qwen2.5-1.5B/
"""
train_name = 'TextGraphImage-medium'
Train = False
init_train = True
# 打印开始时间
print(f"{datetime.now()} Start")

def interact(i, testid, runtype, raindata):
    tem_device_str = 'cuda:' + str(0)
    tem_device = torch.device(tem_device_str)

    # 创建模型副本
    tem_model = DQN(agent_params, device=tem_device).to(tem_device)
    if Train==True:
        tem_model.load_model('./model/' + train_name)
    if Train==False:
        tem_model.load_model('./model/best' + train_name)

    tokenizer = AutoTokenizer.from_pretrained(agent_params["LLM_path"])
    env = SWMM_ENV.SWMM_ENV(env_params, tokenizer)
    
    # 模拟初始化
    s, a, r, s_ = [], [], [], []
    rain = [(entry[0], float(entry[1])) for entry in raindata[i]]  # 确保每个元素是包含两个值的元组
    observation = env.reset(raindata[i], i, testid)
    episode_return, episode_length, episode_CSO, episode_all, episode_in, episode_out, episode_Depth_delta, episode_judge = 0, 0, 0, 0, 0, 0, 0, 0
    done = False
    while not done:
        observation = {k: v.to(tem_device) for k, v in observation.items()}
        action = sample_action(observation, tem_model, runtype, tem_device)

        at = tem_model.action_table[int(action) - 1].tolist()
        observation_new, reward, results, done = env.step(at)
        episode_return += reward
        
        episode_length += 1
        episode_CSO += results['V_cso']
        episode_all += results['V_all']
        episode_in += results['V_in']  
        episode_out += results['V_out']
        episode_Depth_delta += results['Depth_delta']
        episode_judge += results['r_judge']

        s.append(observation)
        a.append(action)
        r.append(reward)
        s_.append(observation_new)

        observation = observation_new
    last_value = 0 if done else tem_model(torch.tensor(observation, dtype=torch.float16).unsqueeze(0).to(tem_device)).max().item()
    del tem_model  # 释放 GPU 内存  
    torch.cuda.empty_cache()
    return s, a, r, s_, last_value, episode_return, episode_length, episode_CSO, episode_all, episode_in, episode_out, episode_Depth_delta, episode_judge, i

def train(model, raindata):
    history = {'episode': [], 'Episode_reward': [], 'Episode_length': [], 'Episode_CSO': [], 'Episode_all': [], 'Episode_in': [], 'Episode_out': [], 'Episode_Depth_delta': [], 'Episode_judge': [], 'epilson': [], 'best_testreward': []}
    final_test_history = {}
    epochs = 0
    if init_train:  
        with open('./Results/train/' + train_name + '.json', 'r') as f:
            history = json.load(f)
        with open('./Results/test/' + train_name + '.json', 'r') as f:
            final_test_history = json.load(f)
        epochs = len(history['episode'])
        model.params['epsilon'] = history['epilson']
        print(model.params['epsilon'])
    else:
        history['best_testreward'] = -100000
    for epoch in range(epochs, model.params['training_step']):
        sum_return = 0
        sum_length = 0
        sum_CSO = 0
        sum_all = 0
        sum_in = 0
        sum_out = 0
        sum_Depth_delta = 0
        sum_judge = 0
        num_episodes = 0
        
        DQNbuffer = DQNBuffer(model.params['state_dim'], int(len(raindata[0]) * model.params['num_rain']))
        
        res = Parallel(n_jobs=30)(delayed(interact)(i, 'MOE', True, raindata) for i in range(model.params['num_rain']))
        
        for i in range(model.params['num_rain']):
            for o, a, r, o_ in zip(res[i][0], res[i][1], res[i][2], res[i][3]):
                DQNbuffer.store(o, a, r, o_)

            DQNbuffer.finish_trajectory(res[i][4])
            sum_return += res[i][5]
            sum_length += res[i][6]
            sum_CSO += res[i][7]
            sum_all += res[i][8]
            sum_in += res[i][9]
            sum_out += res[i][10]
            sum_Depth_delta += res[i][11]
            sum_judge += res[i][12]
            num_episodes += 1

        observation_buffer, action_buffer, observation_next_buffer, reward_buffer, _ = DQNbuffer.get()
        for _ in range(model.params['train_iterations']):
            model=train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model, device=main_device)

        model.save_model('./model/' + train_name)

        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        history['Episode_length'].append(sum_length / num_episodes)
        history['Episode_CSO'].append(sum_CSO)
        history['Episode_all'].append(sum_all)
        history['Episode_in'].append(sum_in)
        history['Episode_out'].append(sum_out)
        history['Episode_Depth_delta'].append(sum_Depth_delta)
        history['Episode_judge'].append(sum_judge)
        history['epilson'] = model.params['epsilon']

        
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
            model.params['epsilon'] *= model.params['ep_decay']

            
        with open('./Results/train/' + train_name + '.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"{datetime.now()}, Epoch {epoch + 1} completed. Return: {sum_return}. Mean Length: {sum_length / num_episodes}, sum_CSO: {sum_CSO}, sum_all: {sum_all}, sum_in: {sum_in}, sum_out: {sum_out}, sum_Depth_delta: {sum_Depth_delta}, sum_judge: {sum_judge}")
        

        if True:
            print(f"Testing RSH {agent_params['num_test']} times")
            test_history = test(model, test_raindata)
            final_test_history[str(epoch)] = test_history
            with open('./Results/test/' + train_name + '.json', 'w') as f:
                json.dump(final_test_history, f, indent=4)
            test_total_return = sum([test_history[i]['return'] for i in range(agent_params['num_test'])])
            if test_total_return > history['best_testreward']:
                history['best_testreward'] = test_total_return
                model.save_model('./model/best' + train_name)
                with open('./Results/train/' + train_name + '.json', 'w') as f:
                    json.dump(history, f, indent=4)

def test(model, raindata):
    sum_return = 0
    sum_length = 0
    sum_CSO = 0
    sum_all = 0
    sum_in = 0
    sum_out = 0
    sum_Depth_delta = 0
    sum_judge = 0
    num_episodes = 0
    res = Parallel(n_jobs=30)(delayed(interact)(i, 'MOE', False, raindata) for i in range(model.params['num_test']))
    
    test_history = {}
    for i in range(model.params['num_test']):
        print("In test" + str(res[i][13]) + " return: " + str(res[i][5]) + " length: " + str(res[i][6]) + " CSO: " + str(res[i][7]) + " all: " + str(res[i][8]) + " in: " + str(res[i][9]) + " out: " + str(res[i][10]) + " Depth_delta: " + str(res[i][11]) + " judge: " + str(res[i][12]))

        sum_return += res[i][5]
        sum_length += res[i][6]
        sum_CSO += res[i][7]
        sum_all += res[i][8]
        sum_in += res[i][9]
        sum_out += res[i][10]
        sum_Depth_delta += res[i][11]
        sum_judge += res[i][12]
        num_episodes += 1

        test_history[res[i][13]] = {'return': res[i][5], 'length': res[i][6], 'CSO': res[i][7], 'all': res[i][8], 'in': res[i][9], 'out': res[i][10], 'Depth_delta': res[i][11], 'judge': res[i][12]}
    
    print(f"{datetime.now()} Return: {sum_return}. Mean Length: {sum_length / num_episodes}, sum_CSO: {sum_CSO}, sum_all: {sum_all}, sum_in: {sum_in}, sum_out: {sum_out}, sum_Depth_delta: {sum_Depth_delta}, sum_judge: {sum_judge}")

    return test_history



model = DQN(agent_params, device=main_device).to(main_device)

if init_train:
    model.load_model('./model/' + train_name)
    print('Model loaded')
model.save_model('./model/' + train_name)
if Train:
    train(model, train_raindata)
else:
    print(len(test_raindata))
    model.load_model('./model/best' + train_name)
    test_history = test(model, test_raindata)

print('Done')