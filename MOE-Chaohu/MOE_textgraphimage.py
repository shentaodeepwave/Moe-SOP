import numpy as np
import SWMM_ENV as SWMM_ENV
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn import Dropout
from torch_geometric.nn import GCNConv
from transformers import GPT2Model, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout1 = Dropout(dropout_rate)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout2 = Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = torch.mean(x, dim=0, keepdim=True)
        return x
    
class CNNModel(nn.Module):
    def __init__(self,output_dim=256):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 512)  # 修改此处以适应256x256的输入图像
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)  # 修改此处以适应256x256的输入图像
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN(nn.Module):
    def __init__(self, params, device):
        super(DQN, self).__init__()
        self.params = params
        self.action_table = pd.read_csv('./DQN_action_table.csv').values[:, 1:]
        self.LLM = AutoModelForCausalLM.from_pretrained(params['LLM_path'], output_hidden_states=True, return_dict_in_generate=True,torch_dtype=torch.float16).to(device)
        if params['LLM'] == 'GPT2':
            self.state_dim = self.LLM.config.n_embd
        else:
            self.state_dim = self.LLM.config.hidden_size
        self.gcn = GCNModel(self.state_dim, self.state_dim, self.state_dim).to(device)
        self.cnn = CNNModel(output_dim=self.state_dim).to(device)
        
        for param in self.LLM.parameters():
            param.requires_grad = False
        self.attention = nn.MultiheadAttention(embed_dim=self.state_dim*3, num_heads=8).to(device)
        self.Q = nn.Sequential(
            nn.Linear(self.state_dim*3, 512),
            nn.Tanh(),
            nn.Linear(512, self.params['action_dim'])
        ).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.device = device

    def forward(self, state):
        # 文本嵌入
        input_id = state['text']["input_ids"].to(self.device)
        attention_mask = state['text']["attention_mask"].to(self.device)
        outputs = self.LLM(input_ids=input_id, attention_mask=attention_mask)
        text_embedding = outputs.hidden_states[-1][:, -1, :]

        x = state['x'].to(self.device)
        edge_index = state['edge_index'].to(self.device)
        node_features = []
        for node in x:
            node = node.unsqueeze(0)
            node_feature = self.LLM(input_ids=node, attention_mask=torch.ones_like(node)).hidden_states[-1]
            node_feature = node_feature[:, -1, :]
            node_features.append(node_feature)
        node_features = torch.stack(node_features).to(self.device).to(torch.float32)
        node_features = node_features.mean(dim=1, keepdim=False).squeeze(0)
        graph_embedding = self.gcn(node_features, edge_index).to(self.device).to(torch.float16)
        
        # 图像嵌入
        images = state['image'].to(self.device) 
        image_embeddings = []
        for image in images:
            image_embedding = self.cnn(image)
            image_embeddings.append(image_embedding)
        image_embeddings = torch.stack(image_embeddings)
        image_embedding = torch.mean(image_embeddings, dim=0)

        embedding = torch.cat((text_embedding,graph_embedding,image_embedding), dim=-1).to(self.device).to(torch.float32)
        embedding = embedding.unsqueeze(0)  # 添加批次维度
        attn_output, _ = self.attention(embedding, embedding, embedding)
        attn_output = attn_output.squeeze(0)  # 移除批次维度
        #attn_output = torch.cat((attn_output,text_embedding), dim=-1).to(self.device).to(torch.float32)
        return self.Q(attn_output)

    def save_model(self, path):
        torch.save(self.gcn.state_dict(), path + 'gcn.pth')
        torch.save(self.cnn.state_dict(), path + 'cnn.pth')
        torch.save(self.attention.state_dict(), path + 'attention.pth')
        torch.save(self.Q.state_dict(), path + 'dqn.pth')

    def load_model(self, path):
        self.gcn.load_state_dict(torch.load(path + 'gcn.pth', map_location=self.device,weights_only=True))
        self.cnn.load_state_dict(torch.load(path + 'cnn.pth', map_location=self.device,weights_only=True))
        self.attention.load_state_dict(torch.load(path + 'attention.pth', map_location=self.device,weights_only=True))
        self.Q.load_state_dict(torch.load(path + 'dqn.pth', map_location=self.device,weights_only=True))

def sample_action(state, model, train_log, device):
    state = {k: v.to(device) for k, v in state.items()}
    if train_log:
        pa = np.random.uniform()
        if model.params['epsilon'] < pa:
            with torch.no_grad():
                action_value = model(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(model.params['action_dim'])
    else:
        model.eval()
        with torch.no_grad():
            action_value = model(state)
        action = torch.argmax(action_value).item()
    return action

def train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model, device):
    model.train()
    observation_buffers = []
    for i in range(len(observation_buffer)):
        observation_buffers.append({k: v.to(device) for k, v in observation_buffer[i].items()})
    action_buffer = torch.tensor(action_buffer, dtype=torch.int64).to(device)
    reward_buffer = torch.tensor(reward_buffer, dtype=torch.float16).to(device)
    observation_next_buffers = []
    for i in range(len(observation_next_buffer)):
        observation_next_buffers.append({k: v.to(device) for k, v in observation_next_buffer[i].items()})
    
    # 计算当前状态的Q值
    q_values = []
    for i in range(len(observation_buffer)):
        q_values.append(model(observation_buffers[i]))
    
    # 计算下一状态的Q值
    next_q_values = []
    for i in range(len(observation_next_buffer)):
        next_q_values.append(model(observation_next_buffers[i]))

    # 计算目标Q值
    target_q_values = []
    for i in range(len(observation_buffer)):
        target_q_value = reward_buffer[i] + model.params['gamma'] * torch.max(next_q_values[i])
        target_q_values.append(target_q_value)

    # 计算损失
    loss = 0
    for i in range(len(observation_buffer)):
        loss += nn.MSELoss()(q_values[i][0][action_buffer[i]], target_q_values[i])
    
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    return model

