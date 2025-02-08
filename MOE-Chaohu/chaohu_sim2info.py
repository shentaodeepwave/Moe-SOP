import torch
import torch.nn.functional as F
import pyswmm
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
node_list = [
    "CC-storage",
    "JK-storage",
    "WS02006229",
    "WS02006116",
    "WS02006235",
    "WS02006251",
    "WS02006253",
    "HS02004478",
    "YS02001907",
    "YS02001757",
    "WSC"
]

edge_name_list = [
    "WS02006115WS02006116",
    "WS02006280WS02006229",
    "WS02006230WS02006229",
    "WS02006238WS02006235",
    "WS02006253WS02006251",
    "WS02006252WS02006251",
    "WS02006256WS02006235",
    "YS02001757CaochengTiaoxuchi",
    "YS02001793JiankangQianchi",
]

edge_list = [
    ("WS02006229", "WS02006116"),
    ("HS02004478", "WS02006229"),
    ("WS02006235", "WS02006229"),
    ("WS02006251", "WS02006235"),
    ("WS02006253", "WS02006251"),
    ("CC-storage", "WS02006251"),
    ("JK-storage", "WS02006235"),
    ("YS02001757", "CC-storage"),
    ("YS02001907", "JK-storage")
]

def add_gaussian_noise(data, noise_level=0.2):
    """为数据添加高斯噪声"""
    """
    noise = np.random.normal(0, noise_level * np.abs(data))
    res=data + noise
    """
    #百分之20返回0,
    res=data
    if np.random.random()<0.2:
        return 0
        
    return res

def generate_graph_data(sim, tokenizer):
    nodes = pyswmm.Nodes(sim)
    nodes = sorted(nodes, key=lambda x: x.nodeid)
    links = pyswmm.Links(sim)
    links = sorted(links, key=lambda x: x.linkid)
    x = []
    x_2 = []
    max_len = 0
    for node in nodes:
        for i in range(len(node_list)):
            if node.nodeid == node_list[i]:
                # 添加高斯噪声
                depth_with_noise = add_gaussian_noise(node.depth)
                flooding_with_noise = add_gaussian_noise(node.flooding)
                
                message = ("The depth is: " + str(round(depth_with_noise, 2)) + " now is " + "not flooding" if flooding_with_noise == 0 else "flooding")
                encoded_message = tokenizer(message, return_tensors='pt')
                input_ids = encoded_message['input_ids'].squeeze(0)
                x.append(input_ids)
                x_2.append([depth_with_noise, flooding_with_noise])
                if input_ids.size(0) > max_len:
                    max_len = input_ids.size(0)
    # 填充所有张量到相同的大小
    x = [F.pad(t, (0, max_len - t.size(0))) for t in x]

    edge_index = [(node_list.index(src), node_list.index(dest)) for src, dest in edge_list]
    # 填充所有张量到相同的大小
    x_tensor = torch.stack(x)  # 堆叠并转换为浮点张量
    x_2_tensor = torch.tensor(x_2, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return x_tensor, edge_index, x_2_tensor

def generate_text_data(original_info, sim, tokenizer):
    control_names = ["CC-R1", "CC-R2", "CC-S1", "CC-S2", "JK-R1", "JK-R2", "JK-S"]
    Tank_names = ["CC-storage", "JK-storage"]
    CSO_names = ['CC-storage', 'JK-storage', 'YS02001757', 'YS02001907', 'WS02006229']
    nodes = pyswmm.Nodes(sim)
    nodes = sorted(nodes, key=lambda x: x.nodeid)
    links = pyswmm.Links(sim)
    links = sorted(links, key=lambda x: x.linkid)
    rgs = RainGages(sim)
    
    # 孔口开度信息，添加高斯噪声
    orifice_opening = [add_gaussian_noise(link.target_setting) for link in links if link.linkid in control_names]
    # 溢流口与水库信息，添加高斯噪声
    overflow = [add_gaussian_noise(node.total_inflow) for node in nodes if node.nodeid in CSO_names]
    # 水库信息，添加高斯噪声
    tank = [add_gaussian_noise(node.depth) for node in nodes if node.nodeid in Tank_names]
    # 雨水径流信息，添加高斯噪声
    runoff = add_gaussian_noise(rgs["RG"].rainfall)
    # 污水处理厂进水信息，添加高斯噪声
    OTW_inflow = add_gaussian_noise(Nodes(sim)["WSC"].total_inflow)
    
    # 把以上内容写成文本
    orifice_opening_text = " ".join([f"Orifice_{control_names[i]}:{round(orifice_opening[i], 3)}" for i in range(len(orifice_opening))])
    overflow_text = " ".join([f"Overflow_{CSO_names[i]}:{round(overflow[i], 3)}" for i in range(len(overflow))])
    tank_text = " ".join([f"Tank_{Tank_names[i]}:{round(tank[i], 3)}" for i in range(len(tank))])
    rain_gage_text = f"Rainfall:{round(runoff, 3)}"
    OTW_inflow_text = f"OTW_inflow:{round(OTW_inflow, 3)}"
    text = f"{original_info}{orifice_opening_text} {overflow_text} {tank_text} {rain_gage_text} {OTW_inflow_text}"
    output = tokenizer(text, return_tensors='pt')
    return output

def generate_image_data():
    image_folder = './Original_data'
    image_tensors = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图片大小到 256x256
        transforms.ToTensor()
    ])
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')  # 确保图片是 RGB 格式
            image_tensor = transform(image)
            image_tensors.append(image_tensor)
    
    image_tensors = torch.stack(image_tensors)  # 将所有图片tensor堆叠成一个tensor
    return image_tensors

def get_info_state(sim, tokenizer):
    original_info = ""
    # 读取原始信息
    with open('./Original_data/chaohu_original_info.txt', 'r', encoding='utf-8') as f:
        for line in f:
            original_info += line.strip() 
    x, edge_index, x_2 = generate_graph_data(sim, tokenizer)
    text = generate_text_data(original_info, sim, tokenizer)
    image_tensors = generate_image_data()
    state = {
        "x": x,
        "x_2": x_2,
        "edge_index": edge_index,
        "text": text,
        "image": image_tensors
    }
    return state