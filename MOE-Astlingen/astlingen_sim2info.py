import torch
import torch.nn.functional as F
import pyswmm
from pyswmm import Nodes, Subcatchments
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

node_list = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "CSO7", "CSO8", "CSO9", "CSO10", "Out_to_WWTP", "T1", "T2", "T3", "T4", "T5", "T6"]
edge_name_list = [
        "C1","C10", "C11", "C12", "C13", "C14", 
        "C15", "C16", "C17", "C18", "C19",
        "C2", "C20","C21", "C22", "C23",
        "C3", "C4", "C5", "C6", "C7", "C8", "C9", 
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
edge_list = [
        ("J2", "J3"), ("J8", "T6"), ("J3", "J15"), ("J9", "J12"), ("J12", "T3"), ("J19", "Out_to_WWTP"),
        ("CSO8", "J11"), ("J11", "J10"), ("J10", "J12"), ("J13", "T2"), ("J17", "J15"), 
        ("J6","J3"), ("J16", "J15"), ("CSO10", "J18"), ("J18", "J15"), ("J15", "T1"), ("J14", "J3"), ("J1", "T5"),
        ("CSO7", "J4"), ("J4", "J5"), ("J5", "T4"), ("CSO9", "J7"), ("J7", "J8"), ("T1", "J19"),
        ("T2", "J17"), ("T3", "J14"), ("T4", "J6"), ("T5", "J2"), ("T6", "J9")
    ]

def add_gaussian_noise(data, noise_level=0.2):
    """为数据添加高斯噪声"""
    """
    noise = np.random.normal(0, noise_level * np.abs(data))
    res=data + noise
    """
    #百分之20返回0，
    res=data
    '''
    if np.random.random()<0.2:
        return 0
    '''
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
                
                message = ("The depth is: " + str(round(depth_with_noise,2)) + " now is " + "not flooding" if flooding_with_noise == 0 else "flooding")
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
    control_names = ["V1", "V2", "V3", "V4", "V5", "V6"]
    CSO_names = ["CSO7", "CSO8", "CSO9", "CSO10"]
    Tank_names = ["T1", "T2", "T3", "T4", "T5", "T6"]
    Subcatchment_names = ["SC01","SC02","SC03","SC04","SC05","SC06","SC07","SC08","SC09","SC010"]
    SC01, SC02, SC03, SC04, SC05, SC06, SC07, SC08, SC09, SC010 = [Subcatchments(sim)[name] for name in Subcatchment_names]
    nodes = pyswmm.Nodes(sim)
    nodes = sorted(nodes, key=lambda x: x.nodeid)
    links = pyswmm.Links(sim)
    links = sorted(links, key=lambda x: x.linkid)
    
    # 孔口开度信息，添加高斯噪声
    orifice_opening = [add_gaussian_noise(link.target_setting) for link in links if link.linkid in control_names]
    # 溢流口与水库信息，添加高斯噪声
    overflow = [add_gaussian_noise(node.flooding) for node in nodes if node.nodeid in CSO_names]
    # 水库信息，添加高斯噪声
    tank = [add_gaussian_noise(node.depth) for node in nodes if node.nodeid in Tank_names]
    # 雨水径流信息，添加高斯噪声
    runoff = [add_gaussian_noise(subcatchment.runoff) for subcatchment in [SC01, SC02, SC03, SC04, SC05, SC06, SC07, SC08, SC09, SC010]]
    # 污水处理厂进水信息，添加高斯噪声
    OTW_inflow = add_gaussian_noise(Nodes(sim)["Out_to_WWTP"].cumulative_inflow)
    
    # 把以上内容写成文本
    orifice_opening_text = " ".join([f"Orifice_{control_names[i]}:{round(orifice_opening[i], 3)}" for i in range(len(orifice_opening))])
    overflow_text = " ".join([f"Overflow_{CSO_names[i]}:{round(overflow[i], 3)}" for i in range(len(overflow))])
    tank_text = " ".join([f"Tank_{Tank_names[i]}:{round(tank[i], 3)}" for i in range(len(tank))])
    runoff_text = " ".join([f"Runoff_{Subcatchment_names[i]}:{round(runoff[i], 3)}" for i in range(len(runoff))])
    OTW_inflow_text = f"OTW_inflow:{round(OTW_inflow, 3)}"
    text = f"{original_info}{orifice_opening_text} {overflow_text} {tank_text} {runoff_text} {OTW_inflow_text}"
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
    with open('./Original_data/Astlingen_original_info.txt', 'r', encoding='utf-8') as f:
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