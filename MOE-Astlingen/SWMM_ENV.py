import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
import pandas as pd

from swmm_api.input_file import read_inp_file
from swmm_api import read_out_file
from pyswmm import Simulation,Links,Nodes,RainGages,Subcatchments,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 
from astlingen_sim2info import get_info_state
import torch
from astlingen_reward import get_reward as get_reward
def add_gaussian_noise(data, noise_level=0.2):
    """为数据添加高斯噪声"""
    """
    noise = np.random.normal(0, noise_level * np.abs(data))
    res=data + noise
    """
    #百分之10返回
    res=data
    return res

class SWMM_ENV:
    def __init__(self,params,tokenizer=None):
        self.params = params
        self.tokenizer = tokenizer
    def reset(self,rain,i,testid):
        self.daction =None
        self.testid = testid
        root = '_teminp'
        destination_file = './' + root + '/' + testid + '/' + self.params['orf'] + str(i) + 'train.inp'
        shutil.copyfile(rain, destination_file)
        self.sim = Simulation(destination_file)
        self.sim.start()
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else: 
            self.sim._model.swmm_stride(self.params['advance_seconds'])
        nodes = Nodes(self.sim)
        sys = SystemStats(self.sim)
        scs = Subcatchments(self.sim)
        links=Links(self.sim)
        rgs=RainGages(self.sim)
        self.flood_volume={}
        self.depths={}
        self.inflows={}
        for n in nodes:
            self.flood_volume[n.nodeid]=n.statistics['flooding_volume']
            self.depths[n.nodeid]=n.depth
            self.inflows[n.nodeid]=n.cumulative_inflow
        self.system_inflow=(sys.routing_stats['dry_weather_inflow']
                  +sys.routing_stats['wet_weather_inflow']
                  +sys.routing_stats['groundwater_inflow']
                  +sys.routing_stats['II_inflow']
                  +sys.routing_stats['external_inflow']
                  )
        states = []
        if self.testid=='baseline':
            node_list = ["T1","T2","T3","T4","T5","T6"]
            for n in node_list:
                states.append(add_gaussian_noise(self.depths[n]))
        elif self.testid=='MOE':
            states=get_info_state(self.sim,self.tokenizer)

        elif self.testid=='DQN':
            state=[]
            node_list = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "CSO7", "CSO8", "CSO9", "CSO10", "Out_to_WWTP", "T1", "T2", "T3", "T4", "T5", "T6"]
            edge_name_list = [
                    "C1","C10", "C11", "C12", "C13", "C14", 
                    "C15", "C16", "C17", "C18", "C19",
                    "C2", "C20","C21", "C22", "C23",
                    "C3", "C4", "C5", "C6", "C7", "C8", "C9", 
                    "V1", "V2", "V3", "V4", "V5", "V6"
                ]
            
            for _temp in node_list: #30*2
                state.append(add_gaussian_noise(nodes[_temp].depth))
                state.append(add_gaussian_noise(nodes[_temp].flooding))
            for _temp in edge_name_list:#29
                state.append(add_gaussian_noise(links[_temp].flow))
            for rg in rgs:#4
                state.append(add_gaussian_noise(rg.rainfall))
            #把整个state转为tensor
            state=torch.tensor(state,dtype=torch.float32)
            states=state
        return states

    def step(self,action):
        

        #获取模拟结果
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        sys = SystemStats(self.sim)
        scs = Subcatchments(self.sim)
        rgs = RainGages(self.sim)
        states = []
        if self.testid=='baseline':
            node_list = ["T1","T2","T3","T4","T5","T6"]
            for n in node_list:
                states.append(add_gaussian_noise(self.depths[n]))
        elif self.testid=='MOE':
            states=get_info_state(self.sim,self.tokenizer)
        elif self.testid=='DQN':
            state=[]
            node_list = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "CSO7", "CSO8", "CSO9", "CSO10", "Out_to_WWTP", "T1", "T2", "T3", "T4", "T5", "T6"]
            edge_name_list = [
                    "C1","C10", "C11", "C12", "C13", "C14", 
                    "C15", "C16", "C17", "C18", "C19",
                    "C2", "C20","C21", "C22", "C23",
                    "C3", "C4", "C5", "C6", "C7", "C8", "C9", 
                    "V1", "V2", "V3", "V4", "V5", "V6"
                ]
            
            for _temp in node_list: #30*2
                state.append(add_gaussian_noise(nodes[_temp].depth))
                state.append(add_gaussian_noise(nodes[_temp].flooding))
            for _temp in edge_name_list:#29
                state.append(add_gaussian_noise(links[_temp].flow))
            for rg in rgs:#4
                state.append(add_gaussian_noise(rg.rainfall))
            #把整个state转为tensor
            state=torch.tensor(state,dtype=torch.float32)
            states=state
        control_names = ["V2", "V3", "V4", "V6"]
        V2, V3, V4, V6 = [Links(self.sim)[name] for name in control_names]
        V2.target_setting = action[0]
        V3.target_setting = action[1]
        V4.target_setting = action[2]
        V6.target_setting = action[3]
        #模拟一步
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])

        Interval_t=self.params['advance_seconds'] #minutes
        done = False if time > 0 else True

        #获取奖励和运行过程信息
        rewards,results = get_reward(self.sim,Interval_t,self.flood_volume,self.depths,self.inflows,self.system_inflow)

        #更新储存信息
        self.flood_volume={}
        self.depths={}
        self.inflows={}
        for n in nodes:
            self.flood_volume[n.nodeid]=n.statistics['flooding_volume']
            self.depths[n.nodeid]=n.depth
            self.inflows[n.nodeid]=n.cumulative_inflow
        self.system_inflow=(sys.routing_stats['dry_weather_inflow']
                  +sys.routing_stats['wet_weather_inflow']
                  +sys.routing_stats['groundwater_inflow']
                  +sys.routing_stats['II_inflow']
                  +sys.routing_stats['external_inflow']
                  )
        if(done):
            self.sim.close()
        return states,rewards,results,done
        