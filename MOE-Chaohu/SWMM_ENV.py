import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
import pandas as pd
#import pyswmm.toolkitapi as tkai

from swmm_api.input_file import read_inp_file
from swmm_api import read_out_file
from pyswmm import Simulation,Links,Nodes,RainGages,Subcatchments,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 
from chaohu_sim2info import get_info_state
import torch
from chaohu_reward import get_reward as get_reward
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

class SWMM_ENV:
    #can be used for every SWMM inp
    def __init__(self,params,tokenizer=None):
        self.params = params
        self.config = yaml.load(open(self.params['parm']+".yaml"), yaml.FullLoader)
        self.tokenizer = tokenizer
        #self.t=[]
    def reset(self,rain,i,testid):
        #分东西的情况raine与rainw为两场降雨，不分的情况两者为同一个
        self.daction = [0,0,0,0]
        self.testid=testid
        root='_teminp'
        shutil.copyfile('./SWMM/'+self.params['orf']+'.inp', './'+root+'/'+testid+'/'+self.params['orf']+str(i)+'.inp')
        inp = read_inp_file('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
        inp.write_file('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'train.inp')

        self.sim=Simulation('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'train.inp',
                            './'+root+'/'+testid+'/'+self.params['orf']+str(i)+'train.rpt',
                            './'+root+'/'+testid+'/'+self.params['orf']+str(i)+'train.out')
        #把结束时间设为降雨结束时间
        start_time_str = rain[0][0]
        end_time_str = rain[-1][0]
        self.sim.start_time = datetime.datetime.strptime(start_time_str, '%m/%d/%Y %H:%M:%S')
        self.sim.end_time = datetime.datetime.strptime(end_time_str, '%m/%d/%Y %H:%M:%S')
        self.sim.start()
        #模拟一步
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:  
            self.sim._model.swmm_stride(self.params['advance_seconds'])
        #记录总体cso和flooding
        self.CSO,self.flooding=0,0
        
        #obtain states and reward term by yaml (config)
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
            node_list = ["CC-storage","JK-storage","YS02001757"]
            for n in node_list:
                states.append(add_gaussian_noise(self.depths[n]))
        elif self.testid=='MOE':
            states=get_info_state(self.sim,self.tokenizer)
        elif self.testid=='DQN':
            state=[]
            for _temp in self.config["states"]:
                if _temp[1] == 'depthN':
                    state.append(add_gaussian_noise(nodes[_temp[0]].depth))
                    state.append(add_gaussian_noise(nodes[_temp[0]].flooding))
                elif _temp[1] == 'flow':
                    state.append(add_gaussian_noise(links[_temp[0]].flow))
                elif _temp[1] == 'inflow':
                    states.append(add_gaussian_noise(nodes[_temp[0]].total_inflow))
                else:
                    state.append(add_gaussian_noise(rgs[_temp[0]].rainfall))
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
            node_list = ["CC-storage","JK-storage","YS02001757"]
            for n in node_list:
                states.append(add_gaussian_noise(self.depths[n]))
        elif self.testid=='MOE':
            states=get_info_state(self.sim,self.tokenizer)
        elif self.testid=='DQN':
            state=[]
            for _temp in self.config["states"]:
                if _temp[1] == 'depthN':
                    state.append(add_gaussian_noise(nodes[_temp[0]].depth))
                    state.append(add_gaussian_noise(nodes[_temp[0]].flooding))
                elif _temp[1] == 'flow':
                    state.append(add_gaussian_noise(links[_temp[0]].flow))
                elif _temp[1] == 'inflow':
                    states.append(add_gaussian_noise(nodes[_temp[0]].total_inflow))
                else:
                    state.append(add_gaussian_noise(rgs[_temp[0]].rainfall))
            #把整个state转为tensor
            state=torch.tensor(state,dtype=torch.float32)
            states=state

        for item,a in zip(self.config['action_assets'],action):
            links[item].target_setting = a
    
        #模拟一步
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])

        #参数设置
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
        