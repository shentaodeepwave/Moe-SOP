from pyswmm import Simulation,Links,Nodes,RainGages,Subcatchments,SystemStats
def get_reward(sim,intervalT,flood_volume,last_depth,last_inflow,last_scs_inflow):
    nodes = Nodes(sim)
    scm=Subcatchments(sim)
    V_in=0
    #总入流量
    for s in scm:
        V_in+=nodes[s.connection].lateral_inflow*intervalT

    #重点排放口排放体积(洪水)
    V_cso=0
    CSO_nodes=['T1','T2','T3','T4','T5','T6']
    for n in CSO_nodes:
        V_cso+=(nodes[n].flooding*intervalT)

    #所有节点的总溢流（CSO)
    V_all=0
    for n in nodes:
        V_all+=n.flooding*intervalT

    #集中排出口的总流出量
    V_out=nodes['Out_to_WWTP'].total_inflow*intervalT

    #储罐高度的变化
    Depth_delta = 0
    storage_nodes=['T1','T2','T3','T4','T5','T6']
    for n in storage_nodes:
        Depth_delta+=abs(nodes[n].depth-last_depth[n])
    Depth_delta=Depth_delta/len(storage_nodes)
    
    #重点溢流口是否溢流
    r_judge=1
    for n in CSO_nodes:
        if (nodes[n].statistics['flooding_volume']-flood_volume[n])>0:
            r_judge=-1
            break
    if V_in==0:
        reward=1
    else:
        reward=(-10*V_cso/V_in) + (-5*V_all/V_in)  + (0.01*V_out/V_in)

    result={
        'V_in':V_in,
        'V_cso':V_cso,
        'V_all':V_all,
        'V_out':V_out,
        'Depth_delta':Depth_delta,
        'r_judge':r_judge
    }
    return reward,result
    