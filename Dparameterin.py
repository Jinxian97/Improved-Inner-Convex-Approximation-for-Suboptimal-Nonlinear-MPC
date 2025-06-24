import numpy as np
from dlqr import *
import scipy.linalg as slg

MissionPara = {}
MissionPara['Num'] = 1
MissionPara['Xcons'] = np.array([[-3,3],[-3,3],[-np.pi,np.pi]])
MissionPara['Ucons'] = np.array([[-0.2,0.7],[-0.3*np.pi/2,0.3*np.pi/2]])
MissionPara['dt'] = 0.5
MissionPara['Q'] = np.diag([1,1,1])
MissionPara['R'] = np.diag([0.01,0.01])
MissionPara['Sstart'] = np.array([[-2,-2,np.pi/4],[-2,2,0],[-3,-3,0],[-1,-3,0]])
MissionPara['Send'] = np.array([[2,2,0],[2,-2,0],[2,1,0],[1,2,0]])
MissionPara['Uend'] = np.array([[1,0],[1,0],[1,0],[1,0]])
MissionPara['A']  = np.array([[1,0,0],[0,1,MissionPara['dt']],[0,0,1]])
# MissionPara['A'] = np.eye(3)
MissionPara['B']  = np.array([[MissionPara['dt']*np.cos(MissionPara['Send'][0,2]),0],[MissionPara['dt']*np.sin(MissionPara['Send'][0,2]),0],[0,MissionPara['dt']]])
MissionPara['Ustart'] = np.zeros((MissionPara['Num'],2))
MissionPara['Err'] = 0.001
MissionPara['L_J'] = np.array(100)
MissionPara['safety_dis'] = 1
P,K = dlqr(MissionPara['A'],MissionPara['B'],MissionPara['Q'],MissionPara['R'])
MissionPara['K'] = -K
MissionPara['Ak'] = MissionPara['A'] + MissionPara['B'] @ MissionPara['K']

MissionPara['P'] = slg.solve_discrete_lyapunov(MissionPara['Ak'].T,2*(MissionPara['Q']+MissionPara['K'].T @ MissionPara['R'] @ MissionPara['K']))
MissionPara['obs'] = [[-0.5,-1],[0,2]]

# x0,y0,a,b,theta
MissionPara['obs_ep'] = [[1,0,1,0.6,-np.pi * 5/11]]
MissionPara['ep_expl'] = []
for o in range(len(MissionPara['obs_ep'])):
    Ro = np.array([[np.cos(MissionPara['obs_ep'][o][4]), -np.sin(MissionPara['obs_ep'][o][4])], [np.sin(MissionPara['obs_ep'][o][4]), np.cos(MissionPara['obs_ep'][o][4])]])
    Po = np.array([[1/MissionPara['obs_ep'][o][2]**2, 0], [0, 1/MissionPara['obs_ep'][o][3]**2]])
    MissionPara['ep_expl'].append(Ro.T @ Po @ Ro)


AlgorithmPara = {}
AlgorithmPara['N'] = 15
AlgorithmPara['Rmin'] = [1,1]


CalculatePara = {}
CalculatePara['k'] = 0
CalculatePara['jk'] = 0
CalculatePara['alpha'] = 1
CalculatePara['state_now'] = MissionPara['Sstart']
CalculatePara['U'] = [np.kron(np.ones(AlgorithmPara['N']),MissionPara['Ustart'].reshape(-1,1)) for _ in range(MissionPara['Num'])]

CalculatePara['warm_start'] = np.load('data_pre/warm_start.npy')
CalculatePara['U_pre'] = [CalculatePara['warm_start'] for _ in range(MissionPara['Num'])]
CalculatePara['h_set'] = np.array([1,1,1])
