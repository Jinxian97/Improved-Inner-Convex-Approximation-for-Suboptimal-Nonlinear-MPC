import numpy as np
import casadi as cs
import scipy.linalg as lng
import scipy,time,copy
import matplotlib.pyplot as plt
from Dparameterin import *
import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout,clear_log=False):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        self.clear_log = clear_log  # 新增参数控制清空日志文件
        # self.log = open(filename, 'a+')

        # 如果 clear_log 为 True，清空日志文件
        if self.clear_log:
            with open(self.filename, 'w') as log:
                log.truncate()  # 清空文件内容

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename,'w') as log:
                self.terminal.write(message)
                log.write(message)
    def flush(self):
        pass



N,Q,R,dt = AlgorithmPara['N'],MissionPara['Q'],MissionPara['R'],MissionPara['dt']
num_agent = 2

wei_k = 0

# sys.stdout = Logger("data_results/all_results_delta_NLP.log", sys.stdout, clear_log=True)

# far warmstart
CalculatePara['warm_start'] = np.load('data_pre/warm_start_25-02-13_delta1.npy')

duizhao = copy.copy(CalculatePara['warm_start'])
jishu = 0
Fk_old = np.zeros(2 * num_agent * N)


opti = cs.Opti()
U = opti.variable(2 * num_agent * N)
u = U.reshape((2 * num_agent,N))

x = opti.parameter(3 * num_agent,1)



for s in range(AlgorithmPara['N']): 
    for i in range(num_agent):
        if i == 0:
            x_req = cs.vertcat( x[3 * i,s] + u [2*i + 0,s]*np.cos(x [3 * i + 2,s])*dt-dt,x[3 * i + 1,s] + u [2*i + 0,s]*np.sin(x [3 * i + 2,s])*dt,x [3 * i + 2,s]+ u [2*i + 1,s]*dt)
        else:
            x_req = cs.vertcat(x_req, x[3 * i,s] + u [2*i + 0,s]*np.cos(x [3 * i + 2,s])*dt-dt,x[3 * i + 1,s] + u [2*i + 0,s]*np.sin(x [3 * i + 2,s])*dt,x [3 * i + 2,s]+ u [2*i + 1,s]*dt)

    x = cs.horzcat(x,  x_req)

J = 0
C = opti.parameter(0)
tilde_g = opti.parameter(0)
g = opti.parameter(0)
gk= opti.parameter(0)

for i in range(num_agent):
    for s in range(N):
        J += (x[3*i:3*(i+1),s]-MissionPara['Send'][i,:]).T @ Q @ (x[3*i:3*(i+1),s]-MissionPara['Send'][i,:]) + (u[2*i:2*(i+1),s]-MissionPara['Uend'][i,:]).T @ R @ (u[2*i:2*(i+1),s]-MissionPara['Uend'][i,:])

        C = cs.vertcat(C,MissionPara['Xcons'][:,0]-x[3*i:3*i+3,s], x[3*i:3*i+3,s]-MissionPara['Xcons'][:,1],
                        MissionPara['Ucons'][:,0]+MissionPara['Uend'][i,:]-u[2*i:2*i+2,s],  u[2*i:2*i+2,s]-MissionPara['Uend'][i,:]- MissionPara['Ucons'][:,1])

        for o in range(len(MissionPara['obs'])):
            g = cs.vertcat(g, AlgorithmPara['Rmin'][o] - (x[3*i:3*i+2,s] - MissionPara['obs'][o]).reshape((1,-1)) @ (x[3*i:3*i+2,s] - MissionPara['obs'][o]).reshape((-1,1)) )

        for o in range(len(MissionPara['obs_ep'])):
            g = cs.vertcat(g, 1 - (x[3*i:3*i+2,s] - MissionPara['obs_ep'][o][:2]).reshape((1,-1)) @ MissionPara['ep_expl'][o] @ (x[3*i:3*i+2,s] - MissionPara['obs_ep'][o][:2]).reshape((-1,1)) )

        for j in range(num_agent):
            if j > i:
                g = cs.vertcat(g, AlgorithmPara['Rmin'][o] - (x[3*i:3*i+2,s] -x[3*j:3*j+2,s]).reshape((1,-1)) @ (x[3*i:3*i+2,s] -x[3*j:3*j+2,s]).reshape((-1,1)) )


    C = cs.vertcat(C, (x[3*i:3*(i+1),N]-MissionPara['Send'][i,:]).T @ MissionPara['P'] @ (x[3*i:3*(i+1),N]-MissionPara['Send'][i,:])-0.2)
    J   += (x[3*i:3*(i+1),N]-MissionPara['Send'][i,:]).T @ MissionPara['P'] @ (x[3*i:3*(i+1),N]-MissionPara['Send'][i,:])

flag = np.zeros((num_agent,1))

t = 0
print('----------------t = {}-----------------'.format(t))
print('x_current =', MissionPara['Sstart'][0:num_agent,:].reshape((1,-1))[0])
print('stage_cost =', (MissionPara['Sstart'][0,:]-MissionPara['Send'][0,:]).T @ Q @ (MissionPara['Sstart'][0,:]-MissionPara['Send'][0,:]) +
                      (MissionPara['Sstart'][1,:]-MissionPara['Send'][1,:]).T @ Q @ (MissionPara['Sstart'][1,:]-MissionPara['Send'][1,:]))


flagexit = np.zeros((num_agent,1))
flagterminallaw = np.zeros((num_agent,1))
x_current = MissionPara['Sstart'][0:num_agent,:].reshape((3*num_agent,1))
opti.subject_to([C<=0.05,g<=0.05])
while t >= 0:
    t += 1
    if t == 20:
        a = 0
    print('----------------t = {}-----------------'.format(t))
    opti.set_value(x[:,0],x_current)
    k = 0
    CalculatePara_old = copy.copy(CalculatePara['warm_start'][0:2*num_agent,:])
    J_set = [1e10]
    Fk = 100

    opti.minimize(J)

    # opti.set_initial(U,CalculatePara['warm_start'][0:2*num_agent,:].reshape((-1,1),order='F'))
    opti.solver('ipopt',{'print_time':1, 'ipopt.print_level': 0,'ipopt.tol':1e-10})
    sol = opti.solve()
    
    import pdb;pdb.set_trace()
    abcdefg = sol.value(u)

    print('J = {}'.format(sol.value(J,[U==abcdefg.reshape((-1,1),order='F')])))

    if sol.value(J,[U==abcdefg.reshape((-1,1),order='F')]) >= sol.value(J,[U==CalculatePara_old.reshape((-1,1),order='F')]):
        abcdefg = copy.copy(CalculatePara_old[0:2*num_agent,:])
    x_current = sol.value(x[:,1],[U==abcdefg.reshape((-1,1),order='F')]).reshape((-1,1))

    for i in range(num_agent):
        if np.linalg.norm(x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1)),ord=2) <= MissionPara['Err']:
            flagexit[i] = 1
        if (x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1))).T @ MissionPara['P'] @ (x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1))) <= 0.5:
            flagterminallaw[i] = 1
    if min(flagexit) == 1:
        break

    x_req = sol.value(x[:,-1],[U==abcdefg.reshape((-1,1),order='F')]).reshape((-1,1))
    kappa = np.vstack((MissionPara['K'] @ (x_req[0:3,:]-MissionPara['Send'][0,:].reshape((-1,1))).reshape((-1,1))+MissionPara['Uend'][0,:].reshape((-1,1)),
            MissionPara['K'] @ (x_req[3:6,:]-MissionPara['Send'][1,:].reshape((-1,1))).reshape((-1,1))+MissionPara['Uend'][1,:].reshape((-1,1))))
    CalculatePara['warm_start'] = np.hstack((np.delete(abcdefg,[0],1),kappa))

    if min(flagterminallaw) == 1:
        print('terminal sets entered')
        warm_start_req = copy.copy(CalculatePara['warm_start'])
        warm_start_req2 = np.zeros(CalculatePara['warm_start'].shape)

        x_req = copy.copy(x_current)
        for s in range(N):
            kappa = np.vstack((MissionPara['K'] @ (x_req[0:3]-MissionPara['Send'][0,:].reshape((-1,1)))+MissionPara['Uend'][0,:].reshape((-1,1)),
                MissionPara['K'] @ (x_req[3:6]-MissionPara['Send'][1,:].reshape((-1,1)))+MissionPara['Uend'][1,:].reshape((-1,1))))
            x0_req = cs.vertcat(x_req[0,-1] + kappa[0]*np.cos(x_req[2,-1])*dt-dt,x_req[1,-1] + kappa[0]*np.sin(x_req[2,-1])*dt,x_req[2,-1]+ kappa[1]*dt)
            x1_req = cs.vertcat(x_req[3,-1] + kappa[2]*np.cos(x_req[5,-1])*dt-dt,x_req[4,-1] + kappa[2]*np.sin(x_req[5,-1])*dt,x_req[5,-1]+ kappa[3]*dt)
            x_req = np.vstack((x0_req,x1_req))
            warm_start_req2[:,s] = copy.copy(kappa.reshape((1,-1)))
        if sol.value(J,[U==warm_start_req.reshape((-1,1),order='F')]) >= sol.value(J,[U==warm_start_req2.reshape((-1,1),order='F')]):
            CalculatePara['warm_start'] = copy.copy(warm_start_req2)
        else:
            CalculatePara['warm_start'] = copy.copy(warm_start_req)

    
    print('x_current =', x_current.reshape((1,-1))[0])
    print('u_current =', abcdefg[:,0].reshape((1,-1))[0])
    a = 0