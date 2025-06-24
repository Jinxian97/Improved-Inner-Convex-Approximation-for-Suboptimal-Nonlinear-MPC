import numpy as np
import casadi as cs
import time,copy
from Dparameterin import *
import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout,clear_log=False):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        self.clear_log = clear_log 
 
        if self.clear_log:
            with open(self.filename, 'w') as log:
                log.truncate() 

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
k = 0

delta = 0.1

# sys.stdout = Logger("data_results/convergence_delta"+str(delta)+".log", sys.stdout, clear_log=True)

CalculatePara['warm_start'] = np.load('data_pre/warm_start.npy')

duizhao = copy.copy(CalculatePara['warm_start'])
jishu = 0
Fk_old = np.zeros(2 * num_agent * N)


opti = cs.Opti()
U = opti.variable(2 * num_agent * N)
u = U.reshape((2 * num_agent,N))
Uk = opti.parameter(2  * num_agent* N)
uk = Uk.reshape((2 * num_agent,N))

x = opti.parameter(3 * num_agent,1)
xk = opti.parameter(3 * num_agent,1)

opti.set_value(x,MissionPara['Sstart'][0:num_agent,:].reshape((3*num_agent,1)))
opti.set_value(xk,MissionPara['Sstart'][0:num_agent,:].reshape((3*num_agent,1)))

for s in range(AlgorithmPara['N']): 
    for i in range(num_agent):
        if i == 0:
            xk_req = cs.vertcat( xk[3*i,s] + uk [2*i + 0,s]*np.cos(xk [3 * i + 2,s])*dt-dt,xk[3 * i + 1,s] + uk [2*i + 0,s]*np.sin(xk [3 * i + 2,s])*dt,xk [3 * i + 2,s]+ uk [2*i + 1,s]*dt)
        else:
            xk_req = cs.vertcat(xk_req, xk[3*i,s] + uk [2*i + 0,s]*np.cos(xk [3 * i + 2,s])*dt-dt,xk[3 * i + 1,s] + uk [2*i + 0,s]*np.sin(xk [3 * i + 2,s])*dt,xk [3 * i + 2,s]+ uk [2*i + 1,s]*dt)

    xk = cs.horzcat(xk, xk_req)

J = 0
J_k = 0
C = opti.parameter(0)
tilde_C = opti.parameter(0)
tilde_g = opti.parameter(0)
g = opti.parameter(0)
gk= opti.parameter(0)

for i in range(num_agent):
    for s in range(N):
        J_k += (xk[3*i:3*(i+1),s]-MissionPara['Send'][i,:]).T @ Q @ (xk[3*i:3*(i+1),s]-MissionPara['Send'][i,:]) + (uk[2*i:2*(i+1),s]-MissionPara['Uend'][i,:]).T @ R @ (uk[2*i:2*(i+1),s]-MissionPara['Uend'][i,:])
        C = cs.vertcat(C,MissionPara['Xcons'][:,0]-xk[3*i:3*i+3,s], xk[3*i:3*i+3,s]-MissionPara['Xcons'][:,1],
                        MissionPara['Ucons'][:,0]+MissionPara['Uend'][i,:]-uk[2*i:2*i+2,s],  uk[2*i:2*i+2,s]-MissionPara['Uend'][i,:]- MissionPara['Ucons'][:,1])
        for o in range(len(MissionPara['obs'])):
            gk= cs.vertcat(gk,AlgorithmPara['Rmin'][o] - (xk[3*i:3*i+2,s] - MissionPara['obs'][o]).reshape((1,-1)) @ (xk[3*i:3*i+2,s] - MissionPara['obs'][o]).reshape((-1,1)) )
        
        for o in range(len(MissionPara['obs_ep'])):
            gk = cs.vertcat(gk, 1 - (xk[3*i:3*i+2,s] - MissionPara['obs_ep'][o][:2]).reshape((1,-1)) @ MissionPara['ep_expl'][o] @ (xk[3*i:3*i+2,s] - MissionPara['obs_ep'][o][:2]).reshape((-1,1)) )

        for j in range(num_agent):
            if j > i:
                gk = cs.vertcat(gk, AlgorithmPara['Rmin'][o] - (xk[3*i:3*i+2,s] -xk[3*j:3*j+2,s]).reshape((1,-1)) @ (xk[3*i:3*i+2,s] -xk[3*j:3*j+2,s]).reshape((-1,1)) )

    C = cs.vertcat(C, (xk[3*i:3*(i+1),N]-MissionPara['Send'][i,:]).T @ MissionPara['P'] @ (xk[3*i:3*(i+1),N]-MissionPara['Send'][i,:])-0.2)
    J_k += (xk[3*i:3*(i+1),N]-MissionPara['Send'][i,:]).T @ MissionPara['P'] @ (xk[3*i:3*(i+1),N]-MissionPara['Send'][i,:])

tilde_J = J_k + cs.jacobian(J_k,Uk).reshape((-1,1)).T @ (U-Uk).reshape((-1,1)) + MissionPara['L_J']/2 * (U-Uk).reshape((-1,1)).T @ (U-Uk).reshape((-1,1))
for ell in range(gk.shape[0]):
    tilde_g = cs.vertcat(tilde_g, gk[ell] + cs.gradient(gk[ell],Uk).reshape((-1,1)).T @ (U-Uk).reshape((-1,1))+ MissionPara['L_J']/2 * (U-Uk).reshape((1,-1)) @ (U-Uk).reshape((-1,1)))
for ell in range(C.shape[0]):
    tilde_C = cs.vertcat(tilde_C, C[ell] + cs.gradient(C[ell],Uk).reshape((-1,1)).T @ (U-Uk).reshape((-1,1))+ MissionPara['L_J']/2 * (U-Uk).reshape((1,-1)) @ (U-Uk).reshape((-1,1)))

hat_J =   tilde_J/ delta - np.ones(tilde_C.shape).T @ cs.log(-tilde_C) - np.ones(tilde_g.shape).T @ cs.log(-tilde_g)
Hessians = np.load('data_pre/hessians.npy',allow_pickle = True).item()
while k <= 200:
    k+=1

    opti.set_value(Uk,CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F'))
    opti.minimize(hat_J)
    opti.set_initial(U,CalculatePara['warm_start'][0:2*num_agent,:].reshape((-1,1),order='F'))
    opti.solver('ipopt',{'print_time':0, 'ipopt.print_level': 0,'ipopt.tol':1e-6})
    sol = opti.solve()

    Fk = CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - sol.value(U) 
    if k==1:
        print('------------------ k = 0------------------')
        true_J = sol.value(J_k)
        print('J = ', abs(true_J))
        print('delta = ', delta)
        Hessian11_value = Hessians['he11J']/delta + Hessians['he11g']
        Hessian12_value = Hessians['he12J']/delta + Hessians['he12g']
        nabla = - np.linalg.inv(Hessian11_value) @ Hessian12_value
        Bk = np.eye(U.shape[0]) - nabla
        Hk = np.linalg.inv(Bk)
    else:
        time_start = time.time()
        yk = (Fk - Fk_old).reshape((-1,1))
        sk = (CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - CalculatePara_old.reshape((1,-1),order='F')).reshape((-1,1))
        Hk -= (Hk@yk-sk)@sk.T@Hk/(sk.T@Hk@yk)
        time_stop = time.time()
    

    pk = - Hk @ (CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - sol.value(U)).T 
    alpha = 1
    while True:
        abcdefg = (CalculatePara['warm_start'][0:2*num_agent,:].reshape((-1,1),order='F') + alpha * pk).reshape((N,2*num_agent)).T
        if np.max(sol.value(cs.mmax(cs.vertcat(C,gk)),[Uk==abcdefg.reshape((-1,1),order='F'), U==abcdefg.reshape((-1,1),order='F')])) >= 0:
            alpha *= 0.5
        else:
            break

    print('------------------ k =',k,'------------------')
    print('alpha = ', alpha)
    CalculatePara_old = copy.copy(CalculatePara['warm_start'][0:2*num_agent,:])
    CalculatePara['warm_start'][0:2*num_agent,:] = copy.copy(abcdefg)
    true_J = sol.value(J_k,[Uk==abcdefg.reshape((-1,1),order='F')])
    print('J = ', abs(true_J))
    print('delta-Phi(delta) = ', np.linalg.norm(Fk))
    Fk_old = copy.copy(Fk)
    if np.linalg.norm(Fk) == 0:
        break
