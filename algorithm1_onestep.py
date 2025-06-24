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
        self.clear_log = clear_log  
        # self.log = open(filename, 'a+')

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

wei_k = 0

delta = 0.7
# sys.stdout = Logger("data_results/all_results_delta_onestep"+str(delta)+".log", sys.stdout, clear_log=True)

CalculatePara['warm_start'] = np.load('data_pre/warm_start_25-02-13_delta1.npy')

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

for s in range(AlgorithmPara['N']): 
    for i in range(num_agent):
        if i == 0:
            xk_req = cs.vertcat( xk[3*i,s] + uk [2*i + 0,s]*np.cos(xk [3 * i + 2,s])*dt-dt,xk[3 * i + 1,s] + uk [2*i + 0,s]*np.sin(xk [3 * i + 2,s])*dt,xk [3 * i + 2,s]+ uk [2*i + 1,s]*dt)
        else:
            xk_req = cs.vertcat(xk_req, xk[3*i,s] + uk [2*i + 0,s]*np.cos(xk [3 * i + 2,s])*dt-dt,xk[3 * i + 1,s] + uk [2*i + 0,s]*np.sin(xk [3 * i + 2,s])*dt,xk [3 * i + 2,s]+ uk [2*i + 1,s]*dt)

    xk = cs.horzcat(xk, xk_req)

J = 0
stage_J = 0
J_k = 0
C = opti.parameter(0)
tilde_C = opti.parameter(0)
tilde_g = opti.parameter(0)
g = opti.parameter(0)
gk= opti.parameter(0)

for i in range(num_agent):
    stage_J += (xk[3*i:3*(i+1),0]-MissionPara['Send'][i,:]).T @ Q @ (xk[3*i:3*(i+1),0]-MissionPara['Send'][i,:]) + (uk[2*i:2*(i+1),0]-MissionPara['Uend'][i,:]).T @ R @ (uk[2*i:2*(i+1),0]-MissionPara['Uend'][i,:])
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



flag = np.zeros((num_agent,1))

t = 0
print('----------------t = {}-----------------'.format(t))
print('x_current =', MissionPara['Sstart'][0:num_agent,:].reshape((1,-1))[0])


flagexit = np.zeros((num_agent,1))
flagterminallaw = np.zeros((num_agent,1))
x_current = MissionPara['Sstart'][0:num_agent,:].reshape((3*num_agent,1))
while t >= 0:
    t += 1
    print('----------------t = {}-----------------'.format(t))
    opti.set_value(xk[:,0],x_current)
    k = 0
    CalculatePara_old = copy.copy(CalculatePara['warm_start'][0:2*num_agent,:])
    J_set = [1e10]
    Fk = 100

    while k == 0:
        k+=1

        time_Newton = 0
        time_Broyden = 0

        opti.set_value(Uk,CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F'))
        if k == 1:
            time_start = time.time()
            Hessian11,jacabian_1=cs.hessian(hat_J, U)
            Hessian12 = cs.jacobian(jacabian_1, Uk)
            time_stop = time.time()
        else:
            Hessian11 = 0
            Hessian12 = 0
        time_Newton += time_stop - time_start

        opti.minimize(hat_J)
        opti.set_initial(U,CalculatePara['warm_start'][0:2*num_agent,:].reshape((-1,1),order='F'))
        opti.solver('ipopt',{'print_time':0, 'ipopt.print_level': 0,'ipopt.tol':1e-6})
        sol = opti.solve()


        Fk = CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - sol.value(U) 
        if k==1:
            Hessian11_value = sol.value(Hessian11)
            Hessian12_value = sol.value(Hessian12)
            time_start = time.time()
            nabla = - np.linalg.inv(Hessian11_value) @ Hessian12_value
            Bk = np.eye(U.shape[0]) - nabla
            Hk = np.linalg.inv(Bk)
            time_stop = time.time()
            time_Newton += time_stop - time_start
        else:
            time_start = time.time()
            yk = (Fk - Fk_old).reshape((-1,1))
            sk = (CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - CalculatePara_req.reshape((1,-1),order='F')).reshape((-1,1))
            Hk -= (Hk@yk-sk)@sk.T@Hk/(sk.T@Hk@yk)
            time_stop = time.time()
            time_Broyden += time_stop - time_start
        
        time_start = time.time()
        pk = - Hk @ (CalculatePara['warm_start'][0:2*num_agent,:].reshape((1,-1),order='F') - sol.value(U)).T 
        time_stop = time.time()
        time_Broyden += time_stop - time_start
        time_Newton += time_stop - time_start

        alpha = 1
        while True:
            abcdefgreq = (CalculatePara['warm_start'][0:2*num_agent,:].reshape((-1,1),order='F') + alpha * pk).reshape((N,2*num_agent)).T
            if np.max(sol.value(cs.mmax(cs.vertcat(C,gk)),[Uk==abcdefgreq.reshape((-1,1),order='F'), U==abcdefgreq.reshape((-1,1),order='F')])) >= 0:
                alpha *= 0.5
            else:
                break
       
        CalculatePara_req = copy.copy(CalculatePara['warm_start'][0:2*num_agent,:] )
        CalculatePara['warm_start'][0:2*num_agent,:] = copy.copy(abcdefgreq)
        Fk_old = copy.copy(Fk)
        if sol.value(J_k,[Uk==abcdefgreq.reshape((-1,1),order='F')]) <= min(J_set):
            abcdefg = copy.copy(abcdefgreq)
        J_set.append(sol.value(J_k,[Uk==abcdefg.reshape((-1,1),order='F')]))
        print('k_{}: Fk = {}'.format(k,np.linalg.norm(Fk)))
        Fk_old = copy.copy(Fk)

    print('J = {}'.format(sol.value(J_k,[Uk==abcdefg.reshape((-1,1),order='F'), U==abcdefg.reshape((-1,1),order='F')])))

    if sol.value(J_k,[Uk==abcdefg.reshape((-1,1),order='F')]) >= sol.value(J_k,[Uk==CalculatePara_old.reshape((-1,1),order='F')]):
        abcdefg = copy.copy(CalculatePara_old[0:2*num_agent,:])
    x_current = sol.value(xk[:,1],[Uk==abcdefg.reshape((-1,1),order='F')]).reshape((-1,1))

    for i in range(num_agent):
        if np.linalg.norm(x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1)),ord=2) <= MissionPara['Err']:
            flagexit[i] = 1
        if (x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1))).T @ MissionPara['P'] @ (x_current[3*i:3*(i+1),:]-MissionPara['Send'][i,:].reshape((-1,1))) <= 0.2:
            flagterminallaw[i] = 1
    if min(flagexit) == 1:
        break

    x_req = sol.value(xk[:,-1],[Uk==abcdefg.reshape((-1,1),order='F')]).reshape((-1,1))
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
        if sol.value(J_k,[Uk==warm_start_req.reshape((-1,1),order='F')]) >= sol.value(J_k,[Uk==warm_start_req2.reshape((-1,1),order='F')]):
            CalculatePara['warm_start'] = copy.copy(warm_start_req2)
        else:
            CalculatePara['warm_start'] = copy.copy(warm_start_req)

    
    print('x_current =', x_current.reshape((1,-1))[0])
    print('u_current =', abcdefg[:,0].reshape((1,-1))[0])


print('Mission Complete')