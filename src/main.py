import readmsh
import SH_greenfunction
import DH_greenfunction
import numpy as np
import sys
import matplotlib.pyplot as plt
import QDsim
from math import *
import time
import argparse
import os
import psutil

file_name = sys.argv[0]
print(file_name)
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / (1024 ** 2)} MB") 

if __name__ == "__main__":
    try:
        #start_time = time.time()
        parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
        parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
        parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

        args = parser.parse_args()

        fnamegeo = args.inputgeo
        fnamePara = args.inputpara
    
    except:
        #fnamegeo='examples/surface0/Surface0.msh'
        #fnamePara='examples/surface0/parameter.txt'
        fnamegeo='examples/bp5t/bp5t.msh'
        fnamePara='examples/bp5t/parameter.txt'
        #fnamegeo='examples/d60_u/d60.msh'
        #fnamePara='examples/d60_u/parameter.txt'
    print('Input msh geometry file:',fnamegeo)
    print('Input parameter file:',fnamePara)
    
    
    
    #fname='bp5t.msh'
    nodelst,elelst=readmsh.read_mshV2(fnamegeo)
    print('Number of Node',nodelst.shape)
    print('Number of Element',elelst.shape)
    #print(boundary_nodes)
    #print(np.max(nodelst[:,0]),np.min(nodelst[:,0]),np.max(nodelst[:,1]),np.min(nodelst[:,1]))
    
    
    #print(eleVec.shape,xg.shape)
    
    sim0=QDsim.QDsim(elelst,nodelst,fnamePara)
    #print(sim0.__dict__)
    
    #print(f"Time taken: {end_time - start_time:.2f} seconds")


    #print(sim0.slipv.shape,sim0.Tt.shape)


    Sliplast=np.mean(sim0.slipv)
    sim0.calc_nucleaszie_cohesivezone()

    x=0
    y=0
    z=-10000
    index1_=sim0.get_value(x,y,z)

    print(sim0.mu*sim0.dc/(sim0.b[0]*1e6*10))
    start_time=time.time()
    
    totaloutputsteps=int(sim0.Para0['totaloutputsteps'])
    
    if(sim0.useGPU==True):
        sim0.initGPUvariable()
        SLIP=[]
        SLIPV=[]
        Tt=[]
        f=open('state.txt','w')
        f1=open('curve.txt','w')
        for i in range(totaloutputsteps):
            print('iteration:',i)
            if(i==0):
                dttry=sim0.htry
            else:
                dttry=dtnext
            dttry,dtnext=sim0.simu_forwardGPU(dttry)

            sim0.slipv=sim0.slipv_gpu.get()
            sim0.slip=sim0.slip_gpu.get()
            sim0.Tt=sim0.Tt_gpu.get()
            sim0.state=sim0.state_gpu.get()

            year=sim0.time/3600/24/365
            #print('self.slipv1',np.max(sim0.slipv1))
            print('dt:',dttry,' Seconds:',sim0.time,'  Days:',sim0.time/3600/24,'year',year)
            print(' max_vel:',np.max(np.abs(sim0.slipv)),' min_Tt:',np.min(sim0.Tt),' min_state:',np.min(sim0.state))  
            f.write('%d %f %.16e %f %f %f %f\n' %(i,dttry,np.max(np.abs(sim0.slipv)),sim0.time,sim0.time/3600.0/24.0,sim0.Relerrormax1,sim0.Relerrormax2))
            f1.write('%d %f %f %f %.6e %.16e\n'%(i,dttry,sim0.time,sim0.time/3600.0/24.0,sim0.Tt[index1_],sim0.slipv[index1_]))
            SLIP.append(sim0.slip)
            SLIPV.append(sim0.slipv)
            Tt.append(sim0.Tt)

            memory_info = process.memory_info()
            print(f"Memory usage: {memory_info.rss / (1024 ** 2)} MB")
            
            outsteps=int(sim0.Para0['outsteps'])
            directory='out'
            if not os.path.exists(directory):
                os.mkdir(directory)
            
            if(i%outsteps==0):
                SLIP=np.array(SLIP)
                SLIPV=np.array(SLIPV)
                Tt=np.array(Tt)
                #np.save('examples/d60_u/slipv/slipv_%d'%i,SLIPV)
                #np.save('examples/d60_u/slip/slip_%d'%i,SLIP)
                #np.save('examples/d60_u/Tt/Tt_%d'%i,Tt)
                SLIP=[]
                SLIPV=[]
                Tt=[]
            
            if(i%outsteps==0):
                
                sim0.slipv1=sim0.slipv1_gpu.get()
                sim0.slipv2=sim0.slipv2_gpu.get()
                
                sim0.slip1=sim0.slip1_gpu.get()
                sim0.slip2=sim0.slip2_gpu.get()
                
                sim0.Tno=sim0.Tno_gpu.get()
                sim0.Tt1o=sim0.Tt1o_gpu.get()
                sim0.Tt2o=sim0.Tt2o_gpu.get()
                
                if(sim0.Para0['outputvtk']=='True'):
                    fname=directory+'/step'+str(i)+'.vtk'
                    sim0.ouputVTK(fname)
                if(sim0.Para0['outputmatrix']=='True'):
                    fname='step'+str(i)
                    sim0.outputtxt(fname)
                
                if(year>1200 or np.max(np.abs(sim0.slipv))>10):
                    break
                
                
        end_time=time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        SLIP=[]
        SLIPV=[]
        Tt=[]
        #print(size_nuclear)
        f=open('state.txt','a')
        f1=open('curve.txt','a')
        
        for i in range(totaloutputsteps):
        #for i in range(10):
            print('iteration:',i)

            # for k in range(len(sim0.arriT)):
            #     if(sim0.slipv[k]>=0.01 and sim0.arriT[k]==1e9):
            #         sim0.arriT[k]=sim0.time

            
            if(i==0):
                dttry=sim0.htry
            else:
                dttry=dtnext
            dttry,dtnext=sim0.simu_forward(dttry)
            year=sim0.time/3600/24/365
            print('dt:',dttry,' max_vel:',np.max(np.abs(sim0.slipv)),' Seconds:',sim0.time,'  Days:',sim0.time/3600/24,
                'year',year)
            f.write('%d %f %.16e %f %f %f %f\n' %(i,dttry,np.max(np.abs(sim0.slipv)),sim0.time,sim0.time/3600.0/24.0,sim0.Relerrormax1,sim0.Relerrormax2))
            f1.write('%d %f %f %f %.6e %.16e\n'%(i,dttry,sim0.time,sim0.time/3600.0/24.0,sim0.Tt[index1_],sim0.slipv[index1_]))
            SLIP.append(sim0.slip)
            SLIPV.append(sim0.slipv)
            Tt.append(sim0.Tt)
            
            # if(sim0.time>60):
            #     break
            outsteps=int(sim0.Para0['outsteps'])
            directory='out'
            if not os.path.exists(directory):
                os.mkdir(directory)
            if(i%outsteps==0):
                SLIP=np.array(SLIP)
                SLIPV=np.array(SLIPV)
                Tt=np.array(Tt)
                #np.save('examples/bp5t/slipv/slipv_%d'%i,SLIPV)
                #np.save('examples/bp5t/slip/slip_%d'%i,SLIP)
                #np.save('examples/bp5t/Tt/Tt_%d'%i,Tt)

                SLIP=[]
                SLIPV=[]
                Tt=[]

                if(sim0.Para0['outputvtk']=='True'):
                    fname=directory+'/step'+str(i)+'.vtk'
                    sim0.ouputVTK(fname)
                if(sim0.Para0['outputmatrix']=='True'):
                    fname='step'+str(i)
                    sim0.outputtxt(fname)
                #if(year>1200 or np.max(np.abs(sim0.slipv))<0.01):
                #if(year>280):
                #    break
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        f.close()
        f1.close()
        
    #sim0.test()
    #fname='test.vtk'
    #sim0.ouputVTK(fname)

    # f=open('arrivt.txt','w')
    # for i in range(len(sim0.arriT)):
    #     f.write('%f %f %f\n' %(sim0.xg[i,0],sim0.xg[i,2],sim0.arriT[i]))
    # f.close()


