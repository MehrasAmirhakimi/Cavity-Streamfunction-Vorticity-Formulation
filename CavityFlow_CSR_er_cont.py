import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy import sparse

def nodeij(i,im):
    inode = i%(im-1)
    jnode = i//(im-1)
    return inode,jnode

tfp=30
nmp=int(tfp/dt)

u1=u.copy()
v1=v.copy()
sai1=sai.copy()
omega1=omega.copy()
u=np.zeros([nm+nmp+1,N])
v=u.copy()
sai=u.copy()
omega=u.copy()
u[0:nm+1,:]=u1
v[0:nm+1,:]=v1
sai[0:nm+1,:]=sai1
omega[0:nm+1,:]=omega1


komega=0.1
ksai=0.1

for n in range(nm,nm+nmp):
    uold=u[n,:]
    vold=v[n,:]
    saiold=sai[n,:]
    omegaold=omega[n,:]
    er=1
    while er>1e-6:
        j=j+1
        # calculating vorticity field
        inz=-1
        for i in range(N):
            inode,jnode = nodeij(i,im)
            inz+=1
            dataomega[inz]=1+dt/dx*(max(uold[i],0)+max(-uold[i],0)+beta*max(vold[i],0)+beta*max(-vold[i],0)+2/(Re*dx)+2*beta**2/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i
            bomega[i]=omega[n,i]   
            if inode==0:
                inz+=1
                dataomega[inz]=-dt/dx*max(-uold[i],0)-dt/(Re*dx**2)
                rowomega[inz]=i
                colomega[inz]=i+1
                bomega[i]+=2/(dx**2)*(dt/dx*max(uold[i],0)+dt/(Re*dx**2))*(saiwall-saiold[i])
            elif inode==im-2:
                inz+=1
                dataomega[inz]=-(dt/dx*max(uold[i],0)+dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i-1
                bomega[i]+=2/(dx**2)*(dt/dx*max(-uold[i],0)+dt/(Re*dx**2))*(saiwall-saiold[i])
            else:
                inz+=1
                dataomega[inz]=-(dt/dx*max(-uold[i],0)+dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i+1
                inz+=1
                dataomega[inz]=-(dt/dx*max(uold[i],0)+dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i-1
                bomega[i]+=0    
            if jnode==0:
                inz+=1
                dataomega[inz]=-(beta*dt/dx*max(-vold[i],0)+beta**2*dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i+(im-1)
                bomega[i]+=2/(dy**2)*(beta*dt/dx*max(vold[i],0)+beta**2*dt/(Re*dx**2))*(saiwall-saiold[i])
            elif jnode==jm-2:
                inz+=1
                dataomega[inz]=-(beta*dt/dx*max(vold[i],0)+beta**2*dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i-(im-1)
                bomega[i]+=2/(dy**2)*(beta*dt/dx*max(-vold[i],0)+beta**2*dt/(Re*dx**2))*(saiwall-saiold[i]-1*dy)
            else:
                inz+=1
                dataomega[inz]=-(beta*dt/dx*max(-vold[i],0)+beta**2*dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i+(im-1)
                inz+=1
                dataomega[inz]=-(beta*dt/dx*max(vold[i],0)+beta**2*dt/(Re*dx**2))
                rowomega[inz]=i
                colomega[inz]=i-(im-1)
                bomega[i]+=0
        a2=sparse.csr_matrix((dataomega, (rowomega, colomega)), shape=(N, N))
        x=spla.gmres(a2,bomega,omegaold)
        omeganew=x[0]
        c1=x[1]
        er_omega=np.sqrt(np.sum((omeganew-omegaold)**2)/np.sum(omeganew**2))
        erj[0,0]=er_omega
        omeganew=omegaold+komega*(omeganew-omegaold)
        
        # calculating stream function
        inz=-1
        for i in range(N):
            inode,jnode = nodeij(i,im)
            inz+=1
            datasai[inz]=-2*(1+beta**2)
            rowsai[inz]=i
            colsai[inz]=i
            bsai[i]=-dx**2*omeganew[i]
            if inode==0:
                inz+=1
                datasai[inz]=1
                rowsai[inz]=i
                colsai[inz]=i+1
                bsai[i]+=-saiwall
            elif inode==im-2:
                inz+=1
                datasai[inz]=1
                rowsai[inz]=i
                colsai[inz]=i-1
                bsai[i]+=-saiwall
            else:
                inz+=1
                datasai[inz]=1
                rowsai[inz]=i
                colsai[inz]=i+1
                inz+=1
                datasai[inz]=1
                rowsai[inz]=i
                colsai[inz]=i-1
                bsai[i]+=0
            if jnode==0:
                inz+=1
                datasai[inz]=beta**2
                rowsai[inz]=i
                colsai[inz]=i+(im-1)
                bsai[i]+=-beta**2*saiwall
            elif jnode==jm-2:
                inz+=1
                datasai[inz]=beta**2
                rowsai[inz]=i
                colsai[inz]=i-(im-1)
                bsai[i]+=-beta**2*saiwall
            else:
                inz+=1
                datasai[inz]=beta**2
                rowsai[inz]=i
                colsai[inz]=i+(im-1)
                inz+=1
                datasai[inz]=beta**2
                rowsai[inz]=i
                colsai[inz]=i-(im-1)
                bsai[i]+=0
        a2=sparse.csr_matrix((datasai, (rowsai, colsai)), shape=(N, N))
        x=spla.gmres(a2,bsai,saiold)
        sainew=x[0]
        c2=x[1]
        er_sai=np.sqrt(np.sum((sainew-saiold)**2)/np.sum(sainew**2))
        erj[1,0]=er_sai
        sainew=saiold+ksai*(sainew-saiold)
            
        # calculating x & y velocities
        for i in range(N):
            inode,jnode = nodeij(i,im)
            if inode==0:
                vnew[i]=-(sainew[i+1]-saiwall)/(2*dx)
            elif inode==im-2:
                vnew[i]=-(saiwall-sainew[i-1])/(2*dx)
            else:
                vnew[i]=-(sainew[i+1]-sainew[i-1])/(2*dx)
            if jnode==0:
                unew[i]=(sainew[i+(im-1)]-saiwall)/(2*dy)
            elif jnode==jm-2:
                unew[i]=(saiwall-sainew[i-(im-1)])/(2*dy)
            else:
                unew[i]=(sainew[i+(im-1)]-sainew[i-(im-1)])/(2*dy)
        er_u=np.sqrt(np.sum((unew-uold)**2)/np.sum(unew**2))
        erj[2,0]=er_u
        er_v=np.sqrt(np.sum((vnew-vold)**2)/np.sum(vnew**2))
        erj[3,0]=er_v
        er_array=np.concatenate((er_array, erj), axis=1)
        er=max(erj)
        print(n,j,er,c1,c2)
        # upgrading old variables for unknowns (k->k+1)
        omegaold=omeganew.copy()
        saiold=sainew.copy()
        uold=unew.copy()
        vold=vnew.copy()
    print(n)
    # assigning most recent calculated variables (last k) to next time step (n+1)
    omega[n+1,:]=omeganew
    sai[n+1,:]=sainew
    u[n+1,:]=unew
    v[n+1,:]=vnew

#x=np.linspace(0,1,im+1)
#y=np.linspace(0,1,jm+1)
#X,Y=np.meshgrid(x,y)
#
#U=np.zeros([im+1,jm+1])
#U[1:im,1:jm]=np.reshape(u[nm,:],[im-1,jm-1])
#U[im,:]=1
#plt.figure(num=0,figsize=(8,6))
#cp=plt.contourf(X,Y,U,35,cmap='jet')
#plt.colorbar()
#
#V=np.zeros([im+1,jm+1])
#V[1:im,1:jm]=np.reshape(v[nm,:],[im-1,jm-1])
#plt.figure(num=1,figsize=(8,6))
#cp=plt.contourf(X,Y,V,35,cmap='jet')
#plt.colorbar()
#
#plt.figure(num=2,figsize=(6,6))
#plt.quiver(X,Y,U,V)
#
#plt.figure(num=3,figsize=(6,6))
#plt.streamplot(X,Y,U,V,color=U**2+V**2,cmap='jet')
#
#SAI=np.zeros([im+1,jm+1])+saiwall
#SAI[1:im,1:jm]=np.reshape(sai[nm,:],[im-1,jm-1])
#plt.figure(num=4,figsize=(6,6))
#levels=np.array([-0.1175,-0.115,-0.11,-0.09,-0.07,-0.05,-0.03,-0.01,-1e-4,-1e-5,-1e-7,-1e-10\
#                 ,1e-8,1e-7,1e-6,1e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,1.5e-3,3e-3],float)
#cp=plt.contour(X,Y,SAI,levels,cmap='jet')
#plt.clabel(cp, inline=True, fontsize=10)
#
#OMEGA=np.zeros([im+1,jm+1])
#OMEGA[1:im,1:jm]=np.reshape(omega[nm,:],[im-1,jm-1])
#OMEGA[:,0]=2/(dx**2)*(saiwall-SAI[:,1])
#OMEGA[:,jm]=2/(dx**2)*(saiwall-SAI[:,jm-1])
#OMEGA[0,:]=2/(dy**2)*(saiwall-SAI[1,:])
#OMEGA[im,:]=2/(dy**2)*(saiwall-SAI[im-1,:]-1*dy)
#plt.figure(num=5,figsize=(6,6))
#levels=np.array([-3,-2,-1,-0.5,0,0.5,1,2,3,4,5])
#cp=plt.contour(X,Y,OMEGA,levels,cmap='jet')
#plt.clabel(cp, inline=True, fontsize=10)
#
#plt.figure(6)
#plt.plot(er_array[:,1::].T)
#plt.yscale('log') ; plt.grid(True)
#plt.xlabel('k') ; plt.ylabel('relative rms')
#plt.legend(('vorticity','stream function','x-velocity','y-velocity'))