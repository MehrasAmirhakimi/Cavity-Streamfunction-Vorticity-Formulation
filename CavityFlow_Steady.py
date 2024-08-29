import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy import sparse

def nodeij(i,im):
    inode = i%(im-1)
    jnode = i//(im-1)
    return inode,jnode

Re=100
im=129
jm=129
H=1
dx=H/im
dy=H/jm
beta=dx/dy
N=(im-1)*(jm-1)

aomega=np.zeros([N,N])
bomega=np.zeros(N)
asai=aomega.copy()
bsai=bomega.copy()
uold=np.zeros(N)
unew=uold.copy()
vold=uold.copy()
vnew=uold.copy()
saiold=uold.copy()
sainew=uold.copy()
omegaold=uold.copy()
omeganew=uold.copy()

u=np.zeros(N)
v=u.copy()
sai=u.copy()
omega=u.copy()

saiwall=0

komega=0.5
ksai=0.5
er_array=np.zeros([4,1])
erj=np.zeros([4,1])

j=0
er=1
while er>1e-7:
    j=j+1
    # calculating vorticity field
    for i in range(N):
        inode,jnode = nodeij(i,im)
        aomega[i,i]=max(uold[i],0)+max(-uold[i],0)+beta*max(vold[i],0)+beta*max(-vold[i],0)+2/(Re*dx)+2*beta**2/(Re*dx)
        bomega[i]=0
        if inode==0:
            aomega[i,i+1]=--max(-uold[i],0)-1/(Re*dx)
            bomega[i]+=2/(dx**2)*(max(uold[i],0)+1/(Re*dx))*(saiwall-saiold[i])
        elif inode==im-2:
            aomega[i,i-1]=-(max(uold[i],0)+1/(Re*dx))
            bomega[i]+=2/(dx**2)*(max(-uold[i],0)+1/(Re*dx))*(saiwall-saiold[i])
        else:
            aomega[i,i+1]=-(max(-uold[i],0)+1/(Re*dx))
            aomega[i,i-1]=-(max(uold[i],0)+1/(Re*dx))
            bomega[i]+=0    
        if jnode==0:
            aomega[i,i+(im-1)]=-(beta*max(-vold[i],0)+beta**2/(Re*dx))
            bomega[i]+=2/(dy**2)*(beta*max(vold[i],0)+beta**2/(Re*dx))*(saiwall-saiold[i])
        elif jnode==jm-2:
            aomega[i,i-(im-1)]=-(beta*max(vold[i],0)+beta**2/(Re*dx))
            bomega[i]+=2/(dy**2)*(beta*max(-vold[i],0)+beta**2/(Re*dx))*(saiwall-saiold[i]-1*dy)
        else:
            aomega[i,i+(im-1)]=-(beta*max(-vold[i],0)+beta**2/(Re*dx))
            aomega[i,i-(im-1)]=-(beta*max(vold[i],0)+beta**2/(Re*dx))
            bomega[i]+=0
    a2=sparse.csr_matrix(aomega)
    x=spla.gmres(a2,bomega)
#    if j==1:
#        a31=a2
#        x31=x
#        print(max(uold[i],0)+max(-uold[i],0)+beta*max(vold[i],0)+beta*max(-vold[i],0)+2/(Re*dx)+2*beta**2/(Re*dx))
    omeganew=x[0]
    omeganew=omegaold+komega*(omeganew-omegaold)
    er_omega=np.sqrt(np.sum((omeganew-omegaold)**2)/np.sum(omeganew**2))
    erj[0,0]=er_omega
    
    # calculating stream function
    for i in range(N):
        inode,jnode = nodeij(i,im)
        asai[i,i]=-2*(1+beta**2)
        bsai[i]=-dx**2*omeganew[i]
        if inode==0:
            asai[i,i+1]=1
            bsai[i]+=-saiwall
        elif inode==im-2:
            asai[i,i-1]=1
            bsai[i]+=-saiwall
        else:
            asai[i,i+1]=1
            asai[i,i-1]=1
            bsai[i]+=0
        if jnode==0:
            asai[i,i+(im-1)]=beta**2
            bsai[i]+=-beta**2*saiwall
        elif jnode==jm-2:
            asai[i,i-(im-1)]=beta**2
            bsai[i]+=-beta**2*saiwall
        else:
            asai[i,i+(im-1)]=beta**2
            asai[i,i-(im-1)]=beta**2
            bsai[i]+=0
    a2=sparse.csr_matrix(asai)
    x=spla.gmres(a2,bsai)
#    if j==1:
#        a31=a2
#        x31=x
#        print(-2*(1+beta**2))
    sainew=x[0]
    sainew=saiold+ksai*(sainew-saiold)
    er_sai=np.sqrt(np.sum((sainew-saiold)**2)/np.sum(sainew**2))
    erj[1,0]=er_sai
        
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
    print(j,er)
    # upgrading old variables for unknowns (k->k+1)
    omegaold=omeganew.copy()
    saiold=sainew.copy()
    uold=unew.copy()
    vold=vnew.copy()
# assigning most recent calculated variables (last k) to final solution
omega=omeganew
sai=sainew
u=unew
v=vnew

x=np.linspace(0,1,im+1)
y=np.linspace(0,1,jm+1)
X,Y=np.meshgrid(x,y)

U=np.zeros([im+1,jm+1])
U[1:im,1:jm]=np.reshape(u,[im-1,jm-1])
U[im,:]=1
plt.figure(num=0,figsize=(8,6))
cp=plt.contourf(X,Y,U,35,cmap='jet')
plt.colorbar()

V=np.zeros([im+1,jm+1])
V[1:im,1:jm]=np.reshape(v,[im-1,jm-1])
plt.figure(num=1,figsize=(8,6))
cp=plt.contourf(X,Y,V,35,cmap='jet')
plt.colorbar()

plt.figure(num=2,figsize=(6,6))
plt.quiver(X,Y,U,V)

plt.figure(num=3,figsize=(6,6))
plt.streamplot(X,Y,U,V,color=U**2+V**2,cmap='jet')

SAI=np.zeros([im+1,jm+1])+saiwall
SAI[1:im,1:jm]=np.reshape(sai,[im-1,jm-1])
plt.figure(num=4,figsize=(6,6))
levels=np.array([-0.1175,-0.115,-0.11,-0.09,-0.07,-0.05,-0.03,-0.01,-1e-4,-1e-5,-1e-7,-1e-10\
                 ,1e-8,1e-7,1e-6,1e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,1.5e-3,3e-3],float)
cp=plt.contour(X,Y,SAI,levels,cmap='jet')
plt.clabel(cp, inline=True, fontsize=10)

OMEGA=np.zeros([im+1,jm+1])
OMEGA[1:im,1:jm]=np.reshape(omega,[im-1,jm-1])
OMEGA[:,0]=2/(dx**2)*(saiwall-SAI[:,1])
OMEGA[:,jm]=2/(dx**2)*(saiwall-SAI[:,jm-1])
OMEGA[0,:]=2/(dy**2)*(saiwall-SAI[1,:])
OMEGA[im,:]=2/(dy**2)*(saiwall-SAI[im-1,:]-1*dy)
plt.figure(num=5,figsize=(6,6))
levels=np.array([-3,-2,-1,-0.5,0,0.5,1,2,3,4,5])
cp=plt.contour(X,Y,OMEGA,levels,cmap='jet')
plt.clabel(cp, inline=True, fontsize=10)