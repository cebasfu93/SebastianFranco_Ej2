import numpy as np
import matplotlib.pyplot as plt

N_iter=1000
desv=0.05
N=6
v=5

time_obs=np.array([3.23, 3.82, 2.27, 3.04, 5.65, 6.57])
posx=np.array([2, -2, 5, 8, 5, 1], dtype='float')
posy=np.array([20, -1, 12, 10, -16, 40], dtype='float')
posz=np.array([0, 0, 0, 0, 0, 0], dtype='float')

time_new=np.zeros(N)
time_old=np.zeros(N)

def modelo(x_p, y_p, z_p):
    r=((posx-x_p)**2+(posy-y_p)**2+(posz-z_p)**2)**0.5
    return r/v
def dist(x_p,y_p,z_p):
    return modelo(x_p, y_p, z_p)*v
def leapfrog(x_p, y_p, z_p, px, py, pz, deltat=0.1, niter=5):
    x_pnew=x_p
    y_pnew=y_p
    z_pnew=z_p
    px_new=px
    py_new=py
    pz_new=pz
    for i in range(niter):
        px_new=px_new+0.5*deltat*gradx_log_like(x_pnew, y_pnew, z_pnew)
        py_new=py_new+0.5*deltat*grady_log_like(x_pnew, y_pnew, z_pnew)
        pz_new=pz_new+0.5*deltat*gradz_log_like(x_pnew, y_pnew, z_pnew)

        x_pnew=x_pnew+0.5*deltat*px_new
        y_pnew=y_pnew+0.5*deltat*py_new
        z_pnew=z_pnew+0.5*deltat*pz_new

        px_new=px_new+0.5*deltat*gradx_log_like(x_pnew, y_pnew, z_pnew)
        py_new=py_new+0.5*deltat*grady_log_like(x_pnew, y_pnew, z_pnew)
        pz_new=pz_new+0.5*deltat*gradz_log_like(x_pnew, y_pnew, z_pnew)
    return x_pnew, y_pnew, z_pnew, px_new, py_new, pz_new
def chi2(x_p, y_p, z_p):
    t_calc=modelo(x_p, y_p, z_p)
    return np.sum(((t_calc-time_obs)/desv)**2)
def like(x_p, y_p, z_p):
    t_calc=modelo(x_p, y_p, z_p)
    chi_2=chi2(t_calc)
    return np.exp(-chi_2)
def log_like(x_p, y_p, z_p):
    t_calc=modelo(x_p, y_p, z_p)
    chi_2=chi2(x_p, y_p, z_p)
    return -chi_2
def gradx_log_like(x_p, y_p, z_p):
    return -1./desv**2*np.sum((modelo(x_p,y_p,z_p)-time_obs)*(x_p-posx)/(v*dist(x_p, y_p, z_p)))
def grady_log_like(x_p, y_p, z_p):
    return -1./desv**2*np.sum((modelo(x_p,y_p,z_p)-time_obs)*(y_p-posy)/(v*dist(x_p, y_p, z_p)))
def gradz_log_like(x_p, y_p, z_p):
    return -1./desv**2*np.sum((modelo(x_p,y_p,z_p)-time_obs)*(z_p-posz)/(v*dist(x_p, y_p, z_p)))
def hamilton(px, py, pz, x_p, y_p, z_p):
    return 0.5*(px**2+py**2+pz**2)-log_like(x_p, y_p, z_p)
def MCMC(steps):
    xs=np.zeros(steps)
    ys=np.zeros(steps)
    zs=np.zeros(steps)
    pxs=np.zeros(steps)
    pys=np.zeros(steps)
    pzs=np.zeros(steps)
    xs[0]=np.random.normal(0,1)
    ys[0]=np.random.normal(0,1)
    zs[0]=np.random.normal(0,1)
    pxs[0]=np.random.normal(0,1)
    pys[0]=np.random.normal(0,1)
    pzs[0]=np.random.normal(0,1)
    for i in range(1, steps):
        pxs[i]=np.random.normal(0,1)
        pys[i]=np.random.normal(0,1)
        pzs[i]=np.random.normal(0,1)
        x_new, y_new, z_new, px_new, py_new, pz_new = leapfrog(xs[i-1], ys[i-1], zs[i-1], pxs[i-1], pys[i-1], pzs[i-1])
        E_new=hamilton(px_new, py_new, pz_new, x_new, y_new, z_new)
        E_old=hamilton(pxs[i-1], pys[i-1], pzs[i-1], xs[i-1], ys[i-1], zs[i-1])
        alpha=min(1.0, np.exp(-E_new+E_old))
        beta=np.random.random()
        if beta < alpha:
            xs[i]=x_new
            ys[i]=y_new
            zs[i]=z_new
        else:
            xs[i]=xs[i-1]
            ys[i]=ys[i-1]
            zs[i]=zs[i-1]
    return xs, ys, zs

x_chain, y_chain, z_chain=MCMC(N_iter)
fig=plt.figure()
a=plt.hist(x_chain[500:], bins=20, normed=True)
plt.xlabel('x')
plt.ylabel('Frecuencia')
plt.title('Distribucion en X')
plt.savefig('x.pdf', format='pdf')
plt.close()

fig=plt.figure()
b=plt.hist(y_chain[500:], bins=20, normed=True)
plt.xlabel('y')
plt.ylabel('Frecuencia')
plt.title('Distribucion en Y')
plt.savefig('y.pdf', format='pdf')
plt.close()

fig=plt.figure()
c=plt.hist(z_chain[500:], bins=20, normed=True)
plt.xlabel('z')
plt.ylabel('Frecuencia')
plt.title('Distribucion en Z')
plt.savefig('z.pdf', format='pdf')
plt.close()


n_rubin=4
chainx={}
chainy={}
chainz={}
for i in range(n_rubin):
    chainx[i], chainy[i], chainz[i]=MCMC(N_iter)

Rx=np.zeros(N_iter-1)
Ry=np.zeros(N_iter-1)
Rz=np.zeros(N_iter-1)

for i in range(N_iter-1):
    n=i+1
    mean_chainx=np.zeros(n_rubin)
    mean_chainy=np.zeros(n_rubin)
    mean_chainz=np.zeros(n_rubin)
    variance_chainx=np.zeros(n_rubin)
    variance_chainy=np.zeros(n_rubin)
    variance_chainz=np.zeros(n_rubin)
    for j in range(n_rubin):
        mean_chainx[j]=chainx[j][:n].mean()
        mean_chainy[j]=chainy[j][:n].mean()
        mean_chainz[j]=chainz[j][:n].mean()
        variance_chainx[j]=chainx[j][:n].std()**2
        variance_chainy[j]=chainy[j][:n].std()**2
        variance_chainz[j]=chainz[j][:n].std()**2

    mean_genx=mean_chainx.mean()
    mean_geny=mean_chainy.mean()
    mean_genz=mean_chainz.mean()

    A=0.0
    B=0.0
    C=0.0
    for j in range(n_rubin):
        A+=(mean_chainx[j]-mean_genx)**2
        B+=(mean_chainy[j]-mean_geny)**2
        C+=(mean_chainz[j]-mean_genz)**2
    A=n*A/(n_rubin-1)
    B=n*B/(n_rubin-1)
    C=n*C/(n_rubin-1)

    Wx=variance_chainx.mean()
    Wy=variance_chainy.mean()
    Wz=variance_chainz.mean()

    if (Wx==0.0 or Wy==0.0 or Wz==0.0):
        print i
    Rx[n-1]=(n-1)/n + (A/Wx)*(n_rubin+1)/(n*n_rubin)
    Ry[n-1]=(n-1)/n + (B/Wy)*(n_rubin+1)/(n*n_rubin)
    Rz[n-1]=(n-1)/n + (C/Wz)*(n_rubin+1)/(n*n_rubin)

plt.figure()
plt.title('Rubin-Gelman')
plt.xlabel('Iteracion')
plt.ylabel(r'$\hat{R}$')
plt.plot(Rx[500:], label='Datos en X')
plt.plot(Ry[500:], label='Datos en Y')
plt.plot(Rz[500:], label='Datos en Z')
plt.legend()
plt.savefig('Rubin.pdf', format='pdf')
plt.close()
