import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

def calc_E(phi_p, theta_p, F_pc, phiD, phiN):
    """
    The function above takes in five arguments in the following order:
    latitude, longitude, magnetic flux, night- and dayside reconnection rate.

    We want to calculate the electric field for a given position.
    We start by calculating lambra_R1 for a given magnetic flux
    value (F_pc). Thereafter we can calculate the potential given
    in table 1 in Milan's article from 2013. After that our goal is
    to calculate the gradients of the electric potential with the
    fourier coefficent Sm. When completed we can finally calculate the
    two components E_lambda and E_theta.
    """
    Beq = 31000e-9
    R_E = 6371e3


    L_R1 = np.arcsin(np.sqrt(F_pc/(2*np.pi*R_E**2*Beq)))

    v_R1 = (1/(2*np.pi*R_E*Beq*np.sin(2*L_R1)))*(phiD-phiN)
    B_R1 = 2*Beq*np.cos(L_R1)

    E_B = -v_R1*B_R1

    thtN = np.deg2rad(30)
    thtD = np.deg2rad(30)

    lD = 2*thtD*R_E*np.sin(L_R1)
    lN = 2*thtN*R_E*np.sin(L_R1)

    E_D = E_B + phiD/lD
    E_N = E_B - phiN/lN

    E_D = E_B + phiD/lD
    E_N = E_B - phiN/lN

    theta = np.linspace(0, 2*np.pi, 120)
    Vpcb = np.zeros(len(theta))

    """
    Calculating values for the potential for a given
    colattitutde (lambda_R1) for all theta values. This
    is done with the parameters L_R1, E_B, E_N and E_D.
    """
    for i in range(len(theta)):
        if theta[i] < thtN:
            Vpcb[i] = -R_E*np.sin(L_R1)*(E_N*theta[i])
        if thtN< theta[i] <np.pi-thtD:
            Vpcb[i] = -R_E*np.sin(L_R1)*((E_N-E_B)*thtN+E_B*theta[i])
        if np.pi-thtD < theta[i] < np.pi+thtD:
            Vpcb[i] = -R_E*np.sin(L_R1)*((E_N-E_B)*thtN+(E_D-E_B)*(thtD-np.pi)+E_D*theta[i])
        if np.pi+thtD < theta[i] < 2*np.pi - thtN:
            Vpcb[i] = -R_E*np.sin(L_R1)*((E_N-E_B)*thtN+2*(E_D-E_B)*thtD+E_B*theta[i])
        if 2*np.pi - thtN < theta[i] < 2*np.pi:
            Vpcb[i] = -R_E*np.sin(L_R1)*(2*(E_N-E_B)*(thtN-np.pi)+2*(E_D-E_B)*thtD+E_N*theta[i])

    """
    Now we calculate the Fourier coefficent (Sm) by using the values
    for Vpcb, theta and dtheta. Each Sm[m] is each its own descrete sum
    and each sum is summized from 0 to 2pi with the same amount of steps
    as theta has elements.
    """
    m = np.linspace(0,20, 21)
    Sm = np.zeros(len(m))

    for m in range(21):
        sum_fourier = 0
        for k in range(len(theta)):
            dtheta = 2*np.pi/(len(theta))
            sum_fourier += (Vpcb[k]*np.sin(m*theta[k])*dtheta)
        Sm[m] = (1/np.pi)*sum_fourier

    """
    Next step is to calculate the gradient of the electric potensials.
    We apply the boundary conditions for the return flow(Between the
    Heppner-Maynard boundary and the open closed field lines boundary) and
    if the position is in the polar cap region. Here we make the substitution
    of Lambda_x = ln(tan(L_x/2)).
    """

    L_p = np.deg2rad(90-phi_p)
    dL = np.pi/18
    L_R2 = L_R1 + dL

    if np.tan(phi_p/2)==0:
        Lambda_p = -np.inf
    else:
        Lambda_p = np.log(np.tan(L_p/2))

    Lambda_R1 = np.log(np.tan(L_R1/2)) #ocb

    Lambda_R2 = np.log(np.tan(L_R2/2)) #hmb

    dphi_T = 0
    dphi_L = 0
    
    for m in np.arange(1, 21, 1):
        if Lambda_p <= Lambda_R2 and Lambda_p > Lambda_R1:
            dphi_L += Sm[m]*m*np.sin(m*theta_p)* (np.cosh(m*(Lambda_p-Lambda_R2))/np.sinh(m*(Lambda_R1-Lambda_R2)))
            dphi_T += Sm[m]*m*np.cos(m*theta_p)* (np.sinh(m*(Lambda_p-Lambda_R2))/np.sinh(m*(Lambda_R1-Lambda_R2)))
        elif Lambda_p <= Lambda_R1:
            dphi_L += Sm[m]*m*np.sin(m*theta_p)*np.exp(m*(Lambda_p-Lambda_R1))
            dphi_T += Sm[m]*m*np.cos(m*theta_p)*np.exp(m*(Lambda_p-Lambda_R1))

    

    E_lambda = -1/(R_E*np.sin(L_p)) * dphi_L
    E_theta = -1/(R_E*np.sin(L_p)) * dphi_T


    return L_R1, v_R1, Vpcb, Sm, E_lambda, E_theta

a = 60
b = 90 #the reason to why there is a 0-value at the first point. 
       #Increase to 90 and solve inf-problem

n = 30
ph_p = np.linspace(b,a, n) #latitude phi
t_p = np.linspace(2*np.pi, 0, n) #longitude theta

E_lam = np.zeros([len(ph_p),len(t_p)])
E_tht = np.zeros([len(ph_p),len(t_p)])


for p in range(len(ph_p)):
    print("step %d of %g" %(p, n))
    for t in range(len(t_p)):
        L, v, Vp, Sm, E_L, E_T = calc_E(ph_p[p],t_p[t], 0.4e9, 0, 50e3)
        E_lam[p][t] = E_L
        E_tht[p][t] = E_T

        
MLT = np.linspace(0, 24, len(Vp))

bobx = np.zeros((n, n))
boby = np.zeros_like(bobx)
r = np.zeros_like(bobx)
theta = np.zeros_like(bobx)

for i in range(len(bobx)):
    for j in range(len(bobx)):
        r[i, j] = 90 - ph_p[i]
        theta[i, j] = t_p[j]
        boby[i, j] = r[i, j]*np.cos(theta[i, j]) 
        bobx[i, j] = r[i, j]*np.sin(theta[i, j])

"""
colors = E_lam
plt.figure()
ax = plt.subplot(polar = "true")
plt.pcolormesh(theta, r,colors ,cmap = "jet")
plt.gca().invert_yaxis()

ax.set_theta_offset(np.deg2rad(90))
ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
ax.set_xticklabels(['12','18','00','06'])
ax.set_yticks(np.arange(10,31,10))
ax.set_yticklabels(['60','70','80']) 

cbar = plt.colorbar()
cbar.set_label('$E_\lambda (mV \: m^{-1})$', rotation=270)

plt.show()
"""


"""
We now want find the E cross B drift for each position. We create 
a function that calculates V for the latitude and longtidue for a single position.
It will have the arguments phi_p.
"""

"""
OBS, maa oppdateres ved generell bruk
"""


def calc_V(FPC, RCR_D, RCR_N):
    Beq = 31000e-9
    

    N = 30
    A = 60
    B = 90
    
    V_ph_p = np.linspace(B, A, N) #latitude phi
    V_t_p = np.linspace(2*np.pi, 0, N) #longitude theta
    V_L_p = np.zeros(N) 
    Bp = np.zeros((N, N))
    
    #VLp går fra 1-30, men skal det være 60-90?
    for ii in range(N):
        V_L_p[ii] = np.deg2rad(90-V_ph_p[ii])
        for jj in range(N):
            Bp[ii][jj] = 2*Beq*np.cos(V_L_p[ii])
    

    E_lam_v = np.zeros([len(V_ph_p),len(V_t_p)])
    E_tht_v = np.zeros([len(V_ph_p),len(V_t_p)])
    
    V_tht = np.zeros([len(V_ph_p),len(V_t_p)])
    V_lam = np.zeros([len(V_ph_p),len(V_t_p)])
    
    
    for p in range(len(V_ph_p)):
        for t in range(len(V_t_p)):
            L, v, Vp, Sm, E_L, E_T = calc_E(V_ph_p[p],V_t_p[t], FPC, RCR_D, RCR_N)
            E_lam_v[p][t] = E_L
            E_tht_v[p][t] = E_T
            V_tht[p][t] = E_lam_v[p][t]/Bp[p][t]
            V_lam[p][t] = E_tht_v[p][t]/Bp[p][t]
    return V_tht, V_lam, Bp

#dmlt = 1.125 #how big the patch is in mlt range

"""
Creating tracers, here sMLT decides starting position in MLT. 
"""

def trco(s_MLT, Rr, x):
    """ converts tracer coordinates in MLT and colatitude into xy-coordinates """
    """ converts between mlt and radians. x = mlt """
    start_theta = s_MLT*2*np.pi/24 #rad
    start_x=Rr*np.cos(start_theta)
    start_y=Rr*np.sin(start_theta)
    return start_x, start_y

def create_tracer(sMLT, L_p, dmlt):
    
    tracer_xy = [] #empty list to append tracer position
    tracer_mlt1 = 12-sMLT
    tracer_mlt2= 12-sMLT+dmlt
    l1 = 7 # number of tracer particles 
    
    mlts = np.linspace(tracer_mlt1,tracer_mlt2,l1)
    for i in np.arange(0,l1):
        tracer_xy.append(trco(mlts[i], L_p, sMLT))   
    return tracer_xy

def tracer_path(V_theta, V_lambda,sMLT, L_P, dmlt, FPC, RCR_D, RCR_N):
    RR = np.zeros((n, n))
    TT = np.zeros((n, n))
    Vx = np.zeros((n, n))
    Vy = np.zeros((n, n))
    
    tracers_pos = create_tracer(sMLT, L_P, dmlt)
    VT, VL, BP = calc_V(FPC, RCR_D, RCR_N)
    
    for c in range(n): 
        for cc in range(n):
            Vx[c, cc] = VL[c]*np.cos*(theta[cc]) - r[c]*VT[cc]*np.sin(theta[cc])
            Vy[c, cc] = VL[c]*np.sin*(theta[cc]) + r[c]*VT[cc]*np.cos(theta[cc])

#    for i in range():
#        #find V at tracer location
#        tr_x[i] = tr_x[i-1] + vx*dt
#        tr_y[i] = tr_y[i-1] + vy*dt
#        
#        RR = np.sqrt(x[i]**2+y[i]**2)
#        TT = np.arctan2(y[i],x[i])      
    return Vx, Vy


"""
Creating synthetic data to simulate the time evolution 
of our electrical field- and flow values.
"""
start_t = 0 
stop_t = 4*60
minutes_skipped = 2
stepss = int((stop_t-start_t)/minutes_skipped)
Nn = np.linspace(start_t, stop_t, stepss)
dt = (Nn[1]-Nn[0])*60

synth_phi_D = np.zeros(len(Nn))
synth_phi_N = np.zeros(len(Nn))
synth_Fpc = np.zeros(len(Nn))
synth_dFpc = np.zeros(len(Nn))
synth_Fpc[0] = 0.3e9


for kk in range(len(Nn)):
    if Nn[kk] < 40:
        synth_phi_D[kk] = 0
    if 40< Nn[kk] <60:
        synth_phi_D[kk] = (synth_phi_D[kk-1] + 60*dt)
    if 60 < Nn[kk] < 150:
        synth_phi_D[kk] = synth_phi_D[kk-1] 
    if 150 < Nn[kk] < 170:
        synth_phi_D[kk] = (synth_phi_D[kk-1] - 60*dt)
    if 170 < Nn[kk] < 241:
        synth_phi_D[kk] = 0

for ll in range(len(Nn)):
    if Nn[ll] < 110:
        synth_phi_N[kk] = 0
    if 110< Nn[ll] <130:
        synth_phi_N[ll] = (synth_phi_N[ll-1] + 90*dt)
    if 130 < Nn[ll] < 210:
        synth_phi_N[ll] = synth_phi_N[ll-1] 
    if 210 < Nn[ll] < 230:
        synth_phi_N[ll] = (synth_phi_N[ll-1] - 90*dt)
    if 230 < Nn[ll] < 241:
        synth_phi_N[ll] = 0

for dd in range(1,len(Nn)):
    synth_dFpc[dd] = (synth_phi_D[dd]-synth_phi_N[dd])*dt
    synth_Fpc[dd] = synth_Fpc[dd-1]+synth_dFpc[dd]    
    

plt.plot(Nn, synth_phi_D/1e3)
plt.plot(Nn, synth_phi_N/1e3)
plt.xlabel("time (minutes)")
plt.ylabel("$\phi_{N/D}$ (kV)")
plt.legend(["synth_phi_D", "synth_phi_N"])
plt.show()

plt.plot(Nn, synth_Fpc/1e9)
plt.xlabel("time (minutes)")
plt.ylabel("$F_{PC}$ (GWb)")
plt.legend(["synth_Fpc"])
plt.show()


V_theta = np.zeros(len(Nn))
V_lambda = np.zeros(len(Nn))

for e in range(len(Nn)):   
    , Vl, Bp = calc_V(synth_Fpc[e], synth_phi_D[e], synth_phi_N[e])
#    V_theta[e] = Vt
#    V_lambda[e] = Vl

sE_lam = np.zeros([len(Nn), len(ph_p),len(t_p)])
sE_tht = np.zeros([len(Nn),len(ph_p),len(t_p)])
ocb_location = np.zeros([len(Nn),len(ph_p),len(t_p)])

for tt in range(len(Nn)):
    print("step %d of %g" %(tt, len(Nn)))
    for pp in range(len(ph_p)):
        for qq in range(len(t_p)):
            sL, sv, sVp, sSm, sE_L, sE_T = calc_E(ph_p[pp],t_p[qq], synth_Fpc[tt], synth_phi_D[tt], synth_phi_N[tt])
            ocb_location[tt][pp][qq] = sL
            sE_lam[tt][pp][qq] = sE_L
            sE_tht[tt][pp][qq] = sE_T
        
    colorss = sE_tht[tt]
    plt.figure()
    ax1 = plt.subplot(polar = "true")
    plt.pcolormesh(theta, r,colorss,cmap = "jet")
    plt.gca().invert_yaxis()
    
    ax1.set_theta_offset(np.deg2rad(90))
    ax1.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    ax1.set_xticklabels(['12','18','00','06'])
    ax1.set_yticks(np.arange(10,31,10))
    ax1.set_yticklabels(['60','70','80']) 
    
    cbar1 = plt.colorbar()
    cbar1.set_label('$E_\theta (mV \: m^{-1})$', rotation=270)
    
    plt.clim(-0.1,0.1)
    abc = "%d" %(tt)
    cb = Nn[tt] * 60
    cba = "%d" %(cb)
    plt.title("Eletric field component in $\theta$ direction t = " + cba + "s")
    plt.savefig("%d synth_ET.jpg" %(tt))
    plt.show()


    colorsss = sE_lam[tt]
    plt.figure()
    ax2 = plt.subplot(polar = "true")
    plt.pcolormesh(theta, r,colorsss,cmap = "jet")
    plt.gca().invert_yaxis()
    
    ax2.set_theta_offset(np.deg2rad(90))
    ax2.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    ax2.set_xticklabels(['12','18','00','06'])
    ax2.set_yticks(np.arange(10,31,10))
    ax2.set_yticklabels(['60','70','80']) 
    
    cbar2 = plt.colorbar()
    cbar2.set_label('$E_\lambda (mV \: m^{-1})$', rotation=270)

    plt.clim(-0.1,0.1)
    plt.title("Eletric field component in $\lambda$ direction t = " + cba + "s")
    plt.savefig("Lambda %d synth.jpg" %(tt))
    plt.show()
 


stop = time.time()
print(stop - start, "s")

LPP = np.zeros(len(ph_p))
for z in range(len(ph_p)):
    LPP[z]= np.deg2rad(90-ph_p[z])

for ee in range(len(Nn)):    
    VX, VY = tracer_path(V_theta[ee], V_lambda[ee],13.0, LPP[ee], 1.125, synth_Fpc[ee], synth_phi_D[ee], synth_phi_N[ee])

#Lag func som oppdaterer
# i funk, kall på make tracer
#Make tracer er i kartesisk, konverter
#prøv å plot func i polar koordinater