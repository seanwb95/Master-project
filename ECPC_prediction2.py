import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

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

    theta = np.linspace(2*np.pi,0, 120)
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
    
    L_p = phi_p
    dL = np.pi/18
    L_R2 = L_R1 + dL
    
    Lambda_p = 0

    if np.tan(L_p/2)==0:
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


    return L_R1, E_lambda, E_theta#,L_p

n = 30

Phi_p = np.deg2rad(np.linspace(0, 30, n)) #latitude phi
t_p = np.linspace(0, 2*np.pi, n) #longitude theta

E_lam = np.zeros([len(Phi_p),len(t_p)])
E_tht = np.zeros([len(Phi_p),len(t_p)])


#for p in range(len(Phi_p)):
 #   print("step %d of %g" %(p, n))
 #   for t in range(len(t_p)):
 #       L, E_L, E_T = calc_E(Phi_p[p],t_p[t], 0.4e9, 0, 50e3)
 #       E_lam[p][t] = E_L
 #       E_tht[p][t] = E_T

r = np.zeros((n, n))
theta = np.zeros((n, n))


for i in range(len(r)):
    for j in range(len(r)):
        r[i, j] = Phi_p[i]
        theta[i, j] = t_p[j]

ph_p2 = np.deg2rad(np.linspace(30, 0, n)) #latitude phi
t_p2 = np.linspace(0,2*np.pi, n) #longitude theta
R, T = np.meshgrid(Phi_p, t_p)

"""
We now want find the E cross B drift for each position. We create 
a function that calculates V for the latitude and longtidue for a single position.
It will have the arguments phi_p.
"""

def calc_Bp(tr_pos):
    Beq = 31000e-9
    tr_Bp = 2*Beq*np.cos(tr_pos)
    return tr_Bp

def calc_V(phi_p,V_t_p,FPC, RCR_D, RCR_N):
    Bp = calc_Bp(phi_p)
    L, E_L, E_T = calc_E(phi_p,V_t_p, FPC, RCR_D, RCR_N)
    V_tht = E_L/Bp
    V_lam = -E_T/Bp
    return V_tht, V_lam

"""
Creating tracers, here sMLT decides starting position in MLT. 
"""

def trco(s_MLT, Rr):
    """ converts between mlt and radians."""
    if s_MLT == 12.0:
        s_MLT = 12.5
    start_theta = s_MLT*2*np.pi/24 #rad
    return Rr, start_theta

def create_tracer(sMLT, dmlt, tr_nr, pc_flux):
    
    Beq = 31000e-9
    R_E = 6371e3

    Rr1 = np.arcsin(np.sqrt(pc_flux/(2*np.pi*R_E**2*Beq)))
    Rr = np.linspace(Rr1, Rr1, tr_nr)
    
    tracer_RT = [] #empty list to append tracer position
    tracer_mlt1 = sMLT
    tracer_mlt2= sMLT+dmlt
    
    mlts = np.linspace(tracer_mlt1,tracer_mlt2,tr_nr)
    
    for i in np.arange(0, tr_nr):
        tracer_RT.append(trco(mlts[i], Rr[i]))
    return tracer_RT


"""
Creating synthetic data to simulate the time evolution 
of our electrical field- and flow values.
"""
start_t = 0 
stop_t = 4*60
minutes_skipped = 2
stepss = int((stop_t-start_t)/minutes_skipped)
Nn = np.linspace(start_t, stop_t, stepss+1)
dt = (Nn[1]-Nn[0])*60

synth_phi_D = np.zeros(len(Nn))
synth_phi_N = np.zeros(len(Nn))
synth_Fpc = np.zeros(len(Nn))
synth_dFpc = np.zeros(len(Nn))
synth_Fpc[0] = 0.4e9


for kk in range(len(Nn)):
    if Nn[kk] < 40:
        synth_phi_D[kk] = 0
    if 40 < Nn[kk] <= 60:
        synth_phi_D[kk] = (synth_phi_D[kk-1] + 60*dt)
    if 60 < Nn[kk] <= 150:
        synth_phi_D[kk] = synth_phi_D[kk-1] 
    if 150 < Nn[kk] <= 170:
        synth_phi_D[kk] = (synth_phi_D[kk-1] - 60*dt)
        if synth_phi_D[kk] < 0:
            synth_phi_D[kk] = 0
    if 170 < Nn[kk] < 241:
        synth_phi_D[kk] = 0

for ll in range(len(Nn)):
    if Nn[ll] < 110:
        synth_phi_N[kk] = 0
    if 110< Nn[ll] <= 130:
        synth_phi_N[ll] = (synth_phi_N[ll-1] + 90*dt)
    if 130 < Nn[ll] <= 190:
        synth_phi_N[ll] = synth_phi_N[ll-1] 
    if 190 < Nn[ll] <= 210:
        synth_phi_N[ll] = (synth_phi_N[ll-1] - 90*dt)
        if synth_phi_N[kk] < 0:
            synth_phi_N[kk] = 0
    if 210 < Nn[ll] < 241:
        synth_phi_N[ll] = 0

for dd in range(1,len(Nn)):
    synth_dFpc[dd] = (synth_phi_D[dd]-synth_phi_N[dd])*dt
    synth_Fpc[dd] = synth_Fpc[dd-1]+synth_dFpc[dd]    
    

# plt.plot(Nn, synth_phi_D/1e3, "orange")
# plt.plot(Nn, synth_phi_N/1e3, "b")
# plt.xlabel("UT")
# plt.ylabel("$\phi_{N/D}$ (kV)")
# plt.legend(["synth $\phi_D$", "synth $\phi_N$"])
# plt.xlim(0, 240)
# plt.xticks([0, 60, 120 , 180, 240], ['00', '01', '02', "03", "04"])
# plt.show()

# plt.plot(Nn, synth_Fpc/1e9)
# plt.xlabel("UT")
# plt.ylabel("$F_{PC}$ (GWb)")
# plt.legend(["synth $F_{PC}$"])
# plt.xlim(0, 240)
# plt.xticks([0, 60, 120 , 180, 240], ['00', '01', '02', "03", "04"])
# plt.show()

R_E = 6371e3

             

#################
##Bekreftelse####
#################

sE_lam = np.zeros([len(Nn), len(Phi_p),len(t_p)])
sE_tht = np.zeros([len(Nn),len(Phi_p),len(t_p)])
fart_t = np.zeros([len(Nn),len(Phi_p),len(t_p)])
fart_l = np.zeros([len(Nn),len(Phi_p),len(t_p)])



Beq = 31000e-9

# has_collided = 0
# counter = 0
# OCB_test = np.zeros([len(Nn), len(Phi_p)])




"""
Creating starting positions for the tracers
"""

# tr_nr = 7 #number of tracers
# sMLT = 10.0
# dmlt = 5 #distance between first- and last tracer

# tracers_spos = create_tracer(sMLT,dmlt, tr_nr, synth_Fpc[0])
# tracer_pos = []

# for j in range(len(tracers_spos)):
#     tracer_pos.append([tracers_spos[j][0], tracers_spos[j][1]])



# tracer_rt = tracers_spos.copy()

# B_tracer = np.zeros(len(tracer_pos))
# tracer_r = np.zeros([len(Nn),len(tracer_rt)])
# tracer_t = np.zeros([len(Nn),len(tracer_rt)])
# tracer_Vr = np.zeros([len(Nn),len(tracer_rt)])
# tracer_Vt = np.zeros([len(Nn),len(tracer_rt)])
# tracer_EL = np.zeros([len(Nn),len(tracer_rt)])
# tracer_ET = np.zeros([len(Nn),len(tracer_rt)])
# path_r = np.zeros([len(Nn),len(tracer_rt)])
# path_t = np.zeros([len(Nn),len(tracer_rt)])

# tracer_VTOT = np.zeros([len(Nn),len(tracer_rt)])

"""
Used to name header in DataFrames
"""
particles = ['particle_0',  'particle_1', 'particle_2', 'particle_3', 'particle_4', 'particle_5', 'particle_6']


# plt.figure()
# ax = plt.subplot(polar = "true")
# ax.plot(tracer_t, tracer_r, "-")
# ax.set_theta_offset(np.deg2rad(270))
# ax.set_rlim(np.deg2rad(0), np.deg2rad(30))
# ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
# ax.set_xticklabels(['00','06','12','18']) #riktig
# plt.title("tracer path")
# plt.show()  


          
    # plt.figure()
    # ax = plt.subplot(polar = "true")
    # ax.plot(tracer_t[i], tracer_r[i], "o")
    # #ax.set_theta_offset(np.deg2rad(270))
    # ax.set_rlim(np.deg2rad(0), np.deg2rad(30))
    # ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    # ax.set_xticklabels(['06','12','18','00'])
    # plt.title("tracer positions after %d timesteps" %(i))
    # #plt.savefig("tracer_pos_%d.jpg" %(i))
    # plt.show()  


#############
#Her plotter du master
#############

#RCRn = pd.read_csv("genrates_-0.5nT.csv")
#RCRp = pd.read_csv("genrates_0.5nT.csv")
#RCRn = pd.read_csv("genrates_-1nT.csv")
#RCRp = pd.read_csv("genrates_1nT.csv")
#RCRn = pd.read_csv("genrates_-2nT.csv")
#RCRp = pd.read_csv("genrates_-2nT.csv")

#RCRp = pd.read_csv("genrates_3nT.csv")

#RCRn = pd.read_csv("genrates_-4nT.csv")
#RCRp = pd.read_csv("genrates_4nT.csv")
#RCRn = pd.read_csv("genrates_-5nT.csv")
#RCRp = pd.read_csv("genrates_5nT.csv")

RCRn = pd.read_csv("Sgenrates_-5nT.csv")
RCRp = pd.read_csv("Sgenrates_5nT.csv")

dfp = pd.DataFrame(RCRp)
dfn = pd.DataFrame(RCRn) 
# dfsp = pd.DataFrame(sRCRp)
# dfsn = pd.DataFrame(sRCRn) 

dataframes = [dfn, dfp]


generated_phi = pd.concat(dataframes,  ignore_index=True)

synth_Fpc0 = np.zeros(len(generated_phi.day_0))
synth_dFpc0 = np.zeros(len(generated_phi.day_0))
synth_Fpc0[0] = 0.4e9

for i in range(1,len(generated_phi.day_0)):
    synth_dFpc0[i] = (generated_phi.day_0[i]*1e3-generated_phi.night_0[i]*1e3)*dt
    synth_Fpc0[i] = synth_Fpc0[i-1]+synth_dFpc0[i]  

day = [generated_phi.day_0,generated_phi.day_1, generated_phi.day_2, generated_phi.day_3, generated_phi.day_4, generated_phi.day_5, generated_phi.day_6, generated_phi.day_7, generated_phi.day_8, generated_phi.day_9]
night = [generated_phi.night_0, generated_phi.night_1, generated_phi.night_2, generated_phi.night_3, generated_phi.night_4, generated_phi.night_5, generated_phi.night_6, generated_phi.night_7, generated_phi.night_8, generated_phi.night_9]

mean_day = np.mean(day, axis = 0)
mean_night = np.mean(night, axis = 0)

mean_day = mean_day+50
mean_night = mean_night+50

mean_Fpc = np.zeros(len(mean_day))
mean_dFpc = np.zeros(len(mean_day))
mean_Fpc[0] = 0.4e9

UT = np.linspace(0, len(mean_day), len(mean_day))

for i in range(1,len(mean_day)):
    mean_dFpc[i] = (mean_day[i]*1e3-mean_night[i]*1e3)*dt
    mean_Fpc[i] = synth_Fpc0[i-1]+synth_dFpc0[i]  

plt.plot(UT, mean_day, "orange")
plt.plot(UT, mean_night, "b")
plt.xlabel("UT")
plt.ylabel("$\phi_{N/D}$ (kV)")
plt.legend(["$\phi_D$", "$\phi_N$"])
plt.xlim(0, 90)
plt.xticks([0, 30, 60, 90], ['00', '01', '02', "03"])
plt.show()

plt.plot(UT, mean_Fpc/1e9)
plt.xlabel("UT")
plt.ylabel("$F_{PC}$ (GWb)")
plt.legend(["$F_{PC}$"])
plt.xlim(0, 90)
plt.xticks([0, 30, 60, 90], ['00', '01', '02', "03"])
plt.show()



"""
Creating starting positions for the tracers
"""

tr_nr =  1000#number of tracers
sMLT = 07.0
dmlt = 10 #distance between first- and last tracer

tracers_spos = create_tracer(sMLT,dmlt, tr_nr, mean_Fpc[0])
tracer_pos = []

for j in range(len(tracers_spos)):
    tracer_pos.append([tracers_spos[j][0], tracers_spos[j][1]])
    
tracer_rt = tracers_spos.copy()

######################
#Vektorfelt of E felt#
######################

# sE_lam = np.zeros([len(mean_day), len(Phi_p),len(t_p)])
# sE_tht = np.zeros([len(mean_day),len(Phi_p),len(t_p)])
# B_tracer = np.zeros(len(tracer_pos))
# tracer_r = np.zeros([len(mean_day),len(tracer_rt)])
# tracer_t = np.zeros([len(mean_day),len(tracer_rt)])
# tracer_Vr = np.zeros([len(mean_day),len(tracer_rt)])
# tracer_Vt = np.zeros([len(mean_day),len(tracer_rt)])
# tracer_EL = np.zeros([len(mean_day),len(tracer_rt)])
# tracer_ET = np.zeros([len(mean_day),len(tracer_rt)])
# path_r = np.zeros([len(mean_day),len(tracer_rt)])
# path_t = np.zeros([len(mean_day),len(tracer_rt)])
# OCB_test = np.zeros([len(mean_day), len(Phi_p)])

sE_lam = np.zeros([len(Nn), len(Phi_p),len(t_p)])
sE_tht = np.zeros([len(Nn),len(Phi_p),len(t_p)])
B_tracer = np.zeros(len(tracer_pos))
tracer_r = np.zeros([len(Nn),len(tracer_rt)])
tracer_t = np.zeros([len(Nn),len(tracer_rt)])
tracer_Vr = np.zeros([len(Nn),len(tracer_rt)])
tracer_Vt = np.zeros([len(Nn),len(tracer_rt)])
tracer_EL = np.zeros([len(Nn),len(tracer_rt)])
tracer_ET = np.zeros([len(Nn),len(tracer_rt)])
path_r = np.zeros([len(Nn),len(tracer_rt)])
path_t = np.zeros([len(Nn),len(tracer_rt)])
OCB_test = np.zeros([len(Nn), len(Phi_p)])


##HUSK###
#unhashtag de over
#legg til 1e3 på synthFpc og ratene i E_calc
#bytt len Nn til len meanday

# plt.plot(Nn, synth_phi_D/1e3, "orange")
# plt.plot(Nn, synth_phi_N/1e3, "b")
# plt.xlabel("UT")
# plt.ylabel("$\phi_{N/D}$ (kV)")
# plt.legend(["synth $\phi_D$", "synth $\phi_N$"])
# plt.xlim(0, 240)
# plt.xticks([0, 60, 120 , 180, 240], ['00', '01', '02', "03", "04"])
# plt.show()

# plt.plot(Nn, synth_Fpc/1e9)
# plt.xlabel("UT")
# plt.ylabel("$F_{PC}$ (GWb)")
# plt.legend(["synth $F_{PC}$"])
# plt.xlim(0, 240)
# plt.xticks([0, 60, 120 , 180, 240], ['00', '01', '02', "03", "04"])
# plt.show()

# for tt in range(len(mean_day)):
#     print("step %d of %g" %(tt, len(mean_day)))
#     for pp in range(len(Phi_p)):
#         for qq in range(len(t_p)):
#             sL, sE_L, sE_T = calc_E(Phi_p[pp],t_p[qq],  mean_Fpc[tt], mean_day[tt]*1e3 , mean_night[tt]*1e3)
#             sE_lam[tt][qq][pp] = sE_L
#             sE_tht[tt][qq][pp] = sE_T


# #####################
# ##Plotting av E felt#
# #####################
#     colorss = sE_tht[tt]
#     plt.figure()
#     ax1 = plt.subplot(polar = "true")
#     plt.pcolormesh(T, np.rad2deg(R),colorss,cmap = "jet")
    
#     ax1.set_theta_offset(np.deg2rad(270))
#     ax1.set_rlim(0, 30)
#     ax1.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax1.set_xticklabels(['00','06','12','18'])
#     ax1.set_yticks(np.arange(10,31,10))
#     ax1.set_yticklabels(['80','70','60']) 
    
#     cbar1 = plt.colorbar()
#     cbar1.set_label('$E_\\theta (mV \: m^{-1})$', rotation=270)

#     plt.clim(-0.035,0.035)
#     cb = tt * 120
#     bc = "%d" %(tt)
#     cba = "%d" %(cb)
#     plt.title("Eletric field component in $\\theta$ direction t = " + cba + "s")
#     #plt.savefig("Offset05_Theta_E %d.jpg" %(tt))
#     #plt.show()
    
    
#     colorsss = sE_lam[tt]
#     plt.figure()
#     ax2 = plt.subplot(polar = "true")
#     plt.pcolormesh(T, np.rad2deg(R),colorsss,cmap = "jet")
    
#     ax2.set_theta_offset(np.deg2rad(270))
#     ax2.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax2.set_xticklabels(['00','06','12','18'])
#     ax2.set_yticks(np.arange(10,31,10))
#     ax2.set_yticklabels(['80','70','60']) 
    
#     cbar2 = plt.colorbar()
#     cbar2.set_label('$E_\lambda (mV \: m^{-1})$', rotation=270)
#     bc = "%d" %(tt)

#     plt.clim(-0.035,0.035)
#     cb = tt * 120
#     bc = "%d" %(tt)
#     cba = "%d" %(cb)
#     plt.title("Eletric field component in $\lambda$ direction t = " + cba + "s")
#     #plt.savefig("Offset05_Lambda_E %d synth.jpg" %(tt))
#     #plt.show()


             

############
##Lise CSV##
############

############
## day 0  ##
############

"""
STOP
"""
collision_times = []
crash_MLT = []

V_tot = np.zeros([len(Nn),len(tracer_rt)])
has_collided = 0
counter = 0
OCB_test = np.zeros([len(Nn), len(Phi_p)])

for p in range(len(tracer_pos)):
    has_collided = 0
    for i in range(len(Nn)):
    # print("Tracer positions")
    # print("step %d of %g" %(i, len(Nn)))           
            OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc[i], synth_phi_D[i] , synth_phi_N[i])
            
            Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
            VT_p = E_L/Bp#(B_tracer) #ExB2/B
            VL_p = -E_T/Bp#(B_tracer)
            
            tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

            tracer_EL[i][p] = E_L
            tracer_ET[i][p] = E_T
            
            tracer_Vr[i][p] = VL_p  
            tracer_Vt[i][p] = VT_p
            V_tot[i][p] = np.sqrt(VT_p**2+VL_p**2)
            
            tracer_r[i][p] = tracer_rt[p][0]
            tracer_t[i][p] = tracer_rt[p][1]

            OCB_test[i]= OCB
            
            
            if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
                counter = i

            if has_collided == 0:
                #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
                if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
                    tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)

                    crashvel = np.sqrt(VT_p**2+VL_p**2)
                    collision_time = i*120      
                    if tracercrash_MLT>=24:
                        tracercrash_MLT=np.round(tracercrash_MLT-24)
            
                    collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
                    collision_times.append(i*120)
                    crash_MLT.append(tracercrash_MLT)
                    print(collision_point)
                    has_collided = 1
    
    plt.figure()
    ax = plt.subplot(polar = "true")
    ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o", label='_nolegend_')
    ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
    #ax.legend(["#0", "#1", "#2", "#3", "#4", "#5", "#6"])
    ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

    ax.set_theta_offset(np.deg2rad(270))
    ax.set_rlim(0, 30)
    ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    ax.set_xticklabels(['00','06','12','18'])
    ax.set_yticks(np.arange(10,31,10))
    ax.set_yticklabels(['80','70','60']) 
    cb = i * 120
    cba = "%d" %(cb)
    plt.title("tracer positions t = " + cba + "s")
    #plt.savefig("Offset05_tracer_pos_%d.jpg" %(i))
    plt.show()  
    
meanV = np.mean(V_tot, axis = 0)    
    
# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# #dfCV = pd.DataFrame(crashvel) #Only when tracers reach the OCB
# #dfMLT = pd.DataFrame(tracercrash_MLT) #Only when tracers reach the OCB
# #dfTime = pd.DataFrame(collision_time) #Only when tracers reach the OCB
# dfMeanV = pd.DataFrame(meanV)
dfTIMES = pd.DataFrame(collision_times)

dfCM = pd.DataFrame(crash_MLT)

# dfr.to_csv('Offset05__tracerposition_lambda.csv', index = False, header = particles)
# dft.to_csv('Offset05__tracerposition_theta.csv', index = False, header = particles)
# dfVr.to_csv('Offset05__veloceties_lambda.csv', index = False, header = particles)
# dfVt.to_csv('Offset05__veloceties_theta.csv', index = False, header = particles)
# #dfCV.to_csv('-5_5_-05_05__collision_velocity.csv', index = False, header = particles) #Only when tracers reach the OCB
# #dfMLT.to_csv('-5_5_-05_05__collisionMLT.csv', index = False, header = particles) #Only when tracers reach the OCB
# #dfTime.to_csv('-5_5_-05_05__traveltime.csv', index = False, header = particles) #Only when tracers reach the OCB
# dfMeanV.to_csv('Offset05__Mean_vel.csv', index = False)
dfTIMES.to_csv('synth_7_17_col_time.csv', index = False)
dfCM.to_csv('synth_7_17_crash_MLT.csv', index = False)

########STOOOOOOPPPPPPP    
    
#day 1    
    
# synth_Fpc1 = np.zeros(len(dfD.day_1))
# synth_dFpc1 = np.zeros(len(dfN.day_1))
# synth_Fpc1[0] = 0.4e9

# for i in range(1,len(dfN.day_1)):
#     synth_dFpc1[i] = (dfD.day_1[i]*1e3-dfN.night_1[i]*1e3)*dt
#     synth_Fpc1[i] = synth_Fpc1[i-1]+synth_dFpc1[i]      
    
# for i in range(len(dfD.day_1)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc1[i], dfD.day_1[i]*1e3, dfN.night_1[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()      
    
# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day1.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day1.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day1.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day1.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day1.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day1.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day1.csv', index = False, header = particles)

    
#day 2
# synth_Fpc2 = np.zeros(len(dfD.day_2))
# synth_dFpc2 = np.zeros(len(dfN.day_2))
# synth_Fpc2[0] = 0.4e9

# for i in range(1,len(dfN.day_2)):
#     synth_dFpc2[i] = (dfD.day_2[i]*1e3-dfN.night_2[i]*1e3)*dt
#     synth_Fpc2[i] = synth_Fpc2[i-1]+synth_dFpc2[i]      
    
# for i in range(len(dfD.day_2)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc2[i], dfD.day_2[i]*1e3, dfN.night_2[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()    


# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day2.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day2.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day2.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day2.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day2.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day2.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day2.csv', index = False, header = particles)


#day 3
# synth_Fpc3 = np.zeros(len(dfD.day_3))
# synth_dFpc3 = np.zeros(len(dfN.day_3))
# synth_Fpc3[0] = 0.4e9

# for i in range(1,len(dfN.day_1)):
#     synth_dFpc3[i] = (dfD.day_3[i]*1e3-dfN.night_3[i]*1e3)*dt
#     synth_Fpc3[i] = synth_Fpc3[i-1]+synth_dFpc3[i]      
    
# for i in range(len(dfD.day_3)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc3[i], dfD.day_3[i]*1e3, dfN.night_3[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            

#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day3.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day3.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day3.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day3.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day3.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day3.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day3.csv', index = False, header = particles)


#day 4
# synth_Fpc4 = np.zeros(len(dfD.day_4))
# synth_dFpc4 = np.zeros(len(dfN.day_4))
# synth_Fpc4[0] = 0.4e9

# for i in range(1,len(dfN.day_1)):
#     synth_dFpc4[i] = (dfD.day_4[i]*1e3-dfN.night_4[i]*1e3)*dt
#     synth_Fpc4[i] = synth_Fpc4[i-1]+synth_dFpc4[i]      
    
# for i in range(len(dfD.day_4)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc4[i], dfD.day_4[i]*1e3, dfN.night_4[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            

#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day4.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day4.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day4.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day4.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day4.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day4.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day4.csv', index = False, header = particles)




#day 5
# synth_Fpc5 = np.zeros(len(dfD.day_5))
# synth_dFpc5 = np.zeros(len(dfN.day_5))
# synth_Fpc5[0] = 0.4e9

# for i in range(1,len(dfN.day_1)):
#     synth_dFpc5[i] = (dfD.day_5[i]*1e3-dfN.night_5[i]*1e3)*dt
#     synth_Fpc5[i] = synth_Fpc5[i-1]+synth_dFpc5[i]      
    
# for i in range(len(dfD.day_5)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc5[i], dfD.day_5[i]*1e3, dfN.night_5[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day5.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day5.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day5.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day5.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day5.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day5.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day5.csv', index = False, header = particles)
 


#day 6
# synth_Fpc6 = np.zeros(len(dfD.day_6))
# synth_dFpc6 = np.zeros(len(dfN.day_6))
# synth_Fpc6[0] = 0.4e9

# for i in range(1,len(dfN.day_6)):
#     synth_dFpc6[i] = (dfD.day_6[i]*1e3-dfN.night_6[i]*1e3)*dt
#     synth_Fpc6[i] = synth_Fpc6[i-1]+synth_dFpc6[i]      
    
# for i in range(len(dfD.day_6)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc6[i], dfD.day_6[i]*1e3, dfN.night_6[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()


# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day6.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day6.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day6.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day6.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day6.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day6.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day6.csv', index = False, header = particles)



#day 7
# synth_Fpc7 = np.zeros(len(dfD.day_7))
# synth_dFpc7 = np.zeros(len(dfN.day_7))
# synth_Fpc7[0] = 0.4e9

# for i in range(1,len(dfN.day_7)):
#     synth_dFpc7[i] = (dfD.day_7[i]*1e3-dfN.night_7[i]*1e3)*dt
#     synth_Fpc7[i] = synth_Fpc7[i-1]+synth_dFpc7[i]      
    
# for i in range(len(dfD.day_7)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc7[i], dfD.day_7[i]*1e3, dfN.night_7[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day7.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day7.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day7.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day7.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day7.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day7.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day7.csv', index = False, header = particles)


#day 8
# synth_Fpc8 = np.zeros(len(dfD.day_8))
# synth_dFpc8 = np.zeros(len(dfN.day_8))
# synth_Fpc8[0] = 0.4e9

# for i in range(1,len(dfN.day_8)):
#     synth_dFpc8[i] = (dfD.day_8[i]*1e3-dfN.night_8[i]*1e3)*dt
#     synth_Fpc8[i] = synth_Fpc8[i-1]+synth_dFpc8[i]      
    
# for i in range(len(dfD.day_8)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc8[i], dfD.day_8[i]*1e3, dfN.night_8[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()
    

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day8.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day8.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day8.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day8.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day8.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day8.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day8.csv', index = False, header = particles)



#day 9
# synth_Fpc9 = np.zeros(len(dfD.day_9))
# synth_dFpc9 = np.zeros(len(dfN.day_9))
# synth_Fpc9[0] = 0.4e9

# for i in range(1,len(dfN.day_9)):
#     synth_dFpc9[i] = (dfD.day_9[i]*1e3-dfN.night_9[i]*1e3)*dt
#     synth_Fpc9[i] = synth_Fpc9[i-1]+synth_dFpc9[i]      
    
# for i in range(len(dfD.day_9)):
#     # print("Tracer positions")
#     # print("step %d of %g" %(i, len(Nn)))
#     for p in range(len(tracer_pos)):             
#             OCB, E_L, E_T = calc_E(tracer_rt[p][0],tracer_rt[p][1], synth_Fpc9[i], dfD.day_9[i]*1e3, dfN.night_9[i]*1e3)
            
#             Bp = 2*Beq*np.cos(tracer_rt[p][0])
            
#             VT_p = E_L/Bp#(B_tracer) #ExB2/B
#             VL_p = -E_T/Bp#(B_tracer)
            
#             tracer_rt[p] = [tracer_rt[p][0]+VL_p*(dt/(R_E)),tracer_rt[p][1]+VT_p*(dt/((R_E*np.sin(tracer_rt[p][0]))))]
            

#             tracer_EL[i][p] = E_L
#             tracer_ET[i][p] = E_T
            
#             tracer_Vr[i][p] = VL_p  
#             tracer_Vt[i][p] = VT_p
            
#             tracer_r[i][p] = tracer_rt[p][0]
#             tracer_t[i][p] = tracer_rt[p][1]

#             OCB_test[i]= OCB
            
            
#             if ((tracer_r[i-1][p] - tracer_r[i][p] != 0) & counter == 0) or ((tracer_t[i-1][p] - tracer_t[i][p] !=0) & counter == 0):
#                 counter = i

#             if has_collided == 0:
#                 #print(i, " L1: ", tracer_r[i][p], " T: ",tracer_t[i][p])
#                 if (((tracer_r[i][p] >= OCB) & (tracer_t[i][p] < np.pi/2)) or ((tracer_r[i][p]>= OCB) & (3*np.pi/2 < tracer_t[i][p]))):
#                     tracercrash_MLT = (tracer_t[i][p])*24/(2*np.pi)
            
#                     crashvel = np.sqrt(VT_p**2+VL_p**2)
#                     collision_time = i*120  

#                     if tracercrash_MLT>=24:
#                         tracercrash_MLT=np.round(tracercrash_MLT-24)
            
#                     collision_point = {'Tracer nr:':p,'Collision_MLT':tracercrash_MLT, 'Collision_Colatitude':tracer_r[i][p], 'delta-T' : 120, 'Velocity[m/s]':np.sqrt(VT_p**2+VL_p**2),'time[s]': i*120, 'time since first tracer moved until first tracer arrived at the nightside OCB[s]': (i-counter)*120 }                    
#                     print(collision_point)
#                     has_collided = 1
#     path_r = np.zeros([len(Nn),len(tracer_rt)])
#     path_T = np.zeros([len(Nn),len(tracer_rt)])
    
#     plt.figure()
#     ax = plt.subplot(polar = "true")
#     ax.plot(tracer_t[i], np.rad2deg(tracer_r[i]), "o")
#     ax.plot(tracer_t[0:i, 0:p+1], np.rad2deg(tracer_r[0:i, 0:p+1]), "-")
#     ax.plot(t_p2, np.rad2deg(OCB_test[i]), "-")

#     ax.set_theta_offset(np.deg2rad(270))
#     ax.set_rlim(0, 30)
#     ax.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
#     ax.set_xticklabels(['00','06','12','18'])
#     ax.set_yticks(np.arange(10,31,10))
#     ax.set_yticklabels(['80','70','60']) 
#     plt.title("tracer positions after %d timesteps" %(i))
#     #plt.savefig("tracer_pos_%d.jpg" %(i))
#     plt.show()

# dft = pd.DataFrame(tracer_t)
# dfr = pd.DataFrame(tracer_r)
# dfVt = pd.DataFrame(tracer_Vt)
# dfVr = pd.DataFrame(tracer_Vr)
# dfCV = pd.DataFrame(crashvel)
# dfMLT = pd.DataFrame(tracercrash_MLT)
# dfTime = pd.DataFrame(collision_time)

# dfr.to_csv('tracerposition_lambda_day9.csv', index = False, header = particles)
# dft.to_csv('tracerposition_theta_day9.csv', index = False, header = particles)
# dfVr.to_csv('veloceties_lambda_day9.csv', index = False, header = particles)
# dfVt.to_csv('veloceties_theta_day9.csv', index = False, header = particles)
# dfCV.to_csv('collision_velocity_day9.csv', index = False, header = particles)
# dfMLT.to_csv('collisionMLT_day9.csv', index = False, header = particles)
# dfTime.to_csv('traveltime_day9.csv', index = False, header = particles)
 

stop = time.time()
print(stop - start, "s")

"""
Sanity check for Milan plot

colors = E_lam
plt.figure()
ax = plt.subplot(polar = "true")
plt.pcolormesh(theta, np.rad2deg(r),colors ,cmap = "jet")
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

######################
#Vektorfelt of E felt#
######################

# for tt in range(len(Nn)):
#     print("step %d of %g" %(tt, len(Nn)))
#     for pp in range(len(Phi_p)):
#         for qq in range(len(t_p)):
#             sL, sE_L, sE_T = calc_E(Phi_p[pp],t_p[qq], synth_Fpc[tt], synth_phi_D[tt], synth_phi_N[tt])
#             sE_lam[tt][qq][pp] = sE_L
#             sE_tht[tt][qq][pp] = sE_T ###
#            Vt, Vl =  calc_V(Phi_p[pp],t_p[qq], synth_Fpc[tt], synth_phi_D[tt], synth_phi_N[tt])
            # Bp = 2*Beq*np.cos(Phi_p[pp])

            # Vt = sE_L/Bp
            # Vl = -sE_T/Bp
            # fart_t[tt][qq][pp] = Vt
            # fart_l[tt][pp][qq] = Vl
    
            
            
#     dr = fart_l[tt]
#     dt = fart_t[tt]
    
#     f = plt.figure()
#     ax3 = f.add_subplot(111, polar=True)
#     ax3.quiver(theta, np.rad2deg(r), dr*np.cos(theta)-dt*np.sin(theta), dr*np.sin(theta)+dt * np.cos(theta))
#     #ax3.set_rlim(np.deg2rad(60), np.deg2rad(90))
#     ax3.set_theta_offset(np.deg2rad(90))
#     cb = tt * 120
#     cba = "%d" %(cb)
#     plt.title("Vector field lines for the velocity[m/s] whent = " + cba + "s")
#     #plt.savefig("synth_vel_%d.jpg" %(tt))
#     plt.show()

####################
#Plotting av E felt#
####################
    # colorss = sE_tht[tt]
    # plt.figure()
    # ax1 = plt.subplot(polar = "true")
    # plt.pcolormesh(T, np.rad2deg(R),colorss,cmap = "jet")
    
    # ax1.set_theta_offset(np.deg2rad(270))
    # ax1.set_rlim(0, 30)
    # ax1.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    # #ax1.set_xticklabels(['0','$\pi$ / 2','$\pi$','$\pi$ 3/2'])
    # ax1.set_xticklabels(['00','06','12','18'])
    # ax1.set_yticks(np.arange(10,31,10))
    # ax1.set_yticklabels(['80','70','60']) 
    
    # cbar1 = plt.colorbar()
    # cbar1.set_label('$E_\\theta (mV \: m^{-1})$', rotation=270)

    # plt.clim(-0.035,0.035)
    # cb = tt * 60
    # bc = "%d" %(tt)
    # cba = "%d" %(cb)
    # plt.title("Eletric field component in $\\theta$ direction t = " + cba + "s")
    # #plt.savefig("%d synth_ET.jpg" %(tt))
    # plt.show()
    
    
    # colorsss = sE_lam[tt]
    # plt.figure()
    # ax2 = plt.subplot(polar = "true")
    # plt.pcolormesh(T, np.rad2deg(R),colorsss,cmap = "jet")
    
    # ax2.set_theta_offset(np.deg2rad(270))
    # ax2.set_xticks((0,np.pi/2,np.pi,3*np.pi/2))
    # ax2.set_xticklabels(['00','06','12','18'])
    # ax2.set_yticks(np.arange(10,31,10))
    # ax2.set_yticklabels(['80','70','60']) 
    
    # cbar2 = plt.colorbar()
    # cbar2.set_label('$E_\lambda (mV \: m^{-1})$', rotation=270)
    # bc = "%d" %(tt)

    # plt.clim(-0.05,0.05)
    # plt.title("Eletric field component in $\lambda$ direction t = " + bc + "min")
    # #plt.savefig("Lambda %d synth.jpg" %(tt))
    
    # plt.show()