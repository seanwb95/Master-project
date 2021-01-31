import numpy as np
import matplotlib.pyplot as plt



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

    if np.tan(theta_p/2)==0:
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
b = 90

ph_p = np.linspace(a, b, 31) #latitude
t_p = np.linspace(0, 2*np.pi, 31) #longitude
E_lam = np.zeros([len(ph_p),len(t_p)])
E_tht = np.zeros([len(ph_p),len(t_p)])



for p in range(len(ph_p)):
    print("step %d of 30" %(p))
    for t in range(len(t_p)):
        L, v, Vp, Sm, E_L, E_T = calc_E(ph_p[p], t_p[t], 0.7e9, 30e3, 70e3)
        E_lam[p][t] = E_L
        E_tht[p][t] = E_T
        
MLT = np.linspace(0, 24, len(Vp))

r = 90-ph_p
theta = np.rad2deg(t_p)
colors = E_lam

q,w = np.meshgrid(r,theta)


fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(q, w, c=colors, cmap='hsv', alpha=0.75)




"""
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(MLT, Vp)
ax.set_xlabel('MLT')
ax.set_ylabel('$\phi_{R1}$ [V]')
ax.set_title('Example 1')
plt.show()

#example2
#L, LA1, LA2, v, Vp, Sm=calc_E(77, 77, 0.7e9, 30e3, 70e3)
#MLT = np.linspace(0, 24, len(Vp))
#ax.plot(MLT, Vp)
#ax.set_xlabel('MLT')
#ax.set_ylabel('$\phi_{R1}$ [V]')
#ax.set_title('Example 2')
#plt.show()
"""
