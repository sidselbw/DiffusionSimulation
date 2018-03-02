from GR20 import GR20
from GR20LinkRoute import GR20LinkRoute as GLR
import numpy as np
import time, os

print '\n\t Running the script...\n'
t1, for_directory = time.time(), time.strftime('%d-%m-%Y_%H:%M:%S', time.localtime())

####################################################################


# what = 'Diffusion in single cylinder'
# what = 'Diffusion in single sphere'
# what = 'Free diffusion'
# what = 'Diffusion in cylinders from AxonPacking'
# what = 'Diffusion in permeable cylinders from AxonPacking'
# what = 'Diffusion in permeable cylinders from AxonPacking with reused paths'
# what = 'Check boundary conditions'
# what = 'Investigate magnetic field' ####LATEST
# what = 'Expected from pilot scan'
# what = 'Effect of susceptibility for a single cylinder'
# what = 'Fix that stupid asymmetry' #FIXED!!!!
what = 'Numerical estimation of B-field'

filename = '../../../../GR20/'+for_directory+'_'+what+'/'#+'/TestingGR20'
if 'reused' not in what:
    os.makedirs(filename) #Make directory for randomwalk. Save outside the Dropbox.
# os.makedirs(filename[17:]) #Make directory for figures

if what == 'Diffusion in single cylinder':
    #### Initialize object
    randomwalk = GR20(100000,0.5,2000,'Single cylinder',filename+'single_cylinder_diffusion',single_radius=10e-6)
    randomwalk.single_cylinder_diffusion()

    delta,Delta = 0.08,0.1
    signalstuff = randomwalk.RepeatPGSE_single_direction(delta,Delta,np.arange(0.,1.4e-2,0.1e-2))
    randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    # randomwalk.PlotPaths(10,200,twoD=False,threeD=True)

    # rmsdstuff = randomwalk.ComputeRMSD()
    # randomwalk.PlotRMSD(rmsdstuff)

if what == 'Diffusion in single sphere':
    #### Initialize object
    randomwalk = GR20(10000,0.5,2000,'Single sphere',filename+'single_sphere_diffusion',single_radius=50e-6)
    randomwalk.single_sphere_diffusion()

    delta,Delta = 0.08,0.1
    signalstuff = randomwalk.RepeatPGSE_single_direction(delta,Delta,np.arange(0.,1.4e-2,0.1e-2))
    randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    randomwalk.PlotPaths(10,200,twoD=False,threeD=True)

    rmsdstuff = randomwalk.ComputeRMSD(n_values=100)
    randomwalk.PlotRMSD(rmsdstuff)

if what == 'Free diffusion':
    #### Initialize object
    randomwalk = GR20(10000,0.2,1000,'Free diffusion',filename+'free_diffusion',diff_lim=25*10**-5,signal_lim=20*10**-5)
    randomwalk.free_diffusion()

    delta,Delta = 0.08,0.1
    signalstuff = randomwalk.RepeatPGSE_single_direction(delta,Delta,np.arange(0.,1.4e-2,0.1e-2))
    randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    randomwalk.PlotPaths(10,200,twoD=False,threeD=True)

    rmsdstuff = randomwalk.ComputeRMSD()
    randomwalk.PlotRMSD(rmsdstuff)

if what == 'Diffusion in cylinders from AxonPacking':
    axons,packing = '../../AxonPacking/code/results/axons.mat','../../AxonPacking/code/results/packing.mat'

    randomwalk = GR20(100000,0.2,3000,'Multi cylinder',filename+'multi_cylinder_diffusion') #Initialize object
    diff_lim = 25*10**-6 #[m]
    randomwalk.TrimCylinders(axons,packing,diff_lim)
    # randomwalk.PlotCylinders(threeD=False)

    randomwalk.multi_cylinder_diffusion()
    # randomwalk.PlotPaths(10,200,twoD=True,threeD=False)

    # delta,Delta = 0.08,0.1
    # signalstuff = randomwalk.RepeatPGSE_single_direction(delta,Delta,np.arange(0.,1.4e-2,0.1e-2))
    # randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    rmsdstuff = randomwalk.ComputeRMSD()
    randomwalk.PlotRMSD(rmsdstuff)

if what == 'Diffusion in permeable cylinders from AxonPacking':
    # axons,packing,stats = '../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/axons.mat','../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/packing.mat','../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/stats.mat'
    axons,packing,stats = '../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/axons.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/packing.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/stats.mat'

    # filename = '../../../../GR20/'+'24-01-2018_09:43:37_Diffusion in permeable cylinders from AxonPacking/'
    #1000 walkers
    # filename = '../../../../GR20/'+'24-01-2018_10:11:12_Diffusion in permeable cylinders from AxonPacking/'
    filename = '../../../../GR20/'+'29-01-2018_14:25:15_Diffusion in permeable cylinders from AxonPacking/'
    #100000 walkers
    # filename = '../../../../GR20/'+'24-01-2018_13:32:53_Diffusion in permeable cylinders from AxonPacking/'
    # filename = '../../../../GR20/'+'29-01-2018_13:02:44_Diffusion in permeable cylinders from AxonPacking/'

    #### REUSE
    randomwalk = GR20(1000,0.2,1000,'Multi cylinder',filename+'multi_cylinder_diffusion',reuse=filename,diff_lim = 37*10**-6,signal_lim=37*10**-6,permeability=0.008) #Initialize object
    #### NEW
    # randomwalk = GR20(1000,0.2,1000,'Multi cylinder',filename+'multi_cylinder_diffusion',diff_lim = 37*10**-6,signal_lim=37*10**-6,permeability=0.008) #Initialize object

    delta,Delta = 0.08,0.1
    # randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1./2.,1./2.,1./2.],[1./2.,1./2.,-1./2.],[1./2.,-1./2.,1./2.]])
    randomwalk.directions = np.array([[1.,0.,0.],[0.,0.,1.],[1.,1.,1.]])
    norm_constant = np.linalg.norm(randomwalk.directions,axis=1).reshape(3,1)
    randomwalk.directions = randomwalk.directions/norm_constant

    #### Check boundary effects
    # diff_lim = 37; signal_lims = np.arange(29,diff_lim,2)
    # gs = np.arange(0.,1.4e-2,0.4e-2)
    # randomwalk.MapLimEffectVsb([diff_lim],signal_lims,gs,delta,Delta)

    # raw_input('...')

    randomwalk.TrimCylinders(axons,packing,stats)
    # randomwalk.PlotCylinders(threeD=False)
    # raw_input('...')

    # randomwalk.permeable_multi_cylinder_diffusion()

    # randomwalk.PlotPaths(10,500,twoD=True,threeD=False)
    # raw_input('...')

    # print np.loadtxt('../../../../Desktop/K252_x32322_no_rotation.bvec').T

    # randomwalk.GetDirectionsFrombvecFile('../../../../Desktop/K252_x32322_no_rotation.bvec')
    # randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    # print np.loadtxt('../../../../Desktop/DesktopNotes')
    signalstuff = randomwalk.RepeatPGSE(delta,Delta,np.arange(0.,1.4e-2,0.2e-2))
    randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    randomwalk.PlotPowderAverageVsb(signalstuff,delta,Delta)

    randomwalk.PlotSignalVsDirection(signalstuff,delta,Delta)

    rmsdstuff = randomwalk.ComputeRMSD()
    randomwalk.PlotRMSD(rmsdstuff)

if what == 'Investigate magnetic field':

    #### REUSE
    # 1000
    filename = '../../../../GR20/'+'12-02-2018_12:04:21_Investigate magnetic field/'
    randomwalk = GR20(1000,0.2,1000,'Multi cylinder',filename+'multi_cylinder_diffusion_with_susceptibility',reuse=filename,diff_lim = 37*10**-6,signal_lim=37*10**-6,permeability=0.008) #Initialize object
    #### NEW
    # permeability = 0.008#0.008
    # randomwalk = GR20(1000,0.05,1000,'Multi cylinder',filename+'multi_cylinder_diffusion_with_susceptibility',permeability = permeability, diff_lim = 37*10**-6,signal_lim=37*10**-6) #Initialize object

    # delta,Delta = 0.005, 0.02
    # print 'b: ', randomwalk.gamma**2 * np.arange(0.,0.41,0.06)**2 * delta**2 * (Delta - delta/3.)
    # raw_input('...')

    # randomwalk.PlotMagneticField(coordinates='polar') #Not for use
    # randomwalk.PlotMagneticField(coordinates='cartesian') #Not for use

    # randomwalk.PlotMagneticField(coordinates='CylindricalWartonBowtell')
    # randomwalk.PlotMagneticField(coordinates='CartesianWartonBowtell')

    # axons,packing,stats = '../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/axons.mat','../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/packing.mat','../../AxonPacking/code/results/Axons100_Mean3_Var1_Delta0_Iter10000/stats.mat'
    # axons,packing,stats = '../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/axons.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/packing.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.2_Iter10000/stats.mat'
    # axons,packing,stats = '../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.3_Iter10000/axons.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.3_Iter10000/packing.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.3_Iter10000/stats.mat'
    axons,packing,stats = '../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.4_Iter10000/axons.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.4_Iter10000/packing.mat','../../AxonPacking/code/results/Axons200_Mean3_Var1_Delta0.4_Iter10000/stats.mat'
    randomwalk.TrimCylinders(axons,packing,stats)

    randomwalk.permeable_multi_cylinder_diffusion()
    randomwalk.PlotPaths(10,500,twoD=True,threeD=False)
    raw_input('...')

    randomwalk.chi_A = 0.00000004#0.00000008
    randomwalk.PlotMultiCylinderMagneticField()
    # raw_input('...')
    randomwalk.ComputeMultiCylinderMagneticField()

    # delta,Delta = 0.08,0.1#,[0.4e-2]
    delta,Delta = 0.005, 0.02
    randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1./2.,1./2.,1./2.],[1./2.,1./2.,-1./2.]])
    norm_constant = np.linalg.norm(randomwalk.directions,axis=1).reshape(5,1)
    randomwalk.directions = randomwalk.directions/norm_constant

    signalstuff = randomwalk.RepeatPGSE(delta,Delta,np.arange(0.,0.41,0.06))

    randomwalk.PlotSignalVsb(signalstuff,delta,Delta)

    randomwalk.PlotPowderAverageVsb(signalstuff,delta,Delta)

    randomwalk.PlotSignalVsDirection(signalstuff,delta,Delta)

    rmsdstuff = randomwalk.ComputeRMSD()
    randomwalk.PlotRMSD(rmsdstuff)

if what == 'Effect of susceptibility for a single cylinder':
    randomwalk = GR20(10000,0.25,2000,'Single cylinder',filename+'single_cylinder_susceptibility',diff_lim=25*10**-6,signal_lim=20*10**-6,single_radius=15e-6)
    randomwalk.single_cylinder_diffusion_with_susceptibility()
    g, delta, Delta = [1.0e-2], 0.08, 0.1
    randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(12,6),facecolor = 'w',edgecolor='w')

    alpha = 0./1.*np.pi
    g, delta, Delta = [1.0e-2], 0.08, 0.1
    g,signal_intra,signal_extra,signal_intra_std,signal_extra_std = randomwalk.RepeatPGSE_susceptibility(delta,Delta,g,alpha)
    plt.plot([1,2,3],signal_intra[0,:],c='blue',marker='o',ms=10,mew=2.5,markerfacecolor='None',alpha=0.95,linestyle='--',label=r'$S_{intra}$, $\theta=0$')
    plt.plot([1,2,3],signal_extra[0,:],c='grey',marker='o',ms=10,mew=2.5,markerfacecolor='None',alpha=0.95,linestyle='--',label=r'$S_{extra}$, $\theta=0$')

    alpha = 1./2.*np.pi
    g, delta, Delta = [1.0e-2], 0.08, 0.1
    g,signal_intra,signal_extra,signal_intra_std,signal_extra_std = randomwalk.RepeatPGSE_susceptibility(delta,Delta,g,alpha)
    plt.plot([1,2,3],signal_intra[0,:],c='red', marker='.',ms=16,alpha=0.95,linestyle='--',label=r'$S_{intra}$, $\theta=\frac{\pi}{2}$')
    plt.plot([1,2,3],signal_extra[0,:],c='green',marker='.',ms=16,alpha=0.95,linestyle='--',label=r'$S_{extra}$, $\theta=\frac{\pi}{2}$')

    plt.xticks([1,2,3], ('x','y','z') )
    plt.xlabel('Direction of gradient')
    plt.ylabel('Signal []')
    plt.legend(numpoints=1)
    plt.savefig(randomwalk.filename+'signals.png',bbox_inches='tight',dpi=100)
    plt.close('all')

    randomwalk.alpha = 0./1.*np.pi
    randomwalk.PlotPaths(10,200,twoD=True,threeD=False,PlotSusField=True)
    randomwalk.alpha = 1./2.*np.pi
    randomwalk.PlotPaths(10,200,twoD=True,threeD=False,PlotSusField=True)
    # randomwalk.PlotPaths(10,200,twoD=False,threeD=True)
    # randomwalk.PlotPaths(10,200,twoD=True,threeD=False,PlotSusField=True)

if what == 'Expected from pilot scan':
    N, duration, timesteps = 1000, 0.2, 1000 #number of walkers
    g, delta, Delta = [1.0e-2], 0.08, 0.1
    diff_lim = 25*10**-6 #[m]

    #Corpus callosum
    axons,packing = '../../AxonPacking/code/results/axons.mat','../../AxonPacking/code/results/packing.mat'
    CC = GR20(N, duration, timesteps,'Multi cylinder',filename+'multi_cylinder_diffusion',diff_lim = diff_lim,permeability=0.008) #Initialize object
    CC.TrimCylinders(axons,packing)
    CC.permeable_multi_cylinder_diffusion()
    CC.PlotPaths(10,200,twoD=True,threeD=False)
    CC.GetDirectionsFrombvecFile('../data_pilot/K252_x32322_no_rotation.bvec',rotate='x',angle=np.pi/2.)
    signalstuff_CC = CC.RepeatPGSE(delta,Delta,g)
    CC.PlotDirections(lengths=np.array([1.]*len(CC.directions)))
    CC.PlotDirections(lengths=signalstuff_CC[1][0],add='_scaled_g'+str(signalstuff_CC[0][0][0]))
    #Ventricles
    V = GR20(N, duration, timesteps,'Free diffusion',filename+'free_diffusion',diff_lim = diff_lim) #Initialize object
    V.free_diffusion()
    V.PlotPaths(10,200,twoD=True,threeD=False)
    V.GetDirectionsFrombvecFile('../data_pilot/K252_x32322_no_rotation.bvec',rotate='x',angle=np.pi/2.)
    signalstuff_V = V.RepeatPGSE(delta,Delta,g)
    V.PlotDirections(lengths=signalstuff_V[1][0],add='_scaled_g'+str(signalstuff_V[0][0][0]))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6),facecolor = 'w',edgecolor='w')
    plt.grid(True)
    plt.plot(signalstuff_CC[1][0,:],marker='.',linestyle='--',label='CC')
    plt.plot(signalstuff_V[1][0,:],marker='.',linestyle='--',label='V')
    plt.legend(numpoints=1)
    plt.xlabel('Directions []')
    plt.ylabel('Signal []')
    plt.show()





if what == 'Check boundary conditions':
    randomwalk = GR20(10000,0.5,1000,'Free diffusion',filename+'free_diffusion',diff_lim=25*10**-6,signal_lim=20*10**-6)
    # randomwalk.free_diffusion()

    # randomwalk.PlotPaths(10,200,twoD=False,threeD=True)
    #diff_lims = np.arange(5*10**-6,50*10**-6,5*10**-6)*10**6
    randomwalk.GetDirectionsFrombvecFile('../data_pilot/K252_x32322_no_rotation.bvec',rotate='x',angle=np.pi/2.)
    #Normalize those b-vectors!
    # norm_constant = np.linalg.norm(randomwalk.directions,axis=1).reshape(16,1)
    # randomwalk.directions = randomwalk.directions/norm_constant

    randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1./2.,1./2.,1./2.],[1./2.,1./2.,-1./2.],[1./2.,-1./2.,1./2.]])
    norm_constant = np.linalg.norm(randomwalk.directions,axis=1).reshape(6,1)
    randomwalk.directions = randomwalk.directions/norm_constant

    # min_diff_lim = np.linalg.norm([randomwalk.signal_lim,]*3)
    min_diff_lim = randomwalk.signal_lim
    # diff_lims = np.array([min_diff_lim*10**-6,25*10**-6,50*10**-6])*10**6
    diff_lims = np.linspace(min_diff_lim,min_diff_lim*2,1)
    delta,Delta,g = 0.08,0.1,[1.0e-2]
    randomwalk.ExperimentOnSolidBoundaryEffect(diff_lims,delta,Delta,g)

if what == 'Fix that stupid asymmetry':
    randomwalk = GR20(10000,0.5,2000,'Free diffusion',filename+'free_diffusion',diff_lim=25*10**-6,signal_lim=20*10**-6)
    randomwalk.free_diffusion_old()
    # randomwalk.filename = '../../../../GR20/23-01-2018 13:02:29_Fix that stupid asymmetry/free_diffusion'
    randomwalk.solid_boundary = 'cube'

    # raw_input('...')

    # alpha,beta,gamma = 0.,0.,0.
    timesteps = [500,1000,1500]
    rotations = [[0.,0.,0.,],[2.*np.pi,0.,0.],[0.25*np.pi,0.,0.],[0.33*np.pi,0.,0.],[0.17*np.pi,0.,0.]]
    randomwalk.HistogramsOfRotatedPaths(timesteps,rotations,bins=40,initialize=True)
    raw_input('...')

    filename = randomwalk.filename

    # rotations2 = [rotations[0],rotations[2]]
    # randomwalk.PathHistogram2d(filename,[200,1000,1900],rotations2)
    # raw_input('..')

    randomwalk.directions = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1./2.,1./2.,1./2.],[1./2.,1./2.,-1./2.],[1./2.,-1./2.,1./2.]])
    # randomwalk.directions = rotations
    norm_constant = np.linalg.norm(randomwalk.directions,axis=1).reshape(6,1)
    randomwalk.directions = randomwalk.directions/norm_constant

    # min_diff_lim = np.linalg.norm([randomwalk.signal_lim,]*3)
    # min_diff_lim = randomwalk.signal_lim
    # diff_lims = np.array([min_diff_lim*10**-6,25*10**-6,50*10**-6])*10**6
    # diff_lims = np.linspace(min_diff_lim,min_diff_lim*2,1)
    delta,Delta,g = 0.08,0.1,np.linspace(0.1e-2,1.2e-2,5)
    # randomwalk.ExperimentOnSolidBoundaryEffect(diff_lims,delta,Delta,g)
    signalstuff = randomwalk.RepeatPGSE(delta,Delta,g)
    print np.shape(signalstuff)
    randomwalk.PlotSignalVsb(signalstuff[0],delta,Delta)

    raw_input('...')

    # rmsdstuff = randomwalk.ComputeRMSD()
    # randomwalk.PlotRMSD_several(rmsdstuff,add='_standard',initialize=True,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(rotations[0])
    # randomwalk.PlotRMSD_several(rmsdstuff,add='_0rotation',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(rotations[1])
    # randomwalk.PlotRMSD_several(rmsdstuff,add='_2pirotation',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(rotations[2])
    # randomwalk.PlotRMSD_several(rmsdstuff,'_0dot25pirotation',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(rotations[3])
    # randomwalk.PlotRMSD_several(rmsdstuff,'_0dot33pirotation x',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(rotations[4])
    # randomwalk.PlotRMSD_several(rmsdstuff,'_0dot17pirotation x',initialize=False,savenclose=True)





    # rmsdstuff = randomwalk.ComputeRMSD_rotate(np.pi/5.,0.,0)
    # randomwalk.PlotRMSD_several(rmsdstuff,'pidiv5rotation x',initialize=False,savenclose=False)

    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,np.pi/3.,0)
    # randomwalk.PlotRMSD_several(rmsdstuff,'_pidiv3rotation y',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,np.pi/4.,0)
    # randomwalk.PlotRMSD_several(rmsdstuff,'_pidiv4rotation y',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,np.pi/5.,0)
    # randomwalk.PlotRMSD_several(rmsdstuff,'pidiv5rotation y',initialize=False,savenclose=False)
    #
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,0.,np.pi/3.)
    # randomwalk.PlotRMSD_several(rmsdstuff,'_-pidiv3rotation z',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,0.,np.pi/4.)
    # randomwalk.PlotRMSD_several(rmsdstuff,'_-pidiv4rotation z',initialize=False,savenclose=False)
    # rmsdstuff = randomwalk.ComputeRMSD_rotate(0.,0.,np.pi/5.)
    # randomwalk.PlotRMSD_several(rmsdstuff,'_-pidiv5rotation z',initialize=False,savenclose=True)
    # import matplotlib.pyplot as plt
    # plt.show()

if what == 'Numerical estimation of B-field':
    x = GR20(1,1,1,'None',filename+'numerical_B')

    # x.numerical_estimation_of_B()
    x.numerical_estimation_of_B2()


####################################################################

print '\n\t Script execution time was %.2f s...\n' %(time.time()-t1)
