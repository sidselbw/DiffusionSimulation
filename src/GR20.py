#Classes are awesome

import numpy as np
import random
import time
import h5py
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import scipy


class GR20(object):

    def __init__(self, N, duration, tmax, restriction_type, filename, D=2e-09, g=0.22, color='blue', solid_boundary='cube', reuse=False,diff_lim=None, signal_lim=None, single_radius=None,permeability=0.):
        self.restriction_type   = restriction_type
        self.filename           = filename
        self.reuse              = reuse
        #### Define constants
        self.gamma              = 267.513*10**6 # [rad/(sT)] # gyromagnetic ratio for Hydrogen # 42.58 * 10**6 [Hz/T]
        self.D                  = D #[m^2/s] #1m^2 = 1e12mum^2 #From Hall et al.
        self.permeability       = permeability
        self.g                  = g
        #### Initialize variable parameters
        self.N                  = N
        self.duration           = duration
        self.tmax               = tmax
        self.dt                 = duration/tmax
        if diff_lim != None and signal_lim != None:
            self.diff_lim       = diff_lim*10**6 #[\mu m]
            self.signal_lim     = signal_lim*10**6 #[\mu m]
        else:
            self.diff_lim = diff_lim
            self.signal_lim = signal_lim
        # if reuse != False:
        #     f = h5py.File(filename + '.h5','r')
        #     info = np.array(f.get('info'))
        #     self.diff_lim = info[7]
        #     self.signal_lim = info[8]

        self.solid_boundary = solid_boundary
        #### Something for plotting
        self.color              = color
        #### Other stuff
        self.step_size          = np.sqrt(6.*self.D*self.dt)
        self.single_radius      = single_radius
        #### Print stuff
        print '\t Specs of initialized object:'
        print '\t    - restriction_type: %s' %self.restriction_type
        print '\t    - step_size: %.2e' %self.step_size
        print '\t    - N: %d' %self.N

    ####Move
    def move(self):
        ########################################################################
        # Contains various methods for generating steps.
        # They are commented out to save if statements.
        ########################################################################

        #### Generate x,y,z from uniform distribution (fastest)
        coord_updates = np.random.uniform(-1.,1.,[self.N,3])
        coord_updates_lengths = np.tile(np.linalg.norm(coord_updates,axis=1),(3,1)) #get the length of each coordinate and repeat it 3 times
        self.path += coord_updates/coord_updates_lengths.T * self.step_size #Normalize step length to equal the step_size

        #### Generate x,y,z from Gaussian distribution
        # coord_updates = np.random.uniform(-1.,1.,[self.N,3])
        # coord_updates_lengths = np.tile(np.linalg.norm(coord_updates,axis=1),(3,1)) #get the length of each coordinate and repeat it 3 times
        # self.path += coord_updates/coord_updates_lengths.T * self.step_size #Normalize step length to equal the step_size

        #### Generate spherical coordinates and convert to cartesian (safest)
        #### Uniformly distributed on SPHERE
        # phi = np.random.uniform(0,np.pi*2,[self.N,1])
        # costheta = np.random.uniform(-1,1,[self.N,1])
        # theta = np.arccos(costheta)
        # x = np.sin(theta) * np.cos(phi)
        # y = np.sin(theta) * np.sin(phi)
        # z = np.cos(theta)
        # self.path += self.step_size*np.hstack((x,y,z))

    def initialize_path_file(self):
        f = h5py.File(self.filename + '.h5','w')
        f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size,self.solid_boundary])
        f.create_dataset('tstep0',data=self.path)
        return f

	####Unrestricted random walk
    def free_diffusion(self):
        time_start = time.time()

        print '\t Initializing coordinates...'

        if self.diff_lim == None and self.signal_lim == None:
            self.path = np.zeros([self.N,3])
        else:
            if self.solid_boundary == 'sphere':
                ####Uniformly distributed in SPHERE
                phi = np.random.uniform(0,2*np.pi,[self.N,1])
                costheta = np.random.uniform(-1,1,[self.N,1])
                u = np.random.uniform(0,1,[self.N,1])

                theta = np.arccos(costheta)
                r = self.diff_lim*10**-6 * np.cbrt(u)

                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                self.path = np.hstack((x,y,z))
            elif self.solid_boundary == 'cube':
                ####Uniformly distributed in CUBE
                self.path = np.random.uniform(-self.diff_lim*10**-6,self.diff_lim*10**-6,[self.N,3])

        print '\t Doing random walk...'

        #### Open the file that will be written to
        f = self.initialize_path_file()

        Trues = np.asarray([True]*self.N)

        if self.diff_lim == None and self.signal_lim == None:
            for i in range(1,self.tmax):
                self.move()
                f.create_dataset('tstep'+str(i),data=self.path)
        else:
            if self.solid_boundary == 'sphere':
                for i in range(1,self.tmax):
                    if i%100 == 0:
                        print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

                    self.move()

                    ####For SPHERICAL diff_lim
                    outside_diff_lim = np.linalg.norm(self.path,axis=1) > self.diff_lim*10**-6
                    self.path[outside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[outside_diff_lim == Trues] #if boundary is crossed, move back to previous position

                    f.create_dataset('tstep'+str(i),data=self.path)

            elif self.solid_boundary == 'cube':
                for i in range(1,self.tmax):
                    if i%100 == 0:
                        print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

                    self.move()

                    ####For CUBIC diff_lim
                    #check if walkers crossed the diff_lim in the x-direction
                    inside_diff_lim = np.abs(self.path[:,0]) > self.diff_lim*10**-6
                    self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position
                    #check if walkers crossed the diff_lim in the y-direction
                    inside_diff_lim = np.abs(self.path[:,1]) > self.diff_lim*10**-6
                    self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position

                    f.create_dataset('tstep'+str(i),data=self.path)
        f.close()

    #### Without diffusion boundary
    # def free_diffusion_old(self):
    #     print '\t Initializing coordinates...'
    #
    #     self.path = np.zeros([self.N,3])
    #
    #     print '\t Doing random walk...'
    #
    #     time_start = time.time()
    #
    #     #### Open the file that will be written to
    #     f = h5py.File(self.filename + '.h5','w')
    #     f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size])
    #     f.create_dataset('tstep0',data=self.path)
    #
    #     for i in range(1,self.tmax):
    #         if i%100 == 0:
    #         	print '\t\t %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)
    #
    #         self.move()
    #
    #         f.create_dataset('tstep'+str(i),data=self.path)
    #
    #     f.close()

	####Random walk restricted by sphere
    def single_sphere_diffusion(self):
        print '\t Initializing coordinates...'

        r = self.single_radius

        self.path = np.zeros([self.N,3])

        print '\t Doing random walk...'

        time_start = time.time()

        #### Open the file that will be written to
        f = h5py.File(self.filename + '.h5','w')
        f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size])
        f.create_dataset('tstep0',data=self.path)

        for i in range(1,self.tmax):
            if i%100 == 0:
            	print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

            self.move()

            dist_from_0 = np.linalg.norm(self.path,axis=1)
            #raw_input('...')

            self.path[dist_from_0>=r] = np.array(f.get('tstep'+str(i-1)))[dist_from_0>=r] #if boundary is crossed, move back to previous position

            f.create_dataset('tstep'+str(i),data=self.path)

        f.close()

	####Random walk restricted by cylinder
    def single_cylinder_diffusion(self):
        print '\t Initializing coordinates...'

        r = self.single_radius

        phi = np.random.uniform(0,2*np.pi,self.N)
        R2 = np.concatenate((np.random.uniform(0,r**2,[self.N,1]),np.random.uniform(0,r**2,[self.N,1])),axis=1)
        R = np.sqrt(R2) #Why? From stackoverflow...
        R[:,0],R[:,1] = R[:,0]*np.cos(phi),R[:,1]*np.sin(phi) #computing x and y values
        xy = R
        z = np.random.uniform(-10e-6,10e-6,[self.N,1])
        self.path = np.concatenate((xy,z),axis=1)

        print '\t Doing random walk...'

        time_start = time.time()

        #### Open the file that will be written to
        f = h5py.File(self.filename + '.h5','w')
        f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size])
        f.create_dataset('tstep0',data=self.path)

        for i in range(1,self.tmax):
            if i%100 == 0:
            	print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

            self.move()

            dist_from_0 = np.linalg.norm(self.path[:,:2],axis=1) #in x,y direction

            self.path[dist_from_0>=r] = np.array(f.get('tstep'+str(i-1)))[dist_from_0>=r] #if boundary is crossed, move back to previous position

            f.create_dataset('tstep'+str(i),data=self.path)

        f.close()

    ####Random walk restricted by multiple cylinders generated by AxonPacking
    def multi_cylinder_diffusion(self):
        print '\t Initializing coordinates...'

        self.path = np.random.uniform(-self.diff_lim*10**-6,self.diff_lim*10**-6,[self.N,3])

        assigned_space = np.zeros(self.N) #0 will be extracylinder space
        #Assign each walker to specific cylinder or to extracylinder space
        for i in range(self.n_cylinders):
            inside = (self.path[:,0]-self.pos[i,0]*10**-6)**2 + (self.path[:,1]-self.pos[i,1]*10**-6)**2 < (self.d[i]*10**-6)**2
            assigned_space[inside] = i+1 #0 will be extracylinder space

        print '\t Doing random walk...'

        time_start = time.time()

        #### Open the file that will be written to
        f = self.initialize_path_file()
        # f = h5py.File(self.filename + '.h5','w')
        # f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size])
        # f.create_dataset('tstep0',data=self.path)

        Trues = np.asarray([True]*self.N)

        for i in range(1,self.tmax):
            if i%100 == 0:
            	print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

            self.move()

            current_space = np.zeros(self.N) #0 will be extracylinder space
            #Assign each walker to specific cylinder or to extracylinder space
            for ii in range(self.n_cylinders):
                inside = (self.path[:,0]-self.pos[ii,0]*10**-6)**2 + (self.path[:,1]-self.pos[ii,1]*10**-6)**2 < (self.d[ii]*10**-6)**2
                current_space[inside] = ii+1 #0 will be extracylinder space

            self.path[current_space != assigned_space] = np.array(f.get('tstep'+str(i-1)))[current_space != assigned_space] #if boundary is crossed, move back to previous position

            #check if walkers in the extracylinder space crossed the diff_lim in the x-direction
            inside_diff_lim = np.abs(self.path[:,0]) > self.diff_lim*10**-6
            self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position
            #check if walkers in the extracylinder space crossed the diff_lim in the y-direction
            inside_diff_lim = np.abs(self.path[:,1]) > self.diff_lim*10**-6
            self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position
            f.create_dataset('tstep'+str(i),data=self.path)

        f.close()

    ####Random walk restricted by multiple permeable cylinders generated by AxonPacking
    def permeable_multi_cylinder_diffusion(self):
        if self.reuse != False:
            print '\t STOP! You are not allowed to overwrite a randomwalk file.'
            return None

        print '\t Initializing coordinates...'

        self.path = np.random.uniform(-self.diff_lim*10**-6,self.diff_lim*10**-6,[self.N,3])

        assigned_space = np.zeros(self.N) #0 will be extracylinder space
        #Assign each walker to specific cylinder or to extracylinder space
        for i in range(self.n_cylinders):
            inside = (self.path[:,0]-self.pos[i,0]*10**-6)**2 + (self.path[:,1]-self.pos[i,1]*10**-6)**2 < (self.d[i]*10**-6)**2
            assigned_space[inside] = i+1 #0 will be extracylinder space

        print '\t Doing random walk...'

        time_start = time.time()

        #### Open the file that will be written to
        f = self.initialize_path_file()
        # f = h5py.File(self.filename + '.h5','w')
        # f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size,self.diff_lim,self.signal_lim])
        # f.create_dataset('tstep0',data=self.path)

        # Trues = np.asarray([True]*self.N)

        for i in range(1,self.tmax):
            if i%100 == 0:
            	print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

            self.move()

            current_space = np.zeros(self.N) #0 will be extracylinder space
            #Assign each walker to specific cylinder or to extracylinder space
            for ii in range(self.n_cylinders):
                inside = (self.path[:,0]-self.pos[ii,0]*10**-6)**2 + (self.path[:,1]-self.pos[ii,1]*10**-6)**2 < (self.d[ii]*10**-6)**2
                current_space[inside] = ii+1 #0 will be extracylinder space

            permeate = np.random.uniform(0,1,[self.N]) < self.permeability #if permeability is high, many walkers are not corrected back to previous position
            crossed = current_space != assigned_space
            change = crossed-permeate*crossed #Of those who crossed, those who permeated should not be corrected

            self.path[change] = np.array(f.get('tstep'+str(i-1)))[change] #if boundary is crossed, move back to previous position

            #update assigned_space
            assigned_space[True-change] = current_space[True-change] #Those that were NOT corrected, should now be assigned to their new space

            #check if walkers in the extracylinder space crossed the diff_lim in the x-direction
            inside_x = np.abs(self.path[:,0]) > self.diff_lim*10**-6
            #check if walkers in the extracylinder space crossed the diff_lim in the y-direction
            inside_y = np.abs(self.path[:,1]) > self.diff_lim*10**-6
            #check if walkers in the extracylinder space crossed the diff_lim in the z-direction
            inside_z = np.abs(self.path[:,2]) > self.diff_lim*10**-6
            # make correction
            self.path[inside_x+inside_y+inside_z == True] = np.array(f.get('tstep'+str(i-1)))[inside_x+inside_y+inside_z == True] #if boundary is crossed, move back to previous position

            f.create_dataset('tstep'+str(i),data=self.path)

        f.close()

    def TrimCylinders(self,axons,packing,stats):
        print '\t Trimming cylinders...'

        #self.diff_lim = diff_lim
        #self.diff_lim = diff_lim*10**6 #[\mu m]

        axons = io.loadmat(axons)
        packing = io.loadmat(packing)
        stats = io.loadmat(stats)
        self.d_mean = axons['d_mean'][0][0][0][0] #theoretical mean of axon diameters in um
        self.d_var = axons['d_var'][0][0][0][0] #theoretical variance of axon diameters in um
        self.gap = axons['Delta'][0][0][0][0] #gap between the edge of axons in um

        self.d = axons['d'][0][0] #diameters of the cylinders #radii, right???
        cyl_circum = np.sum(self.d*2.*np.pi*2)
        box_circum = self.diff_lim*4
        cyl_area = cyl_circum*self.diff_lim
        box_area = box_circum*self.diff_lim
        print '\t    box_circumference/sum(cylinder_circumferences) = %.5f'  %(box_circum/cyl_circum)

        self.n_cylinders = axons['N'][0][0][0][0] #number of axons
        self.pos = packing['final_positions'][0][0].T

        #Center the data
        self.pos[:,0] -= np.mean(self.pos[:,0])
        self.pos[:,1] -= np.mean(self.pos[:,1])

        deleted = 0

        for i in range(self.n_cylinders):
            if (np.abs(self.pos[i-deleted,0])-self.d[i-deleted]) > self.diff_lim or (np.abs(self.pos[i-deleted,1])-self.d[i-deleted]) > self.diff_lim:
                self.d = np.delete(self.d,i-deleted,0)
                self.pos = np.delete(self.pos,i-deleted,0)
                deleted += 1

        self.n_cylinders -=  deleted

        print '\t    %d out of %d cylinders are left...' %(self.n_cylinders,self.n_cylinders+deleted)

    def PlotCylinders(self,twoD=True,threeD=True,directions=False):
        print '\t Plotting cylinders...'

        ####2D
        if twoD == True:
            fig=plt.figure(figsize=(11,10),facecolor = 'w',edgecolor='w')
            ax=fig.add_subplot(1,1,1)
            for i in range(self.n_cylinders):
                circle = plt.Circle((self.pos[i,0],self.pos[i,1]), radius=self.d[i][0],color='grey',ec='None',lw=2,alpha=0.9)
                ax.add_artist(circle)
            #Plot signal box
            ax.plot([-self.signal_lim,-self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
            ax.plot([self.signal_lim,self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
            ax.plot([-self.signal_lim,self.signal_lim],[-self.signal_lim,-self.signal_lim],c='blue',linestyle='--',linewidth=3)
            ax.plot([-self.signal_lim,self.signal_lim],[self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
            #Plot diffusion box
            ax.plot([-self.diff_lim,-self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
            ax.plot([self.diff_lim,self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
            ax.plot([-self.diff_lim,self.diff_lim],[-self.diff_lim,-self.diff_lim],c='blue',linestyle='-',linewidth=3)
            ax.plot([-self.diff_lim,self.diff_lim],[self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)

            #ax.axis('equal')
            ax.set_xlim((-self.diff_lim-10, self.diff_lim+10))
            ax.set_ylim((-self.diff_lim-10, self.diff_lim+10))
            plt.xlabel('x [$\mu$m]');plt.ylabel('y [$\mu$m]')
            plt.savefig(self.filename+'Cylinders_2d'+'.png',bbox_inches='tight',dpi=100)
            plt.close('all')

        ####3D
        if threeD == True:
        	fig1 = plt.figure(figsize=(10.5,10),facecolor = 'w',edgecolor='w')
        	ax = fig1.add_subplot(111, projection='3d')
        	ax.set_xlim([-50,50]); ax.set_ylim([-50,50]); ax.set_zlim([-20,20])
        	ax.set_xlabel('x [$\mu$m]'); ax.set_ylabel('y [$\mu$m]'); ax.set_zlabel('z [$\mu$m]')
        	ax.view_init(elev=55., azim=45)

        	for i in range(self.n_cylinders):
        		x = np.linspace(-self.d[i][0],self.d[i][0], 100) #why d and not r??
        		z = np.linspace(-20, 20, 100)
        		Xc,Zc = np.meshgrid(x, z)
        		Yc = np.sqrt((self.d[i][0])**2-Xc**2)
        		rstride,cstride = 20,2
        		ax.plot_surface(Xc+self.pos[i,0], Yc+self.pos[i,1], Zc, color='gray',alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
        		ax.plot_surface(Xc+self.pos[i,0], -Yc+self.pos[i,1], Zc, color='gray', alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)

        	#Save figure with different elevations
        	ax.view_init(elev=55., azim=45)
        	plt.savefig('Cylinders_3d'+'.png',bbox_inches='tight',dpi=100)
        	ax.view_init(elev=90., azim=-90)
        	plt.savefig('Cylinders_3d_birdview'+'.png',bbox_inches='tight',dpi=100)
        	plt.close('all')

        ####Together with directions
        if directions == True:
        	fig1 = plt.figure(figsize=(10.5,10),facecolor = 'w',edgecolor='w')
        	self.ax = fig1.add_subplot(111, projection='3d')
        	self.ax.set_xlim([-50,50]); self.ax.set_ylim([-50,50]); self.ax.set_zlim([-20,20])
        	self.ax.set_xlabel('x [$\mu$m]'); self.ax.set_ylabel('y [$\mu$m]'); self.ax.set_zlabel('z [$\mu$m]')
        	self.ax.view_init(elev=55., azim=45)

        	for i in range(self.n_cylinders):
        		x = np.linspace(-self.d[i][0],self.d[i][0], 100) #why d and not r??
        		z = np.linspace(-20, 20, 100)
        		Xc,Zc = np.meshgrid(x, z)
        		Yc = np.sqrt((self.d[i][0])**2-Xc**2)
        		rstride,cstride = 20,2
        		self.ax.plot_surface(Xc+self.pos[i,0], Yc+self.pos[i,1], Zc, color='gray',alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
        		self.ax.plot_surface(Xc+self.pos[i,0], -Yc+self.pos[i,1], Zc, color='gray', alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)

    def PlotPaths(self,N_walkers_plotted,N_steps_plotted,twoD=False,threeD=False,PlotSusField=False):
        print '\t Plotting paths...'

        f = h5py.File(self.filename + '.h5','r')

        fig = plt.figure(figsize=(10.5,10),facecolor = 'w',edgecolor='w')
        plt.title('First %d steps out of %d for %d out of %d ensembles with step size %.3f $\mu$m \n and permeability %.3f' %(N_steps_plotted,self.tmax,N_walkers_plotted,self.N,self.step_size*10**6,self.permeability))
        # plt.grid(True)

        if threeD == True:

            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x [$\mu$m]'); ax.set_ylabel('y [$\mu$m]'); ax.set_zlabel('z [$\mu$m]')

            for i in range(N_walkers_plotted): #For all ensembles
                # if i == 0:
                #     alpha = 1.
                # else:
                #     alpha = 0.2
                alpha = 0.2

                current = np.asarray([np.array(f.get('tstep'+str(ii)))[i,:] for ii in range(self.tmax)])*10**6
                ax.plot(current[:N_steps_plotted,0],current[:N_steps_plotted,1],current[:N_steps_plotted,2],marker='.',ms=6,alpha=alpha)

                if i%2 == 0:
                    print '\t    %.2f percent done with plotting the paths...' %(i/float(N_walkers_plotted)*100)

        elif twoD == True:
            ax=fig.add_subplot(1,1,1)
            plt.xlabel('x [$\mu$m]'); plt.ylabel('y [$\mu$m]')
            self.add = ''

            if PlotSusField == True:
                self.add = '_alpha'+str(self.alpha)
                plt.axis('equal')
                dim = 80

                # dchi, chi_par, chi_perp = self.dchi, self.chi_par, self.chi_perp
                B_0, alpha, r1, r2, chi_A = 1., self.alpha, self.r1*10**6, self.r2*10**6, self.chi_A
                # B_0_perp = B_0*np.sin(alpha)

                r_i, r_o, theta = r1, r2, alpha

                #outside
                # r_outside = np.linspace(r2,50,dim)
                # theta = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                # r_matrix, phi_matrix = np.meshgrid(r_outside,theta)
                # X = r_matrix * np.cos(phi_matrix)
                # Y = r_matrix * np.sin(phi_matrix)
                # h_par_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(X**2+Y**2) * (2*X**2/(X**2+Y**2)-1)
                # h_perp_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2) * (2*X*Y)/(X**2+Y**2)**2
                # c_outside = (h_par_out+h_perp_out)#/ np.max(np.max(h_par_out+h_perp_out))
                # ctf1 = ax.contourf(X+0., Y+0., c_outside, 100, cmap=cm.jet)#,vmin=-1,vmax=1)
                phi = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                r = np.linspace(r_o,50,dim)
                r_matrix, phi_matrix = np.meshgrid(r,phi)
                X = r_matrix * np.cos(phi_matrix)
                Y = r_matrix * np.sin(phi_matrix)
                field = chi_A*B_0*np.sin(theta)**2*0.125*(r_o**2-r_i**2) * 1./(X*X+Y*Y) * (2*X*X/(X*X+Y*Y)-1)
                ctf1 = ax.contourf(X, Y, field, 100, cmap=cm.jet)

                #inside
                # r_inside = np.linspace(0,r1,dim)
                # theta = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                # r_matrix, phi_matrix = np.meshgrid(r_inside,theta)
                # X = r_matrix * np.cos(phi_matrix)
                # Y = r_matrix * np.sin(phi_matrix)
                # h_par_in = 2*np.pi*B_0_perp*dchi*np.log(r2/r1) * np.ones([dim,dim])
                # h_perp_in = 0.
                # c_inside = (h_par_in + h_perp_in) #/ np.max(np.max(h_par_out+h_perp_out))
                # ctf2 = ax.contourf(X+0., Y+0., c_inside, 100, cmap=cm.jet)#,vmin=0,vmax=1)
                phi = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                r = np.linspace(0,r_i,dim)
                r_matrix, phi_matrix = np.meshgrid(r,phi)
                X = r_matrix * np.cos(phi_matrix)
                Y = r_matrix * np.sin(phi_matrix)
                field = 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i) * np.ones([dim,dim])
                ctf2 = ax.contourf(X, Y, field, 100, cmap=cm.jet)


                cb = plt.colorbar(ctf1)
                cb.set_label('Some intensity of $\mathbf{h}$')

                plt.xlim(-self.diff_lim-5,self.diff_lim+5)
                plt.ylim(-self.diff_lim-5,self.diff_lim+5)


            for i in range(N_walkers_plotted): #For all ensembles
                # if i == 0:
                #     alpha = 1.
                # else:
                #     alpha = 0.2
                alpha = 0.2

                current = np.asarray([np.array(f.get('tstep'+str(ii)))[i,:] for ii in range(self.tmax)])*10**6
                plt.plot(current[:N_steps_plotted,0],current[:N_steps_plotted,1],marker='.',ms=6,alpha=alpha)

                if i%10 == 0:
                    print '\t    %.2f percent done with plotting the paths...' %(i/float(N_walkers_plotted)*100)


        self.PlotRestriction(ax,twoD,threeD)


        # plt.savefig('paths'+'.png',bbox_inches='tight',dpi=100)
        # plt.close('all')


    ####Plot restriction
    def PlotRestriction(self,ax,twoD,threeD):
        print '\t    Plotting restriction...'

        if self.restriction_type == 'Free diffusion':

            if self.solid_boundary == 'sphere':
                signal_boundary = plt.Circle((0, 0), self.signal_lim, color='b',alpha=0.8, fill=False,lw=3,ls='--')
                ax.add_artist(signal_boundary)
                solid_boundary = plt.Circle((0, 0), self.diff_lim, color='b',alpha=0.8, fill=False,lw=3,ls='-')
                ax.add_artist(solid_boundary)
                plt.xlim(-(self.diff_lim+1),(self.diff_lim+1))
                plt.ylim(-(self.diff_lim+1),(self.diff_lim+1))

            elif self.solid_boundary == 'cube':
                #Plot signal box
                ax.plot([-self.signal_lim,-self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([self.signal_lim,self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([-self.signal_lim,self.signal_lim],[-self.signal_lim,-self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([-self.signal_lim,self.signal_lim],[self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                #Plot diffusion box
                ax.plot([-self.diff_lim,-self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([self.diff_lim,self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([-self.diff_lim,self.diff_lim],[-self.diff_lim,-self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([-self.diff_lim,self.diff_lim],[self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)

            plt.savefig(self.filename+'paths.png',bbox_inches='tight',dpi=100)
            plt.close('all')
        elif self.restriction_type == 'Single cylinder':
            if threeD == True:
                single_radius = self.single_radius*10**6
                x=np.linspace(-single_radius, single_radius, 100)
                z=np.linspace(-20, 20, 100)
                Xc, Zc=np.meshgrid(x, z)
                Yc = np.sqrt(single_radius**2-Xc**2)
                rstride = 20
                cstride = 5
                ax.plot_surface(Xc, Yc, Zc, color='gray',alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
                ax.plot_surface(Xc, -Yc, Zc, color='gray', alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
                plt.savefig(self.filename+'paths_3d.png',bbox_inches='tight',dpi=100)
                plt.close('all')
            elif twoD == True:
                circle = plt.Circle((0,0), radius=self.single_radius*10**6,color='None',ec='black',lw=2,alpha=0.9)
                ax.add_artist(circle)
                plt.savefig(self.filename+self.add+'paths_2d.png',bbox_inches='tight',dpi=100)
                plt.close('all')
        elif self.restriction_type == 'Single sphere':
            single_radius = self.single_radius*10**6
            u = np.linspace(0, 2 * np.pi, 300)
            v = np.linspace(0, np.pi, 300)
            x = single_radius * np.outer(np.cos(u), np.sin(v))
            y = single_radius * np.outer(np.sin(u), np.sin(v))
            z = single_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            rstride = 20
            cstride = 10
            ax.plot_surface(x, y, z, color='gray',alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
            plt.savefig(self.filename+'paths.png',bbox_inches='tight',dpi=100)
            plt.close('all')
        elif self.restriction_type == 'Multi cylinder':
            if threeD == True:
                for i in range(self.n_cylinders):
                    x = np.linspace(-self.d[i][0],self.d[i][0], 100) #why d and not r??
                    z = np.linspace(-20, 20, 100)
                    Xc,Zc = np.meshgrid(x, z)
                    Yc = np.sqrt((self.d[i][0])**2-Xc**2)
                    rstride,cstride = 20,2
                    ax.plot_surface(Xc+self.pos[i,0], Yc+self.pos[i,1], Zc, color='gray',alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
                    ax.plot_surface(Xc+self.pos[i,0], -Yc+self.pos[i,1], Zc, color='gray', alpha=0.4, rstride=rstride, cstride=cstride,linewidth=0)
                #Save figure with different elevations
                ax.view_init(elev=55., azim=45)
                plt.savefig(self.filename+'path_3d'+'.png',bbox_inches='tight',dpi=100)
                ax.view_init(elev=90., azim=-90)
                plt.savefig(self.filename+'_paths_3d_birdview'+'.png',bbox_inches='tight',dpi=100)
                plt.close('all')

            elif twoD == True:
                ls = '-' #linestyle
                for i in range(self.n_cylinders):
                    circle = plt.Circle((self.pos[i,0],self.pos[i,1]), radius=self.d[i][0],ls = ls,color='grey',ec='None',lw=2,alpha=0.8)
                    ax.add_artist(circle)
                #Plot signal box
                ax.plot([-self.signal_lim,-self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([self.signal_lim,self.signal_lim],[-self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([-self.signal_lim,self.signal_lim],[-self.signal_lim,-self.signal_lim],c='blue',linestyle='--',linewidth=3)
                ax.plot([-self.signal_lim,self.signal_lim],[self.signal_lim,self.signal_lim],c='blue',linestyle='--',linewidth=3)
                #Plot diffusion box
                ax.plot([-self.diff_lim,-self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([self.diff_lim,self.diff_lim],[-self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([-self.diff_lim,self.diff_lim],[-self.diff_lim,-self.diff_lim],c='blue',linestyle='-',linewidth=3)
                ax.plot([-self.diff_lim,self.diff_lim],[self.diff_lim,self.diff_lim],c='blue',linestyle='-',linewidth=3)

                ax.axis('equal')
                plt.savefig(self.filename+'_paths_2d.png',bbox_inches='tight',dpi=100)
                plt.close('all')
        else:
            print '\t    This kind of restriction has not been defined for plotting...'

    # def RepeatPGSE_single_direction(self,delta_in,Delta_in,gs,filename=None):
    #     print '\t Doing PGSE...'
    #
    #     if filename == None:
    #         filename = self.filename
    #
    #     #Convert from time to number of iterations
    #     delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))
    #     #Print the effect this conversion has on the time
    #     print '\t    Due to temporal resolution: %.4f -> %.4f and %.4f -> %.4f' %(delta_in,delta*self.dt,Delta_in,Delta*self.dt)
    #     if delta*self.dt+Delta*self.dt > self.duration:
    #         print '\t Error: The chosen time intervals suceed the duration of the random walks.'
    #         raw_input('...')
    #
    #     #self.delta_in
    #     g,signal_tot,signal_std = np.zeros(len(gs)),np.zeros(len(gs)),np.zeros(len(gs))
    #     i = 0
    #     for current_g in gs:
    #         g[i],signal_tot[i],signal_std[i] = self.PGSE_single_direction(delta_in,Delta_in,current_g,filename)
    #         i += 1
    #
    #     return g,signal_tot,signal_std
    #
    # def PGSE_single_direction(self,delta_in,Delta_in,g,filename):
    #     print '\t    Doing PGSE for g=%.3f...' %g
    #
    #     #Convert from time to number of iterations
    #     delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))
    #
    #     f = h5py.File(filename + '.h5','r')
    #
    #     self.phases = np.zeros(self.N)
    #
    #     for i in range(0,delta):
    #         x = np.array(f.get('tstep'+str(i)))[:,0]
    #         self.phases += self.gamma*g*x*self.dt
    #
    #     for i in range(Delta,Delta+delta):
    #         x = np.array(f.get('tstep'+str(i)))[:,0]
    #         self.phases -= self.gamma*g*x*self.dt
    #
    #     #Read out the signal
    #     signals = np.cos(self.phases) # phases = [rad]
    #     signal_tot = np.sum(signals)
    #     signal_std = np.std(signals)
    #
    #     print '\t        Signal: %.2f +- %.2f ' %(signal_tot,signal_std)
    #
    #     f.close()
    #
    #     return g,signal_tot,signal_std

    def RepeatPGSE(self,delta_in,Delta_in,gs,filename=None):
        print '\t Doing PGSE...'

        if filename == None:
            filename = self.filename

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))
        #Print the effect this conversion has on the time
        print '\t    Due to temporal resolution: %.4f -> %.4f and %.4f -> %.4f' %(delta_in,delta*self.dt,Delta_in,Delta*self.dt)
        if delta*self.dt+Delta*self.dt > self.duration:
            print '\t Error: The chosen time intervals suceed the duration of the random walks.'
            raw_input('...')

        #self.delta_in
        grows = len(gs)
        drows, dcols = np.shape(self.directions)

        g,signal_tot,signal_std = np.zeros([grows,drows]),np.zeros([grows,drows]),np.zeros([grows,drows])
        gi = 0
        for current_g in gs:
            di = 0
            for current_direction in self.directions:
                # g[gi,di],signal_tot[gi,di],signal_std[gi,di] = self.PGSE(delta_in,Delta_in,current_g,current_direction,filename)
                # g[gi,di],signal_tot[gi,di],signal_std[gi,di] = self.PGSE_susceptibility(delta_in,Delta_in,current_g,current_direction,filename)
                print 'Doing a special type of PGSE!'
                g[gi,di],signal_tot[gi,di],signal_std[gi,di] = self.PGSE_MultiCylinderSusceptibility(delta_in,Delta_in,current_g,current_direction,filename)

                di += 1
            gi += 1

        return g,signal_tot,signal_std

    def PGSE(self,delta_in,Delta_in,g,current_direction,filename):

        print '\t    g=%.3f and direction %s' %(g,str(current_direction))

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))

        f = h5py.File(filename + '.h5','r')

        self.phases = np.zeros(self.N)

        for i in range(0,delta):
            xyz = np.array(f.get('tstep'+str(i)))
            self.phases += np.dot(xyz,current_direction)

        for i in range(Delta,Delta+delta):
            xyz = np.array(f.get('tstep'+str(i)))
            self.phases -= np.dot(xyz,current_direction)

        #Read out the signal
        self.phases = g*self.gamma*self.dt*self.phases
        signals = np.cos(self.phases) # phases = [rad]
        last_position = f.get('tstep'+str(Delta+delta))
        # self.solid_boundary = 'sphere'

        if self.solid_boundary == 'sphere':
            ####Spherical sampling
            inside_signal_lim = np.linalg.norm(last_position,axis=1) < self.signal_lim*10**-6
            signal_tot = np.sum(signals[inside_signal_lim])
            signal_std = np.std(signals[inside_signal_lim])
        elif self.solid_boundary == 'cube':
            ####Cubic sampling
            inside_x = np.abs(last_position[:,0]) < self.signal_lim*10**-6
            inside_y = np.abs(last_position[:,1]) < self.signal_lim*10**-6
            inside_z = np.abs(last_position[:,2]) < self.signal_lim*10**-6
            signal_tot = np.sum(signals[inside_x*inside_y*inside_z])
            signal_std = np.std(signals[inside_x*inside_y*inside_z])

        # signal_tot = np.sum(signals)

        print '\t    => signal = %.2f +- %.2f ' %(signal_tot,signal_std)

        f.close()

        return g,signal_tot,signal_std

    def ComputeMultiCylinderMagneticField(self):

        print '\t Computing field pertubation for multi cylinder substrate...'

        # mem_width = 0.2 #width of all membranes
        B_0, theta, chi_A = 1., 5./10.*np.pi,self.chi_A# 0.00001
        self.field_resolution = 800
        x = y = np.linspace(-self.diff_lim,self.diff_lim,self.field_resolution)

        self.field = np.zeros([self.field_resolution,self.field_resolution])

        # centers = self.pos#[:2]

        threshold = 0.#0.1*10**-16

        #g ratio membrane stuff
        # g = 0.22 #from the internet
        r_os = 2.*self.d/(1.+self.g)
        r_is = 2.*self.d/(1./self.g+1.)

        for center,r_o,r_i in zip(self.pos,r_os,r_is):
            X,Y = np.meshgrid(x,y)
            X -= center[0]
            Y -= center[1]

            contribution = chi_A*B_0*np.sin(theta)**2*0.125*(r_o**2-r_i**2) * 1./(X*X+Y*Y) * (2*X*X/(X*X+Y*Y)-1)
            self.field[abs(contribution)>threshold] += contribution[abs(contribution)>threshold]

        for center,r_o,r_i in zip(self.pos,r_os,r_is):
            X,Y = np.meshgrid(x,y)
            X -= center[0]
            Y -= center[1]

            inside = np.sqrt(X*X+Y*Y) < r_o - (r_o-r_i)/2.

            self.field[inside] = 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i) * np.ones([self.field_resolution,self.field_resolution])[inside]

        self.X,self.Y = np.meshgrid(x,y)

    def PGSE_MultiCylinderSusceptibility(self,delta_in,Delta_in,g,current_direction,filename):

        print '\t    g=%.3f and direction %s' %(g,str(current_direction))

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))

        f = h5py.File(filename + '.h5','r')

        # self.phases =
        gradient_contrib = np.zeros(self.N)
        susceptibility_contrib = np.zeros(self.N)

        lim = self.diff_lim*10**-6

        for i in range(0,delta):
            xyz = np.array(f.get('tstep'+str(i)))

            gradient_contrib += np.dot(xyz,current_direction)

            index = np.round((xyz[:,:2]+lim)/(2.*lim) * (self.field_resolution-1)).astype(int)
            susceptibility_contrib += self.field[index[:,0],index[:,1]]

            # self.phases += self.field[index[:,0],index[:,1]]

        for i in range(Delta,Delta+delta):
            xyz = np.array(f.get('tstep'+str(i)))

            gradient_contrib -= np.dot(xyz,current_direction)

            index = np.round((xyz[:,:2]+lim)/(2.*lim) * (self.field_resolution-1)).astype(int)
            susceptibility_contrib -= self.field[index[:,0],index[:,1]]


            # xy = np.array(f.get('tstep'+str(i)))[:,:2]
            # index = np.round((xy+lim)/(2.*lim) * self.field_resolution-1).astype(int)#.tolist()
            #
            # self.phases -= self.field[index[:,0],index[:,1]]

        #Read out the signal
        self.phases = (g*gradient_contrib+susceptibility_contrib)*self.gamma*self.dt
        signals = np.cos(self.phases) # phases = [rad]
        last_position = f.get('tstep'+str(Delta+delta))
        # self.solid_boundary = 'sphere'

        if self.solid_boundary == 'sphere':
            ####Spherical sampling
            inside_signal_lim = np.linalg.norm(last_position,axis=1) < self.signal_lim*10**-6
            signal_tot = np.sum(signals[inside_signal_lim])
            signal_std = np.std(signals[inside_signal_lim])
        elif self.solid_boundary == 'cube':
            ####Cubic sampling
            inside_x = np.abs(last_position[:,0]) < self.signal_lim*10**-6
            inside_y = np.abs(last_position[:,1]) < self.signal_lim*10**-6
            inside_z = np.abs(last_position[:,2]) < self.signal_lim*10**-6
            signal_tot = np.sum(signals[inside_x*inside_y*inside_z])
            signal_std = np.std(signals[inside_x*inside_y*inside_z])

        # signal_tot = np.sum(signals)

        print '\t    => signal = %.2f +- %.2f ' %(signal_tot,signal_std)

        f.close()

        return g,signal_tot,signal_std

    def RepeatPGSE_susceptibility(self,delta_in,Delta_in,gs,alpha,filename=None):
        print '\t Doing PGSE...'

        if filename == None:
            filename = self.filename

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))
        #Print the effect this conversion has on the time
        print '\t    Due to temporal resolution: %.4f -> %.4f and %.4f -> %.4f' %(delta_in,delta*self.dt,Delta_in,Delta*self.dt)
        if delta*self.dt+Delta*self.dt > self.duration:
            print '\t Error: The chosen time intervals suceed the duration of the random walks.'
            raw_input('...')

        #self.delta_in
        grows = len(gs)
        drows, dcols = np.shape(self.directions)

        g,signal_intra,signal_extra,signal_intra_std,signal_extra_std = np.zeros([grows,drows]),np.zeros([grows,drows]),np.zeros([grows,drows]),np.zeros([grows,drows]),np.zeros([grows,drows])
        gi = 0
        for current_g in gs:
            di = 0
            for current_direction in self.directions:
                # current_direction = np.asarray([0.1,0.2,0.3])
                # g[gi,di],signal_tot[gi,di],signal_std[gi,di] = self.PGSE(delta_in,Delta_in,current_g,current_direction,filename)
                g[gi,di],signal_intra[gi,di],signal_extra[gi,di],signal_intra_std[gi,di],signal_extra_std[gi,di] = self.PGSE_susceptibility(delta_in,Delta_in,current_g,alpha,current_direction,filename)
                di += 1
            gi += 1

        self.g,self.signal_intra,self.signal_extra,self.signal_intra_std,self.signal_extra_std =g,signal_intra,signal_extra,signal_intra_std,signal_extra_std

        return g,signal_intra,signal_extra,signal_intra_std,signal_extra_std

    def PGSE_susceptibility(self,delta_in,Delta_in,g,theta,current_direction,filename,diff_restriction='sphere'):
        print '\t    g=%.3f and direction %s' %(g,str(current_direction))

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))

        ####Susceptibility stuff
        # dchi, chi_par, chi_perp =  0.0005, 0.0005, 0.0005
        # self.dchi, self.chi_par, self.chi_perp = dchi, chi_par, chi_perp
        # B_0, r1, r2 = 1., self.single_radius-1*10**-6, self.single_radius+1*10**-6
        # self.alpha, self.r1, self.r2 = alpha, r1, r2
        # B_0_perp = B_0*np.sin(alpha)

        B_0, r_i, r_o, chi_A = 1., self.single_radius-1*10**-6, self.single_radius+1*10**-6, 0.0005
        self.alpha, self.r1, self.r2, self.chi_A = theta, r_i, r_o, chi_A

        def h_in():
            # h_par_in = 2*np.pi*B_0_perp*dchi*np.log(r2/r1)
            # h_perp_in = 0.
            # return h_par_in + h_perp_in
            return 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i)
        def h_out(X,Y):
            # h_par_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(x**2+y**2) * (2*x**2/(x**2+y**2)-1)
            # h_perp_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2) * (2*x*y)/(x**2+y**2)**2
            # return h_par_out+h_perp_out
            return chi_A*B_0*np.sin(theta)**2*0.125*(r_o**2-r_i**2) * 1./(X*X+Y*Y) * (2*X*X/(X*X+Y*Y)-1)

        f = h5py.File(filename + '.h5','r')

        self.phases = np.zeros(self.N)

        print 'THE PHASES ARE CALCULTED IN A WRONG WAY HERE!!!!'

        for i in range(0,delta):
            xyz             = np.array(f.get('tstep'+str(i))) #xyz
            assigned_space  = np.array(f.get('assigned_space'+str(i)))
            # sus_contrib_in  = h_in(xyz[:,0],xyz[:,1])*(assigned_space==0)
            sus_contrib_in  = h_in()*(assigned_space==0)
            sus_contrib_out = h_out(xyz[:,0],xyz[:,1])*(assigned_space==1)
            sus_contrib     = sus_contrib_in + sus_contrib_out

            #WRONG!!!!
            self.phases += np.linalg.norm(self.gamma*g*current_direction*xyz*self.dt,axis=1) + sus_contrib

        for i in range(Delta,Delta+delta):
            xyz             = np.array(f.get('tstep'+str(i))) #xyz
            assigned_space  = np.array(f.get('assigned_space'+str(i)))
            sus_contrib_in  = h_in()*(assigned_space==0)
            sus_contrib_out = h_out(xyz[:,0],xyz[:,1])*(assigned_space==1)
            sus_contrib     = sus_contrib_in + sus_contrib_out

            #WRONG!!!!
            self.phases -= np.linalg.norm(self.gamma*g*current_direction*xyz*self.dt,axis=1) + sus_contrib

        #Read out the signal
        print np.sum(self.phases)
        signals = np.cos(self.phases) # phases = [rad]
        last_position = f.get('tstep'+str(Delta+delta))
        last_space = np.array(f.get('assigned_space'+str(Delta+delta)))

        if 2==2 or self.solid_boundary == 'cube':
            ####Cubic sampling
            inside_x = np.abs(last_position[:,0]) < self.signal_lim*10**-6
            inside_y = np.abs(last_position[:,1]) < self.signal_lim*10**-6
            inside_z = np.abs(last_position[:,2]) < self.signal_lim*10**-6
            signal_std = np.std(signals[inside_x*inside_y*inside_z])
            signal_tot = np.sum(signals[inside_x*inside_y*inside_z])
            # print np.array(last_space)
            # raw_input('...')
            intra = last_space == 0
            extra = last_space != 0

            signal_intra = np.sum(signals[inside_x*inside_y*inside_z*intra])
            signal_extra = np.sum(signals[inside_x*inside_y*inside_z*extra])
            signal_intra_std = np.std(signals[inside_x*inside_y*inside_z*intra])
            signal_extra_std = np.std(signals[inside_x*inside_y*inside_z*extra])
        # elif self.solid_boundary == 'sphere':
        #     ####Spherical sampling
        #     inside_signal_lim = np.linalg.norm(last_position,axis=1) < self.signal_lim*10**-6
        #     signal_tot = np.sum(signals[inside_signal_lim])
        #     signal_std = np.std(signals[inside_signal_lim])

        print '\t    => signal = %.2f +- %.2f ' %(signal_tot,signal_std)

        f.close()

        return g,signal_intra,signal_extra,signal_intra_std,signal_extra_std

    def PGSE_old(self,delta_in,Delta_in,g,current_direction,filename):
        print '\t    g=%.3f and direction %s' %(g,str(current_direction))

        #Convert from time to number of iterations
        delta,Delta = int(round(delta_in/self.dt)),int(round(Delta_in/self.dt))

        f = h5py.File(filename + '.h5','r')

        self.phases = np.zeros(self.N)

        for i in range(0,delta):
            xyz = current_direction*np.array(f.get('tstep'+str(i)))
            self.phases += np.linalg.norm(self.gamma*g*xyz*self.dt,axis=1)

        for i in range(Delta,Delta+delta):
            xyz = current_direction*np.array(f.get('tstep'+str(i)))
            self.phases -= np.linalg.norm(self.gamma*g*xyz*self.dt,axis=1)

        #Read out the signal
        signals = np.cos(self.phases) # phases = [rad]
        signal_tot = np.sum(signals)
        signal_std = np.std(signals)

        print '\t    => signal = %.2f +- %.2f ' %(signal_tot,signal_std)

        f.close()

        return g,signal_tot,signal_std

    def GetDirectionsFrombvecFile(self,filename,rotate=None,angle=None):
        bvec_in = np.loadtxt(filename).T
        bvec_out = bvec_in[0,:]

        for i in range(1,len(bvec_in)):
            if (bvec_in[i,:].tolist() not in bvec_out.tolist()) and (bvec_in[i,:].tolist() != [0.,0.,0.]):
                # print bvec_in[i,:].tolist()
                # print bvec_out.tolist()
                # print bvec_in[i,:].tolist() not in bvec_out.tolist()
                bvec_out = np.vstack((bvec_out,bvec_in[i,:]))

        self.directions = bvec_out[1:,:] #Stupid. Why is this neccesary??

        if rotate == 'x':
            R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            self.directions = np.dot(self.directions,R)
        if rotate == 'y':
            R = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            self.directions = np.dot(self.directions,R)
        if rotate == 'z':
            R = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            self.directions = np.dot(self.directions,R)

    def PlotDirections(self,lengths,add=''):

        # self.PlotCylinders(twoD=False,threeD=False,directions=True)

        fig = plt.figure(figsize=(8,8),facecolor = 'w',edgecolor='w')
        ax = fig.add_subplot(111, projection='3d')
        lim = 1.2
        ax.set_xlim([-lim,lim]); ax.set_ylim([-lim,lim]); ax.set_zlim([-lim,lim])
        ax.set_xlabel('x [a.u.]'); ax.set_ylabel('y [a.u.]'); ax.set_zlabel('z [a.u.]')
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]);

        print len(lengths), lengths

        if sum(lengths == np.array([1.]*len(self.directions))) != len(self.directions):
            f = h5py.File(self.filename + '.h5','r')
            self.N = np.array(f.get('info'))[2]
            f.close()
            lengths = np.array([1.]*len(lengths)) - lengths/self.N

        i = 0
        for v in self.directions:
            ax.quiver(0.,0.,0.,v[0],v[1],v[2],length=lengths[i],arrow_length_ratio=0.2,linewidth=3)
            i += 1

        # plt.show()

        #Save figure with different elevations
        ax.view_init(elev=90., azim=0.)
        plt.savefig(self.filename+'_Directions_view1'+add+'.png',bbox_inches='tight',dpi=100)
        ax.view_init(elev=0., azim=0.)
        plt.savefig(self.filename+'_Directions_view2'+add+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

    def PlotSignalVsb(self,signalstuff,delta,Delta):
        print '\t Plotting S(b)...'

        gs,signal_tot,signal_std = signalstuff[0][:,0],signalstuff[1],signalstuff[2]
        n_gs, n_directions = np.shape(signal_tot)

        def func(x,a):
            return self.fit_max*np.exp(-x*a)

        fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        plt.grid(True)

        xlim = 4e9 #Not optimal

        b = (self.gamma*gs*delta)**2*(Delta-delta/3.)
        b_smooth = np.arange(0,np.max(b),np.max(b)/100.)

        #Add fake line to get entry in the legend
        sim_label = r'\textbf{Specs of simulated data}:'+' \n - %d ensembles  \n - %d cylinders \n - d$\_$mean = %.2e m \n -  d$\_$var = %.2e m \n - gap = %.2e m \n - $\delta,\Delta$ = %.2f s, %.2f s \n - dt = %.2e s' %(self.N,self.n_cylinders,self.d_mean*10**-6,self.d_var*10**-6,self.gap*10**-6,delta,Delta,self.dt)
        if self.restriction_type == 'Single cylinder' or self.restriction_type == 'Single sphere':
            sim_label += '\n - Radius = %.2e m' %self.single_radius
        elif self.restriction_type == 'Multi cylinder':
            pass
            # sim_label +=  '\n - Add gamma distribution parameters'#'\n - Radius = %.2f m' %self.single_radius
        plt.plot([0],[0],color='white',label=sim_label)

        for i_d in range(n_directions):
            label = 'Gradient direction: %s' %(str(np.round(self.directions[i_d],2)))
            p = plt.errorbar(b,signal_tot[:,i_d],signal_std[:,i_d],marker='.',ms=12,linewidth=2,linestyle='--',alpha=0.8,label=label)

            self.fit_max = np.max(signal_tot[:,i_d])
            # print self.fit_max
            # raw_input('...')

            ####Make exp fit to read out D
            popt, pcov = curve_fit(func, b[b<xlim], signal_tot[:,i_d][b<xlim],p0=[self.D])#,bounds=([self.D-0.01e-9],[self.D+0.01e-9])) #p0=[self.D] is kind of cheating...
            perr = np.sqrt(np.diag(pcov)) #std of D
            print '\t    D from fit: %.2e+-%.2e' %(popt[0],perr)
            color = p[0].get_color()
            plt.plot(b_smooth, func(b_smooth, *popt),linewidth=2, label='Fit: D=(%.2e$\pm$%.2e) m$^2$/s'%(popt[0],perr),c=color)

        #Add theory for free diffusion
        plt.plot(b_smooth, func(b_smooth, self.D), c='gray',linewidth=2, label='Theoretical free diffusion: D=%.2e m$^2$/s' %self.D)


        plt.xlabel('b [s/m$^2$]'); plt.ylabel('S(-bD) []')
        plt.xlim(0,xlim)
        plt.ylim(0-0.05*self.N,self.N+0.05*self.N)
        #Typical b-values available on modern MRI scanners range from 0 to about 4000 s/mm^2 #http://mriquestions.com/what-is-the-b-value.html
        plt.legend(numpoints=1,fontsize=14,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        plt.savefig(self.filename+'_S_vs_b'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

    def PlotPowderAverageVsb(self,signalstuff,delta,Delta):
        print '\t Plotting PowderAverage(b)...'

        gs,signal_tot,signal_std = signalstuff[0][:,0],signalstuff[1],signalstuff[2]
        n_gs, n_directions = np.shape(signal_tot)


        def monoexp(x,a):
            return self.N * np.exp(-x*a)
        def biexp(x,a,b):
            return self.N * np.exp(-x*a) + np.exp(-x*b)

        fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        plt.grid(True)

        xlim = 4e9 #Not optimal

        b = (self.gamma*gs*delta)**2*(Delta-delta/3.)
        b_smooth = np.arange(0,np.max(b),np.max(b)/100.)

        #Add fake line to get entry in the legend
        sim_label = r'\textbf{Specs of simulated data}:'+' \n - %d ensembles  \n - %d cylinders \n - d$\_$mean = %.2e m \n -  d$\_$var = %.2e m \n - gap = %.2e m \n - $\delta,\Delta$ = %.2f s, %.2f s \n - dt = %.2e s' %(self.N,self.n_cylinders,self.d_mean*10**-6,self.d_var*10**-6,self.gap*10**-6,delta,Delta,self.dt)
        if self.restriction_type == 'Single cylinder' or self.restriction_type == 'Single sphere':
            sim_label += '\n - Radius = %.2e m' %self.single_radius
        elif self.restriction_type == 'Multi cylinder':
            pass
            # sim_label +=  '\n - Add gamma distribution parameters'#'\n - Radius = %.2f m' %self.single_radius
        plt.plot([0],[0],color='white',label=sim_label)

        #Add theory for free diffusion
        plt.plot(b_smooth, monoexp(b_smooth, self.D), c='gray',linewidth=2, label='Theoretical free diffusion: D=%.2e m$^2$/s' %self.D)

        powder_average = np.mean(signal_tot,axis=1)
        powder_average_std = np.std(signal_tot,axis=1) #Just quick fix! Not how it should be done!

        label = 'Simulated'
        p = plt.errorbar(b,powder_average,powder_average_std,marker='.',ms=12,linewidth=2,linestyle='--',alpha=0.8,label=label)

        ####Make exp fit to read out D
        popt, pcov = curve_fit(biexp, b[b<xlim], powder_average[b<xlim],p0=[self.D*0.5,self.D*0.5],bounds=([self.D*0.1,self.D*0.1],[self.D,self.D])) #p0=[self.D] is kind of cheating...
        print 'popt', popt
        print 'perr', pcov, pcov[0,0]
        perr = np.sqrt(np.diag(pcov)) #std of D
        print perr, 'perr'
        print '\t    D1 from fit: %.2e+-%.2e' %(popt[0],perr[0])
        print '\t    D2 from fit: %.2e+-%.2e' %(popt[1],perr[1])
        print 'self.N', self.N
        color = p[0].get_color()

        #biexp
        plt.plot(b_smooth, biexp(b_smooth, *popt),linewidth=2, label='Fit: \t D1=(%.2e$\pm$%.2e) m$^2$/s \n\t D2=(%.2e$\pm$%.2e) m$^2$/s'%(popt[0],perr[0],popt[1],perr[1]),c=color)
        #divide biexp into two monoexp
        # plt.plot(b_smooth, monoexp(b_smooth, popt[0]),linewidth=2, label='Fit: D=(%.2e$\pm$%.2e) m$^2$/s'%(popt[0],perr[0,0]))
        # plt.plot(b_smooth, monoexp(b_smooth, popt[1]),linewidth=2, label='Fit: D=(%.2e$\pm$%.2e) m$^2$/s'%(popt[1],perr[1,1]))

        plt.xlabel('b [s/m$^2$]'); plt.ylabel('S(-bD) []')
        plt.xlim(0,xlim+1*10**9)
        plt.yscale('log')
        #Typical b-values available on modern MRI scanners range from 0 to about 4000 s/mm^2 #http://mriquestions.com/what-is-the-b-value.html
        plt.legend(numpoints=1,fontsize=14,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        plt.savefig(self.filename+'_PowderAverage_vs_b'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

    def PlotSignalVsDirection(self,signalstuff,delta,Delta):
        print '\t Plotting signals vs. directions...'

        gs,signal_tot,signal_std = signalstuff[0][:,0],signalstuff[1],signalstuff[2]
        n_gs, n_directions = np.shape(signal_tot)
        b = (self.gamma*gs*delta)**2*(Delta-delta/3.)

        fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        plt.grid(True)


        #Add fake line to get entry in the legend
        sim_label = r'\textbf{Specs of simulated data}:'+' \n - %d ensembles  \n - %d cylinders \n - d$\_$mean = %.2e m \n -  d$\_$var = %.2e m \n - gap = %.2e m \n - $\delta,\Delta$ = %.2f s, %.2f s \n - dt = %.2e s' %(self.N,self.n_cylinders,self.d_mean*10**-6,self.d_var*10**-6,self.gap*10**-6,delta,Delta,self.dt)
        if self.restriction_type == 'Single cylinder' or self.restriction_type == 'Single sphere':
            sim_label += '\n - Radius = %.2e m' %self.single_radius
        elif self.restriction_type == 'Multi cylinder':
            pass
            # sim_label +=  '\n - Add gamma distribution parameters'#'\n - Radius = %.2f m' %self.single_radius
        plt.plot([0],[0],color='white',label=sim_label)

        for i_g in range(n_gs):
            label = 'b = %.2e s/m$^2$' %b[i_g]
            plt.errorbar(np.arange(len(self.directions)),signal_tot[i_g,:],signal_std[i_g,:],marker='.',ms=12,linewidth=2,linestyle='--',alpha=0.8,label=label)

        plt.legend(numpoints=1,fontsize=14,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        #x label stuff
        x = np.arange(len(self.directions))
        # xlabels = [str(np.round(self.directions[i],2)) for i in range(len(self.directions))]
        xlabels = ['[%.2f,%.2f,%.2f]' %(self.directions[i][0],self.directions[i][1],self.directions[i][2]) for i in range(len(self.directions))]
        plt.xticks(x, xlabels, rotation='vertical')
        plt.xlabel('Gradient direction [$\hat{x}$,$\hat{y}$,$\hat{z}$]')

        plt.ylabel('Signal []')
        plt.ylim(0-0.05*self.N,self.N+0.05*self.N)
        plt.savefig(self.filename+'_Signal_vs_directions'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

    def MapLimEffectVsb(self,diff_lim,signal_lims,gs,delta,Delta):

        if len(diff_lim) != 1:
            print '\t The function is only defined for a constant diff_lim...'
            return None

        powder_averages = np.zeros([len(signal_lims),len(gs)])
        powder_average_stds = np.zeros([len(signal_lims),len(gs)])

        for i in range(len(signal_lims)):
            self.signal_lim = signal_lims[i]
            signalstuff = self.RepeatPGSE(delta,Delta,gs)
            gs,signal_tot,signal_std = signalstuff[0][:,0],signalstuff[1],signalstuff[2]
            powder_averages[i,:] = np.mean(signal_tot/np.max(signal_tot),axis=1)
            powder_average_stds[i,:] = np.std(signal_tot,axis=1) #Just quick fix! Not how it should be done!

        plt.figure(figsize=(8,8),facecolor = 'w',edgecolor='w')
        plt.imshow(powder_averages)
        # plt.xlabel('g []')
        # plt.ylabel('diff$\_$lim vs signal$\_$lim')
        # ax = plt.gca()
        # ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.colorbar()
        # plt.show()
        plt.savefig(self.filename+'_Explore_boundary_effects'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

        fig = plt.figure(figsize=(8,8),facecolor = 'w',edgecolor='w')
        plt.imshow(powder_average_stds)
        plt.xlabel('g []')
        plt.ylabel('diff$\_$lim vs signal$\_$lim')
        ax = plt.gca()
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.savefig(self.filename+'_Explore_boundary_effects_stds'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')





    #Compute RootMeanSquaredDistance from initial positions
    def ComputeRMSD(self,n_values=20):
        print '\t Computing msd...'

        rmsd = [None]*(n_values+1)
        rmsd_std = [None]*(n_values+1)
        t = np.zeros(n_values+1)

        rmsd[0],rmsd_std[0],t[0] = 0.,0.,0.

        f = h5py.File(self.filename + '.h5','r')

        p0 = np.array(f.get('tstep0')) #position at time 0

        for i in range(n_values):
            ii = i*int(self.tmax/float(n_values))

            p = np.array(f.get('tstep'+str(ii)))
            d = np.linalg.norm([p0[:,0]-p[:,0],p0[:,1]-p[:,1],p0[:,2]-p[:,2]],axis=0)

            t[i+1] = ii*self.dt
            rmsd[i+1] = np.sqrt(np.mean(d**2))
            rmsd_std[i+1] = np.std(d) / np.sqrt(self.N)

        f.close()

        return [rmsd, rmsd_std, t]

    def ComputeRMSD_rotate(self,rotation,n_values=20):
        print '\t Computing msd...'

        alpha,beta,gamma = rotation[0],rotation[1],rotation[2]

        R_x = np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
        R_y = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
        R_z = np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])

        # rot = R_x*R_y*R_z #NO!!! Causes assymetry
        rot = np.dot(np.dot(R_x,R_y),R_z)

        rmsd = [None]*(n_values+1)
        rmsd_std = [None]*(n_values+1)
        t = np.zeros(n_values+1)

        rmsd[0],rmsd_std[0],t[0] = 0.,0.,0.

        f = h5py.File(self.filename + '.h5','r')

        p0 = np.dot(np.array(f.get('tstep0')),rot) #position at time 0

        for i in range(n_values):
            ii = i*int(self.tmax/float(n_values))

            p = np.dot(np.array(f.get('tstep'+str(ii))),rot)
            d = np.linalg.norm([p0[:,0]-p[:,0],p0[:,1]-p[:,1],p0[:,2]-p[:,2]],axis=0)

            t[i+1] = ii*self.dt
            rmsd[i+1] = np.sqrt(np.mean(d**2))
            rmsd_std[i+1] = np.std(d) / np.sqrt(self.N)

        f.close()

        return [rmsd, rmsd_std, t]

    def PlotRMSD(self,rmsdstuff,add=''):
        print '\t Plotting msd...'

        rmsd,rmsd_std,t = rmsdstuff[0],rmsdstuff[1],rmsdstuff[2]

        t = np.sqrt(t)

        fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        plt.grid(True)
        plt.xlabel('$\sqrt{t}$ [$s$]')
        plt.ylabel('rmsd [m]')

        #Plot data points
        #t = np.arange(0,self.duration+self.duration/self.tmax,self.duration/self.tmax)
        label = 'Data from %d ensembles over %.2f s with %d timesteps' %(self.N,self.duration,self.tmax)
        plt.errorbar(t,rmsd,rmsd_std, marker='.', ms=9, alpha=0.7, linestyle='--',label=label)#,color=self.color)

        def line(x,slope,intercept):
            return x*slope + intercept
        results = sm.OLS(rmsd,t).fit()
        slope, slope_std, intercept, intercept_std = results.params[-1], results.bse[-1], 0, 0
        D, D_std = slope**2/6.,1/3.*slope*slope_std #D_std has been calculated based on the law of combination of errors (4.14 in Barlow)
        print '\t    This slope corresponds to D=%.3e+-%.2e' %(slope**2/6.,1/3.*slope*slope_std)
        print '\t    The data was generated with D=%.3e' %self.D
        #Plot the fit
        #label = 'Fit: (%.3e $\pm$ %.3e) $\cdot$ $\sqrt{t}$ + (%.3e $\pm$ %.3e)' %(slope, slope_std, intercept, intercept_std)
        label = 'Fit: $\sqrt{6Dt}$, with D = (%.3e $\pm$ %.3e)m/s$^2$' %(D, D_std)
        plt.plot(t,line(t,slope,intercept),color='gray',lw=4,alpha=0.8,label=label)

        plt.legend(numpoints=1,loc='lower right')
        plt.xlim([0,np.max(t)])
        plt.ylim([0,np.max(rmsd)])

        plt.savefig(self.filename+'_rmsd_vs_rt'+add+'.png',bbox_inches='tight',dpi=100)
        # plt.close('all')

    def PlotRMSD_several(self,rmsdstuff,add='',initialize=True,savenclose=True):
        print '\t Plotting msd...'

        rmsd,rmsd_std,t = rmsdstuff[0],rmsdstuff[1],rmsdstuff[2]

        t = np.sqrt(t)

        if initialize == True:
            fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
            plt.grid(True)
            plt.xlabel('$\sqrt{t}$ [$s$]')
            plt.ylabel('rmsd [m]')

        #Plot data points
        #t = np.arange(0,self.duration+self.duration/self.tmax,self.duration/self.tmax)
        label = 'Data from %d ensembles over %.2f s with %d timesteps' %(self.N,self.duration,self.tmax)
        plt.errorbar(t,rmsd,rmsd_std, marker='.', ms=9, alpha=0.7, linestyle='--',label=label+add[1:])#,color=self.color)

        def line(x,slope,intercept):
            return x*slope + intercept
        results = sm.OLS(rmsd,t).fit()
        slope, slope_std, intercept, intercept_std = results.params[-1], results.bse[-1], 0, 0
        D, D_std = slope**2/6.,1/3.*slope*slope_std #D_std has been calculated based on the law of combination of errors (4.14 in Barlow)
        print '\t    This slope corresponds to D=%.3e+-%.2e' %(slope**2/6.,1/3.*slope*slope_std)
        print '\t    The data was generated with D=%.3e' %self.D
        #Plot the fit
        #label = 'Fit: (%.3e $\pm$ %.3e) $\cdot$ $\sqrt{t}$ + (%.3e $\pm$ %.3e)' %(slope, slope_std, intercept, intercept_std)
        label = 'Fit: $\sqrt{6Dt}$, with D = (%.3e $\pm$ %.3e)m/s$^2$' %(D, D_std)
        plt.plot(t,line(t,slope,intercept),color='gray',lw=4,alpha=0.8,label=label+add[1:])


        if savenclose == True:
            plt.legend(numpoints=1,loc='lower right')
            plt.xlim([0,np.max(t)])
            plt.ylim([0,np.max(rmsd)])

            plt.savefig(self.filename+'_rmsd_vs_rt'+add+'.png',bbox_inches='tight',dpi=100)
            # plt.close('all')

    def HistogramsOfRotatedPaths(self,timesteps,rotations,bins=40,initialize=True):
        # print '\t Computing msd...'

        def DoPlotStuff(data,lc,histlabel='NaN',fitlabel='NaN'):
            # distribution fitting
            parameters = scipy.stats.norm.fit(data);
            mean = parameters[0]; std = parameters[1]
            #Set large limits
            xlims = [-8*std+mean, 8*std+mean]
            #Plot histogram
            histdata = plt.hist(data,bins=np.arange(-150.,150.,3.),alpha=.5,label=histlabel,color=lc)
            #Get histogram data, in this case bin edges
            xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in range(len(histdata[1])-1)]
            #Get bin width from this
            binwidth = (max(xh) - min(xh)) / len(histdata[1])

            #stuff
            x_values = histdata[1][:-1]+0.5*binwidth
            occurences = histdata[0]

            #Get Y points via Normal PDF with fitted parameters
            pdf_fitted = scipy.stats.norm.pdf(x_values,loc=mean,scale=std)
            #Scale the fitted PDF by area of the histogram
            pdf_fitted = pdf_fitted * (len(data) * binwidth)

            # plt.plot(x_values,occurences,marker='.',ms=12)

            chi2 = np.sum( (occurences-pdf_fitted)**2/pdf_fitted )

            # plt.plot(x_values,pdf_fitted)

            # print x
            # print x_values

            plt.plot(x_values,pdf_fitted,lc,alpha=0.9,lw=2,label=fitlabel+' $\chi^2$=%f' %chi2)

            plt.hold(True)

            return pdf_fitted
        #
        # R_x = np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
        # R_y = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
        # R_z = np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])
        #
        # rot = R_x*R_y*R_z

        # if initialize == True:
        #     fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        #     plt.grid(True)
        #     plt.xlabel('Postition [$\mu m$]')
        #     plt.ylabel(r'Occurence [#]')

        # rmsd[0],rmsd_std[0],t[0] = 0.,0.,0.

        f = h5py.File(self.filename + '.h5','r')

        # p0 = np.dot(np.array(f.get('tstep0')),rot) #position at time 0
        # i = 0

        for timestep in timesteps:
            plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
            plt.grid(True)
            plt.xlabel('Postition [$\mu m$]')
            plt.ylabel(r'Occurence [#]')

            colors = ['r','b','g','gray','black']
            ci = 0

            for rotation in rotations:

                alpha,beta,gamma = rotation[0],rotation[1],rotation[2]

                R_x = np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
                R_y = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
                R_z = np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])

                rot = R_x*R_y*R_z
                rot = np.dot(np.dot(R_x,R_y),R_z)


            # for i in range(n_values):
                # ii = i*int(self.tmax/float(n_values))
                ii = timestep

                data = np.dot(np.array(f.get('tstep'+str(ii))),rot)*10**6
                # d = np.linalg.norm([p0[:,0]-p[:,0],p0[:,1]-p[:,1],p0[:,2]-p[:,2]],axis=0)
                label=histlabel=fitlabel='x, alpha=%.2f at t=%d' %(alpha,timestep)
                DoPlotStuff(data[:,1],lc = colors[ci],histlabel=label,fitlabel=label)
                # DoPlotStuff(np.random.uniform(-1,1,1000),lc = colors[ci],histlabel=label,fitlabel=label)
                # DoPlotStuff(data[:,1],lc = 'g',histlabel='y at t=%d' %timestep,fitlabel='y at t=%d' %timestep)
                # DoPlotStuff(data[:,2],lc = 'b',histlabel='z at t=%d' %timestep,fitlabel='z at t=%d' %timestep)
                ci += 1


                # plt.hist(p[:,0],bins=bins,label='x',alpha=0.7)
                # plt.hist(p[:,1],bins=bins,label='y',alpha=0.7)
                # plt.hist(p[:,2],bins=bins,label='z',alpha=0.7)
            plt.legend()
            plt.show()
            raw_input('...')

                # t[i+1] = ii*self.dt
                # rmsd[i+1] = np.sqrt(np.mean(d**2))
                # rmsd_std[i+1] = np.std(d) / np.sqrt(self.N)

        f.close()

        # return [rmsd, rmsd_std, t]

    def ExperimentOnSolidBoundaryEffect(self,diff_lims,delta_in,Delta_in,g_in):

        drows, dcols = np.shape(self.directions)
        g,signal_tot,signal_std = np.zeros([len(diff_lims),drows]),np.zeros([len(diff_lims),drows]),np.zeros([len(diff_lims),drows])
        i = 0

        for current_diff_lim in range(len(diff_lims)):
            self.diff_lim = diff_lims[current_diff_lim]
            self.free_diffusion()
            # self.PlotPaths(10,200,twoD=True,threeD=False)
            # raw_input('...')
            # g[i],signal_tot[i],signal_std[i] = self.RepeatPGSE(delta_in,Delta_in,g_in,self.filename)
            g_temp,s_tot_temp,s_std_temp = self.RepeatPGSE(delta_in,Delta_in,g_in,self.filename)
            g[i],signal_tot[i],signal_std[i] = g_temp[0],s_tot_temp[0],s_std_temp[0]
            #self.PlotPaths(10,200,twoD=False,threeD=True)
            i += 1
            # raw_input('...')

        #Plot
        fig = plt.figure(figsize=(16,8),facecolor = 'w',edgecolor='w')
        plt.grid(True)
        for i in range(len(diff_lims)):
            # print np.shape(self.directions), np.shape(signal_tot), np.shape(signal_std)
            # raw_input('...')
            x = np.arange(0,len(self.directions))
            plt.errorbar(x,signal_tot[i,:],signal_std[i,:],marker='.',ms=12,linewidth=2,linestyle='--',alpha=0.8,label='diff lim = %.2f' %diff_lims[i])
        # plt.errorbar(diff_lims,signal_tot,signal_std,marker='.',ms=12,linewidth=2,linestyle='--',alpha=0.8)
        plt.xlabel('Direction []'); plt.ylabel('Signal []')
        #plt.xlim(0,xlim)
        plt.legend(numpoints=1,fontsize=14,loc='middle right')
        plt.savefig(self.filename+'_S_vs_diff_lim'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

        for i in range(len(diff_lims)):
            self.PlotDirections(lengths=signal_tot[i,:],add='_scaled_g'+str(g[i]))

    def PlotMagneticField_old(self):

        # a = np.linspace(0,2*np.pi,50) #start,stop,number of sample to generate
        # b = np.linspace(0,40*10e-6,10)
        # A, B = np.meshgrid(a, b)

        # c = 2*np.pi*B_0*np.sin(alpha)*dchi*np.log(r2/r1)*(np.cos(phi)-r*np.sin(phi))

        #From equation [10] in Sukstanskii and Yablonskiy
        # def h():

        import matplotlib.cm as cm
        ax = plt.subplot(111, polar=True)

        dchi, chi_par, chi_perp = 1., 1., 1.
        B_0, alpha, r1, r2 = 3, 1./10.*np.pi, 20*10e-6, 25*10e-6

        a = np.linspace(0,2*np.pi,50) #start,stop,number of sample to generate
        b = np.linspace(r2,40*10e-6,50)
        phi, r = np.meshgrid(a, b)
        c_outside = 2*np.pi*B_0*np.sin(alpha) * (chi_par+chi_perp)/2. * (r2**2-r1**2)/r**2 * np.cos(2*phi) #h_|| r>r2
        ctf = ax.contourf(a, b, c_outside, cmap=cm.jet)

        a = np.linspace(0,2*np.pi,50) #start,stop,number of sample to generate
        b = np.linspace(0,r1,50)
        phi, r = np.meshgrid(a, b)
        c_inside = 2*np.pi*B_0*np.sin(alpha)*dchi*np.log(r2/r1)*np.ones(phi.shape) #h_|| r<r1
        ctf = ax.contourf(a, b, c_inside, cmap=cm.jet)


        #actual plotting
        # ax.set_yticklabels([])
        xT=plt.xticks()[0]
        xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
        plt.xticks(xT, xL)
        ax.set_ylim(0,40*10e-6)
        plt.colorbar(ctf)
        plt.show()

    def PlotMagneticField(self,coordinates='polar'):

        # a = np.linspace(0,2*np.pi,50) #start,stop,number of sample to generate
        # b = np.linspace(0,40*10e-6,10)
        # A, B = np.meshgrid(a, b)

        # c = 2*np.pi*B_0*np.sin(alpha)*dchi*np.log(r2/r1)*(np.cos(phi)-r*np.sin(phi))

        #From equation [10] in Sukstanskii and Yablonskiy
        # def h():

        import matplotlib.cm as cm
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

        dchi, chi_par, chi_perp = 0.0005, 0.0005, 0.0005
        B_0, alpha, r1, r2 = 1., 5./10.*np.pi, 19*10e-6, 21*10e-6
        B_0_perp = B_0*np.sin(alpha)

        if coordinates == 'polar':
            dim = 50

            plt.figure()
            ax = plt.subplot(111, polar=True)
            # plt.ticklabel_format(useOffset=False)
            # axx = plt.gca()
            # axx.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # plt.ticklabel_format(useOffset=False)

            # dchi, chi_par, chi_perp = 1., 1., 1.
            # B_0, alpha, r1, r2 = 1., 1./10.*np.pi, 20*10e-6, 25*10e-6
            # B_0_perp = B_0*np.sin(alpha)

            a = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
            b = np.linspace(r2,50*10e-6,dim)
            phi, r = np.meshgrid(a, b)
            # c_outside = -2*np.pi*B_0*np.sin(alpha) * (chi_par+chi_perp)/2. * (r2**2-r1**2)/r**2 * (np.cos(phi) + np.sin(phi))

            h_par_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/r**2 * np.cos(2*phi)
            h_perp_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/r**2 * np.sin(2*phi)

            # c_outside = np.linalg.norm([h_par_out,h_perp_out],axis=0)
            # c_outside = np.sqrt(h_par_out**2+h_perp_out**2)
            c_outside = (h_par_out+h_perp_out) #/ np.max(np.max(h_par_out+h_perp_out))
            # print np.max(np.max(c_outside))
            # c_outside = h_par_out
            # c_outside = h_perp_out
            # plt.imshow(c_outside)
            # plt.show()

            ctf1 = ax.contourf(a, b, c_outside, cmap=cm.jet)#,vmin=-1,vmax=1)

            a = np.linspace(0,2*np.pi,dim) #start,stop,number of sample to generate
            b = np.linspace(0,r1,dim)
            phi, r = np.meshgrid(a, b)
            # c_inside = 2*np.pi*B_0*np.sin(alpha) * dchi * np.log(r2/r1) * (np.cos(phi) - np.sin(phi))
            h_par_in = 2*np.pi*B_0_perp*dchi*np.log(r2/r1) * np.ones([dim,dim])
            h_perp_in = 0.
            # c_inside = np.linalg.norm([h_par_in,h_perp_in])
            c_inside = (h_par_in + h_perp_in) #/ np.max(np.max(h_par_out+h_perp_out))
            ctf2 = ax.contourf(a, b, c_inside, cmap=cm.jet)#,vmin=0,vmax=1)

            #actual plotting
            # ax.set_yticklabels([])
            xT=plt.xticks()[0]
            xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
            plt.xticks(xT, xL)
            ax.set_ylim(0,50*10e-6)
            cb = plt.colorbar(ctf1)
            cb.set_label('Normalized intensity of $\mathbf{h}$')
            plt.savefig('MagneticField'+'.png',bbox_inches='tight',dpi=100)
            # plt.show()

        if coordinates == 'cartesian':
            dim = 50

            plt.figure()
            ax = plt.subplot(111)



            #outside
            r_outside = np.linspace(r2,50*10e-6,dim)
            theta = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
            r_matrix, phi_matrix = np.meshgrid(r_outside,theta)
            X = r_matrix * np.cos(phi_matrix)
            Y = r_matrix * np.sin(phi_matrix)
            h_par_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(X**2+Y**2) * (2*X**2/(X**2+Y**2)-1)
            h_perp_out = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2) * (2*X*Y)/(X**2+Y**2)**2
            c_outside = (h_par_out+h_perp_out)#/ np.max(np.max(h_par_out+h_perp_out))
            ctf1 = ax.contourf(X+0., Y+0., c_outside,100, cmap=cm.jet)#,vmin=-1,vmax=1)
            # ax.contour(X+0., Y+0., c_outside,10)

            xx, yy = 0.+0.00037,0.+0.00037
            def afunc(x,y):
                par = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(x**2+y**2) * (2*x**2/(x**2+y**2)-1)
                perp = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(x**2+y**2) * (2*x*y)/(x**2+y**2)
                return par+perp/ np.max(np.max(h_par_out+h_perp_out))
            # par = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(xx**2+yy**2) * (2*xx**2/(xx**2+yy**2)-1)
            # perp = 2*np.pi*B_0_perp * (chi_par+chi_perp)/2. * (r2**2-r1**2)/(xx**2+yy**2) * (2*xx*yy)/(xx**2+yy**2)
            print afunc(xx,yy)
            plt.plot(xx,yy,ms=10,marker='.')
            # plt.imshow(c_outside)

            #inside
            r_inside = np.linspace(0,r1,dim)
            theta = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
            r_matrix, phi_matrix = np.meshgrid(r_inside,theta)
            X = r_matrix * np.cos(phi_matrix)
            Y = r_matrix * np.sin(phi_matrix)
            h_par_in = 2*np.pi*B_0_perp*dchi*np.log(r2/r1) * np.ones([dim,dim])
            h_perp_in = 0.
            c_inside = (h_par_in + h_perp_in) #/ np.max(np.max(h_par_out+h_perp_out))
            ctf2 = ax.contourf(X+0., Y+0., c_inside,100, cmap=cm.jet)#,vmin=0,vmax=1)

            cb = plt.colorbar(ctf1)
            cb.set_label('Normalized intensity of $\mathbf{h}$')
            plt.xlabel('x []'); plt.ylabel('y []')
            plt.savefig('MagneticField'+'.png',bbox_inches='tight',dpi=100)
            plt.show()

        if coordinates == 'CylindricalWartonBowtell':

            dim = 50
            plt.figure()
            ax = plt.subplot(111, polar=True)

            # B_0, theta, r_i, r_o, chi_A = 1., 5./10.*np.pi, 19*10e-6, 21*10e-6, 0.0005
            B_0, theta, r_i, r_o, chi_A = 1., 5./10.*np.pi, 15*10e-6, 25*10e-6, 0.0005

            def FieldPerturb_A_out(r_min,r_max):
                a = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                b = np.linspace(r_min,r_max,dim)
                phi, r = np.meshgrid(a, b)
                field = chi_A*B_0*np.sin(theta)**2*np.cos(2*phi)/8. * (r_o**2-r_i**2)/r**2
                return a,b,field

            def FieldPerturb_A_in(r_min,r_max):
                a = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                b = np.linspace(r_min,r_max,dim)
                phi, r = np.meshgrid(a, b)
                field = 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i) * np.ones([dim,dim])
                return a,b,field

            def FieldPerturb_A_mem(r_min,r_max):
                a = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                b = np.linspace(r_min,r_max,dim)
                phi, r = np.meshgrid(a, b)
                field = chi_A*B_0*(np.sin(theta)**2 * (-5./12.-np.cos(2*phi)/8.*(1+r_i**2/r**2) + 3./4.*np.log(r_o/r)) - np.cos(theta**2)/6.)
                return a,b,field


            ####Outside
            a,b,field = FieldPerturb_A_out(r_o,50*10e-6)
            ctf1 = ax.contourf(a, b, field, 100, cmap=cm.jet)
            ####Inside
            a,b,field = FieldPerturb_A_in(0,r_i)
            ctf2 = ax.contourf(a, b, field, 100, cmap=cm.jet)
            ####Membrane
            a,b,field = FieldPerturb_A_mem(r_i,r_o)
            ctf3 = ax.contourf(a, b, field, 100, cmap=cm.jet)



            xT=plt.xticks()[0]
            xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
            plt.xticks(xT, xL)
            ax.set_ylim(0,50*10e-6)
            cb = plt.colorbar(ctf1)
            cb.set_label('Normalized intensity of $\mathbf{h}$')
            # plt.savefig('MagneticField'+'.png',bbox_inches='tight',dpi=100)

        if coordinates == 'CartesianWartonBowtell':

            dim = 50
            plt.figure()
            ax = plt.subplot(111)

            # B_0, theta, r_i, r_o, chi_A = 1., 5./10.*np.pi, 19*10e-6, 21*10e-6, 0.0005
            B_0, theta, r_i, r_o, chi_A = 1., 1./10.*np.pi, 15*10e-6, 25*10e-6, 0.0005

            def FieldPerturb_A_out(r_min,r_max):
                phi = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                r = np.linspace(r_min,r_max,dim)
                r_matrix, phi_matrix = np.meshgrid(r,phi)
                X = r_matrix * np.cos(phi_matrix)
                Y = r_matrix * np.sin(phi_matrix)
                field = chi_A*B_0*np.sin(theta)**2*0.125*(r_o**2-r_i**2) * 1./(X*X+Y*Y) * (2*X*X/(X*X+Y*Y)-1)
                return X,Y,field

            def FieldPerturb_A_in(r_min,r_max):
                phi = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                r = np.linspace(r_min,r_max,dim)
                r_matrix, phi_matrix = np.meshgrid(r,phi)
                X = r_matrix * np.cos(phi_matrix)
                Y = r_matrix * np.sin(phi_matrix)
                field = 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i) * np.ones([dim,dim])
                return X,Y,field

            def FieldPerturb_A_mem(r_min,r_max):
                phi = np.linspace(0,2*np.pi,dim) #start,stop,number of samples to generate
                r = np.linspace(r_min,r_max,dim)
                r_matrix, phi_matrix = np.meshgrid(r,phi)
                X = r_matrix * np.cos(phi_matrix)
                Y = r_matrix * np.sin(phi_matrix)
                field = chi_A*B_0*(np.sin(theta)**2 * (-5./12.- (1*X*X/(X*X+Y*Y))*0.125*(1+r_i*r_i/(X*X+Y*Y)) + 3./4.*np.log(r_o/r)) - np.cos(theta**2)/6.)
                return X,Y,field


            ####Outside
            X,Y,field = FieldPerturb_A_out(r_o,50*10e-6)
            ctf1 = ax.contourf(X, Y, field, 100, cmap=cm.jet)
            ####Inside
            X,Y,field = FieldPerturb_A_in(0,r_i)
            ctf2 = ax.contourf(X, Y, field, 100, cmap=cm.jet)
            ####Membrane
            X,Y,field = FieldPerturb_A_mem(r_i,r_o)
            ctf3 = ax.contourf(X, Y, field, 100, cmap=cm.jet)

            cb = plt.colorbar(ctf1)
            cb.set_label('Some intensity of $\mathbf{h}$')
            # plt.savefig('MagneticField'+'.png',bbox_inches='tight',dpi=100)
            plt.show()

    def PlotMultiCylinderMagneticField(self):

        mem_width = 0.2 #width of all membranes
        B_0, theta, chi_A = 1., 5./10.*np.pi,self.chi_A# 0.00001
        lim,delta = self.diff_lim,800
        x = y = np.linspace(-lim,lim,delta)

        field = np.zeros([delta,delta])

        centers = self.pos#[:2]

        threshold = 0.#0.1*10**-16

        # g = 0.6 #from the internet
        r_os = 2.*self.d/(1.+self.g)
        r_is = 2.*self.d/(1./self.g+1.)

        for center,r_o,r_i in zip(centers,r_os,r_is):
            X,Y = np.meshgrid(x,y)
            X -= center[0]
            Y -= center[1]

            contribution = chi_A*B_0*np.sin(theta)**2*0.125*(r_o**2-r_i**2) * 1./(X*X+Y*Y) * (2*X*X/(X*X+Y*Y)-1)
            field[abs(contribution)>threshold] += contribution[abs(contribution)>threshold]

        for center,r_o,r_i in zip(centers,r_os,r_is):
            X,Y = np.meshgrid(x,y)
            X -= center[0]
            Y -= center[1]

            inside = np.sqrt(X*X+Y*Y) < r_o - (r_o-r_i)/2.

            field[inside] = 3*chi_A*B_0*np.sin(theta)**2/4. * np.log(r_o/r_i) * np.ones([delta,delta])[inside]

        plt.figure(figsize=(12.45,10),facecolor = 'w',edgecolor='w')
        ax = plt.subplot(111)
        X,Y = np.meshgrid(x,y)
        # plot = ax.contourf(X, Y, field, 100, cmap=cm.bone)
        plot = ax.contourf(X, Y, field, 100, cmap=cm.jet)
        # plot = ax.contourf(X, Y, field, 100, cmap=cm.rainbow)
        cb = plt.colorbar(plot)
        cb.set_label('Some intensity of $\Delta\mathbf{B}$')
        plt.axis('equal')
        plt.xlabel('x [$\mu$m]'); plt.ylabel('y [$\mu$m]')
        plt.savefig(self.filename+'multi_cylinder_susceptibility_field'+'.png',bbox_inches='tight',dpi=100)
        plt.close('all')

    def single_cylinder_diffusion_with_susceptibility(self):
        print '\t Initializing coordinates...'

        if 2 == 2 or solid_boundary == 'cube': #!!!!!!!!!!!!!!!!!!!!!!!!!!
            ####Uniformly distributed in CUBE
            self.path = np.random.uniform(-self.diff_lim*10**-6,self.diff_lim*10**-6,[self.N,3])

        #Assign each walker to  cylinder or to extracylinder space
        assigned_space = np.array([0]*self.N) #0 will be extracylinder space
        inside =  np.linalg.norm(self.path[:,:2],axis=1) < self.single_radius
        assigned_space[inside] = 1 #0 will be extracylinder space

        print '\t Doing random walk...'

        time_start = time.time()

        #### Open the file that will be written to
        f = h5py.File(self.filename + '.h5','w')
        f.create_dataset('info',data=[self.gamma,self.D,self.N,self.duration,self.tmax,self.dt,self.step_size])
        f.create_dataset('tstep0',data=self.path)

        Trues = np.asarray([True]*self.N)

        for i in range(1,self.tmax):
            if i%100 == 0:
                print '\t    %.2f percent done after %.2f minutes' %(float(i)/self.tmax*100,(time.time()-time_start)/60.)

            coord_updates = np.random.normal(0,1,[self.N,3])
            coord_updates_lengths = np.tile(np.linalg.norm(coord_updates,axis=1),(3,1)) #get the length of each coordinate and repeat it 3 times
            self.path += coord_updates/coord_updates_lengths.T * self.step_size #Normalize step length to equal the step_size

            dist_from_0 = np.linalg.norm(self.path[:,:2],axis=1) #in x,y direction

            ####Permeability check
            current_space = np.zeros(self.N) #0 will be extracylinder space
            inside =  np.linalg.norm(self.path[:,:2],axis=1) < self.single_radius
            current_space[inside] = 1 #0 will be extracylinder space

            permeate = np.random.uniform(0,1,[self.N]) < self.permeability #if permeability is high, many walkers are not corrected back to previous position
            crossed = current_space != assigned_space
            change = crossed-permeate*crossed #Of those who crossed, those who permeated should not be corrected
            self.path[change] = np.array(f.get('tstep'+str(i-1)))[change] #if boundary is crossed, move back to previous position
            #update assigned_space
            assigned_space[True-change] = current_space[True-change] #Those that were NOT corrected, should now be assigned to their new space

            ####Solid boundary check
            #check if walkers in the extracylinder space crossed the diff_lim in the x-direction
            inside_diff_lim = np.abs(self.path[:,0]) > self.diff_lim*10**-6
            self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position
            #check if walkers in the extracylinder space crossed the diff_lim in the y-direction
            inside_diff_lim = np.abs(self.path[:,1]) > self.diff_lim*10**-6
            self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position
            #check if walkers in the extracylinder space crossed the diff_lim in the z-direction
            inside_diff_lim = np.abs(self.path[:,2]) > self.diff_lim*10**-6
            self.path[inside_diff_lim == Trues] = np.array(f.get('tstep'+str(i-1)))[inside_diff_lim == Trues] #if boundary is crossed, move back to previous position

            f.create_dataset('tstep'+str(i),data=self.path)
            f.create_dataset('assigned_space'+str(i),data=assigned_space)

        f.close()

    def PathHistogram2d(self,filename,timesteps,rotations):

        lim = 0.0002
        bins = 100

        def GaussFunc(mu_x,mu_y,sigma_x,sigma_y):
            x_var = np.linspace(-lim,lim,bins)
            y_var = np.linspace(-lim,lim,bins)
            x,y = np.meshgrid(x_var,y_var)
            # return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            gauss = np.exp( -( (x-mu_x)**2/(2*sigma_x**2) + (y-mu_y)**2/(2*sigma_y**2) ) )
            return x,y,gauss

        from matplotlib.colors import LogNorm

        f = h5py.File(filename + '.h5','r')

        for i in timesteps:
            for rotation in rotations:

                alpha,beta,gamma = rotation[0],rotation[1],rotation[2]

                R_x = np.array([[1,0,0],[0,np.cos(alpha),-np.sin(alpha)],[0,np.sin(alpha),np.cos(alpha)]])
                R_y = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
                R_z = np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])

                # rot = R_x*R_y*R_z
                rot = np.dot(np.dot(R_x,R_y),R_z)


                xyz = np.array(f.get('tstep'+str(i)))
                xyz = np.dot(xyz,rot)

                plt.figure()
                h,_,_,_ = plt.hist2d(xyz[:,0], xyz[:,1], bins=bins,norm=LogNorm()) #xy

                plt.xlim(-lim,lim); plt.ylim(-lim,lim)
                plt.colorbar()


                #fit
                sigma_x = np.std(xyz[:,0])
                sigma_y = np.std(xyz[:,1])
                mu_x = np.mean(xyz[:,0])
                mu_y = np.mean(xyz[:,1])

                x,y,gauss = GaussFunc(mu_x,mu_y,sigma_x,sigma_y)
                print 'gauss', np.max(gauss)
                gauss_n = gauss/np.max(gauss)
                print np.shape(gauss)
                print np.max(gauss_n)
                plt.figure()

                # plt.contourf(x,y,gauss_n,100)
                plt.imshow(gauss_n)
                plt.title('Normalized Gauss fit'+' rotation: '+str(rotation))
                plt.colorbar()

                plt.figure()
                print 'h', np.max(h)
                h_n  = h/np.max(h)
                print np.max(h_n)
                # plt.contourf(x,y,h_n,100)
                plt.imshow(h_n)
                plt.title('Normalized hist data'+' rotation: '+str(rotation))
                plt.colorbar()

                plt.figure()
                plt.imshow(h_n-gauss_n)
                plt.title('Residuals (data-fit)'+' rotation: '+str(rotation))
                plt.colorbar()






                # plt.figure()
                # plt.hist2d(xyz[:,0], xyz[:,2], bins=40,norm=LogNorm()) #xz
                # plt.xlim(-0.00006,0.00006); plt.ylim(-0.00006,0.00006)
                # plt.colorbar()
                # plt.figure()
                # plt.hist2d(xyz[:,1], xyz[:,2], bins=40,norm=LogNorm()) #yz
                # plt.xlim(-0.00006,0.00006); plt.ylim(-0.00006,0.00006)
                # plt.colorbar()
            plt.axis('equal')
            plt.show()

            raw_input('...')

    def numerical_estimation_of_B(self):

        chi_i,chi_o,chi_m = 0.,0.,-1*10**-5 #susceptibility inside, outside, membrane

        width,bins = 20*10**-6,100
        res = dx = width/bins

        C,dtau = 1.,0.000001 ###Should be???

        print '\t The resolution is %.2e' %res

        r = 8*10**-6 #[m]
        r_o = 1.*r/(1.+self.g)
        r_i = 1.*r/(1./self.g+1.)
        print r_o,r,r_i

        #generate the substrate
        chi_map = np.ones([bins,bins]) * chi_o

        x = np.linspace(-width/2.,width/2.,bins)
        y = np.linspace(-width/2.,width/2.,bins)
        X,Y = np.meshgrid(x,y)

        chi_map[(X**2+Y**2 < r_o**2) * (X**2+Y**2 > r_i**2)] = chi_m
        chi_map[X**2+Y**2 <= r_i**2] = chi_i

        #pad the substrate
        #semi-redundant in this simple case...
        chi_map_padded = np.zeros([bins+2,bins+2])
        chi_map_padded[1:-1,1:-1] = chi_map
        # chi_map_padded[:,0],chi_map_padded[:,-1] = chi_map_padded[:,1],chi_map_padded[:,-2] #pad columns
        # chi_map_padded[0,:],chi_map_padded[-1,:] = chi_map_padded[1,:],chi_map_padded[-2,:] #pad rows
        # chi_map_padded[:,0],chi_map_padded[:,-1] = 0.,0.
        # chi_map_padded[0,:],chi_map_padded[-1,:] = 0.,0.

        #plot the substrate
        plt.figure(figsize=(8,8))
        fig = plt.imshow(chi_map,origin='lower')

        nt = 4 #make tick every nt
        xlabels = ['%.2e' %x[i] for i in range(len(x))]
        plt.xticks(np.arange(len(x))[::nt],xlabels[::nt],rotation=90,visible=True)
        plt.xlabel('x [m]')
        ylabels = ['%.2e' %y[i] for i in range(len(y))]
        plt.yticks(np.arange(len(y))[::nt],ylabels[::nt],rotation=0)
        plt.ylabel('y [m]')

        plt.colorbar(fig,fraction=0.046, pad=0.04)
        plt.show()

        # Bis = 2*(1+chi_map) #B for each element in grid

        def get_Bis(row_shift,col_shift):
            return dtau/(C*dx**2) * 2*(1+chi_map_padded[1+col_shift:-1+bins+2+col_shift,1+row_shift:-1+bins+2+row_shift])*(1+chi_map_padded[1:-1,1:-1]) / ((1+chi_map_padded[1+col_shift:-1+bins+2+col_shift,1+row_shift:-1+bins+2+row_shift]) + (1+chi_map_padded[1:-1,1:-1]))

        row_shifts = [0,1,0,-1]#[[0,-1],[1,0],[0,1],[-1,0]]
        col_shifts = [-1,0,1,0]

        B5s = np.zeros(chi_map.shape)
        for row_shift,col_shift in zip(row_shifts,col_shifts):
            B5s += get_Bis(row_shift,col_shift)

        Bis = get_Bis(0,0) #no shift

        Dis = 2*Bis / (2-B5s)
        D5s = (2+B5s) / (2-B5s)

        #### Do something with rolling indicies!!! To only have to store two timepoints at a time

        def Phi_Ms(current_map):
            # print np.shape(current_map[:-2,1:-1])
            # print np.shape(current_map[1:-1,2:])
            # print np.shape(current_map[2:,1:-1])
            # print np.shape(current_map[1:-1,:-2])
            return current_map[:-2,1:-1]+current_map[1:-1,2:]+current_map[2:,1:-1]+current_map[1:-1,:-2]

        iterations = 1000

        Phi_map_padded = np.ones([bins+2,bins+2,iterations+1]) #Stupid. Do the rolling indicies thing!
        Phi_map_padded[:,0],Phi_map_padded[:,-1] = 0.,0.
        Phi_map_padded[0,:],Phi_map_padded[-1,:] = 0.,0.

        for tau in range(1,iterations):
            # print 'tau: ', tau
            #padding should be updated!!!!!
            Phi_map_padded[1:-1,1:-1,tau+1] = D5s * (Phi_Ms(Phi_map_padded[:,:,tau]) + Phi_Ms(Phi_map_padded[:,:,tau-1]))

            if tau%100 == 0:
                plt.figure
                plt.imshow(Phi_map_padded[:,:,tau])
                plt.colorbar()
                plt.title(r'$\tau$=%d' %tau)
                plt.show()

        print Phi_map_padded


        # B1s = 2*(1+chi_map_padded[0:-2,1:-1])*(1+chi_map_padded[1:-1,1:-1]) / ((1+chi_map_padded[0:-2,1:-1]) + (1+chi_map_padded[1:-1,1:-1])

        # Bnn = 2*(1+chi_map_padded[0:-2,1:-1])*(1+chi_map_padded[1:-1,1:-1]) / ((1+chi_map_padded[0:-2,1:-1]) + (1+chi_map_padded[1:-1,1:-1]) #summed over nearest neighbors #4nn since 2d

        #calculate Phi distribution

    def numerical_estimation_of_B2(self):

        from scipy import ndimage

        #set specs
        Chi_i,Chi_o,Chi_m = 0.,0.,-1*10**-5 #susceptibility inside, outside, membrane

        width,bins = 30*10**-6,50
        dx = dy = dz = width/bins

        dT = 0.9 ###Should be???

        print '\t The resolution is %.2e' %dx

        r = 5*10**-6 #[m]
        # r_o = 1.*r/(1.+self.g)
        # r_i = 1.*r/(1./self.g+1.)
        localg = 0.55
        r_o = 1.*r/(1.+localg)
        r_i = 1.*r/(1./localg+1.)
        print r_o,r,r_i

        #generate the substrate
        Chi = np.ones([bins,bins]) * Chi_o

        x = np.linspace(-width/2.,width/2.,bins)
        y = np.linspace(-width/2.,width/2.,bins)
        X,Y = np.meshgrid(x,y)

        Chi[(X**2+Y**2 < r_o**2) * (X**2+Y**2 > r_i**2)] = Chi_m
        Chi[X**2+Y**2 <= r_i**2] = Chi_i

        #pad the substrate
        #semi-redundant in this simple case...
        Chi_padded = np.ones([bins+2,bins+2]) * Chi_o
        Chi_padded[1:-1,1:-1] = Chi

        #plot the substrate
        plt.figure(figsize=(8,8))
        fig = plt.imshow(Chi,origin='lower')

        nt = 4 #make tick every nt
        xlabels = ['%.2e' %x[i] for i in range(len(x))]
        plt.xticks(np.arange(len(x))[::nt],xlabels[::nt],rotation=90,visible=True)
        plt.xlabel('x [m]')
        ylabels = ['%.2e' %y[i] for i in range(len(y))]
        plt.yticks(np.arange(len(y))[::nt],ylabels[::nt],rotation=0)
        plt.ylabel('y [m]')

        plt.colorbar(fig,fraction=0.046, pad=0.04)
        plt.show()

        def dChi(row_shift,col_shift):
            pass

        def dPhi():
            pass

        def FD(ARRAY,axis=None): #get finitie difference
            return (np.roll(ARRAY,1,axis=axis) - np.roll(ARRAY,-1,axis=axis)) / 10.#2*dx #/2*dx???

        def FDy(ARRAY): #get finitie difference
            # print np.shape(ARRAY),'1'
            plt.imshow(ARRAY)
            plt.show()
            ARRAY = np.pad(ARRAY,((1,1),(0,0)),'linear_ramp')#,'edge')
            plt.imshow(ARRAY)
            plt.show()
            # print np.shape(ARRAY),'2'
            # print np.shape(ARRAY[2:,:] - ARRAY[:-2,:]), '3'
            return ARRAY[2:,:] - ARRAY[:-2,:] #/2*dy???

        def FDx(ARRAY): #get finitie difference
            ARRAY = np.pad(ARRAY,((0,0),(1,1)),'linear_ramp')#'edge')
            return ARRAY[:,2:] - ARRAY[:,:-2]  #/2*dx???

        iterations = 10001

        Phi = Y*1000 #/1000. #corresponds to an initial B-field in the y direction
        #The preliminary calculation is started with an initial guess for Phi(t=0)
        #assuming a uniform magnetic field. In this case Phi(t=0) is defined as a
        #function of location that has a constant, non-zero first derivative in
        #the direction of the initially applied homogeneous magnetic field.

        sigma, mode = 1, 'wrap'#'nearest' #'wrap'

        mu = Chi + 1 #Mu_r actually
        mu_dx = ndimage.gaussian_filter1d(mu, axis=0, sigma=sigma, order=1, mode=mode) #/ dx
        mu_dy = ndimage.gaussian_filter1d(mu, axis=1, sigma=sigma, order=1, mode=mode) #/ dy
        # mu_dx = FD(mu,axis=1)
        # mu_dy = FD(mu,axis=0)

        # mu_dx = FDx(mu)
        # print 'mu_dy = FDy(mu)'
        # mu_dy = FDy(mu)

        # plt.imshow(mu_dx)
        # plt.show()
        # plt.imshow(mu_dy)
        # plt.show()
        # raw_input('...')

        for i in range(iterations):
            Phi_dx = ndimage.gaussian_filter1d(Phi, axis=0, sigma=sigma, order=1, mode=mode) #/ dx
            Phi_dxdx = ndimage.gaussian_filter1d(Phi, axis=0, sigma=sigma, order=2, mode=mode) #/ (dx*dx)
            Phi_dy = ndimage.gaussian_filter1d(Phi, axis=1, sigma=sigma, order=1, mode=mode) #/ dy
            Phi_dydy = ndimage.gaussian_filter1d(Phi, axis=1, sigma=sigma, order=2, mode=mode) #/ (dy*dy)

            # Phi_dx = FD(Phi,axis=1)
            # Phi_dxdx = FD(Phi_dx,axis=1)
            # Phi_dy = FD(Phi,axis=0)
            # Phi_dydy = FD(Phi_dy,axis=0)


            # Phi_dx = FDx(Phi)
            # Phi_dxdx = FDx(Phi_dx)
            # print 'Phi_dy = FDy(Phi)'
            # Phi_dy = FDy(Phi)
            # print 'Phi_dydy = FDy(Phi_dy)'
            # Phi_dydy = FDy(Phi_dy)

            # print np.shape(mu_dx), np.shape(Phi_dx), np.shape(Phi_dxdx)

            Phi_x = mu_dx*Phi_dx + mu*Phi_dxdx
            Phi_y = mu_dy*Phi_dy + mu*Phi_dydy
            Phi_z = 0. #For now... 2d...

            Phi += dT * (Phi_x + Phi_y + Phi_z)

            if i%5000 == 0 and i != 0:
                fig,ax = plt.subplots(2,2,figsize=(12,12))

                plot1 = ax[0,0].imshow(dT * (Phi_x + Phi_y + Phi_z))
                ax[0,0].set_title('dT * (Phi_x + Phi_y + Phi_z)')
                plt.colorbar(plot1,ax=ax[0,0])

                # plt.figure()
                plot2 = ax[0,1].imshow(Phi_x)
                ax[0,1].set_title('Phi_x')
                plt.colorbar(plot2,ax=ax[0,1])

                # plt.figure()
                plot3 = ax[1,0].imshow(Phi_y)
                ax[1,0].set_title('Phi_y')
                plt.colorbar(plot3,ax=ax[1,0])

                # plt.figure()
                plot4 = ax[1,1].imshow(Phi)
                ax[1,1].set_title(r'\Phi')
                plt.colorbar(plot4,ax=ax[1,1])

                fig.suptitle('Iteration: %d' %i)

                plt.show()
