# -*- coding: utf-8 -*-
"""
Utility functions for TV tomography

06.15.2015 Initial implementation by R.C. Barnard
06.22.2015 Included FITS input
07.14.2015 Included padding for octopus sinos
08.12.2015 Inlcuded
"""
import numpy as np

#from skimage.util import pad

def generateSheppLogan(imageresolution):
    """
    Generate 2-D Shepp-Logan phantom given x and y resolution in numpy array
    of length at least 2
    """
    data = np.zeros((imageresolution[0],imageresolution[1]))
    x = np.linspace(-1,1,num = imageresolution[0])
    y = np.linspace(-1,1,num = imageresolution[1])
    for i in range(imageresolution[0]):
        for j in range(imageresolution[1]):
            xi1 = (x[i]-.22)*np.cos(.4*np.pi) + y[j]*np.sin(.4*np.pi)
            eta1 = -1.*(x[i]-.22)*np.sin(.4*np.pi) + y[j]*np.cos(.4*np.pi)
            xi2 = (x[i]+.22)*np.cos(.6*np.pi)+y[j]*np.sin(.6*np.pi)
            eta2 = -1.*(x[i]+.22)*np.sin(.6*np.pi)+y[j]*np.cos(.6*np.pi)
            z=0.
            if np.less_equal( ((x[i])/.69)**2 + ((y[j])/.92)**2 ,1):
                z = 2.
            if np.less_equal( ((x[i])/.6624)**2 + ((y[j]+.0184)/.874)**2 ,1):
                z = z-.98
            if  ((np.less_equal( ((xi1)/.31)**2 + ((eta1)/.11)**2 ,1)) or
                np.less_equal( ((xi2)/.41)**2 + ((eta2)/.16)**2 ,1)):
                z = z-.8
            if  (np.less_equal(((x[i])/.21)**2 + ((y[j]-.35)/.25)**2 ,1)):
                 z = z+.4
            if  (np.less_equal(((x[i])/.046)**2 + ((y[j]-.1)/.046)**2 ,1)):
                 z = z+.4
            if  (np.less_equal(((x[i])/.046)**2 + ((y[j]+.1)/.046)**2 ,1)):
                 z = z+.4
            if  (np.less_equal(((x[i]+.08)/.046)**2+((y[j]+.605)/.023)**2 ,1)):
                 z = z+.4
            if (np.less_equal(((x[i]-.06)/.023)**2 +((y[j]+.605)/.023)**2 ,1)):
                 z = z+.4
            if (np.less_equal(((x[i])/.023)**2 +((y[j]+.605)/.023)**2 ,1)):
                 z = z+.4

            data[i,j] = z
    data = data.T
    data = np.flipud(data)
    return data

def tiff_sino_to_image_slice(tiffdir,slice_ind):
    """
    Convert TIFF of sinograms and process to horizontal slices of sinograms.
    Assume structure of files from Octopus, and that files are numbered in
    uniform order
    Need tiffdir to be the path to the director of files, ending with "/"
    slice_ind is a 2-array for the first and last slice (as a fraction of the
    whole list) to be reconstructed. slice_ind[0]=slice_ind[1] is permitted.
    """
    import glob
    from PIL import Image

    files = glob.glob(tiffdir+'*.tif')
    #Read in data
    index=(np.round(slice_ind*len(files))).astype('int')
    if slice_ind[0]==slice_ind[1]:
        files = files[index]
    else:
        files = files[index[0]:index[1]]
    sinos = np.expand_dims(np.array(Image.open(files[0])),2)
    if len(files)>1:
        for i in range(len(files)-1):
            sinos = np.concatenate((sinos,np.expand_dims(np.array
                (Image.open(files[0])),2)),2)
    sinos = np.transpose(sinos,(1,0,2))
    return sinos

def tiff_sinos_pad(sinogram,flag,thetas):
    """
    Pads Octopus sinograms (after shape has been extracted elsewhere) so that
    applying radon to other data won't need a mask and will still be the right
    shape.  NOTE: Very Very hacked together.
    """
    from skimage.transform import radon, iradon
    if flag=='FBP':#apply Radon transform to FBP of sinogram to use as data
        imres=sinogram.shape[0]
        sinogram = radon(iradon(sinogram,thetas,output_size=imres,circle=True),
                         thetas,circle=False)
    elif flag=='pad':#Insert 0's into sinogram on either side
        imres=sinogram.shape[0]
        temp = radon(iradon(0.*sinogram,thetas,output_size=imres,circle=True),
                         thetas,circle=False)
        sizediff = abs(sinogram.shape[0]-temp.shape[0])
        if sizediff>0: #padding is needed
            sinogram = np.concatenate((temp[0:np.ceil(sizediff/2.),:],sinogram,
                                           temp[0:np.floor(sizediff/2.),:]))
    return sinogram


def FITS_to_sinos(fitsdir,slice_frac,sample_freq):
    """
    Takes a path to the directory of FITS files, assumed only having files (no
    directories) and an 2-length array (read as 1d) of the first and last
    vertical portions to store. These are read as fractions of the whole
    (for example, [.1,.9] will read  all data between 10% and 90%--only includ-
    ing the file at 10% and not the one at 90%--of the vertical). We sample the
    files with frequency sample_freq, so downsampling along the
    After reading this data from the FITS files, sinograms are formatted.
    SUudirectories are ignored.
    We assume all files have the same shaped images and that fitsdir DOES
    end with a '/'. Also, assume that slicefrac[1]>=slicefrac[0]
    """
    from astropy.io import fits
    import glob
    #Get list of files
    files = glob.glob(fitsdir+'*.fits')

    #If in Mac, need to remove hidden files, not sure how to do it smart
    hdulist = fits.open(files[0],mode = 'readonly')
    data = hdulist[0].data
    #Determine slices which we store
    ind1 = np.round(slice_frac[0]*data.shape[1]).astype(int)
    ind2 = np.round(slice_frac[1]*data.shape[1]).astype(int)
    if ind1==ind2:
        indices =ind1
    else:
        indices = np.arange(ind1,ind2,1)
    file_ind = range(0,len(files),sample_freq)
    sinos = np.empty([indices.size,data.shape[0],len(file_ind)])
    for i in range(len(file_ind)):
        hdulist = fits.open(files[file_ind[i]])
        sinos[:,:,i]= hdulist[0].data[indices,:]
    sinos = np.transpose(sinos,(1,2,0))
    return sinos

def create_TV(res,m_spa):
    """
    Create TV-related operators (sparse matrices).  Lifted from Rick A.'s code.
    """
    from scipy import sparse
    TV = -sparse.eye(res)+sparse.diags(np.ones((res-1)),-1)
    TV[0,-1] = 1.
#    C_spa = np.ones([m_spa+1,1])
#    for j in range(m_spa+1):
#        for js in  range(m_spa+1):
#            if js!=j:
#                C_spa[j] = C_spa[j]/(j-js)
#    L_spa = sparse.lil_matrix((res,res))
#    m2 = np.floor((m_spa+1)/2.)
#    q_norm = np.sum(C_spa[0:m2,0])
#    C_spa = C_spa.T/q_norm
#    for j in range(res):
#        inds = np.arange(j-m2,j+m_spa-m2+1)
##        inds[inds<0] = inds[inds<0]+res-1
##        inds[inds>res-1] = inds[inds>res-1]-res
#        L_spa[j,inds]=C_spa
    #Convert for efficient multiplications
    TV = TV.asformat('csr')
    TVT = TV.transpose()
    TVTTV = TVT*TV
    return TV,TVT,TVTTV

def outputVTK(filename,recon):
    """
    Creates vtk file for the 3-D reconstruction, using filename.
    """
    import pyevtk.hl
    pyevtk.hl.imageToVTK(filename,origin = (0.0,0.0,0.0),spacing =
                (1.0,1.0,1.0), pointData={"Reconstruction":recon})

def outputFITS(filename,recon):
    """
    Creates FITS file where the reconstruction is stored in the primary array.
    """
    from astropy.io import fits
    hdu = fits.PrimaryHDU(recon)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename,clobber = True)

def clean_dead_pix(sinogram,thresh):
    """
    Filter out dead pixels by in a sinogram via a local filter process.
    The variable sinogram is assumed to be a 2-D array of shape NrxNtheta
    where Ntheta is the number of beams/projections used and Nr is the reso-
    lution of the projections.  However, the code is basically
    agnostic to this ordering.  The scalar thresh gives the filter threshold
    for determining a dead pixel.  Output is an array of the same
    shape as sinogram.
    """
    Nr,Ntheta=sinogram.shape
    f_pat = np.zeros((Nr,Ntheta))
    tmpindr = (np.arange(Nr-2)+1).tolist()
    tmpindt = (np.arange(Ntheta-2)+1).tolist()
    for jr in tmpindr:  #Determine dead pix
        for jt in tmpindt:
            data = sinogram[jr-1:jr+2,jt-1:jt+2]
            data_r = np.ravel(data)
            data_c = data_r[4]
            data_r = np.delete(data_r,4)
            f_pat[jr,jt] = np.abs(np.mean(data_r)-data_c)/(1.+np.mean(np.abs(
                            data_r-np.mean(data_r))))
    thres_pix = np.where(f_pat>thresh)
#    cleaned = np.copy(sinogram) #May need to be more efficient later
    for jtp in range(thres_pix[0].size):
        jr = thres_pix[0][jtp]
        jth = thres_pix[1][jtp]
        data = sinogram[jr-1:jr+2,jth-1:jth+2]
        data_r = np.ravel(data)
        data_r = np.delete(data_r,4)
        sinogram[jr,jth] = np.mean(data_r)


def add_deadpix(image,dp_perc):
    """Add dead pixels to sinogram"""
    dpmask = np.ones((image.shape[0]*image.shape[1]))
    dpmask[:np.round(dp_perc*image.shape[0]*image.shape[1]).astype('int')]=0.
    np.random.shuffle(dpmask)
    dpmask = dpmask.reshape(image.shape)
    return dpmask*image

def make_rt(Ny,thetas):
    """Create Radon Transform. Only valid for odd Ny."""
    Nr = (Ny-1)/2
    Pyind,Pxind = np.meshgrid(np.arange(-Nr,Nr+1),np.arange(-Nr,Nr+1))
    Transform_R = np.zeros((Ny**2,thetas.shape[0]))
#    rampfilter2D = np.sqrt(Pyind[2:Ny,2:Ny]**2+Pxind[2:Ny,2:Ny]**2)
    for i in range(thetas.shape[0]):
        theta = thetas[i]*np.pi/180.
        X = np.cos(theta)*Pxind-np.sin(theta)*Pyind
        Y = np.sin(theta)*Pxind+np.cos(theta)*Pyind
        X = np.round(X+Nr+1)
        Y = np.round(Y+Nr+1)
        X = X.clip(1.,Ny)
        Y = Y.clip(1.,Ny)
        Transform_R[:,i] = X.ravel()+Ny*Y.ravel()-1
    return Transform_R

def make_irt(Ny,thetas):
    """Create inverse Radon Transform for odd Ny."""
    Nr = (Ny-1)/2
    Pyind,Pxind = np.meshgrid(np.arange(-Nr,Nr+1),np.arange(-Nr,Nr+1))
    X_T = np.zeros((Ny,Ny,thetas.shape[0]))
    Xf_T = np.copy(X_T)
    for i in range(thetas.shape[0]):
        theta = thetas[i]*np.pi/180.
        X = np.cos(theta)*Pxind+np.sin(theta)*Pyind
        X = X+Nr
        X = X.clip(1.,2*Nr)
        Xf = np.floor(X)
        X_T[:,:,i] = X
        Xf_T[:,:,i] = Xf.astype('int')
    return X_T,Xf_T

def do_rt_mat(Transform_R,Nr,n_thetas):
    from scipy import sparse
    rt = sparse.lil_matrix((n_thetas*(2*Nr+1),(2*Nr+1)**2))
    for i in range(n_thetas):
        tmat = np.reshape(Transform_R[:,i],(2*Nr+1,2*Nr+1))
        for j in range(2*Nr-1):
            for js in range(2*Nr-1):
                rt[j+(i-1)*(2*Nr+1)-1,tmat[j,js]] += 1./(2*Nr+1)
    return sparse.csc_matrix(rt)

def do_irt_mat(X_T,Xf_T,Nr,n_thetas):
    from scipy import sparse
    irt = sparse.lil_matrix(((2*Nr+1)**2,n_thetas*(2*Nr+1)))
    for i in range(n_thetas):
        lind = Xf_T[:,:,i] +1
        rind = lind-1
        clind = np.pi*(X_T[:,:,i]-Xf_T[:,:,i])/(n_thetas-1.)
        crind = np.pi(Xf_T[:,:,i]+1-X_T[:,:,i])/(n_thetas-1.)
        for j in range(2*Nr):
            for js in range(2*Nr):
                irt[j+(js-1)*(2*Nr+1),lind[j,js]+(i-1)*(2*Nr+1)] += clind[j,js]
                irt[j+(js-1)*(2*Nr+1),rind[j,js]+(i-1)*(2*Nr+1)] += crind[j,js]
    return sparse.csc_matrix(irt)
def do_FD_mat(Nx):
    """Assume square image with sufficient zero-padding"""
    from scipy import sparse
    Dx = sparse.lil_matrix((Nx**2,Nx**2))
    for i in range(Nx-1):
        Dx[i*Nx:(i+1)*Nx,i*Nx:(i+1)*Nx] = sparse.eye(Nx)
        Dx[(i+1)*Nx:(i+2)*Nx,i*Nx:(i+1)*Nx] = -1.*sparse.eye(Nx)
    Dx[(Nx-1)*Nx:Nx**2,(Nx-1)*Nx:Nx**2] = sparse.eye(Nx)
    Dy = sparse.diags([-np.ones(Nx**2),np.ones(Nx**2-1)],[0,1])
    Lap = Dx.T*Dx + Dy.T*Dy
    return Dx.tocsr,Dy,Lap

def data_load(dataflag,Phant_size,TIFF_dir,TIFF_slice,res_freq,theta_freq,
                     theta_count,padflag,FITS_dir,FITS_slice,corruptflag,
                     shift_amount,dp_perc):
        if dataflag == 1:
            image = TV_Split_utilities.generateSheppLogan(Phant_size)
            image_res = image.shape[0]
            thetas = np.linspace(0.,180.,theta_count, endpoint = False)
            image = np.expand_dims(image,axis = 2)
            sinogram = np.empty((image.shape[0],theta_count,image.shape[2]))
            for i in range(image.shape[2]):
                sinogram[:,:,i] = radon(image[:,:,i], thetas,circle = True)
                if corruptflag>0:
                    sinogram[:,:,i] = TV_Split_utilities.add_deadpix(
                                        sinogram[:,:,i],dp_perc)

        if dataflag == 2:
            sinogram = TV_Split_utilities.FITS_to_sinos(FITS_dir,FITS_slice,
                        theta_freq)
            sinores = sinogram.shape
            #Downsample
            ind = np.linspace(0,sinores[0]-1,num = (sinores[0]-1)/res_freq
                    ).astype('int')
            sinogram = sinogram[ind,:]
            ind = np.linspace(0,sinores[1]-1,num =
                    (sinores[1]-1)/theta_freq).astype('int')
            sinogram = sinogram[:,ind]
            sinogram = sinogram.astype(float)
            image_res = sinogram.shape[0]
            theta_count = sinogram.shape[1]
            thetas = np.linspace(0.,180.,theta_count, endpoint = False)
        if dataflag == 4:
            sinogram = TV_Split_utilities.tiff_sino_to_image_slice(TIFF_dir,
                                                                      TIFF_slice)
            sinores = sinogram.shape
            #Downsample
            ind = np.linspace(0,sinores[0]-1,num = (sinores[0]-1)/res_freq
                    ).astype('int')
            sinogram = sinogram[ind,:]
            ind = np.linspace(0,sinores[1]-1,num =
                    (sinores[1]-1)/theta_freq).astype('int')
            sinogram = sinogram[:,ind]
            sinogram = sinogram.astype(float)
            image_res = sinogram.shape[0]
            theta_count = sinogram.shape[1]
            thetas = np.linspace(0.,180.,theta_count, endpoint = False)

        #Normalize for all sinograms
        sinogram = sinogram/np.max(np.abs(sinogram))
        return sinogram,thetas,image_res

def low_L2_objfun(u,sinogram, lambdapen,dkx,dky,bkx,bky,thetas,image_res,
                  Circle_Mask,TV,TVT,TVTTV):
    u = np.reshape(u,(image_res,image_res))
    u[0==Circle_Mask] = 0.
    dtheta = (thetas[1]-thetas[0])/180.     #Assume regular spaced angles
    dx = 1./image_res #Assume square image
    J = .5*np.sum((radon(u,thetas,circle=True)-sinogram)**2)*dtheta*dx
    J += .5*lambdapen*np.sum( (dkx-bkx-u*TV)**2 )*dx*dx
    J += .5*lambdapen*np.sum( (dky-bky-TVT*u)**2 )*dx*dx
    return J

def low_L2_objgrad(u,sinogram, lambdapen,dkx,dky,bkx,bky,thetas,image_res,
                  Circle_Mask,TV,TVT,TVTTV):
    u = np.reshape(u,(image_res,image_res))
    u[0==Circle_Mask] = 0.
    dtheta = (thetas[1]-thetas[0])/180.
    dx = 1./image_res
    grad = (iradon(radon(u,thetas,circle = True)-sinogram,thetas,
                        output_size = image_res, circle = True,filter = None
                        ))*dtheta*dx*(2.*len(thetas))/np.pi
    grad -= dx*dx*((lambdapen*TVTTV*u+lambdapen*u*TVTTV) +lambdapen*(TVT*(dkx-
                bkx)+(dky-bky)*TV))
    return np.ravel(grad)

def proj_one_one(u):
    """projects onto the L^1 unit ball"""
    return u/np.maximum(1.,np.abs(u))

def grad_one(u):
    return np.vstack((np.diff(u,axis=0),np.zeros((1,u.shape[1]))))

def grad_two(u):
    return np.hstack((np.diff(u,axis=1),np.zeros((u.shape[0],1))))

def resolvent_G(u):
    """Identity here"""
    return u

def resolvent_Fstar(p_one_one,p_one_two,p_two,sigma,lambdapen,g):
    v1 = proj_one_one(p_one_one)
    v2 = proj_one_one(p_one_two)
    v3 = (lambdapen*p_two+sigma*lambdapen*g)/(sigma+lambdapen)
    return v1,v2,v3


def Divu(u):
    """Need tweaking if mask is not applied"""
    gradu_one = np.vstack((u[0][0,:,np.newaxis].T,u[0][1:-1,:]-u[0][0:-2,:],
                           -1.*u[0][-2,:,np.newaxis].T))
    gradu_two = np.hstack((u[1][:,0,np.newaxis],u[1][:,1:-1]-u[1][:,0:-2],
                           -1.*u[1][:,-2,np.newaxis]))
    return gradu_one+gradu_two

def FD_approx(f,u,epsilon):
    FD = np.zeros(u.shape)
    for i in range(FD.shape[0]):
        for j in range(FD.shape[1]):
            u[i,j] += epsilon
            FD[i,j] = f(u)
            u[i,j] -= 2.*epsilon
            FD[i,j] -= f(u)
            u[i,j] += epsilon
    return FD/(2.*epsilon)
def PD_denoise(sinogram,rec_FBP,thetas,outerloops,innerloops,
                           CGloops,image_res,convtestnorm,lambdapen,slicenum):

   """
   Solves the problem via the following formulation.  We minimize F(Ku)+G(u),
   where K:= (grad,R)^T and F(x):= ||x_1||_1+lambda/2||x_2-g||^2, and G(u):=0.
   Then, F^*(y) = \delta_p(y_1)+1/(2lambda)||y_2||^2+<g,y_2>.
   We then perform the primal dual algorithm of Chambolle-Pock '11.
   We assume it is better to have repeated G evaluations and fewer thing stored.
   Do not use; needs significant tuning/possible debugging.
   """
   uk = (TV_Split_utilities.generateSheppLogan(Phant_size))
   radius = min(uk.shape) // 2
   c0, c1 = np.ogrid[0:uk.shape[0], 0:uk.shape[1]]
   Circle_Mask = ((c0 - uk.shape[0] // 2) ** 2+ (c1 - uk.shape[1] // 2) ** 2)
   Circle_Mask = Circle_Mask <= 0.95*(radius ** 2)
   Circle_Mask = np.matrix(Circle_Mask)
   uk[0==Circle_Mask]=0.
   y_onek = np.gradient(0.*uk)
   y_twok = 0.*sinogram
   sigma = 1.e-2
   tau = 1.e-2
   theta = 1.
   ubark = uk
   Err_diff = np.zeros((int(.5*rec_FBP.size)))
   for i in range(int(.5*rec_FBP.size)):
        y_onek[0],y_onek[1], y_twok = resolvent_Fstar(y_onek[0]+sigma*
                  grad_one(ubark),y_onek[1]+sigma*grad_two(ubark),
                  y_twok+sigma*radon(ubark,thetas,circle=True),sigma,lambdapen,
                  sinogram)
        ubark = (1+theta)*resolvent_G(uk+tau*(-1.*Divu(y_onek)+iradon(y_twok,
                 thetas,output_size=image_res,circle=True,filter = 'ramp' )))-theta*uk
        ubark[0==Circle_Mask]=0.
        uk = resolvent_G(uk-tau*(-1.*Divu(y_onek)+iradon(y_twok,
                 thetas,output_size=image_res,circle=True,filter = 'ramp' )))
        uk[0==Circle_Mask]=0.
        Err_diff[i] = np.linalg.norm((ubark-uk)/theta)
        plt.imshow(uk);plt.pause(.0001)
        print('Slice %d]' %(slicenum))
        print('Outer Iterate = ' ,i)
        print('Update = ' ,Err_diff[i])
        if Err_diff[i]<1.e-5:
            break
   return uk,Err_diff
