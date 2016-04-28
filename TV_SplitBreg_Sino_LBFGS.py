# -*- coding: utf-8 -*-
"""
Module for parallel TV regularizations of slices via the radon transform and
Split-Bregman iterations as in Goldfarb, Osher 2009.
Accepts either FITS files, python ndarrays, or TIFF of sinograms for the data.
Also can use Shepp-Logan phantom for a single slice test case.

It does not fix errors in rotation axis; meaning "tuning fork" artifacts are
likely to remain.  Additionally, if one wants to run things in parallel, one
must comment out the single slice version of the code and uncomment the
parallel part.

Required packages:
numpy, scipy, scikit-image (skimage), joblib.
####### History #######

06.15.2015 Initial implementation by R.C. Barnard only for Shepp-Logan
06.16.15 Include TIFF files from Octopus normalization/sinogram generation
            R.C. Barnard
06.23.15 Fork, where now we use TV with data term of the form ||R(u)-F||, TV-
        related operators are sparse matrices
        R. C. Barnard
06.30.15 Allow multiple octopus sinograms to be read, and allow parallel
            reconstructions on several slices, code structure from joblib docs
            NOTE: broke Shepp-Logan functionality due to padding issues
07.07.15 FITS reading completed. VTK output not debugged. R.C. Barnard
08.10.15 FITS output implemented. VTK output is completed.
08.12.15 Implemented dead pixel filtering and center of rotation correction.
            Preprocessing not implemented in parallel yet.
13.08.15 Preprocessing implemented in parallel. (R.C. Barnard)
20.08.15 Added Postprocessing.  NOTE: Involves affine transformation of data.
25.08.15 Shepp-Logan working now. Convergence test is now on norm of updates,
            as opposed to distance of radon of iterate from sinogram. Added
            printout if CG converges (is rare in FITS data case) (R.C. Barnard)
01.09.15 Added both steepest-descent and black-box versions of solving the
            subproblem in Split-Bregman. (R.C. Barnard)
08.12.15 Removed Steepest descent and black-box due to no improvements in
        efficiency/speed.  Implemented Primal dual. Included downsampling in
        angle. Primal dual is as in Chambolle-Pock, with minor alterations in
        the dualization of the term with the Radon transform.  Fine tuning of
        the parameters sigma, tau are needed. Note also that the lambdapen is
        placed on the tracking term and not the regularization term for Primal
        Dual. (R.C. Barnard)
02.25.16 Fork where now we use LBFGS for solving the inner L2 problem in the
        Bregman iteration.  Decent results, but choice of regularization weight
        heavily influences performance of the algorithm. (R.C. Barnard)
04.28.16 Cleaned up code, shifting more functions to TV_Split_utilities, added
         more documetnation. Functionality unchanged.  (R.C. Barnard)
"""

import numpy as np
from skimage.transform import radon, iradon
from scipy import misc
from scipy import optimize as opt
from joblib import Parallel, load,dump,delayed
import TV_Split_utilities
import matplotlib.pyplot as plt
import os

#Relevant Parameters and information
outerloops = 20  #Outer loops for Split-Bregman
innerloops = 10     #Innerloops for Split-Bregman
CGloops =    10    #Number of CG steps for inverse in Split-Bregman
convtestnorm = 1.    #Relative reduction in "noise" for termination of TV-SB
lambdapen = 50. #penalty, may be adjusted to speed up algorithm, but given
                #enough loop iterations, should be irrelevant.
deadpixelthreshold = 0.25 #threshold for determining dead pixel

dataflag = 1       #1 if using Shepp-Logan phantom, 2 if importing FITS, 3 if
                    # using python arrays, 4 if using sinograms (TIFF)
corruptflag = 0 # positive integer if Shepp-Logan sinograms should be corrupted
dp_perc = .001  #Fraction of dead pixels

Outputflag = [1,0,0]    #1 if you want the outputs of that type. Entries are
                        # for jpg,vtk,fits (respectively)
low_l2_meth = 0   #Use 0 or 1 if you want to use LBFGS (resp.
                        #Primal Dual) to (partially) solve the lower L-2 problem
parajob = 1 #number of parallel slices to run (only if not using Shepp-Logan)

Phant_size = np.array([129,129]) # relevant if dataflag is 1,must be odd
theta_count = int((Phant_size[0]-1)*np.pi/4 ) #number of evenly-distributed
                                              #beams to use


FITS_output = '/Users/rbc/Desktop/Arch_Hauck_proj/tomographic_data/'
                #Outputted directory for file of output.

padflag = 'FBP' #Use'FBP' if you want to pad by taking the radon of the FBP of
                # sinograms to ensure size matching, use 'pad' to put 0's on
                #either side (may lead to shift by a pixel)

#Below only used if dataflag >1
TIFF_slice = np.array([.5,.501]) #which horizontal slice to use (as a fraction)
FITS_slice = np.array([.5,.5])
#directory where TIFF file sits, MUST end in '/'
TIFF_dir = '/Users/rbc/Desktop/Arch_Hauck_proj/tomographic_data/sinograms/'
FITS_dir = '/Users/rbc/Desktop/Arch_Hauck_proj/tomographic_data/CT_Al_Steelcenter/'
theta_freq  = 4 #sample rate for data in theta regime
res_freq    = 16 #sample rate for data in resolution for sinograms

SVD_Flag = 1 # 1 if Split-Bergmann is to use SVD approx. instead of CG
SVD_rank = 100



def preprocessing(sinoslice,deadpixelthreshold,thetas):
    """ Perform preprocessing on a 2-D slice """
    TV_Split_utilities.clean_dead_pix(sinoslice,
                    deadpixelthreshold)

def Radon_TV_Split_Bregman_LBFGS(sinogram,rec_FBP,thetas,outerloops,innerloops,
                           image_res,convtestnorm,lambdapen,slicenum):
    #Split Bregman
    uk = np.matrix(rec_FBP)
    dkx = np.matrix(np.zeros(uk.shape))
    dky = np.copy(dkx)
    bkx = np.copy(dkx)
    bky = np.copy(dky)
    fk = np.copy(sinogram)
    #Taken from skimage.transform.radon to mask objects so radon won't pad.
    radius = min(uk.shape) // 2
    c0, c1 = np.ogrid[0:uk.shape[0], 0:uk.shape[1]]
    Circle_Mask = ((c0 - uk.shape[0] // 2) ** 2+ (c1 - uk.shape[1] // 2) ** 2)
    Circle_Mask = Circle_Mask <= 0.95*(radius ** 2)
    Circle_Mask = np.matrix(Circle_Mask)
    uk[0==Circle_Mask]=0.
    TV,TVT,TVTTV = TV_Split_utilities.create_TV(uk.shape[0],1)
    Err_diff = np.zeros([outerloops,1])
    for jouter in range(outerloops):
        ruk = radon(uk,thetas,circle=True)
        olduk = np.copy(uk)
        for jinner in range(innerloops):
            #Solve L2 problem
            obj = lambda u:low_L2_objfun(u,sinogram, lambdapen,dkx,dky,bkx,bky,
                                         thetas,image_res,Circle_Mask,TV,TVT,
                                         TVTTV)
            jacgrad = lambda u:low_L2_objgrad(u,sinogram, lambdapen,dkx,dky,
                                              bkx,bky,thetas,image_res,
                                              Circle_Mask,TV,TVT,TVTTV)
            res = opt.minimize(obj,uk.ravel(),method = 'l-BFGS-B',bounds = None,
                               jac = jacgrad,options={'maxiter': CGloops})
            print res.message,res.nit,res.nfev
            uk = np.reshape(res.x,uk.shape)
            uk [0 == Circle_Mask] = 0.
            plt.cla()
            plt.imshow(uk)
            plt.pause(.001)
            sk = np.sqrt(np.square(-TV*uk+bkx))
            ind = np.where(np.asarray(sk)>1./lambdapen)
            dkx[ind] = np.divide(np.multiply(sk[ind]-(1./lambdapen),
                        (TV*uk+bkx)[ind]),sk[ind])
            ind = np.where(np.asarray(sk)<=1./lambdapen)
            dkx[ind] = 0.

            sk = np.sqrt(np.square(-uk*TVT+bky))
            ind = np.where(np.asarray(sk)>1./lambdapen)
            dky[ind] = np.divide(np.multiply(sk[ind]-(1./lambdapen),
                        (uk*TVT+bky)[ind]),sk[ind])
            ind = np.where(np.asarray(sk)<=1./lambdapen)
            dky[ind] = 0.
            bkx = bkx+TV*uk-dkx
            bky = bky+uk*TVT-dky
            ruk = radon(uk,thetas,circle=True) #To handle skimage padding
        fk = fk + sinogram - ruk
        Err_diff[jouter]=np.linalg.norm(olduk-uk)
        plt.cla()
        plt.imshow(uk)
        plt.pause(.001)
        print('Slice %d]' %(slicenum))
        print('Outer Iterate = ' ,jouter)
        print('Update = ' ,Err_diff[jouter])
        if Err_diff[jouter]<1.e-5*Err_diff[0]:
            break
    return uk,Err_diff


def core_denoise(sinogram,image_res,outerloops,innerloops,CGloops,
                     convtestnorm,lambdapen,thetas,jpegflag,
                     i,output,SVD_Flag):
    """ Core run through preprocessor and denoiser algorithm"""
    rec_FBP = iradon(sinogram[:,:,i],thetas,output_size = image_res,
                     circle = True, filter = 'hann')
    if (low_l2_meth == 0):
        TVrec,Err_diff = Radon_TV_Split_Bregman_LBFGS(sinogram[:,:,i],rec_FBP,
                                  thetas, outerloops,innerloops,image_res,
                                   convtestnorm,lambdapen,i)
    elif (low_l2_meth == 1):
        TVrec,Err_diff = PD_denoise(sinogram[:,:,i],rec_FBP,thetas,outerloops,
                                   innerloops,CGloops,image_res,convtestnorm,
                                   lambdapen,i)
    else:
        TVrec = rec_FBP  ,
        Err_diff = 0.
    print("[Slice %d] finished Split-Breg on Radon" )

    print("[Slice %d] is finished" % (i))
    output[:,:,i] = TVrec
    if jpegflag>0: #Want jpeg outputs
        filename = 'Slice'+str(i)+'.jpg'
        misc.imsave(filename,output[:,:,i])
    np.savetxt('Slice'+str(i)+'resid.txt',Err_diff)


if __name__ == '__main__':
    plt.close('all')

    #Load data
    sinogram,thetas,image_res = data_load(dataflag,Phant_size,TIFF_dir,
                                               TIFF_slice,res_freq,theta_freq,
                                               theta_count,padflag,FITS_dir,
                                               FITS_slice,corruptflag,shift_amount,
                                               dp_perc)
    print('Data loaded/generated')


    #Preprocess
    Parallel(n_jobs = parajob)(delayed(preprocessing)(sinogram[:,:,i],
                 deadpixelthreshold,thetas) for i in range(sinogram.shape[2]))

    #Create memory map for the output
    output = np.memmap(os.path.join(TIFF_dir,'output'), dtype=sinogram.dtype,
                       shape = (sinogram.shape[0],sinogram.shape[0],
                                sinogram.shape[2]),mode='w+')
    # Dump input to disk for more memory
    dump(sinogram,os.path.join(TIFF_dir,'tmp_sinos'))

    sinogram = load(os.path.join(TIFF_dir,'tmp_sinos'),mmap_mode='r')

    print('Forking slices and denoising')

    ##Fork the slices to parallelize TV Split-Bregman
    #Parallel(n_jobs = parajob)(delayed(core_denoise)(sinogram,image_res,outerloops,
    #                    innerloops,CGloops,convtestnorm,lambdapen,
    #                    thetas,Outputflag[0],i,output)
    #                    for i in range(output.shape[2]))

    ##Single Slice version
    core_denoise(sinogram,image_res,outerloops,
                        innerloops,CGloops,convtestnorm,lambdapen,thetas,
                        Outputflag[0],0,output,SVD_Flag)

    if Outputflag[1]>0:
        #Store in vtr file if wanted
        VTKname = './TVRecOuter'+str(outerloops)+'lambda'+str(int(lambdapen))
        tmp = np.empty(output.shape)
        tmp[:] = output[:]
        TV_Split_utilities.outputVTK(VTKname,tmp[:])
    if Outputflag[2]>0:
        #Store in fits file if watned
        FITSname = FITS_output+'output_Outer'+str(outerloops) + 'Lambda' +str(int(
                lambdapen))+'.fits'
        tmp = np.empty(output.shape)
        tmp[:] = output[:]
        TV_Split_utilities.outputFITS(FITSname,tmp)
    #Remove temp files
    os.remove(sinogram.filename)
    os.remove(output.filename)
