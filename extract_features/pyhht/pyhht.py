#!/usr/bin/env python
import numpy as np
from numpy import zeros, vstack, array, sin, linspace,hstack, real, imag,diff
from numpy.random import normal
from scipy.interpolate import splrep,splev
from scipy.io.wavfile import read
from scipy.signal import hilbert
from scipy import angle, unwrap, real
from math import pi
from matplotlib.pyplot import plot,show,imshow,colorbar,subplot,grid

def envlps(f):
    x=zeros((len(f),),dtype=int)
    y=x.copy()

    for i in range(1,len(f)-1):
        if (f[i]>f[i-1])&(f[i]>f[i+1]):
            x[i]=1
        if (f[i]<f[i-1])&(f[i]<f[i+1]):
            y[i]=1

    x=(x>0).nonzero()
    y=(y>0).nonzero()
    y=y[0]
    x=x[0]
    x=hstack((0,x,len(f)-1))
    y=hstack((0,y,len(f)-1))


    # print len(y)
    if len(y) <= 3:
        t=splrep(x=y, y=f[y], k = 1)
        bot=splev(array(range(len(f))),t)

    else:
        t=splrep(y,f[y])
        bot=splev(array(range(len(f))),t)

    if len(x) <= 3:
        t=splrep(x,f[x], k = 1)
        top=splev(array(range(len(f))),t)
    else:
        t=splrep(x,f[x])
        top=splev(array(range(len(f))),t)

    return top,bot

def sift(t):
    top,bot=envlps(t)
    c=t-(top+bot)/2
    return c

def localmean(t):
    top,bot=envlps(t)
    return (top+bot)/2


def checkimf(t):

    xtrm=zeros((len(t),),dtype=int)
    zcross=xtrm.copy()

    for i in range(1,len(t)-1):
        if (t[i]>t[i-1])&(t[i]>t[i+1]):
            xtrm[i]=1
        if (t[i]<t[i-1])&(t[i]<t[i+1]):
            xtrm[i]=1
        if t[i-1]*t[i+1]<0:
            zcross[i]=1


    a=(xtrm>0).nonzero()
    b=(zcross>0).nonzero()

    return abs(len(a[0])-len(b[0]))

def siftrun(t,n):
    d=checkimf(t)
    iters=0
    while d>1:
        t=sift(t)
        iters=iters+1
        d=checkimf(t)
        if iters==n:
            break
    return t

def emd(f,n):
    # print 'emd:'
    imfs=zeros((1,len(f)),dtype=float)
    for i in range(n):
        t=f-sum(imfs)
        t=siftrun(t,20)
        # print i
        imfs=vstack((imfs,t))
    imfs = imfs[1:n+1,:]
    imfs=vstack((imfs,f-sum(imfs)))
    return imfs

def demoinit():
    Fs,f=read('sare.wav')
    f=f[30000:31001]
    return Fs,f

def plothilbert(imfs):
    for i in range(imfs.shape[0]):
        h=hilbert(imfs[i,:])
        plot(real(h),imag(h))
    show()

def symmetrydemo():
    a=sin(linspace(-5*pi,5*pi,10000))
    b=a+2
    c=a-0.5
    ah,bh,ch=hilbert(a),hilbert(b),hilbert(c)
    ph_a,ph_b,ph_c=unwrap(angle(ah)),unwrap(angle(bh)),unwrap(angle(ch))
    omega_a=diff(ph_a)
    omega_b=diff(ph_b)
    omega_c=diff(ph_c)
    subplot(311),plot(ph_a),plot(ph_b),plot(ph_c)
    subplot(312),plot(omega_a),plot(omega_b),plot(omega_c)
    subplot(313),plot(a),plot(b),plot(c)

    grid()
    show()
    return a,b,c


def getinstfreq(imfs):
    # print 'freq:'
    omega=zeros((imfs.shape[0],imfs.shape[1]),dtype=float)
    for i in range(imfs.shape[0]):
        h=hilbert(imfs[i,:])
        theta=unwrap(angle(h))
        omega[i,0:diff(theta).shape[0]]=diff(theta)
    # print 'freq:',np.shape(omega)
    return omega

def getinstamp(imfs):
    # print 'amp:'
    f = zeros((imfs.shape[0],imfs.shape[1]),dtype=float)
    for i in range(imfs.shape[0]):
        h=hilbert(imfs[i,:])
        f[i, 0:real(h).shape[0]] = real(h)
    # print 'amp:',np.shape(f)
    return f

def testdemo():
    import numpy as np
    from math import sqrt
    from scipy import signal
    a=sin(linspace(-5*pi,5*pi,10000))
    b=0.5*sin(linspace(-10*pi+2,10*pi+2,10000))+0.5

    c=a+b

    imfs = emd(c, 2)
    omega = getinstamp(imfs)

    omegah = np.ndarray.flatten(omega)

    subplot(8,2,2),plot(a),plot(b), plot(c)
    subplot(8,2,4),plot(np.sum(omega,axis = 0))
    subplot(8,2,6),plot(omegah)
    subplot(8,2,8),plot(omega[0,:])
    subplot(8,2,10),plot(omega[1,:])
    subplot(8,2,12),plot(omega[2,:])
    # subplot(8,2,14),plot(omega[3,:])
    # subplot(8,2,16),plot(omega[4,:])

    subplot(8,2,1),plot(c)
    subplot(8,2,3),plot(np.sum(imfs,axis = 0))
    subplot(8,2,5),plot(np.ndarray.flatten(imfs))
    subplot(8,2,7),plot(imfs[0,:])
    subplot(8,2,9),plot(imfs[1,:])
    subplot(8,2,11),plot(imfs[2,:])
    # subplot(8,2,13),plot(imfs[3,:])
    # subplot(8,2,15),plot(imfs[4,:])


    show()
