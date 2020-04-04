import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio

from torch.autograd import grad

import time

from pykeops.torch import Kernel, kernel_product
from pykeops.torch.kernel_product.formula import *

from scipy.optimize import minimize

import mesh


class Optimization:
    """Contains methods for:
        -optimization of midsurface and w
        -visualization of optimization results
        -surface unfolding

        Args:
            source (numpy arr): points on midsurface arranged as grid (prior to meshing)
            VH (torch tensor): vertices of target surface
            FH (torch tensor): faces of target surface

        Attributes:
            Q (torch tensor): midsurface vertices
            VH (torch tensor): stored target vertices
            FH (torch tensor): stored target faces
            m (int): number of points along u-axis of midsurface after downsampling
            n (int): number of points along v-axis of midsurface after downsampling

            torchdeviceId (str): name of torch device
            torchdtype (datatype): torch datatype

            Qopt (torch tensor): midsurface vertices after optimization
            Wopt (torch tensor): W scalar field after optimization of midsurface
            Wuopt (torch tensor): Widths describing upper surface vertex positions after optimization
            Wlopt (torch tensor): Widths describing lower surface vertex positions after optimization

        """

    def __init__(self, source, VH, FH, m=50, n=50):
        source = mesh.downsample(source, m, n)
        self.Q, self.FS = mesh.meshSource(source)
        self.VH, self.FH = VH, FH
        self.VH, self.FH = VH, FH
        self.m = m
        self.n = n

        use_cuda = torch.cuda.is_available()
        torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
        torchdtype = torch.float32

        self.torchdeviceId = torchdeviceId
        self.torchdtype = torchdtype

    def GaussKernel(self, sigma):
        def K(x, y, b):
            params = {
                'id': Kernel('gaussian(x,y)'),
                'gamma': 1 / (sigma * sigma),
                'backend': 'auto'
            }
            return kernel_product(params, x, y, b)

        return K


    def GaussLinKernel(self, sigma):
        def K(x, y, u, v, b):
            params = {
                'id': Kernel('gaussian(x,y) * linear(u,v)'),
                'gamma': (1 / (sigma * sigma), None),
                'backend': 'auto'
            }
            return kernel_product(params, (x, u), (y, v), b)

        return K

    def sumGaussLinKernel(self, sigmas):

        """Summation of multiple GaussLinKernels with different sigma values"""

        flist = list(map(lambda s: self.GaussLinKernel(s), sigmas))

        def K(x, y, u, v, b):
            Klist = list(map(lambda f: f(x, y, u, v, b), flist))
            return sum(Klist)

        return K

    def sumGaussKernel(self, sigmas):

        """Summation of multiple GaussKernels with different sigma values"""

        flist = list(map(lambda s: self.GaussKernel(s), sigmas))

        def K(x, y, b):
            Klist = list(map(lambda f: f(x, y, b), flist))
            return sum(Klist)

        return K

    def lossHippSurfQ(self, FSj, fmap, VH, FH, K):

        """Data loss for Q optimization step.
            Args:
                FSj (torch tensor): faces of joined upper and lower surfaces
                fmap (dict): dictionary mapping vertices to incident faces
                VH (torch tensor): target vertices
                FH (torch tensor): target faces
                K (func): kernel function

            Returns:
                loss (func): data loss function

        """

        def compCN(V, F):
            """Computation of surface centroids and normals"""
            V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
            C = (V0 + V1 + V2) / 3
            N = 0.5 * torch.cross(V1 - V0, V2 - V0)

            return C, N

        CT, NT = compCN(VH, FH)
        BT = torch.ones([CT.shape[0], 1], dtype=self.torchdtype, device=self.torchdeviceId)
        cst = torch.dot(K(CT, CT, NT, NT, BT).view(-1), torch.ones_like(K(CT, CT, NT, NT, BT)).view(-1))

        def loss(qn, wv):
            """Computes data loss with method of currents.
                Args:
                    qn (torch tensor): vertices of midsurface
                    wv (torch tensor): scalar field of widths
                Returns:
                    cost (float): numerical value of data loss
            """
            Qd = mesh.doubleQ(qn)
            VS = mesh.generateSourceULW(Qd, torch.flatten(wv), FSj, fmap)

            CS, NS = compCN(VS, FSj)

            BS = torch.ones([CS.shape[0], 1], dtype=self.torchdtype, device=self.torchdeviceId)
            a = K(CS, CS, NS, NS, BS)
            CSdot = torch.dot(a.view(-1), torch.ones_like(a).view(-1))
            b = K(CS, CT, NS, NT, BT)
            CSTdot = torch.dot(b.view(-1), torch.ones_like(b).view(-1))

            cost = cst + CSdot - 2 * CSTdot

            return cost

        return loss

    def lossHippSurfW(self, FSj, fmap, VH, FH, K):

        """Data loss for W optimization step.
            Args:
                FSj (torch tensor): faces of joined upper and lower surfaces
                fmap (dict): dictionary mapping vertices to incident faces
                VH (torch tensor): target vertices
                FH (torch tensor): target faces
                K (func): kernel function

            Returns:
                loss (func): data loss function
        """

        def compCN(V, F):
            V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
            C = (V0 + V1 + V2) / 3
            N = 0.5 * torch.cross(V1 - V0, V2 - V0)

            return C, N

        CT, NT = compCN(VH, FH)
        BT = torch.ones([CT.shape[0], 1], dtype=self.torchdtype, device=self.torchdeviceId)
        cst = torch.dot(K(CT, CT, NT, NT, BT).view(-1), torch.ones_like(K(CT, CT, NT, NT, BT)).view(-1))

        def loss(qn, wu, wl):
            """Computes data loss with method of currents.
                    Args:
                        qn (torch tensor): vertices of midsurface
                        wu (torch tensor): scalar field of widths for upper surface
                        wl (torch tensor): scalar field of widths for lower surface
                    Returns:
                        cost (float): numerical value of data loss
            """
            Qd = mesh.doubleQ(qn)
            VS = mesh.generateSourceULnonsymm(Qd, wu, wl, FSj, fmap)

            CS, NS = compCN(VS, FSj)

            BS = torch.ones([CS.shape[0], 1], dtype=self.torchdtype, device=self.torchdeviceId)
            a = K(CS, CS, NS, NS, BS)
            CSdot = torch.dot(a.view(-1), torch.ones_like(a).view(-1))
            b = K(CS, CT, NS, NT, BT)
            CSTdot = torch.dot(b.view(-1), torch.ones_like(b).view(-1))

            cost = cst + CSdot - 2 * CSTdot

            return cost

        return loss


    def RalstonIntegrator():
        def f(ODESystem, x0, nt, deltat=1.0):
            x = tuple(map(lambda x: x.clone(), x0))
            dt = deltat / nt
            l = [x]
            for i in range(nt):
                xdot = ODESystem(*x)
                xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
                xdoti = ODESystem(*xi)
                x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
                l.append(x)
            return l

        return f

    def Hamiltonian(self, K):
        def H(p, q):
            return .5 * (p * K(q, q, p)).sum()
        return H

    def HamiltonianSystem(self, K):
        H = self.Hamiltonian(K)

        def HS(p, q):
            Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
            return -Gq, Gp

        return HS

    def Shooting(self, p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
        return Integrator(self.HamiltonianSystem(K), (p0, q0), nt)

    def TotalLossIntegratedQ(self, K1, K2, dataloss, gamma=0, beta=0):

        """Total loss for Q optimization step. Includes deformation term and data attachment term.
            Args:
                K1 (func): kernel used to compute position of vertices given momentum vector
                K2 (func): kernel used to compute widths given momentum vector
            Returns:
                loss (func): total loss, summation of deformation and data attachment losses
        """

        def loss(p0, q0, a0, w0):
            p, q = self.Shooting(p0, q0, K1)[-1]
            w = w0 + K2(q0, q0, a0)

            return gamma * self.Hamiltonian(K1)(p0, q0) + beta * self.Hamiltonian(K2)(a0, q0) + dataloss(q, w)

        return loss


    def TotalLossW(self, K, dataloss, gamma=0, beta=0):

        """Total loss for W optimization step. Includes deformation term and data attachment term.
            Args:
                K1 (func): kernel used to compute position of vertices given momentum vector
                K2 (func): kernel used to compute widths given momentum vector
            Returns:
                loss (func): total loss, summation of deformation and data attachment losses
        """

        def loss(q0, a0, wu0, b0, wl0):
            wu = wu0 + K(q0, q0, a0)
            wl = wl0 + K(q0, q0, b0)
            wcost = gamma * (self.Hamiltonian(K)(a0, wu0) + self.Hamiltonian(K)(b0, wl0))
            currcost = beta * dataloss(q0, wu, wl)
            return wcost + currcost

        return loss


    class PytorchObjectiveQ(object):

        """Wrapper class to combine sSipy's LBFGS optimizer with Pytorch's autograd for Q optimization step.
            Args:
                objfun (func): loss function
                param (torch tensor): parameters to be optimized (i.e., momentum vectors)
                q (torch tensor): midsurface vertices
                w (torch tensor): initial widths
                dtype (datatype): data type to use for torch tensors
                deviceId (str): torch device

            Attributes:
                f (func): stores loss function
                x (numpy array): stores initial parameters
                q0 (tensor): stores initial midsurface vertices
                w0 (tensor): stores initial widths
                it (int): iteration counter
                feval (int): function evaluation counter
                plist (list): list of momentum vectors for midsurface after each iteration
                alist (list): list of momentum vectors for widths after each iteration
                losslist (list): list of losses after each iteration

                cached_x (numpy array): stores parameter values from previous function evaluation
                cached_f (numpy array): stores value of loss from previous function evaluation
                cached_jac (numpy array): stores gradients from previous function evaluation
        """

        def __init__(self, objfun, param, q, w, dtype, deviceId):
            self.f = objfun
            self.x0 = param.cpu().data.numpy()
            self.q0 = q
            self.w0 = w
            self.dtype = dtype
            self.device = deviceId
            self.it = 0
            self.feval = 0
            self.alist = []
            self.plist = []
            self.losslist = []

        def is_new(self, x):
            # if this is the first thing we've seen
            if not hasattr(self, 'cached_x'):
                return True
            else:
                # compare x to cached_x to determine if we've been given a new input
                x, self.cached_x = np.array(x), np.array(self.cached_x)
                error = np.abs(x - self.cached_x)
                return error.max() > 1e-8

        def conv_param(self, x):
            psect = x[0:self.q0.shape[0] * self.q0.shape[1]]
            convp = torch.from_numpy(psect).view(self.q0.shape[0], self.q0.shape[1])
            asect = x[self.q0.shape[0] * self.q0.shape[1]:]
            conva = torch.from_numpy(asect).view(self.w0.shape[0], self.w0.shape[1])
            return convp, conva

        def cache(self, x):
            self.feval += 1
            # convert x to tensor
            ptensor, atensor = self.conv_param(x)
            ptensor = ptensor.to(self.device).type(self.dtype).requires_grad_(True)
            atensor = atensor.to(self.device).type(self.dtype).requires_grad_(True)
            # store the raw array
            self.cached_x = x
            # calculate the objective
            L = self.f(ptensor, self.q0, atensor, self.w0)
            # backprop the objective
            L.backward()
            self.cached_f = L.item()
            if self.feval == 1:
                print("iteration %d" % (self.it))
            print("loss = %.2f" % self.cached_f)
            pgrad = ptensor.grad.type(torch.float64).cpu().data.numpy().ravel()
            agrad = atensor.grad.type(torch.float64).cpu().data.numpy().ravel()
            self.cached_jac = np.concatenate([pgrad, agrad])

            self.plist.append(ptensor)
            self.alist.append(atensor)
            self.losslist.append(self.cached_f)

        def fun(self, x):
            if self.is_new(x):
                self.cache(x)
            return self.cached_f

        def jac(self, x):
            if self.is_new(x):
                self.cache(x)
            return self.cached_jac

        def callback(self, x):
            self.it += 1
            self.feval = 0

    class PytorchObjectiveW(object):

        """Wrapper class to combine Scipy's LBFGS with Pytorch's autograd for W optimization step
             Args:
                objfun (func): loss function
                param (torch tensor): parameters to be optimized (i.e., momentum vectors)
                q (torch tensor): midsurface vertices
                wu (torch tensor): upper surface widths
                wl (torch tensor): lower surface widths
                dtype (datatype): data type to use for torch tensors
                deviceId (str): torch device

             Attributes:
                f (func): stores loss function
                x (numpy array): stores parameters
                q0 (tensor): stores initial midsurface vertices
                wu0 (tensor): stores initial upper surface widths
                wl0 (tensor): stores initial lower surface widths
                it (int): iteration counter
                feval (int): function evaluation counter
                alist (list): list of momentum vectors corresponding to upper surface after each iteration
                alist (list): list of momentum vectors corresponding to lower surface after each iteration
                losslist (list): list of losses after each iteration

                cached_x (numpy array): stores parameter values from previous function evaluation
                cached_f (numpy array): stores value of loss from previous function evaluation
                cached_jac (numpy array): stores gradients from previous function evaluation

        """
        def __init__(self, objfun, param, q, wu, wl, dtype, deviceId):
            self.f = objfun  # loss function
            self.x0 = param.cpu().data.numpy()
            self.q0 = q
            self.wu0 = wu
            self.wl0 = wl
            self.dtype = dtype
            self.device = deviceId
            self.it = 0
            self.feval = 0
            self.alist = []
            self.blist = []
            self.losslist = []

        def is_new(self, x):
            # if this is the first thing we've seen
            if not hasattr(self, 'cached_x'):
                return True
            else:
                # compare x to cached_x to determine if we've been given a new input
                x, self.cached_x = np.array(x), np.array(self.cached_x)
                error = np.abs(x - self.cached_x)
                return error.max() > 1e-8

        def conv_param(self, x):
            asect = x[0:self.wu0.shape[0] * self.wu0.shape[1]]
            conva = torch.from_numpy(asect).view(self.wu0.shape[0], self.wu0.shape[1])
            bsect = x[self.wu0.shape[0] * self.wu0.shape[1]:]
            convb = torch.from_numpy(bsect).view(self.wu0.shape[0], self.wu0.shape[1])
            return conva, convb

        def cache(self, x):
            self.feval += 1
            # convert x to tensor
            atensor, btensor = self.conv_param(x)
            atensor = atensor.to(self.device).type(self.dtype).requires_grad_(True)
            btensor = btensor.to(self.device).type(self.dtype).requires_grad_(True)
            # store the raw array
            self.cached_x = x
            # calculate the objective
            L = self.f(self.q0, atensor, self.wu0, btensor, self.wl0)
            # backprop the objective
            L.backward()
            self.cached_f = L.item()
            if self.feval == 1:
                print("iteration %d" % (self.it))
            print("loss = %.2f" % self.cached_f)
            agrad = atensor.grad.type(torch.float64).cpu().data.numpy().ravel()
            bgrad = btensor.grad.type(torch.float64).cpu().data.numpy().ravel()
            self.cached_jac = np.concatenate([agrad, bgrad])

            self.alist.append(atensor)
            self.blist.append(btensor)
            self.losslist.append(self.cached_f)

        def fun(self, x):
            if self.is_new(x):
                self.cache(x)
            return self.cached_f

        def jac(self, x):
            if self.is_new(x):
                self.cache(x)
            return self.cached_jac

        def callback(self, x):
            self.it += 1
            self.feval = 0


    def optimizeQ(self, w, sigmacurrs, sigmadiffs, sigmaw, gamma=0, beta=0, iters=20):

        """Q optimization step.
            Args:
                w (torch tensor): scalar field of initial widths
                sigmacurrs (list): list of sigmas to compute kernel for dataloss term
                sigmadiffs (list): list of sigmas to compute kernel for deformation term
                sigmaw (torch tensor): sigma for kernel used to determine w
                gamma (float): coefficient of deformation term
                beta (float): coefficient of dataloss term
                iters (int): maximum number of iterations

            Returns:
                pqlist (2d array): list of p's and q's
        """

        Fjoined = mesh.joinFlip(self.FS, self.m, self.n)
        facemap = mesh.incidentFaceMap(2 * self.m * self.n, Fjoined)

        q0 = self.Q.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId).requires_grad_(True)
        Fjoined = Fjoined.clone().detach().to(dtype=torch.long, device=self.torchdeviceId)
        w0 = w.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId).requires_grad_(True)
        VH = self.VH.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId)
        FH = self.FH.clone().detach().to(dtype=torch.long, device=self.torchdeviceId)

        dataloss = self.lossHippSurfQ(Fjoined, facemap, VH, FH, self.sumGaussLinKernel(sigmacurrs))
        loss = self.TotalLossIntegratedQ(self.sumGaussKernel(sigmadiffs), self.GaussKernel(sigmaw), dataloss, gamma=gamma, beta=beta)

        p0 = torch.zeros(q0.shape, dtype=self.torchdtype, device=self.torchdeviceId, requires_grad=True)
        a0 = torch.zeros(w0.shape, dtype=self.torchdtype, device=self.torchdeviceId, requires_grad=True)
        pa = torch.cat((p0.flatten(), a0.flatten()))

        obj = Optimization.PytorchObjectiveQ(loss, pa, q0, w0, self.torchdtype, self.torchdeviceId)

        minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac, callback=obj.callback, options={'disp': True, 'maxiter': iters})

        qreslist = [self.Shooting(ptens, q0, self.sumGaussKernel(sigmadiffs), nt=10)[-1][1] for ptens in obj.plist]
        wreslist = [w0.cpu() + self.GaussKernel(sigmaw.cpu())(q0.cpu(), q0.cpu(), atens.cpu()) for atens in obj.alist]

        self.Qopt = qreslist[-1]
        self.Wopt = wreslist[-1]

        return qreslist, wreslist

    def optimizeW(self, wu, wl, sigmacurrs, sigmaws, gamma=1, beta=1, iters=50):

        """W optimization (nonsymmetric)"""

        Fjoined = mesh.joinFlip(self.FS, self.m, self.n)
        facemap = mesh.incidentFaceMap(2 * self.m * self.n, Fjoined)

        q0 = self.Qopt.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId).requires_grad_(True)
        Fjoined = Fjoined.clone().detach().to(dtype=torch.long, device=self.torchdeviceId)
        wu0 = wu.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId).requires_grad_(True)
        wl0 = wl.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId).requires_grad_(True)
        VH = self.VH.clone().detach().to(dtype=self.torchdtype, device=self.torchdeviceId)
        FH = self.FH.clone().detach().to(dtype=torch.long, device=self.torchdeviceId)

        dataloss = self.lossHippSurfW(Fjoined, facemap, VH, FH, self.sumGaussLinKernel(sigmacurrs))

        loss = self.TotalLossW(self.sumGaussKernel(sigmaws), dataloss, gamma, beta)

        a0 = torch.zeros(wu0.shape, dtype=self.torchdtype, device=self.torchdeviceId, requires_grad=True)
        b0 = torch.zeros(wl0.shape, dtype=self.torchdtype, device=self.torchdeviceId, requires_grad=True)
        ab = torch.cat((a0.flatten(), b0.flatten()))

        obj = Optimization.PytorchObjectiveW(loss, ab, q0, wu0, wl0, self.torchdtype, self.torchdeviceId)

        res = minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac, callback=obj.callback,
                       options={'disp': True, 'maxiter': iters})

        sigmaws_cpu = [sigmaw.cpu() for sigmaw in sigmaws]

        wureslist = [wu0.cpu() + self.sumGaussKernel(sigmaws_cpu)(q0.cpu(), q0.cpu(), arestens.cpu()) for arestens in obj.alist]
        wlreslist = [wl0.cpu() + self.sumGaussKernel(sigmaws_cpu)(q0.cpu(), q0.cpu(), brestens.cpu()) for brestens in obj.blist]

        self.Wuopt = wureslist[-1]
        self.Wlopt = wlreslist[-1]

        return wureslist, wlreslist


    def visualizeMidsurface(self, Q, color = 'Reds'):
        figMS = mesh.visualize(Q.cpu(), self.FS.cpu(), color)
        fig = go.Figure(data=[figMS.data[0], figMS.data[1], figMS.data[2]])
        return fig


    def joinedsurfaceFigure(self, Q, w, color='Blues'):
        Qd = mesh.doubleQ(Q)

        Fjoined = mesh.joinFlip(self.FS, self.m, self.n)
        facemap = mesh.incidentFaceMap(2 * Q.shape[0], Fjoined)

        VS = mesh.generateSourceULW(Qd.cpu(), w.flatten().cpu(), Fjoined.cpu(), facemap)

        figS = mesh.visualize(VS.cpu(), Fjoined.cpu(), color)

        return figS


    def joinedsurfaceFigureNonsymm(self, Q, wu, wl, color='Blues'):
        Qd = mesh.doubleQ(Q)

        Fjoined = mesh.joinFlip(self.FS, self.m, self.n)
        facemap = mesh.incidentFaceMap(2 * Q.shape[0], Fjoined)

        VS = mesh.generateSourceULnonsymm(Qd.cpu(), wu.flatten().cpu(), wl.flatten().cpu(), Fjoined.cpu(), facemap)

        figS = mesh.visualize(VS.cpu(), Fjoined.cpu(), color)

        return figS


    def visualizeJoinedsurface(self, Q, w, midsurface=True):
        figS = self.joinedsurfaceFigure(Q, w)

        if midsurface:
            figMS = mesh.visualize(Q.cpu(), self.FS.cpu(), color = 'Reds')
            fig = go.Figure(data=[figMS.data[0], figMS.data[1], figMS.data[2],
                                  figS.data[0], figS.data[1], figS.data[2]])

        else:
            fig = go.Figure(data=[figS.data[0], figS.data[1], figS.data[2]])

        return fig


    def visualizeJoinedsurfaceNonsymm(self, Q, wu, wl, midsurface=True):
        figS = self.joinedsurfaceFigureNonsymm(Q, wu, wl)

        if midsurface:
            figMS = mesh.visualize(Q.cpu(), self.FS.cpu(), color = 'Reds')
            fig = go.Figure(data=[figMS.data[0], figMS.data[1], figMS.data[2],
                                  figS.data[0], figS.data[1], figS.data[2]])

        else:
            fig = go.Figure(data=[figS.data[0], figS.data[1], figS.data[2]])

        return fig


    def visualizeSourceTarget(self, Q, w, VHds, FHds):
        figMS = mesh.visualize(Q.cpu(), self.FS.cpu(), color = 'Reds')
        figS = self.joinedsurfaceFigure(Q, w)

        figT = mesh.visualize(VHds.cpu(), FHds.cpu(), 'Portland')

        figcomb = go.Figure(data=[figMS.data[0], figMS.data[1], figMS.data[2],
                                  figS.data[0], figS.data[1], figS.data[2],
                                  figT.data[0], figT.data[1], figT.data[2]])

        return figcomb


    def visualizeSourceTargetNonsymm(self, Q, wu, wl, VHds, FHds):
        figMS = mesh.visualize(Q.cpu(), self.FS.cpu(), color = 'Reds')
        figS = self.joinedsurfaceFigureNonsymm(Q, wu, wl)

        figT = mesh.visualize(VHds.cpu(), FHds.cpu(), 'Portland')

        figcomb = go.Figure(data=[figMS.data[0], figMS.data[1], figMS.data[2],
                                  figS.data[0], figS.data[1], figS.data[2],
                                  figT.data[0], figT.data[1], figT.data[2]])

        return figcomb

    def gridDim(self, num_s):
        dim = 50
        while num_s > 0:
            dim = 2*dim - 1
        return dim

    def unfold(self, Q, wu, wl):
        """Unfold joined upper and lower surface to produce thickness map
            Args:
                Q: vertices of optimized midsurface
                wu: optimized upper surface widths
                wl: optimized lower surface widths
            Returns:
                uvw_upper: vertices of upper surface, arranged as a grid
                uvw_lower: vertices of lower surface, arranged as a grid
                uvw_thickness: thickness (distance between upper and lower surfaces), arranged as a grid
        """

        Qgrid = Q.reshape(50, 50, 3)

        du = torch.mean(torch.norm((Qgrid[:, 1:] - Qgrid[:, :-1]), dim=2), dim=0)
        dv = torch.mean(torch.norm((Qgrid[1:] - Qgrid[:-1]), dim=2), dim=1)

        ducum = np.cumsum(du)
        ducum = np.insert(ducum, 0, 0)
        dvcum = np.cumsum(dv)
        dvcum = np.insert(dvcum, 0, 0)

        dugrid, dvgrid = np.meshgrid(ducum, dvcum)

        uvw_upper = np.vstack((dvgrid.flatten(), dugrid.flatten(), wu)).transpose().reshape(50, 50, 3)
        uvw_lower = np.vstack((dvgrid.flatten(), dugrid.flatten(), -1 * wl)).transpose().reshape(50, 50, 3)
        uvw_thickness = np.vstack((dvgrid.flatten(), dugrid.flatten(), wu + wl)).transpose().reshape(50, 50, 3)

        return uvw_upper, uvw_lower, uvw_thickness


    def visualizeUnfolded(self, uvw_upper, uvw_lower, uvw_thickness):
        xmin, xmax = uvw_thickness[..., 0].min(), uvw_thickness[..., 0].max()
        ymin, ymax = uvw_thickness[..., 1].min(), uvw_thickness[..., 1].max()
        zumin, zumax = uvw_upper[..., 2].min().item(), uvw_upper[..., 2].max().item()
        zlmin, zlmax = uvw_lower[..., 2].min().item(), uvw_upper[..., 2].max().item()
        zmin, zmax = min(zumin, zlmin), max(zumax, zlmax)

        Vwuppermesh, Fwuppermesh = mesh.meshSource(uvw_upper)
        figwupper = mesh.visualize(Vwuppermesh, Fwuppermesh, 'Blues')

        Vwlowermesh, Fwlowermesh = mesh.meshSource(uvw_lower)
        figwlower = mesh.visualize(Vwlowermesh, Fwlowermesh,'Blues')

        data = [figwupper.data[0], figwupper.data[1], figwupper.data[2],
                figwlower.data[0], figwlower.data[1], figwlower.data[2]]

        layout = go.Layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
                zaxis=dict(range=[zmin, zmax]),
            ),
            width=1200,
            margin=dict(
                r=20, l=10,
                b=10, t=10)
        )

        fig = go.Figure(data=data, layout=layout)

        fig['layout']['scene'].update(go.layout.Scene(
            aspectmode='manual',
            aspectratio=go.layout.scene.Aspectratio(
                x=(xmax - xmin) / 3, y=(ymax - ymin) / 3, z=(zmax - zmin) / 3
            )
        ))

        Vthickness, Fthickness = mesh.meshSource(uvw_thickness)
        figthickness = mesh.visualize(Vthickness, Fthickness, 'RdBu')
        zthickmin, zthickmax = uvw_thickness[..., 2].min().item(), uvw_thickness[..., 2].max().item()

        datathickness = [figthickness.data[0], figthickness.data[1], figthickness.data[2]]

        layoutthickness = go.Layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
                zaxis=dict(range=[zthickmin, zthickmax]),
            ),
            width=1200,
            margin=dict(
                r=20, l=10,
                b=10, t=10)
        )

        figthickness = go.Figure(data=datathickness, layout=layoutthickness)

        figthickness['layout']['scene'].update(go.layout.Scene(
            aspectmode='manual',
            aspectratio=go.layout.scene.Aspectratio(
                x=(xmax - xmin) / 3, y=(ymax - ymin) / 3, z=(zthickmax - zthickmin) / 3
            )
        ))

        return fig, figthickness

if __name__ == "__main__":
    with open("PycharmProjects/hippocampus/dataframes/spline_splines_4_100_ras.df", "rb") as input:
        surface = pickle.load(input)
    with open("PycharmProjects/hippocampus/dataframes/targetV_ras", "rb") as input:
        VH = pickle.load(input)
    with open("PycharmProjects/hippocampus/dataframes/targetF_ras", "rb") as input:
        FH = pickle.load(input)
    num_points = 50

    source = mesh.downsample(surface, m, n)
    Q, FS = mesh.meshSource(source)



    opt = Optimization(source, VH, FH)

    w = 0.48 * torch.ones(num_points ** 2, 1)
    sigmacurrs = [torch.tensor([.96], dtype=opt.torchdtype, device=opt.torchdeviceId),
                  torch.tensor([0.48], dtype=opt.torchdtype, device=opt.torchdeviceId)]
    sigmadiffs = [torch.tensor([2.4], dtype=opt.torchdtype, device=opt.torchdeviceId),
                  torch.tensor([1.2], dtype=opt.torchdtype, device=opt.torchdeviceId)]
    sigmaw = torch.tensor([3.6], dtype=opt.torchdtype, device=opt.torchdeviceId)
    gamma = 0.12
    beta = 6

    pqlist, wreslist = opt.optimizeQ(w, sigmacurrs, sigmadiffs, sigmaw, gamma, beta)

