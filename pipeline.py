import PointCloud
import Midsurface
import mesh
import Optimization
import torch

from plotly.offline import plot

"""Example pipeline from reading of data to completion of optimization """

# Read binary data
pc = PointCloud('ENS_summer_2019/ca_sub_combined.img')
pc.Cartesian(311, 399, system = "RAS")

# Create midsurface
ms = Midsurface(pc, system = "RAS")
ms.curves(4)
surface = ms.surface(100, 100)

ms.plot_splines()
ms.plot_surface()

# Create target mesh
VH, FH = mesh.meshTarget('ENS_summer_2019/ca_sub_combined.img', 311, 399, system = "RAS")

# Optimize midsurface
m = 50
n = 50
opt = Optimization.Optimization(source, VH, FH)
w = 0.48 * torch.ones(m*n, 1)
sigmacurrs = [torch.tensor([.96], dtype=opt.torchdtype, device=opt.torchdeviceId),
              torch.tensor([0.48], dtype=opt.torchdtype, device=opt.torchdeviceId)]
sigmadiffs = [torch.tensor([2.4], dtype=opt.torchdtype, device=opt.torchdeviceId),
              torch.tensor([1.2], dtype=opt.torchdtype, device=opt.torchdeviceId)]
sigmaw = torch.tensor([3.6], dtype=opt.torchdtype, device=opt.torchdeviceId)
gamma = 0.12
beta = 6

qreslist, wreslist = opt.optimizeQ(w, sigmacurrs, sigmadiffs, sigmaw, gamma, beta)

# Visualize midsurface optimization results
figMS = opt.visualizeMidsurface(qreslist[-1])
figJoined = opt.visualizeJoinedsurface(qreslist[-1], wreslist[-1].flatten())
figSourceTarget = opt.visualizeSourceTarget(qreslist[-1], wreslist[-1].flatten(), VHds, FHds)

plot(figMS)
plot(figJoined)
plot(figSourceTarget)

# Visualize unfolded surface and thickness map after midsurface optimization
uvw_upper, uvw_lower, uvw_thickness = opt.unfold(qreslist[-1].detach().cpu(), wreslist[-1].flatten().detach().cpu(), wreslist[-1].flatten().detach().cpu())
figW, figThickness = opt.visualizeUnfolded(uvw_upper, uvw_lower, uvw_thickness)

plot(figW)
plot(figThickness)

# Optimize W scalar field
wu = wreslist[-1].clone()
wl = wreslist[-1].clone()
sigmacurrs = [torch.tensor([1], dtype = opt.torchdtype, device = opt.torchdeviceId),
              torch.tensor([0.6], dtype = opt.torchdtype, device = opt.torchdeviceId)]
sigmaws = [torch.tensor([3], dtype = opt.torchdtype, device = opt.torchdeviceId),
           torch.tensor([1], dtype = opt.torchdtype, device = opt.torchdeviceId),
          torch.tensor([0.3], dtype = opt.torchdtype, device = opt.torchdeviceId)]
gamma = 1
beta = 1

# Visualize W optimization results
wureslist, wlreslist = opt.optimizeW(wu, wl, sigmacurrs, sigmaws, gamma, beta)
figJoinedNonsymm = opt.visualizeJoinedsurfaceNonsymm(qreslist[-1], wureslist[-1].flatten(), wlreslist[-1].flatten())
figSourceTargetNonsymm = opt.visualizeSourceTargetNonsymm(qreslist[-1], wureslist[-1].flatten(), wlreslist[-1].flatten(),VHds, FHds)

# Visualize unfolded surface and thickness map after W optimization
uvw_upper, uvw_lower, uvw_thickness = opt.unfold(qreslist[-1].detach().cpu(), wureslist[-1].flatten().detach().cpu(), wlreslist[-1].flatten().detach().cpu())
figW, figThickness = opt.visualizeUnfolded(uvw_upper, uvw_lower, uvw_thickness)