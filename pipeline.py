from PointCloud import PointCloud
from Midsurface import Midsurface
import mesh
import Optimization
import torch
import pickle
from plotly.offline import plot

"""Example pipeline from reading of data to completion of optimization """

# Read binary data
pc = PointCloud('/cis/project/exvivohuman_11T/data/subfield_masks/brain_3/eileen_brain3_segmentations/', combined = False, rc_axis = 1)
pc.Cartesian(315, 455, system = "RAS")

# Create midsurface
ms = Midsurface(pc, system = "RAS")
ms.curves(4)
source = ms.surface(100, 100)
#ms.plot_splines(ms.usplines)
#ms.plot_surface()

# Create target mesh
VH, FH = mesh.meshTarget('hippocampus/BrainData/brain3/caSubBrain3.img', 315, 455, system = "RAS")

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


with open('/hippocampus/thicknessMap/dataframes/brain3/cartesian_pc_ras', 'rb') as input:
    cartesian_data_ras = pickle.load(input)

Vthick, Fthick = mesh.meshSource(uvw_thickness)
Q = qreslist[-1].detach().cpu()
l, lw = mesh.kNN(Q, cartesian_data_ras, 5)
bd_sub = mesh.surfaceIsocontour(Vthick, Fthick, lw, 0, t = 0.5)
bd_ca1 = mesh.surfaceIsocontour(Vthick, Fthick, lw, 1, t = 0.5)
bd_ca2 = mesh.surfaceIsocontour(Vthick, Fthick, lw, 2, t = 0.5)
bd_ca3 = mesh.surfaceIsocontour(Vthick, Fthick, lw, 3, t = 0.5)

fig, ax = plt.subplots(figsize = (10, 10))
tcf = ax.tricontourf(Vthick[:, 0].detach().numpy(),
                     Vthick[:, 1].detach().numpy(),
                     Vthick[:, 2].detach().numpy(),
                     cmap = 'RdBu_r', levels = 30)
ax.plot(bd_sub[:, 0], bd_sub[:, 1], c = 'black')
ax.plot(bd_ca1[:, 0], bd_ca1[:, 1], c = 'black')
ax.plot(bd_ca2[:, 0], bd_ca2[:, 1], c = 'black')
ax.plot(bd_ca3[:, 0], bd_ca3[:, 1], c = 'black')
fig.colorbar(tcf).set_label('Thickness (mm)')
ax.set_xlabel('Rostral-Caudal Axis (mm)')
ax.set_ylabel('Proximal-Distal Axis (mm)')
ax.set_aspec('equal')
plt.show()