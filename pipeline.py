from PointCloud import PointCloud
from Midsurface import Midsurface
import mesh
import Optimization
import torch
import pickle
from plotly.offline import plot
import argparse as ap

"""Example pipeline from reading of data to completion of optimization """

def get_args():
    p = ap.ArgumentParser()

    p.add_argument("--brain", type = str, required = True, choices = ["2", "3", "4"])
    p.add_argument("--first_slice", type = int, required = True)
    p.add_argument("--last_slice", type = int, required = True)
    p.add_argument("--cached_surface", type = int, required = True, choices = [0, 1])
    p.add_argument("--rc_axis", type = int, required = True, choices = [0,1])

    return p.parse_args()



def build(args):
    if args.cached_surface == 0:
        # Read binary data
        pc = PointCloud('/cis/project/exvivohuman_11T/data/subfield_masks/brain_' + args.brain + '/eileen_brain' + args.brain + '_segmentations/', combined = False, rc_axis = args.rc_axis)
        pc.Cartesian(args.first_slice, args.last_slice, system = "RAS")

        with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/cartesian_pc_ras_brain' + args.brain, 'wb') as output:
            pickle.dump(pc_uc.cartesian_data_ras, output)

        # Create midsurface
        ms = Midsurface(pc, system = "RAS")
        ms.curves(4)
        source = ms.surface(100, 100)

    else:
        with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/sourcePC', 'rb') as input:
            source = pickle.load(input)

    # Create target mesh
    VH, FH = mesh.meshTarget('hippocampus/BrainData/brain' + args.brain + '/caSubBrain' + args.brain + '.img', args.first_slice, args.last_slice, system = "RAS", rc_axis = args.rc_axis)
    VHds, FHds = mesh.meshTarget('hippocampus/BrainData/brain' + args.brain + '/caSubBrain' + args.brain + '.img', args.first_slice, args.last_slice, system = "RAS", rc_axis = args.rc_axis, step = 2)

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

    #qreslist, wreslist = opt.optimizeQ(w, sigmacurrs, sigmadiffs, sigmaw, gamma, beta)
    opt.optimizeQ(w, sigmacurrs, sigmadiffs, sigmaw, gamma, beta)

    # Visualize midsurface optimization results
    figMS = opt.visualizeMidsurface(opt.Qopt)
    figJoined = opt.visualizeJoinedsurface(opt.Qopt, opt.Wopt.flatten())
    figSourceTarget = opt.visualizeSourceTarget(opt.Qopt, opt.Wopt.flatten(), VHds, FHds)

    plot(figMS)
    plot(figJoined)
    plot(figSourceTarget)

    # Visualize unfolded surface and thickness map after midsurface optimization
    #uvw_upper, uvw_lower, uvw_thickness = opt.unfold(qreslist[-1].detach().cpu(), wreslist[-1].flatten().detach().cpu(), wreslist[-1].flatten().detach().cpu())
    uvw_upper, uvw_lower, uvw_thickness = opt.unfold(opt.Qopt.detach().cpu(), opt.Wopt.flatten().detach().cpu(), opt.Wopt.flatten().detach().cpu())
    figW, figThickness = opt.visualizeUnfolded(uvw_upper, uvw_lower, uvw_thickness)

    plot(figW)
    plot(figThickness)

    # Optimize W scalar field
    #wu = wreslist[-1].clone()
    #wl = wreslist[-1].clone()
    sigmacurrs = [torch.tensor([1], dtype = opt.torchdtype, device = opt.torchdeviceId),
                  torch.tensor([0.6], dtype = opt.torchdtype, device = opt.torchdeviceId)]
    sigmaws = [torch.tensor([3], dtype = opt.torchdtype, device = opt.torchdeviceId),
               torch.tensor([1], dtype = opt.torchdtype, device = opt.torchdeviceId),
              torch.tensor([0.3], dtype = opt.torchdtype, device = opt.torchdeviceId)]
    gamma = 1
    beta = 1

    # Visualize W optimization results
    #wureslist, wlreslist = opt.optimizeW(wu, wl, sigmacurrs, sigmaws, gamma, beta)
    opt.optimizeW(opt.Wopt, opt.Wopt, sigmacurrs, sigmaws, gamma, beta)
    #figJoinedNonsymm = opt.visualizeJoinedsurfaceNonsymm(qreslist[-1], wureslist[-1].flatten(), wlreslist[-1].flatten())
    #figSourceTargetNonsymm = opt.visualizeSourceTargetNonsymm(qreslist[-1], wureslist[-1].flatten(), wlreslist[-1].flatten(),VHds, FHds)

    # Visualize unfolded surface and thickness map after W optimization
    #uvw_upper, uvw_lower, uvw_thickness = opt.unfold(qreslist[-1].detach().cpu(), abs(wureslist[-1].flatten().detach().cpu()), abs(wlreslist[-1].flatten().detach().cpu()))
    uvw_upper, uvw_lower, uvw_thickness = opt.unfold(opt.Qopt.detach().cpu(), abs(opt.Wuopt.flatten().detach().cpu()),
                                                     abs(opt.Wlopt.flatten().detach().cpu()))
    #figW, figThickness = opt.visualizeUnfolded(uvw_upper, uvw_lower, uvw_thickness)

    #Q = qreslist[-1].detach().cpu()
    Qd = mesh.doubleQ(opt.Qopt.detach().cpu())

    Fjoined = mesh.joinFlip(opt.FS, 50, 50)
    facemap = mesh.incidentFaceMap(2 * 50 * 50, Fjoined)
    VS = mesh.generateSourceULnonsymm(Qd, abs(opt.Wuopt.detach().flatten()), abs(opt.Wlopt.detach().flatten()),
                                      Fjoined, facemap)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/sourceUpper_optimized_brain' + args.brain, 'wb') as output:
        pickle.dump(VS[0:2500].detach(), output)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/sourceLower_optimized_brain' + args.brain, 'wb') as output:
        pickle.dump(VS[2500:], output)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/sourceFaces_brain' + args.brain, 'wb') as output:
        pickle.dump(opt.FS, output)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/uvw_thickness_brain' + args.brain, 'wb') as output:
        pickle.dump(uvw_thickness, output)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/sourceQ_optimized_brain' + args.brain, 'wb') as output:
        pickle.dump(opt.Qopt.detach().cpu(), output)

    with open('hippocampus/thicknessMap/dataframes/brain' + args.brain + '/opt_brain' + args.brain, 'wb') as output:
        pickle.dump(opt, output)


if __name__ == "__main__":
    args = get_args()
    build(args)