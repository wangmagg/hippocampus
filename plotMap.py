import numpy
import pickle
from matplotlib import pyplot as plt, colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import mesh
import argparse as ap
import chart_studio.plotly as py
import plotly.figure_factory as FF
import plotly.graph_objs as go

def get_args():
    p = ap.ArgumentParser()

    p.add_argument("--brain", type = str, required = True, choices = ["2", "3", "4"])

    return p.parse_args()

def visualizeMap(brain, type, ax):

    norm = colors.Normalize(vmin = 0, vmax = 3)

    with open('hippocampus/thicknessMap/dataframes/brain' + brain + '/cartesian_pc_ras_brain' + brain, 'rb') as input:
        cartesian_data_ras = pickle.load(input)

    with open('hippocampus/thicknessMap/dataframes/brain' + brain + '/uvw_thickness_brain' + brain, 'rb') as input:
        uvw_thickness = pickle.load(input)

    with open('hippocampus/thicknessMap/dataframes/brain' + brain + '/sourceQ_optimized_brain' + brain,'rb') as input:
        Q = pickle.load(input)

    Vthickness, Fthickness = mesh.meshSource(uvw_thickness)
    l, lw = mesh.kNN(Q, cartesian_data_ras, 5)
    bd_presub = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 0, t=0.5)
    bd_sub = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 1, t=0.5)
    bd_parasub = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 2, t=0.5)
    bd_ca1 = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 3, t=0.5)
    bd_ca2 = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 4, t=0.5)
    bd_ca3 = mesh.surfaceIsocontour(Vthickness, Fthickness, lw, 5, t=0.5)

    #fig, ax = plt.subplots(figsize = (10, 10))
    tcf = ax.tricontourf(Vthickness[:, 0].detach().numpy(),
                         Vthickness[:, 1].detach().numpy(),
                         Vthickness[:, 2].detach().numpy(),
                         cmap = 'RdBu_r', levels = 100, norm = norm)

    ax.plot(bd_presub[:, 0], bd_presub[:, 1], c='black')
    x, y = np.nanmean(bd_presub[:, 0]), np.mean((np.nanmean(bd_presub[:, 1]), 0))
    #ax.text(x, y, 'PRESUB', fontsize = 18)

    ax.plot(bd_sub[:, 0], bd_sub[:, 1], c='black')
    x, y = np.nanmean(bd_sub[:, 0]), (np.nanmean(bd_sub[:, 1]))
    #ax.text(x, y, 'SUB', fontsize = 18)

    ax.plot(bd_parasub[:, 0], bd_parasub[:, 1], c='black')
    x, y = np.nanmean(bd_parasub[:, 0]), np.mean((np.nanmean(bd_parasub[:, 1]), 0))
    #ax.text(x, y, 'PARASUB', fontsize = 18)

    ax.plot(bd_ca1[:, 0], bd_ca1[:, 1], c='black')
    x, y = np.nanmean(bd_ca1[:, 0]), np.nanmean(bd_ca1[:, 1])
    #ax.text(x, y, 'CA1', fontsize = 18)

    ax.plot(bd_ca2[:, 0], bd_ca2[:, 1], c='black')
    x, y = np.nanmean(bd_ca2[:, 0]), np.nanmean(bd_ca2[:, 1])
    #ax.text(x, y, 'CA2', fontsize = 18)


    ax.plot(bd_ca3[:, 0], bd_ca3[:, 1], c='black')
    x, y = np.nanmean(bd_ca3[:, 0]), np.mean((np.nanmean(bd_ca3[:, 1]), np.max(Vthickness[:, 1].detach().numpy())))
    #ax.text(x, y, 'CA3', fontsize = 18)

    #fig.colorbar(tcf).set_label('Thickness (mm)')
    ax.tick_params(labelsize = 16)
    ax.set_title('Brain ' + brain + ' (' + type + ')', fontsize = 28)
    ax.set_xlabel('Rostral-Caudal Axis (mm)', fontsize = 24)
    ax.set_ylabel('Proximal-Distal Axis (mm)', fontsize = 24)
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_frame_on(False)
    #plt.tight_layout()
    #plt.show()

    return tcf

if __name__ == "__main__":
    #args = get_args()
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True, figsize = (12, 12))
    tcf1 = visualizeMap("3", "Alzheimer's", ax1)
    tcf2 = visualizeMap("4", "Control", ax2)

    cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    cb = fig.colorbar(tcf2, cax = cax, orientation = 'horizontal')
    cb.set_label('Thickness(mm)', fontsize = 24)
    cb.ax.tick_params(labelsize = 16)
    fig.set_tight_layout(True)
    plt.subplots_adjust(bottom = 0.15)
    plt.show()

    #visualizeMapPlotly("3")