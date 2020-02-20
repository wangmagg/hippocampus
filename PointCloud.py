import sys
import numpy as np
import pandas as pd
from io import BytesIO
import os
import nibabel as nib
from nibabel.affines import apply_affine
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import pickle
import plotly
import plotly.graph_objs as go

class PointCloud:
    """For reading binary point cloud data and storing as Cartesian coordinates.

    Args:
        rawdatapath (str): path name containing binary data segmentation files
        combined (bool): True if rawdatapath is a single file of combined segmentations outputted from
                        Daniel's combine.m function; False otherwise

    Attributes:
        path (str): stores rawdatapath
        comb (bool): stores combined
        M (arr): rotation/scaling part of image affine matrix (for performing conversion to RAS coordinates)
        abc (arra): translation part of image affine matrix (for performing conversion to RAS coordinates)
        cartesian_data (pandas): dataframe storing cartesian data in voxel index space
        cartesian_data_ras (pandas): dataframe storing cartesian data in RAS coordinates

    """



    def __init__(self, rawdatapath, combined = True, rc_axis = 0):
        self.path = rawdatapath
        self.comb = combined
        self.rc_axis = rc_axis

    def _toCartesian(self, fname):
        """Read binary data from single file and store in Cartesian space

            Args:
                fname: name of file with binary data

            Returns:
                data_df: dataframe of Cartesian data in voxel space
                data_df_ras: dataframe of Cartesian data in RAS space
        """

        img = nib.load(fname)
        data = img.get_data()

        #Voxel space coordinates
        data_all = np.array(np.nonzero(data))

        #RAS coordinate conversion
        self._M = abs(img.affine[:3, :3])
        self._abc = img.affine[:3, 3]
        ras = self._M.dot(data_all[:-1]) + np.tile(self._abc, (data_all.shape[1], 1)).transpose()

        #Store as pandas dataframes
        data_df = pd.DataFrame(np.transpose(data_all))[[0, 1, 2]]
        data_df_ras = pd.DataFrame(np.transpose(ras))

        return data_df, data_df_ras

    def _joinCartesian(self, min, max):
        """Convert all binary files in path to Cartesian space and select desired section
            Args:
                xmin (int): Voxel space x-coordinate of first slice in desired section
                xmax (int): Voxel space x-coordinate of last slice in desired section
            Returns:
                data_img_df (pandas): Voxel space Cartesian data from desired section
                data_img_ras_df (pandas): RAS space Cartesian data from desired section
        """
        img_files = ['ca1', 'ca2', 'ca3', 'presubiculum','subiculum','parasubiculum']

        #Temporary holding lists for data from each file
        data_img_list = []
        data_img_ras_list = []

        #For each file, convert to Cartesian and append to list
        for img in img_files:
            data_df, data_ras_df = self._toCartesian(self.path + img + '.img')
            data_df['label'] = img
            data_ras_df['label'] = img
            data_img_list.append(data_df)
            data_img_ras_list.append(data_ras_df)

        #Concatenate dataframes in lists
        data_img_df = pd.concat(data_img_list, ignore_index = True)
        data_img_ras_df = pd.concat(data_img_ras_list, ignore_index = True)
        data_img_df = data_img_df.loc[(data_img_df[self.rc_axis] >= min) & (data_img_df[self.rc_axis] <= max)]
        data_img_ras_df = data_img_ras_df.loc[data_img_df.index]

        return data_img_df, data_img_ras_df

    def Cartesian(self, min, max, system = "voxel"):
        """
        Public method invoked by user to perform conversion from binary data to Cartesian space.

            Args:
                xmin (int): Voxel space x-coordinate of first slice in desired section
                xmax (int): Voxel space x-coordinate of last slice in desired section
                system (str): specify coordinate system (voxel or RAS); default voxel
        """

        if self.comb:
            data, data_ras = self._toCartesian(self.path)
            self.cartesian_data = data.loc[(data[self.rc_axis] >= min) & (data[self.rc_axis] <= max)]
            self.cartesian_data_ras = data_ras.loc[self.cartesian_data.index]

        else:
            self.cartesian_data, self.cartesian_data_ras = self._joinCartesian(min, max)

        if system == "voxel":
            return self.cartesian_data

        elif system == "RAS":
            return self.cartesian_data_ras

        else:
            print("Error: unrecognized coordinate system")

    def _plotUncombined(self, cartesian_data):
        """Plotting function for data that was not combined into single binary file
            Args:
                cartesian_data (pandas): dataframe with Cartesian data
        """
        colors = {'ca1': 'pink', 'ca2': 'red', 'ca3': 'gold', 'subiculum': 'fuchsia', 'presubiculum':'darkviolet', 'parasubiculum':'indigo'}
        data = []

        #for col in cartesian_data.columns:
        trace = go.Scatter3d(
            x=cartesian_data[0],
            y=cartesian_data[1],
            z=cartesian_data[2],
            mode='markers',
            marker=dict(
                size=2,
                color=[colors[cartesian_data['label'][idx]] for idx in cartesian_data.index],
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                )
            )
        )

        data.append(trace)

        layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title='Anterior-Posterior (mm)'),
                yaxis=dict(
                    title='Left-Right (mm)'),
                zaxis=dict(
                    title='Superior-Inferior (mm)')
            )
        )

        fig = go.Figure(data=data, layout = layout)
        plotly.offline.plot(fig)

    def _plotCombined(self, cartesian_data):
        """Plotting function for data that was combined into single binary file (using combine.m)
                Args:
                    cartesian_data (pandas): dataframe with Cartesian data
        """
        data = []

        trace = go.Scatter3d(
            #x=np.concatenate([cartesian_data[0][idx] for idx in cartesian_data.index]),
            #y=np.concatenate([cartesian_data[1][idx] for idx in cartesian_data.index]),
            #z=np.concatenate([cartesian_data[2][idx] for idx in cartesian_data.index]),
            x = cartesian_data[0],
            y = cartesian_data[1],
            z = cartesian_data[2],
            mode='markers',
            marker=dict(
                size=2,
                color='firebrick',
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                )
            )
        )

        data.append(trace)

        layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title='Proximal-Distal (mm)'),
                yaxis=dict(
                    title='Anterior-Posterior (mm)'),
                zaxis=dict(
                    title='Superior-Inferior (mm)')
            )
        )

        fig = go.Figure(data=data, layout = layout)
        plotly.offline.plot(fig)

    def plot(self, system = "voxel"):
        """Public method invoked by user to plot Cartesian data.

            Args:
                system: coordinate system to plot in (voxel or RAS); default voxel
        """
        if system == "voxel":
            if self.comb:
                self._plotCombined(self.cartesian_data)
            else:
                self._plotUncombined(self.cartesian_data)

        elif system == "RAS":
            if self.comb:
                self._plotCombined(self.cartesian_data_ras)
            else:
                self._plotUncombined(self.cartesian_data_ras)

        else:
            print("Error: unrecognized coordinate system")




if __name__ == "__main__":
    # brain 2: (311, 399)
    # brain 3: (315, 455)
    # brain 4: (224, 361)

    pc = PointCloud('hippocampus/BrainData/brain' + sys.argv[1] + '/caSubBrain' + sys.argv[1] + '.img')
    pc.Cartesian(sys.argv[2], sys.argv[3])
    pc.plot(system = "RAS")


    pc_uc = PointCloud('/cis/project/exvivohuman_11T/data/subfield_masks/brain_' + sys.argv[1] + '/eileen_brain' + sys.argv[1] + '_segmentations/', combined = False, rc_axis = 1)
    pc_uc.Cartesian(sys.argv[2], sys.argv[3])
    pc_uc.plot(system = 'RAS')


    with open('hippocampus/thicknessMap/dataframes/brain' + sys.argv[1] + '/cartesian_pc_ras', 'wb') as output:
        pickle.dump(pc_uc.cartesian_data_ras, output)