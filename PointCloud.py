import sys
import numpy as np
import pandas as pd
from io import BytesIO
import os
import nibabel as nib
from nibabel.affines import apply_affine
from matplotlib import pyplot as plt
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



    def __init__(self, rawdatapath, combined = True):
        self.path = rawdatapath
        self.comb = combined

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
        self._M = img.affine[:3, :3]
        self._abc = img.affine[:3, 3]
        ras = self._M.dot(data_all[:-1]) + np.tile(self._abc, (data_all.shape[1], 1)).transpose()

        #Store as pandas dataframes
        data_df = pd.DataFrame(np.transpose(data_all))[[0, 1, 2]]
        data_df_ras = pd.DataFrame(np.transpose(ras))

        return data_df, data_df_ras

    def _arrangeBySlice(self, xmin, xmax, data_df, data_df_ras):
        """Select section from Cartesian data that lies between the desired slice bounds.
         Arrange dataframe for selected section so that each row corresponds to a single coronal cross-section.

            Args:
                xmin (int): Voxel space x-coordinate of first slice in desired section
                xmax (int): Voxel space x-coordinate of last slice in desired section

            Returns:
                df_arranged (pandas): data from selected section arranged by slice, Voxel space
                df_arranged_ras (pandas): data from selected section arranged by slice, RAS space
        """

        #Define range of x-coordinates (the x-coordinate is constant across a coronal slice)
        xvals = np.arange(xmin, xmax)
        index = xvals
        index_ras = np.zeros_like(xvals, dtype = float)

        #Initialize empty dataframes to store arranged slices
        df_arranged = pd.DataFrame(index = index, columns = [0])
        df_arranged_ras = pd.DataFrame(index = index, columns = [0])

        #Locate "slices" in dataframe where the x-value is constant and store in new dataframes
        for i, x in enumerate(xvals):

            df = data_df.loc[data_df[0] == x]
            dfidx = df.index.tolist()
            rasdf = data_df_ras.loc[dfidx]
            index_ras[i] = list(rasdf[0])[0]

            df_arranged[0][x] = [list(df[0]), list(df[1]), list(df[2])]
            df_arranged_ras[0][x] = [list(rasdf[0]), list(rasdf[1]), list(rasdf[2])]

        df_arranged_ras.index = index_ras

        return df_arranged, df_arranged_ras


    def _joinCartesian(self, xmin, xmax):
        """Convert all binary files in path to Cartesian space and select desired section
            Args:
                xmin (int): Voxel space x-coordinate of first slice in desired section
                xmax (int): Voxel space x-coordinate of last slice in desired section
            Returns:
                data_img_df (pandas): Voxel space Cartesian data from desired section
                data_img_ras_df (pandas): RAS space Cartesian data from desired section
        """
        img_files = ['ca1', 'ca2', 'ca3', 'subiculum']

        #Temporary holding dictionaries for data from each file
        #data_img = {'ca1': [], 'ca2': [], 'ca3': [], 'subiculum': []}
        #data_img_ras = {'ca1': [], 'ca2': [], 'ca3': [], 'subiculum': []}
        data_img_list = []
        data_img_ras_list = []

        #For each file, convert to Cartesian and arrange by slice
        for img in img_files:
            data_df, data_ras_df = self._toCartesian(self.path + img + '.img')
            data_df['label'] = img
            data_ras_df['label'] = img
            data_img_list.append(data_df)
            data_img_ras_list.append(data_ras_df)
            #data_df_arr, data_ras_df_arr = self._arrangeBySlice(xmin, xmax, data_df, data_ras_df)
            #data_img[img] = data_df_arr[0]
            #data_img_ras[img] = data_ras_df_arr[0]

        #Convert dictionaries to dataframes
        #data_img_df = pd.DataFrame(data_img)
        #data_img_ras_df = pd.DataFrame(data_img_ras)
        data_img_df = pd.concat(data_img_list, ignore_index = True)
        data_img_ras_df = pd.concat(data_img_ras_list, ignore_index = True)
        data_img_df = data_img_df.loc[(data_img_df[0] >= xmin) & (data_img_df[0] <= xmax)]
        data_img_ras_df = data_img_ras_df.loc[data_img_df.index]

        return data_img_df, data_img_ras_df

    def Cartesian(self, xmin, xmax, system = "voxel"):
        """
        Public method invoked by user to perform conversion from binary data to Cartesian space.

            Args:
                xmin (int): Voxel space x-coordinate of first slice in desired section
                xmax (int): Voxel space x-coordinate of last slice in desired section
                system (str): specify coordinate system (voxel or RAS); default voxel
        """

        if self.comb:
            data, data_ras = self._toCartesian(self.path)
            self.cartesian_data, self.cartesian_data_ras = self._arrangeBySlice(xmin, xmax, data, data_ras)

        else:
            self.cartesian_data, self.cartesian_data_ras = self._joinCartesian(xmin, xmax)

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
        colors = {'ca1': 'orangered', 'ca2': 'darkorange', 'ca3': 'gold', 'subiculum': 'firebrick'}
        data = []

        for col in cartesian_data.columns:
            trace = go.Scatter3d(
                x=np.concatenate([cartesian_data[col][idx][0] for idx in cartesian_data.index]),
                y=np.concatenate([cartesian_data[col][idx][1] for idx in cartesian_data.index]),
                z=np.concatenate([cartesian_data[col][idx][2] for idx in cartesian_data.index]),
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors[col],
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
            x=np.concatenate([cartesian_data[0][idx][0] for idx in cartesian_data.index]),
            y=np.concatenate([cartesian_data[0][idx][1] for idx in cartesian_data.index]),
            z=np.concatenate([cartesian_data[0][idx][2] for idx in cartesian_data.index]),
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
                    title='Anterior-Posterior (mm)'),
                yaxis=dict(
                    title='Left-Right (mm)'),
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
    pc = PointCloud('Documents/research/ENS/ca_sub_combined.img')
    pc.Cartesian(311, 399)
    #print(pc.cartesian_data)
    #pc.plot(system = "RAS")

    pc_uc = PointCloud('Documents/research/ENS/brain_2/eileen_brain2_segmentations/', combined = False)
    pc_uc.Cartesian(311, 399)

    with open('PycharmProjects/hippocampus/dataframes/cartesian_pc_ras', 'wb') as output:
        pickle.dump(pc_uc.cartesian_data_ras, output)

    #pc_uc.plot(system = "RAS")