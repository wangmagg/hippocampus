import sys
import numpy as np
import pandas as pd
from io import BytesIO
import os
from matplotlib import pyplot as plt
import math
import pickle
import plotly
import plotly.graph_objs as go
import scipy.interpolate as interpolate
from PointCloud import PointCloud

class Midsurface:
    """Create midsurface from Cartesian point cloud data using manual selection of control points
        and B-spline interpolation.

        Args:
            pointcloud: PointCloud object

        Attributes:
            data: store PointCloud object
    """

    class LineBuilderUpdate:
        """Provides interface for midcurve point selection

        Args:
            line (obj): matplotlib ax object with single arbitrary initial point

        Attributes:
            line (obj): ax object with selected midcurve points
            xs (arr): array of x-coordinates of points on midcurve
            ys (arr): array of y-coordinates of points on midcurve

         """
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            #self.click = None
            #self.push = None

        def connect(self):
            """Links canvas updates to keyboard and mouse actions"""

            self.cidclick = self.line.figure.canvas.mpl_connect('button_press_event', self.on_click)
            self.cidpush = self.line.figure.canvas.mpl_connect('key_press_event', self.on_push)

        def on_click(self, event):
            """Append x-, y- coordinate data of cursor position when mouse is clicked"""
            if event.inaxes != self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs[1:], self.ys[1:])
            self.line.figure.canvas.draw()

        def on_push(self, event):
            """Delete most recently added coordinate when key is pressed"""
            if event.inaxes != self.line.axes: return
            self.xs.pop(-1)
            self.ys.pop(-1)
            self.line.set_data(self.xs[1:], self.ys[1:])
            self.line.figure.canvas.draw()

        def points(self):
            """Get points on midcurve (with initial arbitrary point removed)"""
            return self.xs[1:], self.ys[1:]

    def __init__(self, pointcloud, system = "voxel"):
        self.pc = pointcloud
        self.sys = system

    def subsample(self, num_slices):
        """Reduce number of slices by subsampling.

            Args:
                num_slices (int): number of desired slices
                cartesian_data: dataframe with Cartesian coordinates

            Returns:
                data_slices (int): subsampled dataframe

        """

        if self.sys == "voxel":
            data = self.pc.cartesian_data
        else:
            data = self.pc.cartesian_data_ras

        #total = data.shape[0]
        total = data[self.pc.rc_axis].unique().shape[0]
        slice_idxs = [math.floor(i) for i in np.linspace(0, total-1, num_slices)]
        #data_slices = data.iloc[slice_idxs]
        data_slices = data.loc[data[self.pc.rc_axis].isin(data[self.pc.rc_axis].unique()[slice_idxs])]
        self.data_ds = data_slices

        return data_slices

    def _curvesUncombined(self, cartesian_data_ds):

        curves_y = []
        curves_z = []

        if self.pc.rc_axis == 0:
            self.ax1, self.ax2 = [1, 2]
        else:
            self.ax1, self.ax2 = [0, 2]

        for num, xval in enumerate(cartesian_data_ds[self.pc.rc_axis].unique()):
            print(xval)
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111)

            slice_data = cartesian_data_ds.loc[cartesian_data_ds[self.pc.rc_axis] == xval]

            ax.plot(slice_data.loc[slice_data['label'] == 'ca1'][self.ax1], slice_data.loc[slice_data['label'] == 'ca1'][self.ax2],
                    color = 'pink', marker='o', markersize=2, markerfacecolor='None', linestyle="None")
            ax.plot(slice_data.loc[slice_data['label'] == 'ca2'][self.ax1], slice_data.loc[slice_data['label'] == 'ca2'][self.ax2],
                    color = 'red', marker='o', markersize=2, markerfacecolor='None', linestyle="None")
            ax.plot(slice_data.loc[slice_data['label'] == 'ca3'][self.ax1], slice_data.loc[slice_data['label'] == 'ca3'][self.ax2],
                    color = 'gold', marker='o', markersize=2, markerfacecolor='None', linestyle="None")
            ax.plot(slice_data.loc[slice_data['label'] == 'subiculum'][self.ax1], slice_data.loc[slice_data['label'] == 'subiculum'][self.ax2],
                    color = 'fuchsia', marker='o', markersize=2, markerfacecolor='None', linestyle="None")
            ax.plot(slice_data.loc[slice_data['label'] == 'presubiculum'][self.ax1],
                    slice_data.loc[slice_data['label'] == 'presubiculum'][self.ax2],
                    color='darkviolet', marker='o', markersize=2, markerfacecolor='None', linestyle="None")
            ax.plot(slice_data.loc[slice_data['label'] == 'parasubiculum'][self.ax1],
                    slice_data.loc[slice_data['label'] == 'parasubiculum'][self.ax2],
                    color='indigo', marker='o', markersize=2, markerfacecolor='None', linestyle="None")

            start_pt, = ax.plot(slice_data.iloc[0][self.ax1], slice_data.iloc[0][self.ax2], marker='o', markersize=4, color='blue')

            linebuilder = Midsurface.LineBuilderUpdate(start_pt)
            linebuilder.connect()
            ax.set_title("Slice %d of %d" % (num, cartesian_data_ds[self.pc.rc_axis].unique().shape[0] - 1))
            plt.show()

            curve_y, curve_z = linebuilder.points()
            curves_y.append(curve_y)
            curves_z.append(curve_z)
            #bound_y.append([curve_y[8], curve_y[13], curve_y[15]])
            #bound_z.append([curve_z[8], curve_z[13], curve_z[15]])

        idx = cartesian_data_ds[self.pc.rc_axis].unique()
        curvesy_df = pd.DataFrame(curves_y, index= idx)
        curvesz_df = pd.DataFrame(curves_z, index = idx)
        #boundy_df = pd.DataFrame(bound_y, index = idx)
        #boundz_df = pd.DataFrame(bound_z, index = idx)

        self.curvesy = curvesy_df
        self.curvesz = curvesz_df
        #self.boundy = boundy_df
        #self.boundz = boundz_df

        return curvesy_df, curvesz_df

    def _curvesUncombined_v0(self, cartesian_data_ds):
        """Manually select points from each slice to create mid-curves; for uncombined binary data files

            Args:
                cartesian_data_ds (pandas): subsampled dataframe

            Returns:
                curvesy_df (pandas): y-coordinates of all points on mid-curves
                curvesz_df (pandas): z-coordinates of all points on mid-curves
        """
        colors = {'ca1': 'pink', 'ca2': 'red', 'ca3': 'gold', 'subiculum': 'fuchsia', 'presubiculum':'darkviolet', 'parasubiculum':'indigo'}

        curves_y = []
        curves_z = []
        bound_y = []
        bound_z = []

        for num, slice in enumerate(cartesian_data_ds.index):

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111)

            for col in cartesian_data_ds.columns:
                ax.plot(cartesian_data_ds[col][slice][1], cartesian_data_ds[col][slice][2],
                        color=colors[col], marker='o', markersize=2, markerfacecolor='None', linestyle="None")

            start_pt, = ax.plot([cartesian_data_ds['ca1'][slice][1][0]], [cartesian_data_ds['ca1'][slice][2][0]],
                                marker='o', markersize=4, color='blue')
            linebuilder = Midsurface.LineBuilderUpdate(start_pt)
            linebuilder.connect()
            ax.set_title("Slice %d of %d" % (num, cartesian_data_ds.shape[0] - 1))
            plt.show()

            curve_y, curve_z = linebuilder.points()
            curves_y.append(curve_y)
            curves_z.append(curve_z)
            bound_y.append([curve_y[8], curve_y[13], curve_y[15]])
            bound_z.append([curve_z[8], curve_z[13], curve_z[15]])

        curvesy_df = pd.DataFrame(curves_y, index=cartesian_data_ds.index)
        curvesz_df = pd.DataFrame(curves_z, index=cartesian_data_ds.index)
        boundy_df = pd.DataFrame(bound_y, index = cartesian_data_ds.index)
        boundz_df = pd.DataFrame(bound_z, index = cartesian_data_ds.index)

        self.curvesy = curvesy_df
        self.curvesz = curvesz_df
        self.boundy = boundy_df
        self.boundz = boundz_df

        with open("PycharmProjects/hippocampus/dataframes/test_curvesy", "wb") as output:
            pickle.dump(curvesy_df, output)
        with open("PycharmProjects/hippocampus/dataframes/test_curvesz", "wb") as output:
            pickle.dump(curvesz_df, output)
        with open("PycharmProjects/hippocampus/dataframes/test_boundy", "wb") as output:
            pickle.dump(boundy_df, output)
        with open("PycharmProjects/hippocampus/dataframes/test_boundz", "wb") as output:
            pickle.dump(boundz_df, output)

        return curvesy_df, curvesz_df

    def _curvesCombined(self, cartesian_data_ds):
        """Manually select points from each slice to create mid-curves; for combined binary data files

            Args:
                cartesian_data_ds (pandas): subsampled dataframe

            Returns:
                curvesy_df (pandas): y-coordinates of all points on mid-curves
                curvesz_df (pandas): z-coordinates of all points on mid-curves
        """

        curves_y = []
        curves_z = []

        for num, slice in enumerate(cartesian_data_ds.index):
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111)

            ax.plot(cartesian_data_ds[0][slice][1], cartesian_data_ds[0][slice][2], color='red', marker='o',
                    markersize=2,
                    markerfacecolor='None', linestyle="None")

            start_pt, = ax.plot([cartesian_data_ds[0][slice][1][0]], [cartesian_data_ds[0][slice][2][0]],
                                marker='o', markersize=4, color='blue')
            linebuilder = Midsurface.LineBuilderUpdate(start_pt)
            linebuilder.connect()
            ax.set_title("Slice %d of %d" % (num, cartesian_data_ds.shape[0] - 1))
            plt.show()

            curve_y, curve_z = linebuilder.points()
            curves_y.append(curve_y)
            curves_z.append(curve_z)

        curvesy_df = pd.DataFrame(curves_y, index=cartesian_data_ds.index)
        curvesz_df = pd.DataFrame(curves_z, index=cartesian_data_ds.index)

        self.curvesy = curvesy_df
        self.curvesz = curvesz_df

        return curvesy_df, curvesz_df

    def curves(self, num_slices):
        """Public method invoked by user to generate mid-curves.

            Args:
                num_slices (int): Number of slices to downsample data to and create mid_curves for
                system (str): system used for Cartesian coordinates (voxel or RAS); default voxel

            Returns:
                curvesy_df (pandas): y-coordinates of all points on mid-curves
                curvesz_df (pandas): z-coordinates of all points on mid-curves

        """

        self.subsample(num_slices)

        if not self.pc.comb:
            curvesy_df, curvesz_df = self._curvesUncombined(self.data_ds)
        else:
            curvesy_df, curvesz_df = self._curvesCombined(self.data_ds)

        return curvesy_df, curvesz_df

    def curves_cached(self, num_slices):
        self.subsample(num_slices)

        with open('/hippocampus/thicknessMap/dataframes/test_curvesy', 'rb') as input:
            curvesy_df = pickle.load(input)
        with open('/hippocampus/thicknessMap/dataframes/test_curvesz', 'rb') as input:
            curvesz_df = pickle.load(input)

        self.curvesy = curvesy_df
        self.curvesz = curvesz_df

        return curvesy_df, curvesz_df

    def plot_curves(self):
        """Plot selected mid-curve points in 3D

            Args:
                curvesy_df (pandas): y-coordinates of all points on mid-curves
                curvesz_df (pandas): z-coordinates of all points on mid-curves

        """
        if self.pc.rc_axis == 0:
            x = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            y = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        else:
            x = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            y = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        trace = go.Scatter3d(
            x = coord[0],
            y = coord[1],
            z = coord[2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue'
            ),
            line=dict(
                color='black',
                width=2
            )
        )

        data = [trace]
        fig = go.Figure(data=data)
        plotly.offline.plot(fig)

    def _spline_u(self, num_points):
        """Interpolate manually selected midcurve points with cubic B-spline interpolation along u-axis

            Args:
                num_points (int): number of interpolating points

            Returns:
                coord: numpy array containing the coordinates of the interpolating B-splines
        """
        coord = np.zeros((self.curvesy.shape[0], num_points, 3))

        for level, i in enumerate(self.curvesy.index):
            y = np.array(self.curvesy.loc[i])
            z = np.array(self.curvesz.loc[i])

            y = y[~np.isnan(y)]
            z = z[~np.isnan(z)]

            tck, u = interpolate.splprep([y, z], s=0)
            yi, zi = interpolate.splev(np.linspace(0, 1, num_points), tck)
            x = [i for j in range(num_points)]

            '''
            coord[level, ..., 0] = x
            coord[level, ..., 1] = yi
            coord[level, ..., 2] = zi
            '''
            coord[level, ..., self.pc.rc_axis] = x
            coord[level, ..., self.ax1] = yi
            coord[level, ..., self.ax2] = zi

        self.usplines = coord
        #self.pt_num = np.floor(num_points/self.pt_num)

        return coord

    def _spline_v(self, num_points):
        """Interpolate manually selected midcurve points with cubic B-spline interpolation along v-axis

            Args:
                coord (numpy ndarr): coordinates from B-spline interpolation along u-axis
                num_points (int): number of interpolating points

            Returns:
                coord_interp: numpy array containing coordinates of B-splines (both u- and v- axes)
        """
        coord_interp = np.zeros((num_points, self.usplines.shape[1], self.usplines.shape[2]))

        for i in range(self.usplines.shape[1]):
            x_knots = self.usplines[..., i, self.pc.rc_axis]
            y_knots = self.usplines[..., i, self.ax1]
            z_knots = self.usplines[..., i, self.ax2]

            tck, u = interpolate.splprep([x_knots, z_knots], s=0)
            xi, zi = interpolate.splev(np.linspace(0, 1, num_points), tck)

            tck, u = interpolate.splprep([x_knots, y_knots], s=0)
            xi_other, yi = interpolate.splev(np.linspace(0, 1, num_points), tck)

            coord_interp[..., i, self.pc.rc_axis] = xi
            coord_interp[..., i, self.ax1] = yi
            coord_interp[..., i, self.ax2] = zi

            self.surf = coord_interp

        return coord_interp


    def surface(self, num_u, num_v):
        """Function invoked by user to generate interpolated surface

            Args:
                num_u (int): number of interpolating points along u-axis
                num_v (int): number of interpolating points along v-axis

            Returns:
                coord_interp: interpolated surface
        """
        self._spline_u(num_u)
        self._spline_v(num_v)

        return self.surf

    def _plot_splines_uncombined(self, splines):
        """Plot splines in u-axis overlayed on slices where original binary data was uncombined"""

        if self.pc.rc_axis == 0:
            x = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            y = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        else:
            x = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            y = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        trace1 = go.Scatter3d(
            x= splines[..., 0].flatten(),
            y= splines[..., 1].flatten(),
            z= splines[..., 2].flatten(),
            mode='markers',
            marker=dict(
                size=2,
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                ),
                colorscale='Viridis'
            )
        )
        trace2 = go.Scatter3d(
            x=coord[0],
            y=coord[1],
            z=coord[2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                )
            )
        )

        data = [trace1, trace2]
        labels = ['ca1', 'ca2', 'ca3', 'subiculum', 'presubiculum', 'parasubiculum']
        colors = {'ca1': 'pink', 'ca2': 'red', 'ca3': 'gold', 'subiculum': 'fuchsia', 'presubiculum': 'darkviolet',
                  'parasubiculum': 'indigo'}

        for l in labels:
            trace = go.Scatter3d(
                x = self.data_ds.loc[self.data_ds['label'] == l][0],
                y = self.data_ds.loc[self.data_ds['label'] == l][1],
                z = self.data_ds.loc[self.data_ds['label'] == l][2],

                mode='markers',
                marker=dict(
                    size=2,
                    color=colors[l],
                    opacity=0.5,
                )

            )

            data.append(trace)

        fig = go.Figure(data=data)
        plotly.offline.plot(fig)

    def _plot_splines_combined(self, splines):
        """Plot splines in u-axis overlayed on slices where original binary data was combined"""

        if self.pc.rc_axis == 0:
            x = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            y = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        else:
            x = np.concatenate([self.curvesy.loc[idx] for idx in self.curvesy.index])
            y = np.repeat(self.curvesy.index, self.curvesy.shape[1])
            z = np.concatenate([self.curvesz.loc[idx] for idx in self.curvesz.index])
            coord = [x,y,z]

        trace1 = go.Scatter3d(
            x=splines[..., 0].flatten(),
            y=splines[..., 1].flatten(),
            z=splines[..., 2].flatten(),
            mode='markers',
            marker=dict(
                size=2,
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                ),
                colorscale='Viridis'
            )
        )

        trace2 = go.Scatter3d(
            x=coord[0],
            y=coord[1],
            z=coord[2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=1,
                line=dict(
                    color='black',
                    width=0.5
                )
            )
        )

        data = [trace1, trace2]

        for col in self.data_ds.columns:
            trace = go.Scatter3d(
                x=np.concatenate([self.data_ds[col][idx][0] for idx in self.data_ds.index]),
                y=np.concatenate([self.data_ds[col][idx][1] for idx in self.data_ds.index]),
                z=np.concatenate([self.data_ds[col][idx][2] for idx in self.data_ds.index]),
                mode='markers',
                marker=dict(
                    size=2,
                    color='firebrick',
                    opacity=0.5,
                )
            )

            data.append(trace)
        
        layout = go.Layout(
                    scene = dict(
                        xaxis = dict(
                            title='Posterior-Anterior (mm)'),
                        yaxis = dict(
                            title='Right-Left (mm)'),
                        zaxis = dict(
                            title = 'Superior-Inferior (mm)')
                    )
                )
        
        fig = go.Figure(data=data, layout = layout)
        plotly.offline.plot(fig)

    def plot_splines(self, splines):
        """Function invoked by user to plot splines along u-axis"""
        if self.pc.comb:
            self._plot_splines_combined(splines)
        else:
            self._plot_splines_uncombined(splines)

    def plot_surface(self):
        """Function invoked by user to plot surface"""

        trace = go.Scatter3d(
            x=self.surf[..., 0].flatten(),
            y=self.surf[..., 1].flatten(),
            z=self.surf[..., 2].flatten(),

            mode='markers',
            marker=dict(
                size=2,
                opacity=1,
                color=self.surf[..., 1].flatten(),
                colorscale='Reds'
            )

        )

        data = [trace]
        fig = go.Figure(data=data)
        plotly.offline.plot(fig)


if __name__ == "__main__":
    pc = PointCloud('/cis/project/exvivohuman_11T/data/subfield_masks/brain_3/eileen_brain3_segmentations/', combined = False, rc_axis = 1)
    pc.Cartesian(315, 455)

    ms = Midsurface(pc, system = "RAS")
    ms.curves(4)
    #ms.curves_cached(4)

    surface = ms.surface(100, 100)
    ms.plot_splines(ms.usplines)

    #with open("PycharmProjects/hippocampus/dataframes/spline_splines_4_100_ras_biasedlow.df", "wb") as output:
    #    pickle.dump(surface, output)

    #ms.plot_surface()