import numpy as np
import pickle
from scipy.spatial import Delaunay
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as FF
import nibabel as nib

import math
from skimage import measure

import torch
import time

"""Collection of functions for mesh operations"""

def downsample(S, num_v, num_u):
    """Reduce the number of points in the surface prior to meshing
        Args:
            S (numpy ndarray): surface
            num_v (int): number of desired points along v-axis
            num_u (int): number of desired points along u-axis
        Returns:
            S (numpy ndarray): downsampled surface
    """
    index_v = [math.floor(i) for i in np.linspace(0, S.shape[0]-1, num_v)]
    index_u = [math.floor(i) for i in np.linspace(0, S.shape[1] - 1, num_u)]

    S = S[index_v, :, :][:, index_u, :]

    return S

def meshSource(S):
    """Mesh source (midsurface) through Delaunay triangulation.

        Args:
            S (numpy ndarray): surface, downsampled if desired

        Returns:
            tV (torch tensor): mesh vertices (equivalent to flattening the S array)
            tF (torch tensor): mesh faces
    """

    # Points on the midsurface form a grid
    x_grid_axis = np.arange(S.shape[0])
    y_grid_axis = np.arange(S.shape[1])

    x_grid, y_grid = np.meshgrid(x_grid_axis, y_grid_axis)

    # Flatten and concatenate points on the grid
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    points_to_triangulate = np.vstack([x_grid, y_grid]).transpose()

    # Perform Delaunay triangulation
    F = Delaunay(points_to_triangulate).simplices
    tF = torch.as_tensor(F, dtype = torch.long)

    V = S.reshape(S.shape[0]*S.shape[1], S.shape[2])
    tV = torch.as_tensor(V, dtype = torch.float32)

    return tV, tF

def meshSynthTarget(m, n, a, w):
    """Create simple synthetic target and midsurface for testing purposes.

        Args:
            m (int): length of rectangular grid
            n (int): width of rectangular grid
            a (float): amplitude of sine wave
            w (float): distance between reference grid and upper and lower surfaces of the target

        Returns:
            tV (torch tensor): target mesh vertices
            tF (torch tensor): target mesh faces
            tVmid (torch tensor): midsurface mesh vertices
            tFmid (torch tensor): midsurface mesh faces
    """

    # Elevation function that assigns a z-value to each x,y coordinate pair
    def felev(x,y):
        z = a*np.sin(np.pi*x)
        return z, a*np.pi*np.cos(np.pi*x), np.zeros_like(z)

    # Create rectangular mxn grid used to construct the target mesh
    x_grid_axis = np.linspace(0, 2, m)
    y_grid_axis = np.linspace(0, 1, n)

    # Triangulate the rectangular grid
    x_grid, y_grid = np.meshgrid(x_grid_axis, y_grid_axis)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    points_to_triangulate = np.vstack([x_grid, y_grid]).transpose()
    F = Delaunay(points_to_triangulate).simplices

    # Compute z-values on the grid
    z_grid, dxz, dyz = felev(x_grid, y_grid)

    # Create a flat mid-surface defined by the plane z = 0
    midz_grid = np.zeros_like(z_grid)
    midvtcs = np.hstack([x_grid, y_grid, midz_grid])

    # Use the (x,y,z) coordinates defined on the grid as reference vertices to construct an upper and lower surface
    refvtcs = np.hstack([x_grid, y_grid, z_grid])
    numrefvtcs = np.shape(x_grid)[0]

    # Compute the normals along the grid
    normals = np.ones((numrefvtcs, 3))
    normals[:, 0] = -dxz
    normals[:, 1] = -dyz
    norm = np.sqrt(np.sum(normals**2, axis = 1))
    normals = normals/norm

    # Compute the vertex values on the upper and lower surfaces
    uppervtcs = refvtcs + w*normals
    lowervtcs = refvtcs - w*normals
    vtcsfull = np.vstack([uppervtcs, lowervtcs])

    # Define the faces on the lower surface. Flip two points in each triangles so that the normal vector points outwards
    lowerF = F.copy()
    lowerF[:, 0] = F[:, 1]
    lowerF[:, 1] = F[:, 0]

    # Create faces along the edges of the upper and lower surfaces to join the two together
    # Right edge
    upr_edgevtcs = m*np.arange(n) + m - 1
    lowr_edgevtcs = upedgevtcs + numrefvtcs
    redgeFa = np.vstack([upr_edgevtcs[0:-2], lowr_edgevtcs[0:-2], lowr_edgevtcs[1:]]).transpose()
    redgeFb = np.vstack([lowr_edgevtcs[1:], upr_edgevtcs[1:], upr_edgevtcs[0:-2]]).transpose()
    redgeF = np.vstack([redgeFa, redgeFb])

    # Left edge
    upl_edgevtcs = m*np.arange(n)
    lowl_edgevtcs = upedgevtcs + numrefvtcs
    ledgeFa = np.vstack([upl_edgevtcs[0:-2], lowl_edgevtcs[0:-2], lowl_edgevtcs[1:]]).transpose()
    ledgeFb = np.vstack([lowl_edgevtcs[1:], upl_edgevtcs[1:], upl_edgevtcs[0:-2]]).transpose()
    ledgeF = np.vstack([ledgeFa, ledgeFb])

    # Flip vertices to make normal vectors point outwards
    ledgeF_flip = ledgeF.copy()
    ledgeF_flip[:, 0] = ledgeF[:, 1]
    ledgeF_flip[:, 1] = ledgeF[:, 0]

    # Stack all faces to create complete collection of faces
    Ffull = np.vstack([F, lowerF + numrefvtcs, redgeF, ledgeF])

    # Convert everything to torch tensors
    tV = torch.as_tensor(vtcsfull, dtype = torch.float)
    tF = torch.as_tensor(Ffull, dtype = torch.long)
    tVmid = torch.as_tensor(midvtcs, dtype = torch.float)
    tFmid = torch.as_tensor(F, dtype = torch.long)
    
    return tV, tF, tVmid, tFmid
    
def meshTarget(img_file, first_slice, last_slice, system = "voxel"):
    """Mesh binary data through marching cubes.

        Args:
            img_file (str): binary data file
            first_slice (int): Voxel space x-coordinate of first slice (most anterior) in desired section
            last_slice (int): Voxel space x-coordinate of last slice (most posterior) in desired section
            system (str): specify coordinate system (voxel or RAS); default voxel

        Returns:
            tV (torch tensor): target mesh vertices
            tF (torch tensor): target mesh faces
    """

    # Load data
    fname = img_file
    img = nib.load(fname)
    hdr = img.header

    nxi = hdr['dim'][1:4]

    # Select the desired section of data
    data = img.get_data()
    data = data.reshape((nxi[0], nxi[1], nxi[2]))
    data = data[first_slice:last_slice]

    # Create mesh using a marching cubes algorithm
    # V and F are the vertices and faces, respectively
    # To downsample so that mesh is less fine, use larger step_size
    V, F, N, v = measure.marching_cubes_lewiner(data, level = 0, step_size = 2)

    # Add the x-value of the first slice to the first coordinate of all vertices
    V[:, 0] += first_slice

    if system == "RAS":
        M = img.affine[:3, :3]
        abc = img.affine[:3, 3]
        V = V.dot(M) + np.tile(abc, (V.shape[0], 1))

    Vtri = V[F]
    uniqueVidx = []  # array for storing the indices of faces with all three vertices unique

    for i in range(F.shape[0]):
        if np.unique(Vtri[i], axis = 0).shape[0] == 3:
            uniqueVidx.append(i)

    # Find indices of unique faces (two faces are considered unique if one or more of their vertices is different)
    uniqueT, uniqueTidx = np.unique(Vtri, axis = 0, return_index = True)

    # Compute intersection of the indices and keep only these faces
    keep = np.intersect1d(uniqueVidx, uniqueTidx)
    F = F[keep]

    # Flip orientation of normal vectors so that they point outwards
    if system == "voxel":
        Fflip = F.copy()
        Fflip[:, 0] = F[:, 1]
        Fflip[:, 1] = F[:, 0]
        F = Fflip

    # Convert to torch tensors
    tV = torch.as_tensor(V.copy(), dtype = torch.float32)
    tF = torch.as_tensor(F.copy(), dtype = torch.long)

    return tV, tF

def compCN(V, F):
    """Compute centroids and normals for a mesh.

        Args:
            V (torch tensor): mesh vertices
            F (torch tensor): mesh faces

        Returns:
            C (torch tensor): centroids
            N (torch tensor): normal vectors (unnormalized, i.e. magnitude = area of face)

    """

    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    C = (V0 + V1 + V2) / 3
    N = 0.5*torch.cross(V1-V0, V2-V0)

    return C, N

def sumAreas(Nlist):
    """Helper function that computes the sum of the areas of all faces incident to a particular vertex.

        Args:
            Nlist (array): normal vectors of the the faces incident to a particular vertex

        Returns:
            lt (float): sum of the areas of the faces incident to the vertex
    """

    # Euclidean norm of a vector
    def normEuclidean(N):
        return (N ** 2).sum().sqrt()

    # Compute Euclidean norm of all the normal vectors in Nlist
    l = list(map(lambda x: normEuclidean(x), Nlist))

    # Sum the norms
    lt = torch.as_tensor(l).sum()

    return lt

def incidentFaceMap(num_points, FSj):
    """Generate dictionary mapping each vertex to its incident faces.

        Args:
            num_points (int): total number of vertices
            FSj (int): total number of faces

        Returns:
            index_map (dict): dictionary mapping a vertex to its incident faces (faces containing the vertex)
    """

    # Initialize dictionary where each vertex maps to an empty array
    # Indices of incident faces will be added to the array
    index_map = dict((i, []) for i in range(num_points))

    # Each face is a 3x1 array containing the indices of the vertices belonging to the face
    # Each face has a position/index in the face array
    # For every face, append the face's own index to the array mapped to by each vertex belonging to the face
    for i in range(FSj.shape[0]):
        for index in FSj[i]:
            index_map[int(index)].append(i)

    return index_map

# rename to verticesUL_wconst
def surfaceULfast(VSj, index_map, N, w):
    """Compute vertices for the upper and lower surfaces, each at distance w from the midsurface

        Args:
            VSj (torch tensor): Duplicated midsurface vertices
            index_map (dict): dictionary mapping vertices to incident faces
            N (torch tensor): normals
            w (float): distance from midsurface to the upper/lower surface. Constant scalar.

        Returns:
            VSj (torch tensor): vertices of upper and lower surface
    """

    for i in range(int(VSj.shape[0]/2)):
        nN = N[index_map[i]].sum(dim=0) / sumAreas(N[index_map[i]])
        VSj[i + int(VSj.shape[0]/2)] = VSj[i + int(VSj.shape[0]/2)] - w*nN
        VSj[i] = VSj[i] + w * nN

    return VSj

# rename to verticesUL_wfield
def surfaceULW(VSj, W, index_map, N):
    """Compute vertices for the upper and lower surfaces, each at distance W[i] from the midsurface

        Args:
            VSj (torch tensor): Duplicated midsurface vertices
            W (torch tensor): distance from midsurface to the upper/lower surface. Scalar field.
            index_map (dict): dictionary mapping vertices to incident faces
            N (torch tensor): normals

        Returns:
            VSj (torch tensor): vertices of upper and lower surface
    """

    for i in range(int(VSj.shape[0]/2)):
        nN = N[index_map[i]].sum(dim=0) / sumAreas(N[index_map[i]])
        VSj[i + int(VSj.shape[0]/2)] = VSj[i] - W[i]*nN
        VSj[i] = VSj[i] + W[i] * nN

    return VSj


def surfaceULnonsymm(VSj, Wu, Wl, index_map, N):
    """Compute vertices for the upper and lower surfaces, each at a distance Wu[i], Wl[i] from the midsurface, respectively

        Args:
            VSj (torch tensor): Duplicated midsurface vertices
            Wu (torch tensor): distance from midsurface to the upper surface. Scalar field.
            Wl (torch tensor): distance from midsurface to lower surface. Scalar field.
            index_map (dict): dictionary mapping vertices to incident faces
            N (torch tensor): normals

        Returns:
            VSj (torch tensor): vertices of upper and lower surface
    """

    for i in range(int(VSj.shape[0] / 2)):
        nN = N[index_map[i]].sum(dim=0) / sumAreas(N[index_map[i]])
        VSj[i + int(VSj.shape[0] / 2)] = VSj[i] - Wu[i] * nN
        VSj[i] = VSj[i] + Wl[i] * nN

    return VSj


def doubleQ(Q):
    """Duplicate the midsurface vertices. Required for optimization

        Args:
            Q: vertices on the midsurface

        Returns:
            Qd: duplicated midsurface vertices (two midsurfaces concatenated)

    """
    Qd = torch.cat((Q, Q), 0)
    return Qd

# rename to facesUL
def joinULfast(Fc, num_points):
    """Compute faces that join the upper and lower surfaces together.

        Args:
            Fc (torch tensor): faces of duplicated midsurface
            num_points (int): number of vertices on the midsurface

        Returns:
            Fall (torch tensor): all faces for the joined surfaces
    """

    # Compute faces on the edges that join the upper and surfaces
    pts = np.arange(num_points - 1)
    f1 = np.array([num_points*pts, num_points*(pts + 1), num_points*pts + num_points**2])
    f2 = np.array([num_points*pts + num_points**2, num_points*(pts + 1) + num_points**2, num_points*(pts + 1)])
    f3 = f1 + num_points - 1
    f4 = f2 + num_points - 1

    # Concatenate faces
    newF = np.concatenate([f2.transpose(), f3.transpose(), f1.transpose(), f4.transpose()], axis = 0)
    Fall = torch.cat((Fc, torch.as_tensor(newF)), 0)

    return Fall

def joinFlip(F, m, n):
    """Compute faces for joined upper and lower surfaces, with orientation flipping.

        Args:
            F (torch tensor): faces of midsurface
            m (int): number of vertices along u-axis of midsurface grid
            n (int): number of vertices along v-axis of midsurface grid

        Returns:
            tFjoined (torch tensor): all faces for the joined surfaces
    """

    # Create faces for lower surface
    lowerF = F.clone()
    lowerF[:, 0] = F[:, 1]
    lowerF[:, 1] = F[:, 0]

    # Create faces for right border
    upr_edgevtcs = m*np.arange(n) + m - 1
    lowr_edgevtcs = upr_edgevtcs + m*n
    redgeFa = np.vstack([upr_edgevtcs[0:-1], lowr_edgevtcs[0:-1], lowr_edgevtcs[1:]]).transpose()
    redgeFb = np.vstack([lowr_edgevtcs[1:], upr_edgevtcs[1:], upr_edgevtcs[0:-1]]).transpose()
    redgeF = np.vstack([redgeFa, redgeFb])

    # Create faces for left border
    upl_edgevtcs = m*np.arange(n)
    lowl_edgevtcs = upl_edgevtcs + m*n
    ledgeFa = np.vstack([upl_edgevtcs[0:-1], lowl_edgevtcs[0:-1], lowl_edgevtcs[1:]]).transpose()
    ledgeFb = np.vstack([lowl_edgevtcs[1:], upl_edgevtcs[1:], upl_edgevtcs[0:-1]]).transpose()
    ledgeF = np.vstack([ledgeFa, ledgeFb])
    
    ledgeF_flip = ledgeF.copy()
    ledgeF_flip[:, 0] = ledgeF[:, 1]
    ledgeF_flip[:, 1] = ledgeF[:, 0]

    # Join all faces together
    Fjoined = np.vstack([F, lowerF + m*n, redgeF, ledgeF_flip])
    tFjoined = torch.as_tensor(Fjoined, dtype = torch.long)
    
    return tFjoined

def joinFlipSubdivided(F, m, n, t, edge_oddv_map):
    """Compute faces for joined upper and lower surfaces for subdivided mesh, with orientation flipping.

        Args:
            F (torch tensor): faces of subdivided midsurface
            m (int): number of vertices along u-axis of midsurface grid, before subdivision
            n (int): number of vertices along v-axis of midsurface grid, before subdivision
            t (int): total number of vertices of subdivided midsurface
            edge_oddv_map (dict): maps each edge in the original midsurface to the index of the vertex placed on that edge
                                    during subdivision. Ex: '[0 50]': 3000

        Returns:
            tFjoined (torch tensor): all faces for the joined surfaces
    """

    # Create faces for lower surface
    lowerF = F.clone()
    lowerF[:, 0] = F[:, 1]
    lowerF[:, 1] = F[:, 0]

    # Create faces for right border
    upr_edgevtcs = m*np.arange(n) + m - 1
    lowr_edgevtcs = upr_edgevtcs + t
    
    urk = [str(np.array([50*(i+1) - 1, 50*(i+2)-1])) for i in range(49)]  # all of the edges on the right border
    upr_edgevtcs_sub = np.array([edge_oddv_map[k] for k in urk])
    lowr_edgevtcs_sub = upr_edgevtcs_sub + t
    
    redgeFa = np.vstack([upr_edgevtcs[0:-1], lowr_edgevtcs[0:-1], lowr_edgevtcs_sub]).transpose()
    redgeFb = np.vstack([lowr_edgevtcs_sub, upr_edgevtcs_sub, upr_edgevtcs[0:-1]]).transpose()
    redgeFc = np.vstack([upr_edgevtcs_sub, lowr_edgevtcs_sub, lowr_edgevtcs[1:]]).transpose()
    redgeFd = np.vstack([lowr_edgevtcs[1:], upr_edgevtcs[1:], upr_edgevtcs_sub]).transpose()
    redgeF = np.vstack([redgeFa, redgeFb, redgeFc, redgeFd])

    # Create faces for left border
    upl_edgevtcs = m*np.arange(n)
    lowl_edgevtcs = upl_edgevtcs + t

    ulk = [str(np.array([50*(i+1), 50*i])) for i in range(49)] # all of the edges on the left border
    upl_edgevtcs_sub = np.array([edge_oddv_map[k] for k in ulk])
    lowl_edgevtcs_sub = upl_edgevtcs_sub + t
    
    ledgeFa = np.vstack([upl_edgevtcs[0:-1], lowl_edgevtcs[0:-1], lowl_edgevtcs_sub]).transpose()
    ledgeFb = np.vstack([lowl_edgevtcs_sub, upl_edgevtcs_sub, upl_edgevtcs[0:-1]]).transpose()
    ledgeFc = np.vstack([upl_edgevtcs_sub, lowl_edgevtcs_sub, lowl_edgevtcs[1:]]).transpose()
    ledgeFd = np.vstack([lowl_edgevtcs[1:], upl_edgevtcs[1:], upl_edgevtcs_sub]).transpose()
    ledgeF = np.vstack([ledgeFa, ledgeFb, ledgeFc, ledgeFd])
    
    ledgeF_flip = ledgeF.copy()
    ledgeF_flip[:, 0] = ledgeF[:, 1]
    ledgeF_flip[:, 1] = ledgeF[:, 0]

    # Join all faces together
    Fjoined = np.vstack([F, lowerF + t, redgeF, ledgeF_flip])
    tFjoined = torch.as_tensor(Fjoined, dtype = torch.long)
    
    return tFjoined

def generateSourceULfast(Qd, w, Fjpre, map):
    """Function called by user to generate upper and lower surfaces with constant w.

        Args:
            Qd (torch tensor): duplicated midsurface vertices
            w (float): distance from midsurface to the upper/lower surface. Constant scalar.
            Fjpre (torch tensor): faces of joined upper and lower surfaces
            map (dict): maps each vertex to incident faces

        Returns:
            Vul: vertices of upper and lower surfaces
    """

    C, N = compCN(Qd, Fjpre)
    Vul = surfaceULfast(Qd, map, N, w)

    return Vul

def generateSourceULW(Qd, W, Fjpre, map):
    """Function called by user to generate upper and lower surfaces with scalar field w.

       Args:
           Qd (torch tensor): duplicated midsurface vertices
           w (torch tensor): distance from midsurface to the upper/lower surface. Scalar field.
           Fjpre (torch tensor): faces of joined upper and lower surfaces
           map (dict): maps each vertex to incident faces

       Returns:
           Vul: vertices of upper and lower surfaces
    """

    C, N = compCN(Qd, Fjpre)
    Vul = surfaceULW(Qd, W, map, N)

    return Vul

def generateSourceULnonsymm(Vj, Wu, Wl, FSj, fmap):
    """Function called by user to generate upper and lower surfaces with scalar fields Wu, Wl.

       Args:
           Qd (torch tensor): duplicated midsurface vertices
           Wu (torch tensor): distance from midsurface to the upper surface. Scalar field.
           Wl (torch tensor): distance from midsurface to the lower surface. Scalar field.
           Fjpre (torch tensor): faces of joined upper and lower surfaces
           map (dict): maps each vertex to incident faces

       Returns:
           Vul: vertices of upper and lower surfaces
    """

    C, N = compCN(Vj, FSj)
    Vul = surfaceULnonsymm(Vj, Wu, Wl, fmap, N)

    return Vul

def generateTarget(img_file, f, l, system = "voxel"):
    """Function called by user to generate target mesh and compute its centroids and normals.

        Args:
            img_file (str): name of file with binary data
            f (int): Voxel space x-coordinate of first slice (most anterior) in desired section
            l (int): Voxel space x-coordinate of last slice (most posterior) in desired section

        Returns:
            V (torch tensor): vertices of target mesh
            F (torch tensor): faces of target mesh
            C (torch tensor): centroids of target mesh
            N (torch tensor): normals of target mesh
    """
    V, F = meshTarget(img_file, f, l, system = system)
    C, N = compCN(V, F)

    return V, F, C, N


def flatten(mesh, m, n):
    """Create flattened grid from mesh vertices, preserving global distances

        Args:
            mesh (torch tensor): mesh vertices
            m (int): number of vertices in u-axis of grid
            n (int): number of vertices in v-axis of grid

        Returns:
            dugrid.flatten() (numpy array): vertices on u-axes in flattened grid
            dvgrid.flatten() (numpy array): vertices on v-axes in flattened grid

    """

    mesh = mesh.reshape(m, n, 3)
    du = torch.mean(torch.norm((mesh[:, 1:] - mesh[:, :-1]), dim=2), dim=0)
    dv = torch.mean(torch.norm((mesh[1:] - mesh[:-1]), dim=2), dim=1)

    ducum = np.cumsum(du)
    ducum = np.insert(ducum, 0, 0)
    dvcum = np.cumsum(dv)
    dvcum = np.insert(dvcum, 0, 0)

    dugrid, dvgrid = np.meshgrid(ducum, dvcum)

    return dugrid.flatten(), dvgrid.flatten()


def plot_mesh(V, F, color):

    fig_mesh = FF.create_trisurf(x=V[:, 0].detach().numpy(),
                                 y=V[:, 1].detach().numpy(),
                                 z=V[:, 2].detach().numpy(),
                                 colormap=color,
                                 simplices=F)

    fig_mesh['data'][0].update(opacity=1)

    return fig_mesh

def plot_mesh_with_normals(V, F, N, C, color):
    fig_mesh = plot_mesh(V, F, color)
    T = C.detach().numpy() + N.detach().numpy()
    padded = np.zeros((3 * T.shape[0], 3))

    for i in range(T.shape[0]):
        padded[3 * i] = C[i].detach().numpy()
        padded[3 * i + 1] = T[i]
        padded[3 * i + 2] = [None, None, None]

    trace = go.Scatter3d(
        x=[pt[0] for pt in padded],
        y=[pt[1] for pt in padded],
        z=[pt[2] for pt in padded],
        mode='lines + markers',
        line=dict(
            color='blue',
            width=4
        ),
        marker=dict(
            color='red',
            size=1
        )
    )
    data = [trace, fig_mesh.data[0], fig_mesh.data[1], fig_mesh.data[2]]
    fig_combined = go.Figure(data=data)

    return fig_combined

def visualize(V, F, color, normals = False):

    if normals:
        C, N = compCN(V, F)
        fig = plot_mesh_with_normals(V, F, N, C, color)
    else:
        fig = plot_mesh(V, F, color)

    return fig


if __name__ == "__main__":
    with open("PycharmProjects/hippocampus/dataframes/spline_splines_4_100_ras.df", "rb") as input:
        surface = pickle.load(input)

    surface = downsample(surface, 50, 50)
    W = 0.48*torch.ones(50**2, 1)

    Q, F = meshSource(surface)
    C, N = compCN(Q, F)
    Qd = doubleQ(Q)

    Fjoined = joinFlip(F, 50, 50)
    facemap = incidentFaceMap(2 * 50 * 50, Fjoined)
    VS = generateSourceULW(Qd, W, Fjoined, facemap)
    CS, NS = compCN(VS, Fjoined)

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

    figMS = visualize(Q, F, 'Reds')
    figMSonly = go.Figure(data = [figMS.data[0], figMS.data[1], figMS.data[2]], layout = layout)
    plotly.offline.plot(figMSonly)

    figS = visualize(VS, Fjoined, 'Blues')
    figSonly = go.Figure(data = [figS.data[0], figS.data[1], figS.data[2]], layout = layout)
    plotly.offline.plot(figSonly)

    figMS_S = go.Figure(data = [figMS.data[0], figMS.data[1], figMS.data[2],
                                figS.data[0], figS.data[1], figS.data[2]], layout = layout)
    plotly.offline.plot(figMS_S)

    img = "ENS_summer_2019/ca_sub_combined.img"
    #VT, FT, CT, NT = generateTarget(img, 311, 399, system = "RAS")

    #with open("PycharmProjects/hippocampus/dataframes/targetVds_ras", "wb") as output:
        #pickle.dump(VT, output)

    #with open("PycharmProjects/hippocampus/dataframes/targetFds_ras", "wb") as output:
        #pickle.dump(FT, output)

    with open("PycharmProjects/hippocampus/dataframes/targetVds_ras", "rb") as input:
        VT = pickle.load(input)

    with open("PycharmProjects/hippocampus/dataframes/targetFds_ras", "rb") as input:
        FT = pickle.load(input)

    figT = visualize(VT, FT, 'Portland', normals = False)

    figTonly = go.Figure(data = [figT.data[0], figT.data[1], figT.data[2]], layout = layout)

    plotly.offline.plot(figTonly)
    
    figcomb = go.Figure(data = [figMS.data[0], figMS.data[1], figMS.data[2],
                                figS.data[0], figS.data[1], figS.data[2],
                                figT.data[0], figT.data[1], figT.data[2]], layout = layout)

    plotly.offline.plot(figcomb)

