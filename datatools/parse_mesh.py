import numpy as np
import matplotlib.pyplot as plt
from fluidfoam import readscalar, readfield
from fluidfoam.readof import OpenFoamFile
import matplotlib.tri as tri
import torch
from torch_geometric.data import Data
import contextlib
with contextlib.redirect_stdout(None):
    from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
import seaborn as sns
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_scatter import scatter_sum
from scipy.spatial import cKDTree

def compute_signed_distances_vectorized(points, airfoil_segments, airfoil_tree=None):
    """
    Vectorized computation of signed distances from points to airfoil surface
    params:
        points: np.array of shape (N, 2) containing query points
        airfoil_segments: np.array of shape (M, 2, 2) containing line segments
        airfoil_tree: optional pre-computed KDTree for airfoil points
    returns:
        np.array of shape (N,) containing signed distances
    """
    N = len(points)
    M = len(airfoil_segments)
    
    # Convert inputs to numpy arrays if they aren't already
    points = np.asarray(points)
    airfoil_segments = np.asarray(airfoil_segments)
    
    # Use KD-tree to find nearest airfoil points for each query point
    if airfoil_tree is None:
        airfoil_points = airfoil_segments[:, 0]  # Use start points of segments
        airfoil_tree = cKDTree(airfoil_points)
    
    # Find nearest segment for each point
    _, nearest_idx = airfoil_tree.query(points)
    nearest_idx = np.clip(nearest_idx, 0, M-1)
    
    # Compute distances to nearest segments
    p1 = airfoil_segments[nearest_idx, 0]
    p2 = airfoil_segments[nearest_idx, 1]
    
    # Vector from p1 to p2 for each segment
    segment_vectors = p2 - p1
    # Vector from p1 to query points
    point_vectors = points - p1
    
    # Compute projection
    segment_lengths = np.sum(segment_vectors**2, axis=1)
    t = np.sum(point_vectors * segment_vectors, axis=1) / segment_lengths
    t = np.clip(t, 0, 1)
    
    # Find closest points on segments
    closest_points = p1 + (t[:, np.newaxis] * segment_vectors)
    
    # Compute distances
    distances = np.sqrt(np.sum((points - closest_points)**2, axis=1))
    
    # Compute signs using winding number algorithm (vectorized)
    signs = np.ones(N)
    
    # Prepare arrays for vectorized winding number calculation
    y = points[:, 1][:, np.newaxis]  # Shape (N, 1)
    y1 = airfoil_segments[:, 0, 1]  # Shape (M,)
    y2 = airfoil_segments[:, 1, 1]  # Shape (M,)
    x = points[:, 0][:, np.newaxis]  # Shape (N, 1)
    x1 = airfoil_segments[:, 0, 0]  # Shape (M,)
    x2 = airfoil_segments[:, 1, 0]  # Shape (M,)
    
    # Broadcast for vectorized computation
    mask1 = (y1 <= y) & (y2 > y)  # Shape (N, M)
    mask2 = (y1 > y) & (y2 <= y)  # Shape (N, M)
    
    # Compute x-intersections where y crosses segment
    with np.errstate(divide='ignore', invalid='ignore'):
        intersect_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
        intersect_x = np.where(y2 == y1, x1, intersect_x)  # Handle vertical segments
    
    # Count winding number contributions
    winding = np.sum(np.where(mask1, (intersect_x < x), 0), axis=1) - \
              np.sum(np.where(mask2, (intersect_x < x), 0), axis=1)
    
    # Points with non-zero winding numbers are inside
    signs[winding != 0] = -1
    
    return signs * distances


def parse_mesh_to_graph(sim_dir, sim_time, max_radius=None, roughness_feat=False, area_feat=False,  add_farfield_node=True, plot=False):

    # read the different mesh files
    owners = OpenFoamFile(sim_dir + "/constant/polyMesh/", name="owner")
    neighbours = OpenFoamFile(sim_dir + "/constant/polyMesh/", name="neighbour")
    faces = OpenFoamFile(sim_dir + "/constant/polyMesh/", name="faces")
    points =  OpenFoamFile(sim_dir + "/constant/polyMesh/", name="points", precision=15)
    # fix for blank lines affecting fluidfoam parsing
    delete_blank_lines(sim_dir + "/constant/polyMesh/" + "boundary")
    bounfile =  OpenFoamFile(sim_dir + "/constant/polyMesh/", name="boundary")

    # back faces
    id0_back_faces = int(bounfile.boundaryface[str.encode('BACK')][b"startFace"])
    n_back_faces = int(bounfile.boundaryface[str.encode('BACK')][b"nFaces"])

    # airfoil faces
    id0_airfoil_faces = int(bounfile.boundaryface[str.encode('AIRFOIL')][b"startFace"])
    n_airfoil_faces = int(bounfile.boundaryface[str.encode('AIRFOIL')][b"nFaces"])

    # rough bot airfoil faces
    if os.path.exists(sim_dir + "/constant/polyMesh/sets/AIRFOIL_Rough_Bot"):
        n_rough_airfoil_bot_faces = len(get_face_set(sim_dir + "/constant/polyMesh/sets/AIRFOIL_Rough_Bot"))
    else:
        n_rough_airfoil_bot_faces = 0
    id0_rb = id0_airfoil_faces + n_airfoil_faces - n_rough_airfoil_bot_faces
    if n_rough_airfoil_bot_faces != 0:
        k_dict = ParsedParameterFile(os.path.join(sim_dir, '0', 'k'))
        r_bot =  k_dict['boundaryField']['AIRFOIL_Rough_Bot']['Ks']
    else:
        r_bot = 0

    # rough top airfoil faces
    if os.path.exists(sim_dir + "/constant/polyMesh/sets/AIRFOIL_Rough_Top"):
        n_rough_airfoil_top_faces = len(get_face_set(sim_dir + "/constant/polyMesh/sets/AIRFOIL_Rough_Top"))
    else:
        n_rough_airfoil_top_faces = 0
    id0_rt = id0_rb - n_rough_airfoil_top_faces
    if n_rough_airfoil_top_faces!= 0:
        k_dict = ParsedParameterFile(os.path.join(sim_dir, '0', 'k'))
        r_top = k_dict['boundaryField']['AIRFOIL_Rough_Top']['Ks']
    else:
        r_top = 0

    # farfield faces
    id0_farfield_faces = int(bounfile.boundaryface[str.encode('FARFIELD')][b"startFace"])
    n_farfield_faces = int(bounfile.boundaryface[str.encode('FARFIELD')][b"nFaces"])

    elements = [] #list of faces for plotting
    nodes_t = [] # node types
    nodes_a = [] # area of cell for each node
    nodes_r = [] # roughness
    node_pos_dict = {}
    cells_dict = {} # dict which relates cell face num to the cell
    cell_connectivity = []
    cell_face_s = [] # list of the surface area of each cell face
    interior_cells = []
    interior_node_count = 0
    all_node_count = 0
    airfoil_node_count = 0
    airfoil_face_pts_dict = {}

    # main loop through all the of the faces
    for i in range(round(faces.nfaces)):
        face_pts = faces.faces[i]['id_pts']
        face_pts_coords = np.array([points.values_x[face_pts], points.values_y[face_pts], points.values_z[face_pts]])

        # loop only through faces which are within the max radius of interest
        if not max_radius or (np.sqrt((face_pts_coords[0, :]-0.5)**2 + face_pts_coords[1, :]** 2) < max_radius).all():

            # first faces give the internal connectivity of the mesh (neighbour file)
            if i < neighbours.nb_faces:
                cell_connectivity += [[owners.values[i], neighbours.values[i]]]
                cell_face_s += [get_polygon_area(face_pts_coords.T)]

            # get the 2D triangular mesh on the back face
            elif i >= id0_back_faces and i< id0_back_faces+n_back_faces:
                elements.append(faces.faces[i]['id_pts'])
                cells_dict[owners.values[i]] = interior_node_count
                interior_cells.append(owners.values[i])
                interior_node_count+=1

                node_pos_dict[all_node_count] = [np.mean(face_pts_coords[0, :]), np.mean(face_pts_coords[1, :])]
                nodes_t.append(0)
                # add the area of the element
                nodes_a.append(0.5*((face_pts_coords[0,0]*(face_pts_coords[1,1]-face_pts_coords[1,2])) +
                                    (face_pts_coords[0,1]*(face_pts_coords[1,2]-face_pts_coords[1,0])) +
                                    (face_pts_coords[0,2]*(face_pts_coords[1,0]-face_pts_coords[1,1])) ))
                nodes_r.append(0.0)
                all_node_count+=1

            # create airfoil nodes and get connectivity and properties of these nodes
            elif i >= id0_airfoil_faces and i< id0_airfoil_faces+n_airfoil_faces:
                cell_connectivity += [[owners.values[i], 'airfoil_{}'.format(airfoil_node_count)]]
                cell_face_s += [get_polygon_area(face_pts_coords.T)]
                cells_dict['airfoil_{}'.format(airfoil_node_count)] = all_node_count
                airfoil_face_pts_dict[all_node_count] = face_pts
                nodes_t.append(1)
                nodes_a.append(0.0)
                if i >= id0_rt and i < id0_rt+n_rough_airfoil_top_faces:
                    nodes_r.append(r_top)
                elif i >= id0_rb and i < id0_rb+n_rough_airfoil_bot_faces:
                    nodes_r.append(r_bot)
                else:
                    nodes_r.append(0.0)
                node_pos_dict[all_node_count] = [np.mean(face_pts_coords[0, :]), np.mean(face_pts_coords[1, :])]
                all_node_count+=1
                airfoil_node_count += 1

            # get the connectivity of the farfield faces if the farfield node is used
            elif i >= id0_farfield_faces and i< id0_farfield_faces+n_farfield_faces and add_farfield_node:
                cell_connectivity += [[owners.values[i], 'farfield']]
                cell_face_s += [get_polygon_area(face_pts_coords.T)]

    # convert the nodes type list into array
    nodes_t = np.array(nodes_t)

    # retrieve the pressure for all nodes
    nodes_p = np.zeros_like(nodes_t)
    nodes_p[nodes_t==0] = readscalar(sim_dir, sim_time, 'p')[interior_cells]
    nodes_p[nodes_t==1] = readscalar(sim_dir, sim_time, 'p', boundary='AIRFOIL')

    # retrieve the velocity vector for all nodes
    nodes_u = np.zeros((3, len(nodes_t)))
    nodes_u[:, nodes_t == 0] = readfield(sim_dir, sim_time, 'U')[:, interior_cells]
    nodes_u[:, nodes_t == 1] = readfield(sim_dir, sim_time, 'U', boundary='AIRFOIL')

    # retrieve the farfield bc data
    p_dict = ParsedParameterFile(os.path.join(sim_dir, '0', 'p'))
    p_farfield = float(p_dict['internalField'].val)
    u_dict = ParsedParameterFile(os.path.join(sim_dir, '0', 'U'))
    u_farfield = list(u_dict['boundaryField']['FARFIELD']['freestreamValue'].val)[:2]
    k_dict = ParsedParameterFile(os.path.join(sim_dir, '0', 'k'))
    k =  float(k_dict['internalField'].val)
    Ti = np.sqrt(2/3*k/(u_farfield[0]**2 + u_farfield[1]**2))*100
    global_feat_labels = ['x-velocity [m/s]', 'y-velocity [m/s]', 'turb-intensity [%]']
    global_features = torch.tensor(u_farfield + [Ti])

    # add the single node for the farfield if not using a global variable for the farfield
    if add_farfield_node:
        cells_dict['farfield'] = all_node_count
        nodes_t = np.append(nodes_t, 2)
        nodes_a.append(0.0)
        nodes_r.append(0.0)
        nodes_p = np.append(nodes_p, p_farfield)
        nodes_u = np.append(nodes_u, np.array([u_farfield + [0.0]]).T, axis=1)
        node_pos_dict[all_node_count] = [np.min(points.values_x[np.unique(np.array(elements))])-0.1, 0]
        all_node_count+=1

    # compute the edges, accounting for missing farfield edges (due to max radius)
    edges = []
    edges_l = [] # list of edge lengths
    edges_rd = [] # list of edge relative displacement vectors
    edges_s = [] # list of face surface values represented by the edges
    for i, cc in enumerate(cell_connectivity):
        try:
            n1 = cells_dict[cc[0]]
        except KeyError:
            if add_farfield_node:
                n1 = cells_dict['farfield']
            else:
                continue
        try:
            n2 = cells_dict[cc[1]]
        except KeyError:
            if add_farfield_node:
                n2 = cells_dict['farfield']
            else:
                continue
        edges.append([n1, n2])
        n1_pos = np.array(node_pos_dict[n1])
        n2_pos = np.array(node_pos_dict[n2])
        diff = (n2_pos - n1_pos)
        edge_len = np.linalg.norm(diff)
        edges_rd.append(diff/edge_len)
        edges_l.append(edge_len)
        edges_s.append(cell_face_s[i])

    # create edges in between adjacent airfoil nodes, check if they share 2 points
    for n1 in list(airfoil_face_pts_dict.keys()):
        for n2 in list(airfoil_face_pts_dict.keys()):
            if len(np.intersect1d(airfoil_face_pts_dict[n1], airfoil_face_pts_dict[n2])) == 2:
                edges.append([n1, n2])
                n1_pos = np.array(node_pos_dict[n1])
                n2_pos = np.array(node_pos_dict[n2])
                diff = (n2_pos - n1_pos)
                edge_len = np.linalg.norm(diff)
                edges_rd.append(diff / edge_len)
                edges_l.append(edge_len)
                edges_s.append(0.0)

    # create edges tensor and make graph undirected
    edges = torch.tensor(edges)
    edges = torch.cat((edges, torch.stack((edges[:,1], edges[:,0]), dim=1)), dim=0)

    # create edge feature matrix
    edges_rd = torch.cat((torch.tensor(np.array(edges_rd)), -torch.tensor(np.array(edges_rd))))
    edges_l = torch.cat((torch.tensor(edges_l), torch.tensor(edges_l)))
    edges_s = torch.cat((torch.tensor(edges_s), torch.tensor(edges_s)))
    edge_features = torch.cat((edges_rd, edges_l.unsqueeze(1), edges_s.unsqueeze(1)), dim=1)
    edge_feat_labels = ['rd_x [-]', 'rd_y [-]', 'edge_length [m]', 'face_surface [m^2]']
    
    # create array of airfoil segments for SDF calculation
    airfoil_nodes = np.where(nodes_t == 1)[0]
    airfoil_segments = np.zeros((len(airfoil_nodes), 2, 2))
    for i in range(len(airfoil_nodes)):
        node1_idx = airfoil_nodes[i]
        node2_idx = airfoil_nodes[(i + 1) % len(airfoil_nodes)]
        airfoil_segments[i, 0] = node_pos_dict[node1_idx]
        airfoil_segments[i, 1] = node_pos_dict[node2_idx]
    
    # create array of all points
    all_node_points = np.array([node_pos_dict[i] for i in range(all_node_count)])
    
    # create KD-tree for airfoil points (can be reused for multiple queries)
    airfoil_tree = cKDTree(airfoil_segments[:, 0])
    
    # compute signed distances for all points at once
    nodes_sdf = compute_signed_distances_vectorized(all_node_points, airfoil_segments, airfoil_tree)
    
    # create the node feature matrix: area, pressure, x-velocity, y-velocity,sdf, roughness, area of cell
    node_features = [nodes_p, nodes_u[0,:], nodes_u[1,:], nodes_sdf]
    node_feat_labels = ['pressure [Pa]', 'x-velocity [m/s]', 'y-velocity [m/s]', 'signed_distance [m]']
    if roughness_feat:
        node_features.append(nodes_r)
        node_feat_labels.append('roughness [m]')
    if area_feat:
        node_features.append(nodes_a)
        node_feat_labels.append('cell_area [m^2]')
    node_features = torch.tensor(np.array(node_features)).T
    nodes_pos = torch.tensor(list(node_pos_dict.values()))

    # create the pytorch geometric graph data structure
    data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, pos=nodes_pos)
    data.node_feat_labels = node_feat_labels
    data.edge_feat_labels = edge_feat_labels
    data.globals = global_features
    data.global_feat_labels = global_feat_labels
    data.triangles = torch.tensor(elements)
    data.triangle_points = torch.tensor(np.array([points.values_x, points.values_y]))
    data.node_type = torch.tensor(nodes_t)

    if plot:
        xmin = np.min(points.values_x[np.unique(np.array(elements))])
        xmax = np.max(points.values_x[np.unique(np.array(elements))])
        ymin = np.min(points.values_y[np.unique(np.array(elements))])
        ymax = np.max(points.values_y[np.unique(np.array(elements))])

        # plot graph and mesh
        fig, ax = plt.subplots(figsize = (12,8))
        G = to_networkx(data, edge_attrs=['edge_attr'])
        edges, weights = zip(*nx.get_edge_attributes(G, 'edge_attr').items())
        nx.draw(G, pos=node_pos_dict, node_color = nodes_t, edgelist=edges, edge_color=np.array(weights)[:,3], edge_cmap= plt.cm.jet, node_size=20,linewidths=6)
        triangulation = tri.Triangulation(x=data.triangle_points[0,:], y=data.triangle_points[1,:], triangles=data.triangles)
        ax.triplot(triangulation, color='black', alpha=0.2)
        ax.axis('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)

        # plot mesh with pressure
        fig, ax = plt.subplots(figsize = (12,8))
        tpc = ax.tripcolor(triangulation, data.x[data.node_type==0, 0], shading='flat')
        fig.colorbar(tpc)
        ax.triplot(triangulation, color='black', alpha=0.2)
        ax.axis('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)

        # plot mesh with velocity magnitude
        fig, ax = plt.subplots(figsize = (12,8))
        tpc = ax.tripcolor(triangulation, torch.norm(data.x[data.node_type==0,1:3], dim=1), shading='flat')
        fig.colorbar(tpc)
        ax.triplot(triangulation, color='black', alpha=0.2)
        ax.axis('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)

        # plot divergence at mesh each cell
        # get the velocity for each first node of all edges
        u = data.x[data.edge_index[0,:], 1:3]
        # calculate for each edge the flux j= U dot edge_rd x edge_s
        j = torch.bmm(u.view(data.edge_index.shape[1], 1, -1), edge_features[:, 0:2].view(data.edge_index.shape[1], -1, 1))
        j = j.squeeze() * edges_s /0.1
        # for each node sum all the sender edge fluxes
        div = scatter_sum(j, data.edge_index[0,:], dim=0)
        fig, ax = plt.subplots(figsize=(12, 8))
        tpc = ax.tripcolor(triangulation, div[data.node_type==0], shading='flat')
        fig.colorbar(tpc)
        ax.triplot(triangulation, color='black', alpha=0.2)
        ax.axis('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        
        # plot for signed distance function
        fig, ax = plt.subplots(figsize=(12, 8))
        tpc = ax.tripcolor(triangulation, data.x[data.node_type==0, 3], shading='flat', cmap='RdBu')
        fig.colorbar(tpc, label='Signed Distance [m]')
        ax.triplot(triangulation, color='black', alpha=0.2)
        ax.axis('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.title('Signed Distance Function')

        # plot the roughness
        if roughness_feat:
            r_pos = node_feat_labels.index('roughness [m]')
            fig, ax = plt.subplots(figsize=(12, 8))
            airfoil_nodes = data.node_type == 1
            sns.scatterplot(x=data.pos[airfoil_nodes, 0], y=data.pos[airfoil_nodes, 1], hue= data.x[airfoil_nodes, r_pos], ax=ax)
            ax.triplot(triangulation, color='black', alpha=0.2)
            ax.axis('equal')
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim(xmin, xmax)
            plt.gca().set_ylim(ymin, ymax)
        plt.show()

    return data

def get_face_set(path):
    with open(path) as f:
        faces = []
        for line in f:
            line = line.rstrip()
            try:
                faces.append(int(line))
            except ValueError:
                continue

    return faces[1:]

def get_polygon_area(point_coords):
    #shape (N, 3)
    if isinstance(point_coords, list):
        point_coords = np.array(point_coords)
    #all edges
    edges = point_coords[1:] - point_coords[0:1]
    # row wise cross product
    cross_product = np.cross(edges[:-1],edges[1:], axis=1)
    #area of all triangles
    area = np.linalg.norm(cross_product, axis=1)/2
    return sum(area)

def delete_blank_lines(file):
    output = ""
    with open(file, 'r') as f:
        for line in f:
            if not line.isspace():
                output += line

    f = open(file, 'w')
    f.write(output)

if __name__ == '__main__':
    sim_dir = '../datasets/simulation_02'
    sim_time = 'latestTime'
    graph = parse_mesh_to_graph(sim_dir, sim_time, max_radius=0.7, add_farfield_node=False, plot=True)