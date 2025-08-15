import os
import contextlib
import shutil
import numpy as np
from fluidfoam.readof import _find_latesttime
import argparse
import click
import zipfile
from parse_mesh import parse_mesh_to_graph
import tempfile
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import math

def process_single_simulation(sim_path, temp_dir, sim_num, tot_num_sims, max_radius, roughness, area, max_nodes, tol):
    # find the latest time
        try:
            latest_time = _find_latesttime(sim_path)
        except IndexError as e:
            print('Simulation ' + sim_path + ' failed')
            return

        # if latest time is 0, simulation failed
        if latest_time == '0':
            print('Simulation ' + sim_path + ' failed')
            return

        # get final residuals from the PyFoam Analyzed folder present in each sim
        res_log_files = ['linear_p', 'linear_Ux', 'linear_Uy', 'linear_k', 'linear_omega', 'continuity']
        residuals = np.zeros(len(res_log_files))
        for i, log_file in enumerate(res_log_files):
            file_path = os.path.join(sim_path, 'PyFoamSolve.analyzed', log_file)
            res = np.loadtxt(file_path, usecols=1, skiprows=int(latest_time))
            try:
                residuals[i] = res if res.ndim == 0 else res[0]
            except IndexError as e:  # if the file is empty beyond the latest_time line, something went wrong
                print('Simulation ' + sim_path + ' failed')
                return

        # if residuals are not sufficiently converged, skip the simulation
        if (residuals > tol).any():
            print('Simulation ' + sim_path + ' is not sufficiently converged')
            return
        else:
            # parse the graph and save it to the temp directory
            with open(os.devnull, "w") as t, contextlib.redirect_stdout(t):
                graph = parse_mesh_to_graph(sim_dir=sim_path, sim_time='latestTime', max_radius=max_radius,
                                            roughness_feat=roughness, area_feat=area,  add_farfield_node=False, plot=False)
            if graph.num_nodes > max_nodes:
                print('Simulation ' + sim_path + ' has too many nodes, not saving')
                return
            graph_path = os.path.join(temp_dir, 'graph_{}.pt'.format(str(sim_num).zfill(len(str(tot_num_sims)))))
            # graph_path = os.path.join(tempdir, 'graph_{}.pt'.format(os.path.basename(sim_path)))
            torch.save(graph, graph_path)
            print('Simulation ' + sim_path + ' successfully processed')


def process_simulations(run_folder, output_dir, split_ratios, overwrite, num_threads, tol, max_radius, roughness_on, area_on, max_nodes):
    # check that the split ratios sum to 1
    assert math.fsum(split_ratios) == 1.0

    # create the output directory if it doesn't exist
    if os.path.exists(output_dir):
        if overwrite or click.confirm('Do you want to overwrite {}?'.format(output_dir), abort=True):
            shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # create a list of the simulations
    sim_list = []
    for item in os.listdir(run_folder):
        item_path = os.path.join(run_folder, item)
        if os.path.isdir(item_path) and {'etc', 'meshes', 'simulations'}.issubset(os.listdir(item_path)):
            for sim in os.listdir(os.path.join(item_path, 'simulations')):
                sim_list.append(os.path.join(item_path, 'simulations', sim))
    tot_num_sims = len(sim_list)

    # create a temp directory to store each graph
    tempdir = tempfile.mkdtemp()

    print('Starting to process simulations')

    # loop through the simulations, processing them in parallel
    Parallel(n_jobs=num_threads)(delayed(process_single_simulation)(sim_path, tempdir, i, tot_num_sims, max_radius,
                              roughness_on, area_on, max_nodes, tol) for i, sim_path in enumerate(sim_list))

    # split the converged sims into train, validate, test sets
    all_graphs = os.listdir(tempdir)
    print('Number of converged sims: ', len(all_graphs))
    if split_ratios[0] == 1.0:
        train_graphs = all_graphs
        valid_graphs = []
        test_graphs = []
    else:
        train_graphs, valid_graphs = train_test_split(all_graphs, test_size=1-split_ratios[0])
        valid_graphs, test_graphs = train_test_split(valid_graphs, test_size=split_ratios[2]/(split_ratios[1]+split_ratios[2]))
    print('Creating train dataset with {} graphs'.format(len(train_graphs)))
    print('Creating validation dataset with {} graphs'.format(len(valid_graphs)))
    print('Creating test dataset with {} graphs'.format(len(test_graphs)))

    sets = {'train':train_graphs, 'valid':valid_graphs, 'test':test_graphs}
    for key in sets:
        # zip the files
        with zipfile.ZipFile(os.path.join(output_dir, key+'_dataset.zip'), mode='w') as zf:
            for i, name in enumerate(sets[key]):
                arcname = 'graph_{}.pt'.format(str(i).zfill(len(str(tot_num_sims))))
                # arcname = name
                zf.write(os.path.join(tempdir, name), arcname=arcname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', '-r', help="path to the folder containing pipeline runs", type=str)
    parser.add_argument('--output_dir', '-o', help="path for the output directory containing train validation and test zip files", type=str)
    parser.add_argument('--split_ratios', '-s', nargs="+", default=[0.8, 0.1, 0.1], help="list with train, validate and test ratios", type=float)
    parser.add_argument('--overwrite', '-v', help="if the output file should be overwritten", action='store_true')
    parser.add_argument('--num_threads', '-n', help="the number of threads to use", type=int, default=4)
    parser.add_argument('--tol', '-t', help="minimum tolerance for the final residues to be considered converged", type=float, default=5e-5)
    parser.add_argument('--max_radius', '-m', help="maximum radius around airfoil to include", type=float, default=None)
    parser.add_argument('--roughness_on', '-ro', help="if the roughness should be parsed", action='store_true')
    parser.add_argument('--area_on', '-ao', help="if the cell area should be parsed", action='store_true')
    parser.add_argument('--max_nodes', '-mn', help="maximum number of nodes allowed for a graph", type=float, default=150000)
    args = parser.parse_args()
 
    process_simulations(run_folder=args.run_folder, output_dir=args.output_dir, split_ratios=args.split_ratios, overwrite=args.overwrite, 
                        num_threads=args.num_threads, tol=args.tol, max_radius=args.max_radius, roughness_on=args.roughness_on, area_on=args.area_on,
                        max_nodes=args.max_nodes)