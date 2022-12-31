#!/usr/bin/env python
import os
from itertools import repeat
from multiprocessing import Pool
from subprocess import run
import glob
from functools import partial
import numpy as np


def init(env):
    os.environ = env


class DockingVina(object):
    def __init__(self, config):
        super(DockingVina, self).__init__()
        self.config = config
        self.temp_dir = config['temp_dir']
        self.seed = config['seed']
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        self.results = dict()

    def predict(self, smiles_list):
        smiles_set = list(set(smiles_list) - set(self.results.keys()))

        if smiles_set:
            binding_affinities = list()
            fnames = list(map(str, range(len(smiles_set))))
            for i in range(self.config['n_conf']):
                child_env = os.environ.copy()
                child_env['OB_RANDOM_SEED'] = str(self.seed + i)
                with Pool(processes=self.config['num_sub_proc'], initializer=init, initargs=(child_env,)) as pool:
                    binding_affinities.append(pool.starmap(self.docking, zip(smiles_set, fnames)))

            binding_affinities = dict(zip(smiles_set, np.minimum.reduce(binding_affinities)))
            self.results = {**self.results, **binding_affinities}

        files = glob.glob(f"{self.temp_dir}/*")
        for file in files:
            os.remove(file)

        return self.postprocess([self.results[smile] for smile in smiles_list])
    
    def postprocess(self, affinities):
        return self.config['alpha'] * -np.minimum(affinities, 0.0)

    def docking(self, smi, fname):
        return DockingVina._docking(smi, fname, **self.config)

    @staticmethod
    def _docking(smi, fname, *, vina_program, receptor_file, temp_dir, box_center,
            box_size, error_val, seed, num_modes, exhaustiveness,
            timeout_dock, timeout_gen3d, **kwargs):

        ligand_file = os.path.join(temp_dir, "ligand_{}.pdbqt".format(fname))
        docking_file = os.path.join(temp_dir, "dock_{}.pdbqt".format(fname))

        run_line = "obabel -:{} --gen3D -h -O {}".format(smi, ligand_file)
        try:
            result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_gen3d, env=os.environ)
        except:
            return error_val
        
        if "Open Babel Error" in result.stdout or "3D coordinate generation failed" in result.stdout:
            return error_val

        run_line = vina_program
        run_line += " --receptor {} --ligand {} --out {}".format(receptor_file, ligand_file, docking_file)
        run_line += " --center_x {} --center_y {} --center_z {}".format(*box_center)
        run_line += " --size_x {} --size_y {} --size_z {}".format(*box_size)
        run_line += " --num_modes {}".format(num_modes)
        run_line += " --exhaustiveness {}".format(exhaustiveness)
        run_line += " --seed {}".format(seed)
        try:
            result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_dock)
        except:
            return error_val
        
        return DockingVina.parse_output(result.stdout, error_val)
    
    @staticmethod
    def parse_output(result, error_val):
        result_lines = result.split('\n')
        check_result = False
        affinity = error_val

        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            break
        return affinity
