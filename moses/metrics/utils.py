import os
from collections import Counter
from shutil import rmtree
import sqlite3
from functools import partial
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer
from moses.utils import mapper, get_mol
from crem.fragmentation import main as fragmentation
from crem.frag_to_env_mp import main as frag_to_env
from crem.import_env_to_db import main as env_to_db 

_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, 'mcf.csv'))
_pains = pd.read_csv(os.path.join(_base_dir, 'wehi_pains.csv'),
                     names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in
            _mcf.append(_pains, sort=True)['smarts'].values]


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


class CremFragmenter:
    """
    CReM Fragmenter fragment set of molecules using CReM
    """
    def __init__(self, ncpu=1, fragdb_dir=None, radius=2, verbose=False):
        self.ncpu = ncpu
        self.fragdb_dir = fragdb_dir
        self.radius = radius
        self.verbose = verbose
        os.makedirs(fragdb_dir, exist_ok=True)

    def remove_duplicates(self):
        att = Chem.MolFromSmiles('*')
        H = Chem.MolFromSmiles('[H]')

        def canon_smi(smile):
            return Chem.MolToSmiles(Chem.RemoveHs(Chem.ReplaceSubstructs(
                Chem.MolFromSmiles(smile), att, H, replaceAll=True)[0]))

        def num_att(smile):
            return smile.count('*')

        class MaxAtt:
            def __init__(self):
                self.max = 0
                self.smi = ''
                self.cnt = 0

            def step(self, smi, n_att):
                self.cnt += 1
                if n_att > self.max:
                    self.max = n_att
                    self.smi = smi

            def finalize(self):
                return f"{self.cnt},{self.smi}"

        query = """
            CREATE TABLE radius2_filtered AS 
                WITH radius2 AS (
                    SELECT * 
                    FROM (
                        main.radius2 
                        JOIN 
                        (SELECT rowid, _canon_smi(core_smi) as canon_smi FROM main.radius2) as canon_smi 
                        ON 
                        main.radius2.rowid = canon_smi.rowid
                    )
                )
                        
                SELECT _max_att(core_smi, num_att) as core_smi
                FROM (   
                    radius2 
                    JOIN 
                    (SELECT rowid, _num_att(core_smi) as num_att FROM radius2) as num_att 
                    ON 
                    num_att.rowid = radius2.rowid
                )
                GROUP BY canon_smi
        """
        db_path = self.fragdb_dir + "/fragments.db"
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("DROP TABLE IF EXISTS radius2_filtered")
            con.create_function("_num_att", 1, num_att)
            con.create_function("_canon_smi", 1, canon_smi)
            con.create_aggregate("_max_att", 2, MaxAtt)

            cur.execute(query)
            con.commit()

    def fragment(self, smiles):
        with open(os.path.join(self.fragdb_dir, 'smiles.smi'), 'wt') as f:
            f.write('\n'.join(smiles))
        
        fragmentation(
            input_fname=os.path.join(self.fragdb_dir, 'smiles.smi'),
            output_fname=os.path.join(self.fragdb_dir, 'frags.txt'),
            sep=None,
            ncpu=self.ncpu,
            verbose=self.verbose
        )

        frag_to_env(
            input_fname=os.path.join(self.fragdb_dir, 'frags.txt'),
            output_fname=os.path.join(self.fragdb_dir, f'r{self.radius}.txt'),
            keep_mols=None,
            radius=self.radius,
            keep_stereo=False,
            max_heavy_atoms=20,
            ncpu=self.ncpu,
            store_comp_id=False,
            verbose=self.verbose
        )

        with open(os.path.join(self.fragdb_dir, f'r{self.radius}.txt'), 'rt') as f:
            text = f.read()
        
        with open(os.path.join(self.fragdb_dir, f'r{self.radius}_c.txt'), 'wt') as f:
            for frag, freq in Counter(text.split('\n')[:-1]).items():
                f.write(f'{freq} {frag}\n')

        env_to_db(
            input_fname=os.path.join(self.fragdb_dir, f'r{self.radius}_c.txt'),
            output_fname=os.path.join(self.fragdb_dir, 'fragments.db'),
            radius=self.radius,
            counts=True,
            ncpu=self.ncpu,
            verbose=self.verbose
        )
       
        self.remove_duplicates()

        with sqlite3.connect(os.path.join(self.fragdb_dir, 'fragments.db')) as con:
            cur = con.cursor()
            frags = cur.execute(f"SELECT core_smi FROM radius{self.radius}_filtered")
            frags = frags.fetchall()

        rmtree(self.fragdb_dir)

        return [[frag[0].split(",")[1] for frag in frags]]


def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles


def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True
