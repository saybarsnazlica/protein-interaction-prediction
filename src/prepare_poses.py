#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO


CURRENT_DIR = Path().cwd().resolve()
ATOM_NAMES = "atom_names"
FEAT_BINARY = "lib/dock_atom_feat3_sym1_pdb_lrmsd_withnat/feat"
EXPECTED = "/u21/aybars/scratch/contacts-heterodimer"
TOP_DIR = CURRENT_DIR / 'work' / 'training_data'
DIST = "5"
TAKE_LOG = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--query", "-q", help="query name")
args = parser.parse_args()


class Query:
    def __init__(self, name):
        self.name = name
        self.chain_a = name[-2]
        self.chain_b = name[-1]
        self.base_name = name[:4]

    def get_path(self):
        return TOP_DIR / self.name

    def get_reference(self):
        return f"{self.name}.pdb"

    def set_reference(self):
        parser = PDBParser(QUIET=1)
        struct_a = parser.get_structure(
            self.name, f"{self.base_name}{self.chain_a}.pdb"
        )
        struct_b = parser.get_structure(
            self.name, f"{self.base_name}{self.chain_b}.pdb"
        )  # Random rotated
        struct_chain_b = struct_b[0][self.chain_b]
        struct_chain_b.detach_parent()
        struct_a[0].add(struct_chain_b)

        io = PDBIO()
        io.set_structure(struct_a)
        io.save(f"{self.name}.pdb")

    def get_expected(self):
        exp_query = Path(EXPECTED) / self.name
        receptor = exp_query / f"{self.name}-000-00_rec-norm.tsv"
        ligand = exp_query / f"{self.name}-000-00_lig-norm.tsv"

        return str(receptor), str(ligand)


def make_output_dir(output_dir: str):
    output_directory = Path(output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def extract_tar_file(query_path, query_idx):
    make_output_dir(f"{query_idx}_tmp")

    with tarfile.open(next(query_path.glob("*.tar.xz"))) as tgz:
        tgz.extractall(f"{query_idx}_tmp")


def run_feat(query, piper_tmp_folder, output_file):
    reference_pdb = query.get_reference()
    rec_expected, lig_expected = query.get_expected()
    piper_ft_file = str(Path(piper_tmp_folder) / "ft.000.00")
    rots_file = str(Path(piper_tmp_folder) / "fft_rotset")
    rec_file = str(Path(piper_tmp_folder) / "rec.pdb")
    lig_file = str(Path(piper_tmp_folder) / "lig.000.00.pdb")

    feat_command = [
        FEAT_BINARY,
        "--piper",
        piper_ft_file,
        "--rots",
        rots_file,
        "--rec",
        rec_file,
        "--lig",
        lig_file,
        "--d",
        DIST,
        "--ref_pdb",
        reference_pdb,
        "--atom_names",
        ATOM_NAMES,
        "--rec_exp",
        rec_expected,
        "--lig_exp",
        lig_expected,
        "--log",
        TAKE_LOG,
    ]

    print(" ".join(feat_command))

    with open(output_file, "w") as out:
        subprocess.run(feat_command, stdout=out, stderr=out, text=True, check=True)


def main():
    query = Query(args.query.strip())
    query_path = query.get_path()
    pose_out_tmp = "poses_sym1_000-00"
    archive = "poses.tar.gz"
    tmp_dir = f"{query.name}_tmp"

    os.chdir(query_path)
    query.set_reference()
    extract_tar_file(query_path, query.name)
    run_feat(query, tmp_dir, pose_out_tmp)

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pose_out_tmp)

    if os.path.exists(pose_out_tmp):
        os.remove(pose_out_tmp)

    os.chdir(CURRENT_DIR)
    shutil.rmtree(query_path / tmp_dir)


if __name__ == "__main__":
    main()
