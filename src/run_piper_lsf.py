#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--query", "-q", help="query name")
args = parser.parse_args()

PIPER_BINARY = "lib/piper-1.2.0b10_nl/piper_package/bin/run_piper"
OUT_DIR = "/u21/aybars/scratch/training_data"
PDB_CLEAN = "/databases/pdb_clean/pdb/"
RANDROT = "lib/randrot.pl"


class Query:
    def __init__(self, name):
        self.name = name

    def get_receptor_path(self):
        parent_path = Path(PDB_CLEAN) / f"{self.name[1:3]}"
        return parent_path / f"{self.name[:4]}{self.name[-2]}.pdb"

    def get_ligand_path(self):
        parent_path = Path(PDB_CLEAN) / f"{self.name[1:3]}"
        return parent_path / f"{self.name[:4]}{self.name[-1]}.pdb"


def make_output_dir(query):
    output_dir = Path(OUT_DIR) / query
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_piper(receptor, ligand, output_dir):
    cmd = [
        PIPER_BINARY,
        "--rec",
        str(receptor),
        "--lig",
        str(ligand),
        "--dont-minimize",
        "--output-dir",
        output_dir,
        "--advanced",
        "/u21/aybars/scratch/advanced_params.json",
    ]

    subprocess.run(cmd, check=True)


def run_randrot(ligand):
    tmp = Path(ligand).parent / "tmp"
    cmd = [RANDROT, ligand, ">", str(tmp)]
    subprocess.call(" ".join(cmd), shell=True)

    tmp.rename(ligand)

    if tmp.exists():
        os.remove(tmp)


def main():
    query_name = args.query
    query = Query(query_name)
    receptor = query.get_receptor_path()
    ligand = query.get_ligand_path()
    output_dir = make_output_dir(query_name)
    receptor_copy = shutil.copy(receptor, output_dir)
    ligand_copy = shutil.copy(ligand, output_dir)

    run_randrot(ligand=ligand_copy)
    run_piper(output_dir / receptor_copy, output_dir / ligand_copy, output_dir)


if __name__ == "__main__":
    main()
