#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path

from Bio.PDB import PDBParser

DOCKQ = "lib/DockQ/DockQ.py"
PDB_CLEAN = "/databases/pdb_clean/pdb/"
OUT_DIR = "/u21/aybars/scratch/piper_scwrl"

parser = argparse.ArgumentParser()
parser.add_argument("--query", "-q", help="query name")
args = parser.parse_args()


def get_chain_id(path_to_pdb):
    pdb = PDBParser(QUIET=True).get_structure("PDB", path_to_pdb)
    chains = [chain.id for chain in pdb.get_chains()]
    return chains[0], chains[1]


def calc_dockq_score(query_model, native_model):
    chain1_query, chain2_query = get_chain_id(query_model)
    chain1_native, chain2_native = get_chain_id(native_model)

    command = [
        DOCKQ,
        query_model,
        native_model,
        "-native_chain1",
        chain1_native,
        "-native_chain2",
        chain2_native,
        "-model_chain1",
        chain1_query,
        "-model_chain2",
        chain2_query,
    ]
    
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
    except Exception as err:
        print(err)
    else:
        for line in proc.stdout.splitlines():
            if line.startswith("DockQ"):
                score = float(line.split()[1])
            elif line.startswith("iRMS"):
                irms = float(line.split()[1])

    return score, irms


def prep_model_id(query, in_model_path, clustering):
    if clustering == "cluster":
        model_id = f"{query}_{in_model_path.stem}"
    elif clustering == "piper":
        model_id = f"{query}_{'_'.join(in_model_path.stem.split('.')[1:])}"
    return model_id


def write_output(query_name, data, clustering):
    if clustering == "piper":
        p = Path(OUT_DIR) / query_name / f"{query_name}_dockq_output_piper.tsv"
    elif clustering == "cluster":
        p = Path(OUT_DIR) / query_name / f"{query_name}_dockq_output_cluster.tsv"

    with open(p, "w") as handle:
        for model_id, (dockq_score, dockq_rank, irmsd) in data.items():
            handle.write("\t".join([model_id, dockq_score, dockq_rank, irmsd]) + "\n")


def rank_dockq(score):
    if score >= 0.80:
        rank = "High"
    elif 0.49 <= score < 0.80:
        rank = "Medium"
    elif 0.23 <= score < 0.49:
        rank = "Acceptable"
    else:
        rank = "Incorrect"
    return rank


def calculate_dockq(query_name, in_models, ref, clustering):
    data = {}
    for in_model in in_models:
        model_id = prep_model_id(query_name, in_model, clustering=clustering)
        dockq_score, irmsd = calc_dockq_score(str(in_model), str(ref))
        dockq_rank = rank_dockq(dockq_score)
        data[model_id] = (str(dockq_score), dockq_rank, str(irmsd))

    return data


def main():
    query_name = args.query.strip()
    query_path = Path(OUT_DIR) / query_name
    ref = query_path / f"{query_name}.pdb"

    in_models_cluster = query_path.glob("*/*_cluster.000.*.pdb")
    in_models_piper = query_path.glob("*/model.000.*.pdb")

    data_cluster = calculate_dockq(query_name, in_models_cluster, ref, "cluster")
    data_piper = calculate_dockq(query_name, in_models_piper, ref, "piper")

    write_output(query_name, data_cluster, clustering="cluster")
    write_output(query_name, data_piper, clustering="piper")


if __name__ == "__main__":
    main()
