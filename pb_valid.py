import os

import pandas as pd
from posebusters import PoseBusters
import subprocess
import re
buster = PoseBusters(config="dock")

check_cols = [
	'mol_pred_loaded',
	'mol_cond_loaded',
	'sanitization',
	'inchi_convertible',
	'all_atoms_connected',
	'no_radicals',
	'bond_lengths',
	'bond_angles',
	'internal_steric_clash',
	'aromatic_ring_flatness',
	'non-aromatic_ring_non-flatness',
	'double_bond_flatness',
	'internal_energy',
	'protein-ligand_maximum_distance',
	'minimum_distance_to_protein',
	'minimum_distance_to_organic_cofactors',
	'minimum_distance_to_inorganic_cofactors',
	'minimum_distance_to_waters',
	'volume_overlap_with_protein',
	'volume_overlap_with_organic_cofactors',
	'volume_overlap_with_inorganic_cofactors',
	'volume_overlap_with_waters',
]
physical_cols = [
	'bond_lengths',
	'bond_angles',
	'internal_steric_clash',
	'aromatic_ring_flatness',
	'non-aromatic_ring_non-flatness',
	'double_bond_flatness',
	'internal_energy',
	'protein-ligand_maximum_distance',
	'minimum_distance_to_protein',
	'volume_overlap_with_protein',
]



from rdkit import Chem
from rdkit.Geometry import Point3D
from pathlib import Path

def write_pred_pose_from_ref(ref_ligand_sdf, ligand_pose, out_sdf):
	if str(ref_ligand_sdf).endswith('.pdb'):
		mol = Chem.MolFromPDBFile(str(ref_ligand_sdf), removeHs=False)
	else:
		suppl = Chem.SDMolSupplier(str(ref_ligand_sdf), removeHs=False)
		mol = suppl[0] if len(suppl) > 0 else None

	if mol is None:
		raise ValueError(f"Failed to load reference ligand: {ref_ligand_sdf}")

	if mol.GetNumAtoms() != len(ligand_pose):
		raise ValueError(
			f"Atom number mismatch: mol has {mol.GetNumAtoms()} atoms, "
			f"but ligand_pose has {len(ligand_pose)} atoms."
		)

	conf = mol.GetConformer()

	for i, xyz in enumerate(ligand_pose):
		x, y, z = map(float, xyz)
		conf.SetAtomPosition(i, Point3D(x, y, z))
	if not os.path.exists(os.path.dirname(out_sdf)):
		os.makedirs(os.path.dirname(out_sdf))
	writer = Chem.SDWriter(str(out_sdf))
	writer.write(mol)
	writer.close()

	return out_sdf



from pathlib import Path

def get_failed_checks(row, cols):
	failed = []
	for c in cols:
		val = row[c]
		if pd.isna(val) or val != True:
			failed.append(c)
	return failed



def PB_valid(complex_name, ligand_pose,pose_file = None):
	base = Path(
		f"D:/PythonProject medicine/project_mine/data/DOCKGEN/{complex_name}"
	)

	#ref_ligand = base / f"{complex_name}_ligand.sdf"
	ref_ligand = base / f"{complex_name}_ligand.pdb"     ##DOCKGEN
	protein = base / f"{complex_name}_protein.pdb"
	pred_ligand = base / f"{complex_name}_pred_pose.sdf"

	# 1. 先把预测坐标写成 SDF
	if str(ref_ligand).endswith('.pdb'):
		mol = Chem.MolFromPDBFile(str(ref_ligand), removeHs=False)
		if mol is not None:
			writer = Chem.SDWriter(base / f"{complex_name}_ligand.sdf")
			writer.write(mol)
			writer.close()
	write_pred_pose_from_ref(
		ref_ligand_sdf=ref_ligand,
		ligand_pose=ligand_pose,
		out_sdf=pred_ligand,
	)
	if pose_file is not None:
		rescore_pose_file = base / "rescoring_poses" / pose_file
		write_pred_pose_from_ref(
			ref_ligand_sdf=ref_ligand,
			ligand_pose=ligand_pose,
			out_sdf=rescore_pose_file,
		)
	# 2. 再用 PoseBusters 检查
	import contextlib
	import io
	import sys

	# 用一个空的 StringIO 或直接 /dev/null 来接收 stderr
	stderr_capture = io.StringIO()
	with contextlib.redirect_stderr(stderr_capture):
		df = buster.bust(
			mol_pred=str(pred_ligand),
			mol_true=str(ref_ligand),
			mol_cond=str(protein),
			full_report=True,
		)
	# 如果你想知道警告内容，可以打印 stderr_capture.getvalue()，否则直接忽略
	check_cols_exist = [c for c in check_cols if c in df.columns]
	physical_cols_exist = [c for c in physical_cols if c in df.columns]

	# 注意：这里如果你希望 NaN 算失败，建议用 fillna(False)
	df["posebusters_valid"] = df[check_cols_exist].fillna(False).all(axis=1)
	df["physical_valid"] = df[physical_cols_exist].fillna(False).all(axis=1)

	df["failed_checks"] = df.apply(
		lambda row: get_failed_checks(row, check_cols_exist),
		axis=1
	)
	df["num_failed_checks"] = df["failed_checks"].apply(len)

	valid_rate = df["posebusters_valid"].mean()
	physical_valid_rate = df["physical_valid"].mean()


	print("PoseBusters valid rate:", valid_rate)
	print("Physical valid rate:", physical_valid_rate)
	# for c in df.columns:
	#     if "distance" in c.lower() or "overlap" in c.lower() or "protein" in c.lower():
	#         print(c, df[c].iloc[0])
	# print("\nExisting check columns:", len(check_cols_exist))
	# print(check_cols_exist)
	#
	# missing_cols = [c for c in check_cols if c not in df.columns]
	# print("\nMissing check columns:", len(missing_cols))
	# print(missing_cols)
	#
	# print("\nFailed checks per row:")
	# for idx, row in df.iterrows():
	#     if row["num_failed_checks"] == 0:
	#         print(f"Row {idx}: all checks passed")
	#     else:
	#         print(f"Row {idx}: failed {row['num_failed_checks']} checks")
	#         for c in row["failed_checks"]:
	#             print(f"  - {c}: {row[c]}")



	return df, valid_rate, physical_valid_rate

def run_gnina_score(protein_path, ligand_path, ref_ligand_path, gnina_path):
	cmd = [
		gnina_path,
		"-r", protein_path,
		"-l", ligand_path,
		"--autobox_ligand", ref_ligand_path,
		"--score_only",
	]

	proc = subprocess.run(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)

	out = proc.stdout
	match = re.search(r"Affinity:\s*([-\d\.]+)\s*$kcal/mol$", out)

	if match is None:
		raise RuntimeError(f"GNINA affinity not found for {ligand_path}\n{out}")

	return float(match.group(1))
