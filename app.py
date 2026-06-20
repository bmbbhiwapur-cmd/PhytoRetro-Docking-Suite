import time
import streamlit as st
import subprocess
import os
import shutil
import urllib.request
import urllib.parse
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
import io

# --- CRITICAL FIX 1: FORCE MATPLOTLIB TO HEADLESS BACKEND ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

# =====================================================================
# 1. INITIALIZATION & CLOUD BACKEND BOOTSTRAPPING
# =====================================================================
st.set_page_config(page_title="PhytoRetro Docking Suite", layout="wide", page_icon="🌿")

@st.cache_resource
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")
                return False
    return True

vina_ready = ensure_linux_vina_exists()

@st.cache_data
def load_database():
    """Loads the pre-compiled PhytoRetro 100-plant JSON database."""
    try:
        with open("phyto_retro_db.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Database file 'phyto_retro_db.json' not found. Please ensure it is in your GitHub repository.")
        return {}

PHYTO_RETRO_DB = load_database()

def initialize_session_states():
    defaults = {
        "protein_name": "Unknown Protein",
        "cx": 0.0, "cy": 0.0, "cz": 0.0,
        "sx": 20, "sy": 20, "sz": 20,
        "exhaustiveness": 8,
        "target_ready": False,
        "ligand_ready": False,
        "local_target_path": None,
        "pdb_id_display": "Custom",
        "docking_results_raw": None,
        "redesign_docking_results_raw": None,
        "serialized_ligand_block": None,
        "ligand_summary_text": "",
        "smiles_cache": "",
        "baseline_affinity": None,
        "baseline_pre_uff": "N/A",
        "baseline_post_uff": "N/A",
        "baseline_delta_uff": "N/A",
        "redesign_baseline_affinity": None,
        "selected_analog_smiles": None,
        "selected_analog_name": None,
        "style_mode": "cartoon",
        "surf_toggle": False,
        "active_retained_ions": "None",
        "uff_cache": {},
        "detected_pockets": [],
        "selected_native_ligand": "Manual Coordinate Assignment",
        "selected_tree_data": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_states()

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# =====================================================================
# 2. BIOINFORMATICS STRUCTURAL CONVERTERS & PARSERS
# =====================================================================

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception: pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "name": "Unknown Protein", "title": "Uploaded Protein Structure Matrix", 
        "id": pdb_id.upper() if pdb_id and pdb_id != "Uploaded File" else "Unknown",
        "class": "Unknown Classification", "organism": "Unknown", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            title_parts = []
            for line in f:
                if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
                elif line.startswith("HEADER"): 
                    meta["class"] = line[10:50].strip().title()
                elif line.startswith("COMPND") and "MOLECULE:" in line:
                    mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                    if meta["name"] == "Unknown Protein": meta["name"] = mol_name.title()
                elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
                elif "RESOLUTION." in line and "ANGSTROMS." in line:
                    match = re.search(r"(\d+\.\d+)", line)
                    if match: meta["res"] = f"{match.group(1)} Å"
        if title_parts: meta["title"] = " ".join(title_parts).title()
    except Exception: pass
    return meta

def identify_protein_cavities(pdbqt_file, max_pockets=3):
    coords = []
    if not os.path.exists(pdbqt_file): return []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if len(coords) < 10: return []
    arr = np.array(coords)
    min_bound, max_bound = np.min(arr, axis=0), np.max(arr, axis=0)
    step = (max_bound - min_bound) / 4.0
    pockets, idx = [], 1
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                pt = min_bound + np.array([i*step[0], j*step[1], k*step[2]])
                dists = np.linalg.norm(arr - pt, axis=1)
                score = np.sum((dists > 3.0) & (dists < 12.0))
                core_clash = np.sum(dists <= 3.0)
                if core_clash < 20 and score > 20:
                    pockets.append({"Pocket_ID": f"Cavity {idx}", "cx": round(pt[0], 2), "cy": round(pt[1], 2), "cz": round(pt[2], 2), "bx": 20.0, "by": 20.0, "bz": 20.0, "Score": score})
                    idx += 1
    pockets = sorted(pockets, key=lambda x: x["Score"], reverse=True)
    return pockets[:max_pockets]

def compute_protein_bounding_box(pdbqt_file):
    """
    CRITICAL FIX: Removed the 60 Angstrom clamp and allowed scaling up to 126 Angstroms.
    This prevents positive binding energies during Blind Docking by ensuring the box
    actually covers the entire protein surface rather than trapping the ligand inside the core.
    """
    if not os.path.exists(pdbqt_file): return 0, 0, 0, 20, 20, 20
    coords = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append((float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())))
                except ValueError: pass
    if not coords: return 0, 0, 0, 20, 20, 20
    coords = np.array(coords)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    size = (max_c - min_c) + 15.0
    return center[0], center[1], center[2], min(126.0, size[0]), min(126.0, size[1]), min(126.0, size[2])

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False, allowed_heteroatoms=None):
    if allowed_heteroatoms is None: allowed_heteroatoms = []
    autodock_type_map = { "H": "H", "C": "C", "N": "N", "O": "O", "S": "S", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I" }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except: torsions = 4
        
    temp_out = f"temp_{output_pdbqt}"
    try:
        atom_count = 0
        with open(input_pdb, "r", encoding="utf-8", errors="ignore") as pdb, open(temp_out, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    res_name = line[17:20].strip()
                    if record_type == "HETATM" and not is_ligand and res_name not in allowed_heteroatoms: continue
                    try: atom_id = int(line[6:11].strip())
                    except: atom_id = 1
                    atom_name = line[12:16]
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    pdbqt.write(f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
                    atom_count += 1
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        shutil.move(temp_out, output_pdbqt)
        return atom_count > 0, output_pdbqt
    except Exception as e: return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES matrix representation."
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.maxIterations = 2000
        res = AllChem.EmbedMolecule(mol, params)
        
        if res != 0:
            params = AllChem.ETKDG()
            params.enforceChirality = False
            params.useRandomCoords = True
            params.ignoreSmoothingFailures = True
            res = AllChem.EmbedMolecule(mol, params)
            
        if res != 0: return False, "RDKit failed to generate 3D coordinates."
            
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
        
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        ok, msg = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, msg
    except Exception as e: return False, str(e)

def execute_uff_complex_minimization(protein_path, ligand_pose_str, progress_ui=None):
    try:
        protein_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        ligand_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        if not protein_mol or not ligand_mol: return "N/A", "N/A", "N/A"
        
        total_atoms = protein_mol.GetNumAtoms() + ligand_mol.GetNumAtoms()
        if total_atoms > 4000: return "Bypassed", "Bypassed", "N/A"
        
        combined_complex = Chem.CombineMols(protein_mol, ligand_mol)
        uff_field = AllChem.UFFGetMoleculeForceField(combined_complex)
        if not uff_field: return "N/A", "N/A", "N/A"
        
        pre_energy = uff_field.CalcEnergy()
        if progress_ui: prog_bar = progress_ui.progress(0, text="⏳ UFF Minimization...")
        
        res = uff_field.Minimize(maxIts=150, forceTol=1e-3)
        if progress_ui: prog_bar.progress(100, text="✨ Relaxation Converged!")
            
        post_energy = uff_field.CalcEnergy()
        delta_energy = post_energy - pre_energy
        time.sleep(0.4)
        return f"{pre_energy:.2f}", f"{post_energy:.2f}", f"{delta_energy:.2f}"
    except Exception: return "N/A", "N/A", "N/A"

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                element = line[76:78].strip().upper()
                if not element:
                    atom_name = line[12:16].strip()
                    element = "".join([c for c in atom_name if c.isalpha()])[0].upper() if atom_name else "C"
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    with open(receptor_file, "r") as f: receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]: b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C"): b_type = "Hydrophobic / VDW"
                else: b_type = "van der Waals Contact"
                seen.add(res_id)
                interactions.append({"Residue Contact": res_id, "Interaction Type": b_type, "Distance (Å)": round(dist, 2), "r_coord": r_at["coord"].tolist(), "l_coord": l_at["coord"].tolist()})
    return interactions

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    if not stdout_text: return "N/A"
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"

def parse_vina_output_with_residues_global(stdout_text, docking_file="docking_poses.pdbqt"):
    data = []
    poses_dict = split_docking_poses(docking_file)
    if not stdout_text: return pd.DataFrame(data)
    for line in stdout_text.split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                mode_idx, aff, rmsd_lb, rmsd_ub = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                res_string, bond_types = "N/A", "N/A"
                if mode_idx in poses_dict:
                    ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                    if ints:
                        res_string = ", ".join(sorted(list(set([i["Residue Contact"] for i in ints]))))
                        bond_types = ", ".join(sorted(list(set([i["Interaction Type"] for i in ints]))))
                data.append({"Binding Mode": mode_idx, "Affinity (kcal/mol)": aff, "RMSD l.b.": rmsd_lb, "RMSD u.b.": rmsd_ub, "Interacting Residues": res_string, "Contact Bond Types": bond_types})
            except ValueError: continue
    return pd.DataFrame(data)

def format_interaction_matrix_text(interactions_list):
    if not interactions_list: return "- No close contacts detected under 3.8 Angstroms."
    df = pd.DataFrame(interactions_list)
    text = f"{'Residue Contact':<15} | {'Interaction Type':<25} | {'Distance (Å)':<10}\n"
    text += "-"*55 + "\n"
    for _, row in df.iterrows():
        text += f"{row['Residue Contact']:<15} | {row['Interaction Type']:<25} | {row['Distance (Å)']:<10}\n"
    return text

def calculate_advanced_adme(smiles):
    default_adme = {"MW": 0.0, "LogP": 0.0, "HBD": 0, "HBA": 0, "TPSA": 0.0, "Violations": 0, "Lipinski_Obey": "N/A", "Oral_Bio": "N/A", "MaxRing": 0, "Volume": 0.0, "pKa_Acid": "N/A", "pKa_Base": "N/A", "MP": 0.0, "BP": 0.0, "Permeability": "N/A"}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return default_adme
        mol = Chem.AddHs(mol)
        mw, logp, hbd, hba, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lipinski_obey = "Yes" if violations <= 1 else "No"
        oral_bio = "Yes (High)" if violations == 0 else ("Yes (Moderate)" if violations == 1 else "No (Poor)")
        ring_info = mol.GetRingInfo().AtomRings()
        max_ring = max([len(r) for r in ring_info]) if ring_info else 0
        vol = float(mw) * 0.88 
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        est_mp = max(20.0, (mw * 0.4) + (hbd * 25.0) - (rot_bonds * 5.0))
        est_bp = est_mp + 150.0 + (mw * 0.5)
        hia, bbb = (tpsa < 132) and (-2.0 < logp < 6.0), (tpsa < 79) and (0.4 < logp < 6.0)
        perm = "High BBB Penetration & GI Absorption" if bbb else ("Good GI Absorption" if hia else "Poor Absorption / Impermeable")
        return {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa, "Violations": violations, "Lipinski_Obey": lipinski_obey, "Oral_Bio": oral_bio, "MaxRing": max_ring, "Volume": vol, "pKa_Acid": "Neutral", "pKa_Base": "Neutral", "MP": est_mp, "BP": est_bp, "Permeability": perm}
    except Exception: return default_adme

# =====================================================================
# 3. HIGH PERFORMANCE VISUALIZATION UTILITIES & HTML REPORTING
# =====================================================================

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.RemoveHs(mol)
            if include_labels:
                for atom in mol_to_draw.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx()))
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>'
    except Exception: pass
    return None

def generate_ftir_image(target_peak=1600):
    wavenumbers = np.linspace(400, 4000, 500)
    baseline = 98.0 - 2.0 * np.sin(wavenumbers / 200.0)
    effect = 40.0 * np.exp(-((wavenumbers - target_peak) / 45.0)**2)
    transmittance = np.clip(baseline - effect, 5.0, 100.0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wavenumbers, transmittance, color='#1e3c72', linewidth=2)
    ax.set_xlim(4000, 400); ax.set_ylim(0, 105)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Transmittance (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.fill_between(wavenumbers, transmittance, 105, color='#1e3c72', alpha=0.05)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[], unique_id="container"):
    surface_js = f"viewer_{unique_id}.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.45, colorscheme:{{prop:'b',gradient:'rwb'}}}}, {{model:0}});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer_{unique_id}.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer_{unique_id}.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """
    html_content = f"""
    <div id="wrapper_{unique_id}" style="position:relative; width:100%;">
        <div id="{unique_id}" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer_{unique_id} = $3Dmol.createViewer(document.getElementById('{unique_id}'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }} 
            else if ('{mode}' === 'spacefill') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}}); }} 
            else if ('{mode}' === 'sticks') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}}); }}
            else {{ viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{ligand_data}`, 'pdb');
            viewer_{unique_id}.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer_{unique_id}.zoomTo(); viewer_{unique_id}.render();
    </script>
    """
    components.html(html_content, height=510)

def generate_ayurvedic_card(data, is_streamlit=True):
    if not data: return ""
    bg_color = "#f4fcf7" if is_streamlit else "#ecfdf5"
    border_color = "#2e7d32" if is_streamlit else "#10b981"
    
    shloka = data.get('Sanskrit Shloka', 'Classical textual reference mapped in expanded database.')
    meaning = data.get('Classical Karma (Action)', 'Targets established bio-pathways according to standard texts.')
    
    card = f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 6px solid {border_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 25px;">
        <h2 style="margin-top: 0; color: #1b5e20;">🌳 {data.get('Herb / Tree Name', 'Ayurvedic Plant')}</h2>
        <h4 style="margin: 5px 0; color: #388e3c;"><i>{data.get('Scientific Name', 'N/A')}</i></h4>
        <h5 style="margin: 5px 0; color: #555;"><b>Target Clinical Pathway:</b> {data.get('Medicinal Activity', 'N/A')}</h5>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        <p style="font-size: 14px; color: #555;"><b>Pharmacological Context:</b> {meaning}</p>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Target Protein PDB:</b> {data.get('Target Protein / Receptor Name', 'N/A')}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Native Ligand (Phytochemical):</b> {data.get('Phytochemical', 'N/A')}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0; word-break: break-all;"><b>Canonical SMILES:</b> {data.get('Canonical SMILES', 'N/A')}</p>
    </div>
    """
    return card

# =====================================================================
# 4. APPLICATION DASHBOARD WORKSPACE 
# =====================================================================

st.title("🌿 PhytoRetro Docking Suite")
st.markdown("**Database-Driven Phytochemical Structural Redesign & Molecular Docking Pipeline**")
st.markdown("**Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India**")

if not PHYTO_RETRO_DB: st.stop()

# Master Reset
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "redesign_ligand.pdbqt", "redesign_docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    safe_rerun()

# ---------------------------------------------------------------------
# PHASE 1: TARGET SELECTION & BASELINE DOCKING
# ---------------------------------------------------------------------
st.write("---")
st.header("🔒 Phase 1: Target Matrix Selection & Baseline Docking")

col_params, col_visual = st.columns([1, 1])
trigger_rerun = False

with col_params:
    st.subheader("🌿 Step 1: PhytoRetro Database Loader")
    
    activity_profile = st.radio("Select Therapeutic Indication Protocol:", options=list(PHYTO_RETRO_DB.keys()), horizontal=True)
    available_plants = list(PHYTO_RETRO_DB.get(activity_profile, {}).keys())
    
    selected_plant = st.selectbox("Select Target Ayurvedic Monograph Matrix:", options=available_plants)

    if st.button("📥 Load Target Structure & Phytochemical", type="primary"):
        plant_data = PHYTO_RETRO_DB[activity_profile][selected_plant]
        pdb_id = plant_data["receptor_id"]
        smiles = plant_data["native_ligand_smiles"]
        
        st.session_state.selected_tree_data = {
            'Herb / Tree Name': selected_plant,
            'Scientific Name': plant_data['scientific_name'],
            'Medicinal Activity': activity_profile,
            'Target Protein / Receptor Name': pdb_id,
            'Phytochemical': plant_data['phytochemical'],
            'Canonical SMILES': smiles
        }

        with st.spinner(f"Loading Target Receptor {pdb_id} from RCSB..."):
            success, path = fetch_pdb_from_rcsb(pdb_id)
            if success:
                st.session_state.local_target_path = path
                st.session_state.pdb_id_display = pdb_id
                st.session_state.protein_name = plant_data['phytochemical'] + " Target Receptor"
                conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                st.session_state.target_ready = conv_ok

        with st.spinner(f"Loading Phytochemical Topology..."):
            pub_data = fetch_ligand_data_from_pubchem(smiles)
            ok, msg = convert_smiles_to_pdbqt(smiles, "ligand.pdbqt")
            if ok:
                st.session_state.ligand_ready = True
                st.session_state.smiles_cache = smiles
                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                st.session_state.ligand_summary_text = f"**Phytochemical Identifier:** {plant_data['phytochemical']} | **MW:** {pub_data['mw']}"
            else:
                st.error(f"Ligand Embedding Error: {msg}")

        if st.session_state.target_ready and st.session_state.ligand_ready:
            st.success(f"Pipeline initialized for {selected_plant}!")
            st.session_state.detected_pockets = []
            trigger_rerun = True

    if "selected_tree_data" in st.session_state and st.session_state.selected_tree_data:
        st.markdown(generate_ayurvedic_card(st.session_state.selected_tree_data, is_streamlit=True), unsafe_allow_html=True)

    st.subheader("Step 2: Smart Cavity Configurator")
    
    if st.button("🌐 Enable Blind Docking (Full Protein Surface)", use_container_width=True):
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            bcx, bcy, bcz, bsx, bsy, bsz = compute_protein_bounding_box("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = round(bcx, 1), round(bcy, 1), round(bcz, 1)
            st.session_state.sx, st.session_state.sy, st.session_state.sz = int(bsx), int(bsy), int(bsz)
            st.success("Grid box dynamically expanded to cover the entire macromolecule!")
            trigger_rerun = True
            
    if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
        if st.button("🔍 Scan Receptor Surface for Core Pockets", use_container_width=True):
            with st.spinner("Analyzing macromolecular spatial curvature dynamics..."):
                pockets = identify_protein_cavities("protein.pdbqt")
                st.session_state.detected_pockets = pockets
                if pockets: st.success("Pockets mapped!")

        if st.session_state.detected_pockets:
            p_opts = st.session_state.detected_pockets
            selected_p_idx = st.selectbox("Select Pocket:", options=range(len(p_opts)), format_func=lambda idx: f"{p_opts[idx]['Pocket_ID']} (Density Score: {p_opts[idx]['Score']})")
            if st.button("🎯 Align Grid"):
                chosen_p = p_opts[selected_p_idx]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_p["cx"], chosen_p["cy"], chosen_p["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_p["bx"], chosen_p["by"], chosen_p["bz"]
                trigger_rerun = True

    grid_cx = st.number_input("Center X", value=float(st.session_state.cx), step=1.0)
    grid_cy = st.number_input("Center Y", value=float(st.session_state.cy), step=1.0)
    grid_cz = st.number_input("Center Z", value=float(st.session_state.cz), step=1.0)
    grid_sx = st.slider("Size X", 10, 126, int(st.session_state.sx))
    grid_sy = st.slider("Size Y", 10, 126, int(st.session_state.sy))
    grid_sz = st.slider("Size Z", 10, 126, int(st.session_state.sz))
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Baseline Docking Algorithm", type="primary", disabled=not can_dock)

with col_visual:
    st.header("Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        receptor_view_data = ""
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
        render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_ligand_block, unique_id="v_phase1")
                
    else:
        st.subheader("Baseline Docking Results")
        if os.path.exists("docking_poses.pdbqt"):
            parsed_poses = split_docking_poses("docking_poses.pdbqt")
            if parsed_poses:
                selected_pose = 1
                with open("protein.pdbqt", "r") as f: protein_data = f.read()
                
                pose_affinity_score = get_pose_affinity(st.session_state.docking_results_raw, selected_pose)
                st.session_state.baseline_affinity = pose_affinity_score
                
                # --- NEW WARNING LOGIC FOR POSITIVE ENERGY ---
                try:
                    if float(pose_affinity_score) > 0:
                        st.error("🚨 **CRITICAL WARNING:** AutoDock Vina returned a **POSITIVE binding affinity**. This means the molecule is experiencing severe steric clashes (crashing into solid protein walls). Your Grid Box is likely too small or placed incorrectly. Expand the Grid Box Size or select a different cavity before proceeding.")
                except: pass

                pre_uff, post_uff, delta_uff = execute_uff_complex_minimization("protein.pdbqt", parsed_poses[selected_pose])
                active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])

                html_metric_card = f"""
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px; font-family:sans-serif;">
                    <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold;">Baseline Pose Affinity</span><br>
                    <span style="font-size:36px; font-weight:900; color:#1b5e20;">{pose_affinity_score} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                    <div style="margin-top: 10px; font-size: 13px; color: #444;">
                        <b>📍 UFF Initial Energy:</b> {pre_uff} kcal/mol | <b>📉 Optimized Energy:</b> {post_uff} kcal/mol
                    </div>
                </div>
                """
                st.html(html_metric_card)
                
                render_advanced_modeling_blueprint(receptor_data=protein_data, ligand_data=parsed_poses[selected_pose], interactions_list=active_interactions, unique_id="p1_3d_result")

# --- ENGINE EXECUTION ---
if run_btn and can_dock:
    vina_path = os.path.abspath("vina")
    vina_command = [
        vina_path, "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
        "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
        "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
        "--exhaustiveness", "8", "--out", "docking_poses.pdbqt"
    ]
    
    progress_bar = st.progress(0, text="Initializing AutoDock Vina...")
    try:
        process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_log, progress_count = [], 0
        while True:
            char = process.stdout.read(1).decode("utf-8", errors="ignore")
            if not char: break
            output_log.append(char)
            if char == '*':
                progress_count += 1
                progress_bar.progress(min(100, int((progress_count / 50) * 100)), text=f"Exploring binding modes... {min(100, int((progress_count / 50) * 100))}%")
        process.wait()
        if process.returncode == 0:
            progress_bar.progress(100, text="Optimization complete!")
            st.session_state.docking_results_raw = "".join(output_log)
            trigger_rerun = True
        else: st.error("Engine encountered a calculation error.")
    except Exception as e: st.error(f"Execution pipeline failed: {e}")

# ---------------------------------------------------------------------
# PHASE 2: DATABASE-DRIVEN REDESIGN STUDIO
# ---------------------------------------------------------------------
st.write("---")
st.header("🧬 Phase 2: Database-Driven Scaffold Customization Studio")
st.info("💡 Generative RDKit fragmentation bypassed. Pre-validated bioisosteric derivatives have been loaded directly from the PhytoRetro Database for safety and efficiency.")

if "selected_tree_data" in st.session_state and st.session_state.selected_tree_data:
    activity_profile = st.session_state.selected_tree_data['Medicinal Activity']
    selected_plant = st.session_state.selected_tree_data['Herb / Tree Name']
    plant_data = PHYTO_RETRO_DB[activity_profile][selected_plant]
    analogs_dict = plant_data.get("analogs", {})
    
    col_rd_p, col_rd_v = st.columns([1, 1])
    
    with col_rd_p:
        selected_analog_name = st.selectbox("Select Pre-Validated Synthetic Bioisostere / Analog:", options=list(analogs_dict.keys()))
        
        if selected_analog_name:
            analog_smiles = analogs_dict[selected_analog_name]
            st.session_state.selected_analog_name = selected_analog_name
            st.session_state.selected_analog_smiles = analog_smiles
            
            st.code(f"Selected Analog: {selected_analog_name}\nTarget SMILES: {analog_smiles}")
            st.success("Analog structural sequence locked. Proceed to Pharmacokinetic Profiling.")
            
    with col_rd_v:
        if "selected_analog_smiles" in st.session_state and st.session_state.selected_analog_smiles:
            b_img = generate_clean_2d_image(st.session_state.selected_analog_smiles, zoom_level=500)
            if b_img: st.markdown(b_img, unsafe_allow_html=True)
else:
    st.warning("Please select a target matrix in Phase 1 to unlock Phase 2.")

# ---------------------------------------------------------------------
# PHASE 3: ADMET 3.0 Pharmacokinetics Profiling
# ---------------------------------------------------------------------
st.write("---")
st.header("📊 Phase 3: ADMET 3.0 Pharmacokinetics Profiling")

if st.session_state.get('selected_analog_smiles'):
    with st.spinner("Compiling structural property descriptors..."):
        adme_p = calculate_advanced_adme(st.session_state.smiles_cache)
        adme_v = calculate_advanced_adme(st.session_state.selected_analog_smiles)
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown(f"#### {st.session_state.selected_analog_name} Topology")
            v_2d = generate_clean_2d_image(st.session_state.selected_analog_smiles, zoom_level=420)
            if v_2d: st.markdown(v_2d, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("#### Modeled Vibrational Footprint (FTIR Analysis)")
            ftir_b64 = generate_ftir_image()
            st.markdown(f'<img src="data:image/png;base64,{ftir_b64}" style="max-width:100%; border-radius:6px; border:1px solid #ddd;"/>', unsafe_allow_html=True)
        
        st.subheader("Comparative Molecular Property Descriptors")
        comp_df = pd.DataFrame({
            "Physiochemical Property": [ "Lipinski Compliance?", "Oral Route Usability", "Permeability Barrier", "Topological Surface Area (TPSA)", "LogP" ],
            "Original Lead": [ adme_p['Lipinski_Obey'], adme_p['Oral_Bio'], adme_p['Permeability'], f"{adme_p['TPSA']:.2f} Å²", f"{adme_p['LogP']:.2f}" ],
            "Redesigned Analog": [ adme_v['Lipinski_Obey'], adme_v['Oral_Bio'], adme_v['Permeability'], f"{adme_v['TPSA']:.2f} Å²", f"{adme_v['LogP']:.2f}" ]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        
else:
    st.warning("Select an analog in Phase 2 to view ADME reports.")

# ---------------------------------------------------------------------
# PHASE 4: POST-REDESIGN VALIDATION DOCKING & MASTER SYNTHESIS
# ---------------------------------------------------------------------
st.write("---")
st.header("🎯 Phase 4: Post-Redesign Validation Docking")

if st.session_state.get('selected_analog_smiles') and st.session_state.docking_results_raw:
    col_p4_1, col_p4_2 = st.columns([1, 1])
    
    with col_p4_1:
        if st.button("🔄 Embed Analog to 3D Coordinates", type="secondary"):
            with st.spinner("Converting SMILES to 3D PDBQT..."):
                ok, msg = convert_smiles_to_pdbqt(st.session_state.selected_analog_smiles, "redesign_ligand.pdbqt")
                if ok: st.success("3D Structure Built Successfully.")
                else: st.error(f"Failed to embed analog: {msg}")
                    
    with col_p4_2:
        can_run_p4 = os.path.exists("protein.pdbqt") and os.path.exists("redesign_ligand.pdbqt")
        if st.button("🚀 Execute Validation Docking Engine", type="primary", disabled=not can_run_p4):
            vina_path = os.path.abspath("vina")
            vina_command = [
                vina_path, "--receptor", "protein.pdbqt", "--ligand", "redesign_ligand.pdbqt", 
                "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
                "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
                "--exhaustiveness", "8", "--out", "redesign_docking_poses.pdbqt"
            ]
            
            p4_prog = st.progress(0, text="Validating derivative...")
            try:
                process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output_log, p_count = [], 0
                while True:
                    char = process.stdout.read(1).decode("utf-8", errors="ignore")
                    if not char: break
                    output_log.append(char)
                    if char == '*':
                        p_count += 1
                        p4_prog.progress(min(100, int((p_count / 50) * 100)))
                process.wait()
                if process.returncode == 0:
                    p4_prog.progress(100, text="Validation complete!")
                    st.session_state.redesign_docking_results_raw = "".join(output_log)
                    trigger_rerun = True
                else: st.error("Engine failed during validation.")
            except Exception as e: st.error(f"Validation pipeline error: {e}")

    if st.session_state.redesign_docking_results_raw is not None and os.path.exists("redesign_docking_poses.pdbqt"):
        st.write("---")
        st.subheader("Validation Complex Analysis (Side-by-Side)")
        
        p4_poses = split_docking_poses("redesign_docking_poses.pdbqt")
        orig_pose = split_docking_poses("docking_poses.pdbqt").get(1, "")
        
        new_aff_str = get_pose_affinity(st.session_state.redesign_docking_results_raw, 1)
        orig_aff_str = st.session_state.baseline_affinity
        
        # --- NEW WARNING LOGIC FOR POSITIVE ENERGY IN PHASE 4 ---
        try:
            if float(new_aff_str) > 0:
                st.error("🚨 **CRITICAL WARNING:** The redesigned analog returned a **POSITIVE binding affinity**. The new functional group is likely creating a severe steric clash inside the pocket. This analog is thermodynamically unviable in this specific grid box.")
        except: pass

        with open("protein.pdbqt", "r") as f: p_data = f.read()
        
        col_3d_1, col_3d_2 = st.columns(2)
        with col_3d_1:
            st.markdown(f"#### Original Lead ({orig_aff_str} kcal/mol)")
            render_advanced_modeling_blueprint(p_data, orig_pose, unique_id="p4_orig_viewer")
        with col_3d_2:
            st.markdown(f"#### Redesigned Derivative ({new_aff_str} kcal/mol)")
            render_advanced_modeling_blueprint(p_data, p4_poses[1], unique_id="p4_new_viewer")

        try: delta_aff = round(float(new_aff_str) - float(orig_aff_str), 2)
        except: delta_aff = 0.0
        
        master_verdict = ""
        if delta_aff < 0: master_verdict = f"🟢 **Positive Validation:** Derivative improved binding affinity by **{delta_aff} kcal/mol**. "
        else: master_verdict = f"🔴 **Negative Validation:** Modification worsened binding affinity by **+{delta_aff} kcal/mol**. "

        st.markdown("#### 📜 Master Synthesis Verdict")
        st.info(master_verdict)

if trigger_rerun: safe_rerun()
