"""Shared code for the ICLR27 corrected series (iclr27-x*corrected-* and iclr27-x18 onward).

Every corrected notebook pulls this file with curl and imports it as `common`. It holds the pieces that must not drift
between notebooks: census loading, the three graph lifts and their circuits, the readout blocks, the fold-local
within-bucket probe, the paired classical-versus-classical-plus-quantum comparison, the classical descriptors, the
retrieval scoring, and the configuration digest. Notebooks keep their own config cell, arm table, loop, and tables.

Conventions: L0(G), L1(G), L2(G) are lift0, lift1, lift2 in code. Settings live in module globals set by `configure`.
"""

import csv
import hashlib
import itertools
import json
import pickle
import time

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ---------------- settings (module globals; notebooks call configure to change them) ----------------
ETA = 23 / 8                                  # flat preparation angle on every qubit
GAMMA = 2.0                                   # interaction scale on every RZZ
BETA = 0.1                                    # mixer angle; arms set this per run through configure
TORSION_WEIGHT = 1.0                          # scale of the dihedral edge angle at L2(G)
LINEAR_SIN_TOL = 0.05                         # angles this close to linear are skipped when forming dihedrals
L2_QUBIT_CAP = 12                             # heavy-atom angle registers larger than this are dropped
MAX_LIFT_DEGREE = 24                          # degree sectors span 0..MAX_LIFT_DEGREE at every lift
PROFILE_HEAD = 16                             # entries kept from each sector-conditioned sorted profile
CLASS_PROFILE_LIFTS = ("lift0",)              # class-conditioned profiles only where the class-sector vocabulary is small
CLASS_PROFILE_HEAD = 8                        # entries kept from each class-sector profile; the sector union is wide, so the head is short
RIDGE_ALPHAS = np.logspace(-10, 2, 13)        # ridge grid shared by every probe
RANK_TOLERANCE = 1e-14
EIGENVALUE_FLOOR = 1e-12
RESIDUAL_VARIANCE_FLOOR = 1e-10               # residual targets with less variance than this fraction are flagged exact
DENSE_CHECK_TOL = 1e-12                       # qiskit against the independent tensor simulation
BOND_ORDER = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5}
BOND_TYPE_INDEX = {"single": 0, "double": 1, "triple": 2, "aromatic": 3}


def configure(**overrides):
    """Set module settings by name, e.g. configure(BETA=1.0); unknown names raise"""
    for name, value in overrides.items():
        assert name in globals() and name.isupper(), f"unknown setting {name}"
        globals()[name] = value


def module_settings():
    """The module settings that enter the digest"""
    return {"ETA": ETA, "GAMMA": GAMMA, "BETA": BETA, "TORSION_WEIGHT": TORSION_WEIGHT, "LINEAR_SIN_TOL": LINEAR_SIN_TOL, "L2_QUBIT_CAP": L2_QUBIT_CAP,
            "MAX_LIFT_DEGREE": MAX_LIFT_DEGREE, "PROFILE_HEAD": PROFILE_HEAD, "CLASS_PROFILE_HEAD": CLASS_PROFILE_HEAD, "CLASS_PROFILE_LIFTS": list(CLASS_PROFILE_LIFTS), "RIDGE_ALPHAS": list(np.asarray(RIDGE_ALPHAS).tolist()),
            "RANK_TOLERANCE": RANK_TOLERANCE, "EIGENVALUE_FLOOR": EIGENVALUE_FLOOR, "RESIDUAL_VARIANCE_FLOOR": RESIDUAL_VARIANCE_FLOOR}


def config_digest(notebook_settings):
    """Digest of the module settings, the notebook's own settings, and the qiskit version; every notebook records this"""
    payload = {"module": module_settings(), "notebook": notebook_settings, "qiskit_version": qiskit.__version__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------- census loading ----------------

def load_census(freeze_pkl, geometry_pkl):
    """The freeze and geometry pickles as one dict of the arrays every notebook needs, indexed by probe position"""
    with open(freeze_pkl, "rb") as handle:
        freeze = pickle.load(handle)
    with open(geometry_pkl, "rb") as handle:
        geometry_pickle = pickle.load(handle)
    assert geometry_pickle["freeze_digests"] == freeze["digests"]
    molecules = freeze["molecules"]
    geometry = geometry_pickle["geometry"]
    position_by_row = {molecule["row"]: position for position, molecule in enumerate(molecules)}
    probe_positions = np.array([position_by_row[row] for row in freeze["probe_rows"]])
    addition_names = list(geometry_pickle["panel_additions"]["names"])
    addition_matrix = np.asarray(geometry_pickle["panel_additions"]["matrix"], dtype=float)
    bank_matrix = np.asarray(geometry_pickle["geometric_bank"]["matrix"], dtype=float)
    census = {
        "freeze": freeze, "geometry_pickle": geometry_pickle, "molecules": molecules, "geometry": geometry,
        "probe_positions": probe_positions, "probe_index_of_position": {position: index for index, position in enumerate(probe_positions)},
        "probe_fold": np.asarray(freeze["folds"]["probe_fold"]), "num_outer_folds": len(freeze["folds"]["fold_sizes"]),
        "longest_bond": freeze["census_constants"]["longest_bond_length"],
        "target_names": list(freeze["targets"]["names"]) + addition_names,
        "target_matrix": np.hstack([np.asarray(freeze["targets"]["matrix"], dtype=float), addition_matrix]),
        "cip_label": addition_matrix[:, addition_names.index("handedness_rs")] if "handedness_rs" in addition_names else np.full(len(probe_positions), np.nan),
        "lift2_register": np.array([geometry[position]["registers"]["lift2_heavy"] for position in probe_positions]),
        "element_classes": [int(z) for z in freeze["vocabularies"]["element_classes"]],
        "degree_classes": [int(d) for d in freeze["vocabularies"]["degree_classes"]],
        "joint_classes": [tuple(int(v) for v in c) for c in freeze["vocabularies"]["joint_classes"]],
        "bond_classes": [tuple(int(v) for v in c) for c in geometry_pickle["vocabularies"]["lift1_bond_class"]["classes"]],
        "angle_classes": [tuple(int(v) for v in c) for c in geometry_pickle["vocabularies"]["lift2_angle_class"]["classes"]],
        "bank_matrix": bank_matrix,                                                  # probe-indexed, all columns; subsets drop constant columns themselves
        "bank_names": list(geometry_pickle["geometric_bank"].get("names", [f"bank_{i}" for i in range(bank_matrix.shape[1])])),
    }
    census["fits_cap"] = (census["lift2_register"] > 0) & (census["lift2_register"] <= L2_QUBIT_CAP)
    return census


# ---------------- graphs and lifts ----------------

def aromatic_rings(molecule):
    """Rings of the aromatic-bond subgraph, as atom lists, from its minimum cycle basis"""
    aromatic = nx.Graph()
    for edge, bond_type in zip(molecule["edges"], molecule["bond_types"]):
        if bond_type == "aromatic":
            aromatic.add_edge(int(edge[0]), int(edge[1]))
    return [list(ring) for ring in nx.minimum_cycle_basis(aromatic)] if aromatic.number_of_edges() else []


def underlying_graph(molecule, positions, aromatic_setting, multiplicity_setting):
    """Edges of G as (atom a, atom b, weight, length, kind) under the aromatic and multiplicity settings"""
    rings = aromatic_rings(molecule) if aromatic_setting == "complete" else []
    ring_bonds = set()
    for ring in rings:
        for a, b in itertools.combinations(ring, 2):
            ring_bonds.add((min(a, b), max(a, b)))
    graph_edges = []
    for edge, bond_type, length in zip(molecule["edges"], molecule["bond_types"], molecule["bond_lengths"]):
        a, b = int(edge[0]), int(edge[1])
        if bond_type == "aromatic" and (a, b) in ring_bonds:
            continue                                            # replaced by the ring's complete graph below
        order = BOND_ORDER[bond_type]
        if multiplicity_setting == "parallel" and bond_type in ("double", "triple"):
            for _ in range(int(order)):
                graph_edges.append((a, b, 1.0, float(length), bond_type))
        else:
            graph_edges.append((a, b, order, float(length), bond_type))
    ring_weight = {}
    for ring in rings:
        n = len(ring)
        weight = 1.5 * n / (n * (n - 1) / 2)
        for a, b in itertools.combinations(sorted(ring), 2):
            ring_weight[(a, b)] = ring_weight.get((a, b), 0.0) + weight
    for (a, b), weight in sorted(ring_weight.items()):
        graph_edges.append((a, b, weight, float(np.linalg.norm(positions[a] - positions[b])), "ring"))
    return graph_edges


def bond_angle(positions, i, j, k):
    """Angle at j between i and k, radians"""
    a = positions[i] - positions[j]
    b = positions[k] - positions[j]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def is_linear(positions, i, j, k):
    return np.sin(bond_angle(positions, i, j, k)) < LINEAR_SIN_TOL


def dihedral(positions, i, j, k, l):
    """Signed dihedral about the j-k axis between i and l"""
    b0 = positions[i] - positions[j]
    b1 = positions[k] - positions[j]
    b2 = positions[l] - positions[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def dihedral_angle(phi, sign_setting):
    """The edge angle for a dihedral: cosine only, or cosine plus sine"""
    return TORSION_WEIGHT * (np.cos(phi) + (np.sin(phi) if sign_setting == "signed" else 0.0))


def pack(node_angles, node_encoding, node_classes, edges, mixer):
    """Pack a construction; edges is a dict from qubit pair to summed angle"""
    degrees = np.zeros(len(node_angles), dtype=int)
    edge_list = []
    for (a, b), angle in sorted(edges.items()):
        degrees[a] += 1
        degrees[b] += 1
        edge_list.append((a, b, float(angle)))
    return {"num_qubits": len(node_angles), "node_angles": np.array(node_angles, dtype=float), "node_encoding": node_encoding,
            "node_classes": np.array(node_classes), "node_degrees": degrees, "edges": edge_list, "mixer": mixer}


def add_edge(edges, a, b, angle):
    key = (min(a, b), max(a, b))
    edges[key] = edges.get(key, 0.0) + angle


def build_lift0(molecule, positions, setting, longest_bond):
    """Atoms on qubits, every edge of G on an RZZ; setting["node"] chooses the preparation: flat, binary (H at pi - eta), atomic (eta Z / 9)"""
    graph_edges = underlying_graph(molecule, positions, setting["aromatic"], setting["multiplicity"])
    edges = {}
    for a, b, weight, length, _ in graph_edges:
        if setting["entangler"] == "none":
            continue
        add_edge(edges, a, b, GAMMA * weight * (length / longest_bond if setting["edges"] == "length" else 1.0))
    atomic = [int(z) for z in molecule["atomic_numbers"]]
    preparation = setting.get("node", "flat")
    if preparation == "flat":
        node_angles = [ETA] * len(atomic)
    elif preparation == "binary":
        node_angles = [np.pi - ETA if z == 1 else ETA for z in atomic]
    elif preparation == "atomic":
        node_angles = [ETA * z / 9.0 for z in atomic]
    else:
        raise ValueError(preparation)
    return pack(node_angles, "amplitude", atomic, edges, setting["mixer"])


def build_lift1(molecule, positions, setting, longest_bond):
    """Edges of G on qubits, angles between edges sharing an atom on RZZ"""
    graph_edges = underlying_graph(molecule, positions, setting["aromatic"], setting["multiplicity"])
    atomic = molecule["atomic_numbers"]
    node_angles = [ETA * (length / longest_bond if setting["feature"] == "length" else 1.0) for _, _, _, length, _ in graph_edges]
    node_classes = [(min(int(atomic[a]), int(atomic[b])), max(int(atomic[a]), int(atomic[b])), kind) for a, b, _, _, kind in graph_edges]
    edges = {}
    if setting["entangler"] != "none":
        for (p, (a, b, _, _, _)), (q, (c, d, _, _, _)) in itertools.combinations(enumerate(graph_edges), 2):
            shared = {a, b} & {c, d}
            if not shared:
                continue
            if len(shared) == 2:
                theta = 0.0                                     # parallel copies of one bond
            else:
                j = shared.pop()
                i = a if b == j else b
                k = c if d == j else d
                theta = bond_angle(positions, i, j, k)
            add_edge(edges, p, q, GAMMA * (theta / np.pi if setting["edges"] == "angle" else 1.0))
    return pack(node_angles, setting["encoding"], node_classes, edges, setting["mixer"])


def build_lift2(molecule, positions, setting, longest_bond):
    """Angles of the heavy-atom part of G on qubits, dihedrals about shared edges on RZZ; None when the register exceeds the cap"""
    graph_edges = underlying_graph(molecule, positions, "cycle", setting["multiplicity"])
    atomic = molecule["atomic_numbers"]
    heavy_edges = [(index, a, b) for index, (a, b, _, _, _) in enumerate(graph_edges) if atomic[a] > 1 and atomic[b] > 1]
    edges_at = {}
    for index, a, b in heavy_edges:
        edges_at.setdefault(a, []).append((index, b))
        edges_at.setdefault(b, []).append((index, a))
    nodes = []                                                  # (centre j, edge e1 to outer i, outer i, edge e2 to outer k, outer k)
    for j in sorted(edges_at):
        for (e1, i), (e2, k) in itertools.combinations(sorted(edges_at[j]), 2):
            nodes.append((j, e1, i, e2, k))
    if len(nodes) == 0 or len(nodes) > L2_QUBIT_CAP:
        return None
    thetas = [bond_angle(positions, i, j, k) for j, _, i, _, k in nodes]
    node_angles = [ETA * (theta / np.pi if setting["feature"] == "angle" else 1.0) for theta in thetas]
    node_classes = [(int(atomic[j]), min(int(atomic[i]), int(atomic[k])), max(int(atomic[i]), int(atomic[k]))) for j, _, i, _, k in nodes]
    edges = {}
    if setting["entangler"] != "none":
        nodes_with_edge = {}
        for index, (j, e1, i, e2, k) in enumerate(nodes):
            nodes_with_edge.setdefault(e1, []).append(index)
            nodes_with_edge.setdefault(e2, []).append(index)
        for edge_index, members in nodes_with_edge.items():
            for p, q in itertools.combinations(members, 2):
                jp, e1p, ip, e2p, kp = nodes[p]
                jq, e1q, iq, e2q, kq = nodes[q]
                if setting["edges"] == "flat":
                    add_edge(edges, p, q, GAMMA)
                    continue
                if jp != jq:                                    # proper: centres at the two ends of the shared edge
                    outer_p = ip if e1p != edge_index else kp
                    outer_q = iq if e1q != edge_index else kq
                    if outer_p == jq or outer_q == jp or outer_p == outer_q:
                        continue
                    if is_linear(positions, outer_p, jp, jq) or is_linear(positions, jp, jq, outer_q):
                        continue
                    add_edge(edges, p, q, dihedral_angle(dihedral(positions, outer_p, jp, jq, outer_q), setting["sign"]))
                elif setting["improper"] == "on":               # improper: same centre, dihedral about the shared edge between the outer atoms
                    j = jp
                    shared_outer = ip if e1p == edge_index else kp
                    outer_p = kp if e1p == edge_index else ip
                    outer_q = kq if e1q == edge_index else iq
                    if outer_p == outer_q or outer_p == shared_outer or outer_q == shared_outer:
                        continue
                    key_p = (int(atomic[outer_p]), len(edges_at.get(outer_p, [])))
                    key_q = (int(atomic[outer_q]), len(edges_at.get(outer_q, [])))
                    if key_p < key_q:
                        outer_p, outer_q, key_p, key_q = outer_q, outer_p, key_q, key_p
                    if is_linear(positions, outer_p, j, shared_outer) or is_linear(positions, j, shared_outer, outer_q):
                        continue
                    phi = dihedral(positions, outer_p, j, shared_outer, outer_q)
                    angle = TORSION_WEIGHT * (np.cos(phi) + (np.sin(phi) if setting["sign"] == "signed" and key_p != key_q else 0.0))
                    add_edge(edges, p, q, angle)
    return pack(node_angles, setting["encoding"], node_classes, edges, setting["mixer"])


BUILDERS = {"lift0": build_lift0, "lift1": build_lift1, "lift2": build_lift2}


def build(lift, molecule, positions, setting, census):
    """The construction of one molecule under one arm, or None when it exceeds the cap"""
    return BUILDERS[lift](molecule, positions, setting, census["longest_bond"])


# ---------------- circuits ----------------

def build_circuit(construction, measurement_basis="Z"):
    """The circuit: preparation, RZZ edges, mixer at BETA, then the rotation that makes the chosen Pauli basis the computational one"""
    circuit = QuantumCircuit(construction["num_qubits"])
    for qubit, angle in enumerate(construction["node_angles"]):
        if construction["node_encoding"] == "phase":
            circuit.h(qubit)
            circuit.rz(angle, qubit)
        else:
            circuit.rx(angle, qubit)
    for a, b, angle in construction["edges"]:
        circuit.rzz(angle, a, b)
    for qubit in range(construction["num_qubits"]):
        if construction["mixer"] == "X":
            circuit.rx(BETA, qubit)
        elif construction["mixer"] == "Y":
            circuit.ry(BETA, qubit)
    for qubit in range(construction["num_qubits"]):
        if measurement_basis == "X":
            circuit.h(qubit)
        elif measurement_basis == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        else:
            assert measurement_basis == "Z", measurement_basis
    return circuit


def statevector_of(construction):
    """The pre-measurement state in qiskit's ordering"""
    return Statevector(build_circuit(construction, "Z")).data


def probabilities_of(construction, measurement_basis="Z"):
    return np.abs(Statevector(build_circuit(construction, measurement_basis)).data) ** 2


def single_qubit_matrix(gate, angle):
    if gate == "rx":
        return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)], [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
    if gate == "ry":
        return np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])
    if gate == "rz":
        return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]])
    if gate == "h":
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    raise ValueError(gate)


def dense_probabilities_of(construction):
    """Independent tensor simulation in qiskit's index convention, Z basis; the cross-check for qiskit"""
    n = construction["num_qubits"]
    state = np.zeros((2,) * n, dtype=np.complex128)
    state[(0,) * n] = 1.0

    def apply(matrix, qubit):
        axis = n - 1 - qubit
        return np.moveaxis(np.tensordot(matrix, state, axes=([1], [axis])), 0, axis)

    for qubit, angle in enumerate(construction["node_angles"]):
        if construction["node_encoding"] == "phase":
            state = apply(single_qubit_matrix("h", 0.0), qubit)
            state = apply(single_qubit_matrix("rz", angle), qubit)
        else:
            state = apply(single_qubit_matrix("rx", angle), qubit)
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    z_values = 1 - 2 * bits.astype(float)
    exponent = np.zeros(2 ** n)
    for a, b, angle in construction["edges"]:
        exponent += angle / 2 * z_values[:, a] * z_values[:, b]
    state = (state.reshape(-1) * np.exp(-1j * exponent)).reshape(state.shape)
    if construction["mixer"] in ("X", "Y"):
        for qubit in range(n):
            state = apply(single_qubit_matrix("ry" if construction["mixer"] == "Y" else "rx", BETA), qubit)
    return np.abs(state.reshape(-1)) ** 2


def relabel(construction, permutation):
    inverse = np.argsort(permutation)
    return {**construction, "node_angles": construction["node_angles"][inverse], "node_classes": construction["node_classes"][inverse],
            "node_degrees": construction["node_degrees"][inverse], "edges": [(int(permutation[a]), int(permutation[b]), angle) for a, b, angle in construction["edges"]]}


def sample_probabilities(probabilities, shots, rng):
    """One multinomial realisation of a Born distribution, as empirical probabilities"""
    return rng.multinomial(shots, probabilities / probabilities.sum()) / shots


# ---------------- readout blocks ----------------

def global_class_of_node(lift, node_class):
    """A construction's node class in the census-wide vocabulary"""
    if lift == "lift0":
        return int(node_class)
    if lift == "lift1":
        return (int(node_class[0]), int(node_class[1]), BOND_TYPE_INDEX[node_class[2]])
    return (int(node_class[0]), int(node_class[1]), int(node_class[2]))


def class_vocabulary(lift, census):
    """The census-wide class list for one lift"""
    return {"lift0": census["element_classes"], "lift1": census["bond_classes"], "lift2": census["angle_classes"]}[lift]


def occupancy_keys(bits, ids, num_classes):
    """Mixed-radix key of every outcome's occupancy vector over the classes, and the weights to decode a key"""
    indicator = np.zeros((bits.shape[1], num_classes))
    indicator[np.arange(bits.shape[1]), ids] = 1.0
    occupancy = (bits @ indicator).astype(np.int64)
    radix = occupancy.max(axis=0) + 1
    weights = np.concatenate([np.cumprod(radix[::-1])[::-1][1:], [1]])
    return occupancy @ weights, weights


def decode_key(key, weights):
    counts = []
    remainder = int(key)
    for weight in weights:
        counts.append(remainder // int(weight))
        remainder = remainder % int(weight)
    return tuple(counts)


def readout_blocks(probabilities, construction, lift, census):
    """Every readout block of one distribution.

    sorted: the sorted vector (descending). hamming: masses by Hamming weight. class_sectors / degree_sectors: masses keyed by
    occupancy over the census-wide class list / the degree range. hamming_profiles: for each weight, the first PROFILE_HEAD
    entries of the sorted conditional distribution inside that layer. class_profiles: the same inside each class sector.
    correlators: sorted one-qubit expectations, sorted edge ZZ correlators, and ZZ correlators summed by class pair.
    The product state without entangler is uniform inside every Hamming layer, so the profiles measure interaction directly.
    """
    n = construction["num_qubits"]
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    weights_by_outcome = bits.sum(axis=1)
    blocks = {"sorted": np.sort(probabilities)[::-1], "hamming": np.bincount(weights_by_outcome, weights=probabilities, minlength=n + 1)}
    node_class_ids = [class_vocabulary(lift, census).index(global_class_of_node(lift, c)) for c in construction["node_classes"]]
    assert max(construction["node_degrees"], default=0) <= MAX_LIFT_DEGREE
    node_degree_ids = [int(d) for d in construction["node_degrees"]]
    for label, ids, num_classes in (("class_sectors", node_class_ids, len(class_vocabulary(lift, census))), ("degree_sectors", node_degree_ids, MAX_LIFT_DEGREE + 1)):
        keys, weights = occupancy_keys(bits, ids, num_classes)
        masses = np.bincount(keys, weights=probabilities)
        blocks[label] = {decode_key(key, weights): float(masses[key]) for key in np.nonzero(masses)[0]}
        if label == "class_sectors" and lift in CLASS_PROFILE_LIFTS:
            blocks["class_profiles"] = {}
            for key in np.nonzero(masses)[0]:
                members = probabilities[keys == key]
                profile = np.sort(members / masses[key])[::-1][:CLASS_PROFILE_HEAD]
                for rank, value in enumerate(profile):
                    blocks["class_profiles"][(decode_key(key, weights), rank)] = float(value)
    blocks["hamming_profiles"] = {}
    for weight in range(n + 1):
        members = probabilities[weights_by_outcome == weight]
        if members.sum() <= 0:
            continue
        profile = np.sort(members / members.sum())[::-1][:PROFILE_HEAD]
        for rank, value in enumerate(profile):
            blocks["hamming_profiles"][(weight, rank)] = float(value)
    z_values = 1.0 - 2.0 * bits
    single = probabilities @ z_values                                              # <Z_i>
    blocks["correlators"] = {("z1", rank): float(value) for rank, value in enumerate(np.sort(single)[::-1])}
    edge_values = []
    class_pair_sum = {}
    for a, b, _ in construction["edges"]:
        value = float(probabilities @ (z_values[:, a] * z_values[:, b]))
        edge_values.append(value)
        pair = tuple(sorted((node_class_ids[a], node_class_ids[b])))
        class_pair_sum[pair] = class_pair_sum.get(pair, 0.0) + value
    for rank, value in enumerate(sorted(edge_values, reverse=True)):
        blocks["correlators"][("zz_edge", rank)] = value
    for pair, value in class_pair_sum.items():
        blocks["correlators"][("zz_class", pair)] = value
    return blocks


BLOCK_NAMES = ["sorted", "hamming", "class_sectors", "degree_sectors", "hamming_profiles", "class_profiles", "correlators"]   # class_profiles only for CLASS_PROFILE_LIFTS


def feature_matrix(block_list, block):
    """Rows of one block as a dense matrix: vectors padded with zeros to the longest, dicts over the union of keys"""
    first = block_list[0][block]
    if isinstance(first, np.ndarray):
        longest = max(len(r[block]) for r in block_list)
        matrix = np.zeros((len(block_list), longest))
        for row, r in enumerate(block_list):
            matrix[row, :len(r[block])] = r[block]
        return matrix
    vocabulary = {}
    for r in block_list:
        for key in r[block]:
            if key not in vocabulary:
                vocabulary[key] = len(vocabulary)
    matrix = np.zeros((len(block_list), len(vocabulary)))
    for row, r in enumerate(block_list):
        for key, value in r[block].items():
            matrix[row, vocabulary[key]] = value
    return matrix


def feature_matrices(block_lists_by_realisation, block, float_type=np.float32):
    """One dense matrix per realisation for a block, all sharing one column vocabulary so exact and sampled columns align"""
    realisations = list(block_lists_by_realisation)
    first = block_lists_by_realisation[realisations[0]][0][block]
    if isinstance(first, np.ndarray):
        longest = max(len(r[block]) for name in realisations for r in block_lists_by_realisation[name])
        matrices = {}
        for name in realisations:
            matrix = np.zeros((len(block_lists_by_realisation[name]), longest), dtype=float_type)
            for row, r in enumerate(block_lists_by_realisation[name]):
                matrix[row, :len(r[block])] = r[block]
            matrices[name] = matrix
        return matrices, [f"{block}_{i}" for i in range(longest)]
    vocabulary = {}
    for name in realisations:
        for r in block_lists_by_realisation[name]:
            for key in r[block]:
                if key not in vocabulary:
                    vocabulary[key] = len(vocabulary)
    matrices = {}
    for name in realisations:
        matrix = np.zeros((len(block_lists_by_realisation[name]), len(vocabulary)), dtype=float_type)
        for row, r in enumerate(block_lists_by_realisation[name]):
            for key, value in r[block].items():
                matrix[row, vocabulary[key]] = value
        matrices[name] = matrix
    return matrices, list(vocabulary)


def block_vocabulary(block_list, block, min_support, max_columns):
    """Keys of a dict block that are nonzero on at least min_support rows, ordered by support, at most max_columns of them"""
    support = {}
    for r in block_list:
        for key, value in r[block].items():
            if value != 0:
                support[key] = support.get(key, 0) + 1
    kept = sorted((key for key, rows_with in support.items() if rows_with >= min_support), key=lambda key: (-support[key], str(key)))[:max_columns]
    return kept, len(support)


def block_matrix(block_list, block, vocabulary=None, width=None, float_type=np.float32):
    """Dense matrix of one realisation of a block: dict blocks over a fixed vocabulary (keys outside it are dropped), array blocks padded to width"""
    if vocabulary is None:
        matrix = np.zeros((len(block_list), width), dtype=float_type)
        for row, r in enumerate(block_list):
            values = r[block][:width]
            matrix[row, :len(values)] = values
        return matrix
    column_of = {key: column for column, key in enumerate(vocabulary)}
    matrix = np.zeros((len(block_list), len(vocabulary)), dtype=float_type)
    for row, r in enumerate(block_list):
        for key, value in r[block].items():
            column = column_of.get(key)
            if column is not None:
                matrix[row, column] = value
    return matrix


def audit_construction(construction, lift, census, arm_name, seed):
    """Qiskit against the dense simulation, and relabeling invariance of the blocks; raises on failure"""
    probabilities = probabilities_of(construction)
    worst = float(np.max(np.abs(probabilities - dense_probabilities_of(construction))))
    assert worst < DENSE_CHECK_TOL, f"{arm_name}: dense cross-check {worst:.2e}"
    permuted = relabel(construction, np.random.default_rng(seed).permutation(construction["num_qubits"]))
    original = readout_blocks(probabilities, construction, lift, census)
    relabeled = readout_blocks(probabilities_of(permuted), permuted, lift, census)
    assert np.max(np.abs(original["sorted"] - relabeled["sorted"])) < 1e-10, f"{arm_name}: relabeling changed the sorted vector"
    for block in ("class_sectors", "degree_sectors", "hamming_profiles", "class_profiles", "correlators"):
        if block not in original:
            continue
        assert set(original[block]) == set(relabeled[block]), f"{arm_name}: relabeling changed the keys of {block}"
        assert max(abs(original[block][k] - relabeled[block][k]) for k in original[block]) < 1e-10, f"{arm_name}: relabeling changed {block}"


# ---------------- classical descriptors ----------------

def count_features(census, row_positions):
    """Joint (element, degree), bond-class, and angle-class counts per row"""
    molecules, geometry = census["molecules"], census["geometry"]
    joint = np.array([[int(((molecules[p]["atomic_numbers"] == element) & (molecules[p]["degrees"] == degree)).sum()) for element, degree in census["joint_classes"]] for p in row_positions], dtype=float)
    bonds = np.array([np.bincount(geometry[p]["bond_classes_ids"], minlength=len(census["bond_classes"])) for p in row_positions], dtype=float)
    angles = np.array([np.bincount(geometry[p]["angle_classes_ids"], minlength=len(census["angle_classes"])) for p in row_positions], dtype=float)
    return np.hstack([joint, bonds, angles])


def degree_histogram(census, row_positions):
    molecules = census["molecules"]
    return np.array([[int((molecules[p]["degrees"] == degree).sum()) for degree in census["degree_classes"]] for p in row_positions], dtype=int)


def bank_features(census, row_positions):
    """The geometric bank on the rows, constant columns dropped on these rows"""
    bank = np.array([census["bank_matrix"][census["probe_index_of_position"][p]] for p in row_positions])
    return bank[:, bank.std(axis=0) > 0]


def wl_count_fingerprint(molecule, rounds):
    """Weisfeiler-Leman label counts after the given rounds; initial label (element, aromatic flag), refinement over (bond type, neighbour label)"""
    atoms = range(len(molecule["atomic_numbers"]))
    aromatic_atoms = set()
    incident = {atom: [] for atom in atoms}
    for edge, bond_type in zip(molecule["edges"], molecule["bond_types"]):
        a, b = int(edge[0]), int(edge[1])
        incident[a].append((b, bond_type))
        incident[b].append((a, bond_type))
        if bond_type == "aromatic":
            aromatic_atoms.update((a, b))
    labels = {atom: (int(molecule["atomic_numbers"][atom]), atom in aromatic_atoms) for atom in atoms}
    counts = {}
    for round_index in range(rounds + 1):
        for atom in atoms:
            key = (round_index, labels[atom])
            counts[key] = counts.get(key, 0) + 1
        labels = {atom: (labels[atom], tuple(sorted((bond_type, labels[neighbour]) for neighbour, bond_type in incident[atom]))) for atom in atoms}
    return counts


def lifted_wl_counts(construction, lift, rounds):
    """Weisfeiler-Leman label counts on the lifted graph itself: nodes labelled by class, edges unlabelled; the matched classical lift"""
    n = construction["num_qubits"]
    incident = {node: [] for node in range(n)}
    for a, b, _ in construction["edges"]:
        incident[a].append(b)
        incident[b].append(a)
    labels = {node: (global_class_of_node(lift, construction["node_classes"][node]),) for node in range(n)}
    counts = {}
    for round_index in range(rounds + 1):
        for node in range(n):
            key = (round_index, labels[node])
            counts[key] = counts.get(key, 0) + 1
        labels = {node: (labels[node], tuple(sorted(labels[neighbour] for neighbour in incident[node]))) for node in range(n)}
    return counts


def frequent_count_matrix(count_dicts, min_support, max_columns):
    """Dense matrix of the most frequent count keys: support of at least min_support rows, at most max_columns columns"""
    support = {}
    for counts in count_dicts:
        for key in counts:
            support[key] = support.get(key, 0) + 1
    kept = sorted((key for key, rows_with in support.items() if rows_with >= min_support), key=lambda key: (-support[key], str(key)))[:max_columns]
    matrix = np.zeros((len(count_dicts), len(kept)))
    for row, counts in enumerate(count_dicts):
        for column, key in enumerate(kept):
            matrix[row, column] = counts.get(key, 0)
    return matrix


def morgan_count_fingerprint(molecule, radius):
    """RDKit Morgan count fingerprint of the freeze graph with explicit hydrogens; partial sanitisation when full sanitisation fails"""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    bond_types = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE, "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    editable = Chem.RWMol()
    for z in molecule["atomic_numbers"]:
        editable.AddAtom(Chem.Atom(int(z)))
    for edge, bond_type in zip(molecule["edges"], molecule["bond_types"]):
        editable.AddBond(int(edge[0]), int(edge[1]), bond_types[bond_type])
        if bond_type == "aromatic":
            editable.GetAtomWithIdx(int(edge[0])).SetIsAromatic(True)
            editable.GetAtomWithIdx(int(edge[1])).SetIsAromatic(True)
    built = editable.GetMol()
    if Chem.SanitizeMol(built, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        built = editable.GetMol()
        built.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(built)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    return dict(generator.GetSparseCountFingerprint(built).GetNonzeroElements())


COULOMB_ELEMENTS = [1, 6, 7, 8, 9]


def coulomb_matrix(molecule, positions):
    """Coulomb matrix: 0.5 Z^2.4 on the diagonal, Z_i Z_j / r_ij off it"""
    charges = np.asarray(molecule["atomic_numbers"], dtype=float)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    with np.errstate(divide="ignore"):
        matrix = np.outer(charges, charges) / distances
    np.fill_diagonal(matrix, 0.5 * charges ** 2.4)
    return matrix


def descriptor_3d(molecule, positions, distance_bins=10, distance_min=0.5, distance_max=6.0):
    """Sorted Coulomb eigenvalues, sorted-row Coulomb upper triangle, and per-element-pair distance histograms, concatenated"""
    matrix = coulomb_matrix(molecule, positions)
    eigenvalues = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    order = np.argsort(-np.linalg.norm(matrix, axis=1))
    sorted_matrix = matrix[np.ix_(order, order)]
    upper = sorted_matrix[np.triu_indices(len(order))]
    charges = np.asarray(molecule["atomic_numbers"])
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    edges = np.linspace(distance_min, distance_max, distance_bins + 1)
    histograms = []
    for index, a in enumerate(COULOMB_ELEMENTS):
        for b in COULOMB_ELEMENTS[index:]:
            mask = (charges[:, None] == a) & (charges[None, :] == b)
            mask = np.triu(mask | mask.T, k=1)
            histograms.append(np.histogram(distances[mask], bins=edges)[0].astype(float))
    return np.concatenate([eigenvalues, upper, np.concatenate(histograms)])


# ---------------- rows, buckets, folds ----------------

def bucket_rows(census, positions, min_bucket_size):
    """Rows of the within-bucket design: probe positions in formula buckets of at least min_bucket_size, with bucket ids"""
    members_by_formula = {}
    for position in positions:
        members_by_formula.setdefault(census["molecules"][position]["formula"], []).append(position)
    kept_formulas = [formula for formula, members in members_by_formula.items() if len(members) >= min_bucket_size]
    row_positions = np.array([position for formula in kept_formulas for position in members_by_formula[formula]])
    bucket_id = np.array([kept_formulas.index(census["molecules"][position]["formula"]) for position in row_positions])
    return row_positions, bucket_id


def bucket_folds(bucket_id, num_outer, num_inner, seed):
    """Random balanced folds inside every bucket, outer and inner; each bucket appears in every outer fold's training set"""
    rng = np.random.default_rng(seed)
    outer = np.full(len(bucket_id), -1, dtype=int)
    for bucket in np.unique(bucket_id):
        members = np.where(bucket_id == bucket)[0]
        outer[members] = rng.permutation(np.arange(len(members)) % num_outer)
    inner = np.full((num_outer, len(bucket_id)), -1, dtype=int)
    for fold in range(num_outer):
        train_rows = np.where(outer != fold)[0]
        for bucket in np.unique(bucket_id[train_rows]):
            members = train_rows[bucket_id[train_rows] == bucket]
            inner[fold, members] = rng.permutation(np.arange(len(members)) % num_inner)
    return outer, inner


def formula_folds(census, row_positions, num_inner, seed):
    """The freeze's formula-whole outer folds for cross-formula probing, with inner folds by the same balanced bucket rule"""
    formulas = [census["molecules"][p]["formula"] for p in row_positions]
    outer = np.array([census["probe_fold"][census["probe_index_of_position"][p]] for p in row_positions])
    inner = np.full((census["num_outer_folds"], len(row_positions)), -1, dtype=int)
    rng = np.random.default_rng(seed)
    for fold in range(census["num_outer_folds"]):
        train_rows = np.where(outer != fold)[0]
        sizes = {}
        for row in train_rows:
            sizes[formulas[row]] = sizes.get(formulas[row], 0) + 1
        keys = list(sizes)
        tie_break = rng.permutation(len(keys))
        order = sorted(range(len(keys)), key=lambda index: (-sizes[keys[index]], tie_break[index]))
        load = [0] * num_inner
        assignment = {}
        for index in order:
            lightest = min(range(num_inner), key=lambda f: (load[f], f))
            assignment[keys[index]] = lightest
            load[lightest] += sizes[keys[index]]
        for row in train_rows:
            inner[fold, row] = assignment[formulas[row]]
    return outer, inner


def bucket_centre(matrix, bucket_id, train_rows, apply_rows):
    """Subtract bucket means learned on train_rows from apply_rows; a bucket unseen in training gets the training grand mean; NaN-aware"""
    matrix = np.asarray(matrix, dtype=float)
    result = matrix[apply_rows].copy()
    with np.errstate(invalid="ignore"):
        grand = np.nanmean(matrix[train_rows], axis=0)
    grand = np.where(np.isnan(grand), 0.0, grand)
    for bucket in np.unique(bucket_id[apply_rows]):
        train_members = train_rows[bucket_id[train_rows] == bucket]
        apply_members = np.where(bucket_id[apply_rows] == bucket)[0]
        if len(train_members) == 0:
            result[apply_members] -= grand
            continue
        with np.errstate(invalid="ignore"):
            means = np.nanmean(matrix[train_members], axis=0)
        result[apply_members] -= np.where(np.isnan(means), grand, means)
    return result


# ---------------- probe core ----------------

def centred_kernel(kernel, train_rows, test_rows):
    """Training kernel and test-by-training kernel after centring in feature space by the training mean"""
    kernel_train = kernel[np.ix_(train_rows, train_rows)]
    kernel_test = kernel[np.ix_(test_rows, train_rows)]
    train_row_means = kernel_train.mean(axis=1)
    grand_mean = train_row_means.mean()
    centred_train = kernel_train - train_row_means[:, None] - train_row_means[None, :] + grand_mean
    centred_test = kernel_test - kernel_test.mean(axis=1)[:, None] - train_row_means[None, :] + grand_mean
    return centred_train, centred_test


def ridge_predictions(kernel_train, kernel_test, targets_train):
    """Test predictions for every alpha on the ridge grid, (alphas, tests, targets)"""
    eigenvalues, eigenvectors = np.linalg.eigh(kernel_train)
    kept = eigenvalues > max(RANK_TOLERANCE * eigenvalues.max(), EIGENVALUE_FLOOR)
    eigenvalues = eigenvalues[kept]
    eigenvectors = eigenvectors[:, kept]
    target_means = targets_train.mean(axis=0)
    rotated_targets = eigenvectors.T @ (targets_train - target_means)
    test_rotated = kernel_test @ eigenvectors
    predictions = np.empty((len(RIDGE_ALPHAS), kernel_test.shape[0], targets_train.shape[1]))
    for alpha_index, alpha in enumerate(RIDGE_ALPHAS):
        predictions[alpha_index] = (test_rotated / (eigenvalues + alpha)) @ rotated_targets + target_means
    return predictions


def r_squared(predictions, targets):
    """R2 per target column; NaN where the targets are constant"""
    total = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    residual = ((predictions - targets) ** 2).sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, 1 - residual / total, np.nan)


def best_alpha_index(scores):
    """Per target, the index of the largest score; ties go to the largest alpha"""
    reversed_scores = scores[::-1]
    return len(RIDGE_ALPHAS) - 1 - np.argmax(reversed_scores, axis=0)


def linear_kernel(features):
    return features @ features.T


def gaussian_kernel(features, bandwidth_factor, seed=0, sample_size=1000):
    """Gaussian kernel at a multiple of the median pairwise distance; returns the kernel and the bandwidth"""
    squared_norms = (features ** 2).sum(axis=1)
    squared_distances = np.maximum(squared_norms[:, None] + squared_norms[None, :] - 2 * features @ features.T, 0.0)
    sample = np.random.default_rng(seed).choice(len(features), size=min(sample_size, len(features)), replace=False)
    sample_distances = np.sqrt(squared_distances[np.ix_(sample, sample)])
    median = float(np.median(sample_distances[np.triu_indices(len(sample), k=1)]))
    bandwidth = bandwidth_factor * median if median > 0 else 1.0
    return np.exp(-squared_distances / (2 * bandwidth ** 2)), bandwidth


class Probe:
    """One probe design: rows, buckets, folds, targets, and the fold-local transformations.

    Nothing is centred or fitted on rows outside the training fold. Targets, features, and nuisance matrices are
    bucket-centred with training-bucket means when centring is "bucket", or centred by the training mean only when
    centring is "none" (the cross-formula design). Kernels are computed per fold on the centred features.
    """

    def __init__(self, census, row_positions, bucket_id, outer_fold, inner_fold, centring, target_names=None):
        self.census = census
        self.row_positions = np.asarray(row_positions)
        self.bucket_id = np.asarray(bucket_id)
        self.outer_fold = np.asarray(outer_fold)
        self.inner_fold = np.asarray(inner_fold)
        self.centring = centring
        self.num_outer = int(self.outer_fold.max()) + 1
        self.num_inner = int(self.inner_fold.max()) + 1
        probe_index = [census["probe_index_of_position"][p] for p in self.row_positions]
        self.target_names = list(census["target_names"]) if target_names is None else list(target_names)
        columns = [census["target_names"].index(name) for name in self.target_names]
        self.targets = census["target_matrix"][np.ix_(probe_index, columns)]

    def centre(self, matrix, train_rows, apply_rows):
        """The fold-local centring of any row matrix.

        A block may be a pair (train_realisation, test_realisation): rows in train_rows are taken from the first matrix
        and every other row from the second, so a sampled readout is fitted on one draw and scored on an independent one.
        """
        if isinstance(matrix, tuple):
            train_matrix, test_matrix = matrix
            combined = np.asarray(test_matrix, dtype=float).copy()
            combined[train_rows] = np.asarray(train_matrix, dtype=float)[train_rows]
            matrix = combined
        if self.centring == "bucket":
            return bucket_centre(matrix, self.bucket_id, train_rows, apply_rows)
        matrix = np.asarray(matrix, dtype=float)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(matrix[train_rows], axis=0)
        return matrix[apply_rows] - np.where(np.isnan(means), 0.0, means)

    def split(self, outer):
        train_rows = np.where(self.outer_fold != outer)[0]
        test_rows = np.where(self.outer_fold == outer)[0]
        return train_rows, test_rows

    def inner_split(self, outer, train_rows, inner):
        inner_train = train_rows[self.inner_fold[outer, train_rows] != inner]
        inner_test = train_rows[self.inner_fold[outer, train_rows] == inner]
        return inner_train, inner_test


def defined_rows_for(targets):
    """Groups of target columns that share the same set of rows without NaN, as (rows, columns) pairs"""
    groups = {}
    for column in range(targets.shape[1]):
        key = tuple(np.where(~np.isnan(targets[:, column]))[0].tolist())
        groups.setdefault(key, []).append(column)
    return [(np.array(rows), np.array(columns)) for rows, columns in groups.items() if len(rows) > 0]


def block_kernel(features_train, features_test, kernel_name, seed=0):
    """Kernel of one feature block on the fold: linear, or gaussian_<factor> at a multiple of the median training distance"""
    if kernel_name == "linear":
        return features_train @ features_train.T, features_test @ features_train.T
    factor = float(kernel_name.split("_")[1])
    train_norms = (features_train ** 2).sum(axis=1)
    test_norms = (features_test ** 2).sum(axis=1)
    train_sq = np.maximum(train_norms[:, None] + train_norms[None, :] - 2 * features_train @ features_train.T, 0.0)
    test_sq = np.maximum(test_norms[:, None] + train_norms[None, :] - 2 * features_test @ features_train.T, 0.0)
    sample = np.random.default_rng(seed).choice(len(features_train), size=min(1000, len(features_train)), replace=False)
    median = float(np.median(np.sqrt(train_sq[np.ix_(sample, sample)])[np.triu_indices(len(sample), k=1)]))
    bandwidth = factor * median if median > 0 else 1.0
    return np.exp(-train_sq / (2 * bandwidth ** 2)), np.exp(-test_sq / (2 * bandwidth ** 2))


def scale_block(features_train, features_test):
    """Scale a block so its training rows have unit mean squared norm; blocks then enter a sum kernel on equal footing"""
    scale = np.sqrt((features_train ** 2).sum(axis=1).mean())
    if scale <= 0:
        return features_train, features_test
    return features_train / scale, features_test / scale


def fit_fold(probe, outer, blocks, kernel_name, train_rows, test_rows, targets_train, targets_test):
    """Predictions on test_rows for the alpha chosen by inner CV, using the sum of block kernels, all transformations fold-local"""
    inner_scores = np.full((probe.num_inner, len(RIDGE_ALPHAS), targets_train.shape[1]), np.nan)
    for inner in range(probe.num_inner):
        inner_train, inner_test = probe.inner_split(outer, train_rows, inner)
        if len(inner_test) == 0 or len(inner_train) < 2:
            continue
        kernel_train = np.zeros((len(inner_train), len(inner_train)))
        kernel_test = np.zeros((len(inner_test), len(inner_train)))
        for block in blocks:
            block_train = probe.centre(block, inner_train, inner_train)
            block_test = probe.centre(block, inner_train, inner_test)
            block_train, block_test = scale_block(block_train, block_test)
            k_train, k_test = block_kernel(block_train, block_test, kernel_name)
            kernel_train += k_train
            kernel_test += k_test
        # the targets arrive centred and residualised by the outer training fold; inside the inner loop only the ridge
        # intercept is refit, so alpha selection sees the same target the outer fit will see
        position_in_train = {row: position for position, row in enumerate(train_rows)}
        selection = np.array([position_in_train[r] for r in inner_train])
        selection_test = np.array([position_in_train[r] for r in inner_test])
        predictions = ridge_predictions(kernel_train, kernel_test, targets_train[selection])
        inner_scores[inner] = r_squared(predictions, targets_train[selection_test])
    informative = ~np.isnan(inner_scores)
    fold_count = informative.sum(axis=0)
    mean_scores = np.where(fold_count > 0, np.where(informative, inner_scores, 0.0).sum(axis=0) / np.maximum(fold_count, 1), -np.inf)
    chosen = best_alpha_index(mean_scores)
    kernel_train = np.zeros((len(train_rows), len(train_rows)))
    kernel_test = np.zeros((len(test_rows), len(train_rows)))
    for block in blocks:
        block_train = probe.centre(block, train_rows, train_rows)
        block_test = probe.centre(block, train_rows, test_rows)
        block_train, block_test = scale_block(block_train, block_test)
        k_train, k_test = block_kernel(block_train, block_test, kernel_name)
        kernel_train += k_train
        kernel_test += k_test
    predictions = ridge_predictions(kernel_train, kernel_test, targets_train)
    return predictions[chosen, :, np.arange(targets_train.shape[1])].T


def out_of_fold(probe, blocks, kernel_name, nuisance=None, histogram=None):
    """Out-of-fold predictions of every target from the sum kernel over the blocks.

    nuisance: optional list of feature matrices; the targets are residualised on them with a fold-local linear model first.
    histogram: optional integer matrix; after the linear step the residuals are group-centred by (bucket, histogram) with
    leave-one-out means on training rows, the degree-histogram category.
    Returns out-of-fold predictions, the out-of-fold targets they were scored against (centred and residualised as the
    fold saw them), and a boolean per target marking columns the nuisance model reproduced exactly.
    """
    predictions = np.full(probe.targets.shape, np.nan)
    oof_targets = np.full(probe.targets.shape, np.nan)
    exact = np.zeros(probe.targets.shape[1], dtype=bool)
    for defined, columns in defined_rows_for(probe.targets):
        for outer in range(probe.num_outer):
            train_rows, test_rows = probe.split(outer)
            train_rows = train_rows[np.isin(train_rows, defined)]
            test_rows = test_rows[np.isin(test_rows, defined)]
            if len(test_rows) == 0 or len(train_rows) < 2:
                continue
            targets_train = probe.centre(probe.targets[:, columns], train_rows, train_rows)
            targets_test = probe.centre(probe.targets[:, columns], train_rows, test_rows)
            raw_variance = targets_train.var(axis=0)
            if nuisance is not None:
                design_train = [np.ones((len(train_rows), 1))]
                design_test = [np.ones((len(test_rows), 1))]
                for matrix in nuisance:
                    design_train.append(probe.centre(matrix, train_rows, train_rows))
                    design_test.append(probe.centre(matrix, train_rows, test_rows))
                design_train = np.hstack(design_train)
                design_test = np.hstack(design_test)
                coefficients = np.linalg.lstsq(design_train, targets_train, rcond=None)[0]
                targets_train = targets_train - design_train @ coefficients
                targets_test = targets_test - design_test @ coefficients
            if histogram is not None:
                targets_train, targets_test = category_residuals(probe.bucket_id, histogram, train_rows, test_rows, targets_train, targets_test)
            flagged = (raw_variance <= 0) | (targets_train.var(axis=0) <= RESIDUAL_VARIANCE_FLOOR * np.maximum(raw_variance, 1e-300))
            exact[columns[flagged]] = True
            fold_predictions = fit_fold(probe, outer, blocks, kernel_name, train_rows, test_rows, targets_train, targets_test)
            predictions[np.ix_(test_rows, columns)] = fold_predictions
            oof_targets[np.ix_(test_rows, columns)] = targets_test
    return predictions, oof_targets, exact


def category_residuals(bucket_id, histogram, train_rows, test_rows, targets_train, targets_test):
    """Group-centre by (bucket, histogram row) with leave-one-out means on training rows; unseen test groups get the training grand mean"""
    keys_all = np.unique(np.hstack([bucket_id[:, None], histogram]), axis=0, return_inverse=True)[1].reshape(-1)
    train_keys = keys_all[train_rows]
    test_keys = keys_all[test_rows]
    keys, inverse = np.unique(train_keys, return_inverse=True)
    group_sums = np.zeros((len(keys), targets_train.shape[1]))
    np.add.at(group_sums, inverse, targets_train)
    group_counts = np.bincount(inverse, minlength=len(keys))
    grand_mean = targets_train.mean(axis=0)
    leave_one_out = (group_sums[inverse] - targets_train) / np.maximum(group_counts[inverse] - 1, 1)[:, None]
    leave_one_out[group_counts[inverse] == 1] = grand_mean
    position = np.searchsorted(keys, test_keys)
    clipped = np.minimum(position, len(keys) - 1)
    seen = (position < len(keys)) & (keys[clipped] == test_keys)
    test_means = np.where(seen[:, None], group_sums[clipped] / np.maximum(group_counts[clipped], 1)[:, None], grand_mean)
    return targets_train - leave_one_out, targets_test - test_means


def pooled_r_squared(predictions, oof_targets, exact):
    """Pooled out-of-fold R2 per target; NaN where the nuisance model was exact"""
    scores = np.full(predictions.shape[1], np.nan)
    for column in range(predictions.shape[1]):
        rows = ~np.isnan(oof_targets[:, column])
        if rows.sum() < 2 or exact[column]:
            continue
        scores[column] = r_squared(predictions[rows, column][:, None], oof_targets[rows, column][:, None])[0]
    return scores


def paired_gain(classical_pred, both_pred, oof_targets, exact, bootstrap_draws=500, seed=0):
    """Per target: R2 of each model, the mean paired reduction in squared error as a fraction of the target variance
    (positive means the added block helped), and a bootstrap 95 percent interval of that fraction over rows"""
    rng = np.random.default_rng(seed)
    num_targets = oof_targets.shape[1]
    result = {"classical_r2": pooled_r_squared(classical_pred, oof_targets, exact), "both_r2": pooled_r_squared(both_pred, oof_targets, exact),
              "gain": np.full(num_targets, np.nan), "gain_low": np.full(num_targets, np.nan), "gain_high": np.full(num_targets, np.nan)}
    for column in range(num_targets):
        rows = ~np.isnan(oof_targets[:, column])
        if rows.sum() < 2 or exact[column]:
            continue
        target = oof_targets[rows, column]
        variance = target.var()
        if variance <= 0:
            continue
        paired = ((classical_pred[rows, column] - target) ** 2 - (both_pred[rows, column] - target) ** 2) / variance
        result["gain"][column] = paired.mean()
        draws = rng.choice(len(paired), size=(bootstrap_draws, len(paired)), replace=True)
        means = paired[draws].mean(axis=1)
        result["gain_low"][column], result["gain_high"][column] = np.percentile(means, [2.5, 97.5])
    return result


def paired_comparison(probe, classical_blocks, quantum_blocks, kernel_name, nuisance=None, histogram=None):
    """Classical-only against classical-plus-quantum, same kernel and grid, paired out-of-fold squared errors"""
    classical_pred, oof_targets, exact = out_of_fold(probe, classical_blocks, kernel_name, nuisance, histogram)
    both_pred, oof_targets_both, _ = out_of_fold(probe, classical_blocks + quantum_blocks, kernel_name, nuisance, histogram)
    assert np.allclose(np.nan_to_num(oof_targets), np.nan_to_num(oof_targets_both))
    return paired_gain(classical_pred, both_pred, oof_targets, exact)


# ---------------- retrieval ----------------

def standardise_on_rows(matrix):
    """Standardise columns on these rows, dropping constant columns; asserts nothing is NaN afterwards"""
    matrix = np.asarray(matrix, dtype=float)
    keep = matrix.std(axis=0) > 0
    result = (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / matrix[:, keep].std(axis=0)
    assert not np.isnan(result).any(), "standardisation produced NaN"
    return result


def nearest_within_bucket(matrix, bucket_id, valid):
    """For each valid row, the set of nearest other valid rows in its bucket by Euclidean distance, ties included"""
    nearest = [[] for _ in range(len(matrix))]
    for bucket in np.unique(bucket_id):
        members = np.where((bucket_id == bucket) & valid)[0]
        if len(members) < 2:
            continue
        block = matrix[members]
        squared = (block ** 2).sum(axis=1)
        distances = squared[:, None] + squared[None, :] - 2 * block @ block.T
        np.fill_diagonal(distances, np.inf)
        for local, index in enumerate(members):
            best = distances[local].min()
            nearest[index] = [int(members[j]) for j in np.where(distances[local] <= best + 1e-12)[0]]
    return nearest


def nearest_by_similarity(similarity, bucket_id, valid):
    """For each valid row, the set of most similar other valid rows in its bucket under a pairwise similarity function, ties included"""
    nearest = [[] for _ in range(len(bucket_id))]
    for bucket in np.unique(bucket_id):
        members = np.where((bucket_id == bucket) & valid)[0]
        if len(members) < 2:
            continue
        for index in members:
            values = [(similarity(index, other), other) for other in members if other != index]
            best = max(value for value, _ in values)
            nearest[index] = [int(other) for value, other in values if value >= best - 1e-12]
    return nearest


def retrieval_scores(nearest, values, bucket_id, valid, continuous):
    """Tie-averaged agreement (or mean absolute difference when continuous) with the nearest neighbours, micro and macro over buckets"""
    per_row = np.full(len(values), np.nan)
    for index, neighbours in enumerate(nearest):
        if not valid[index] or not neighbours or np.isnan(values[index]):
            continue
        neighbour_values = np.array([values[j] for j in neighbours if not np.isnan(values[j])])
        if len(neighbour_values) == 0:
            continue
        per_row[index] = np.abs(neighbour_values - values[index]).mean() if continuous else (neighbour_values == values[index]).mean()
    scored = ~np.isnan(per_row)
    micro = per_row[scored].mean() if scored.any() else np.nan
    bucket_means = [per_row[(bucket_id == bucket) & scored].mean() for bucket in np.unique(bucket_id) if ((bucket_id == bucket) & scored).any()]
    macro = float(np.mean(bucket_means)) if bucket_means else np.nan
    return micro, macro


def count_tanimoto(first, second):
    """Tanimoto similarity of two sparse count dictionaries: sum of minima over sum of maxima"""
    keys = set(first) | set(second)
    minima = sum(min(first.get(k, 0), second.get(k, 0)) for k in keys)
    maxima = sum(max(first.get(k, 0), second.get(k, 0)) for k in keys)
    return minima / maxima if maxima else 1.0


# ---------------- producer storage: incremental, sparse, one file per realisation ----------------

def save_realisation(path, dense_blocks, sparse_blocks):
    """Write one realisation of an arm: dense blocks as arrays, dict blocks as (rows, cols, vals) triplets over the arm's vocabulary"""
    payload = {}
    for block, matrix in dense_blocks.items():
        payload[f"{block}__dense"] = np.asarray(matrix, dtype=np.float32)
    for block, (rows, cols, vals) in sparse_blocks.items():
        payload[f"{block}__rows"] = np.asarray(rows, dtype=np.int32)
        payload[f"{block}__cols"] = np.asarray(cols, dtype=np.int32)
        payload[f"{block}__vals"] = np.asarray(vals, dtype=np.float32)
    np.savez_compressed(path, **payload)


class ArmReadouts:
    """Reader for one arm of the producer: the arm index holds settings, rows, and the block vocabularies; each realisation is one npz.

    block(realisation, name) returns a dense matrix over the arm's full vocabulary. For the wide occupancy-keyed blocks pass
    min_support and max_columns to keep only keys nonzero on at least min_support molecules in the exact realisation, at
    most max_columns of them by support; the choice is the consumer's and the producer keeps everything.
    """

    def __init__(self, producer_dir, arm_name):
        self.producer_dir = producer_dir
        self.arm_name = arm_name
        self.index = load_pickle(f"{producer_dir}/iclr27_x9pcorrected_{arm_name}_index.pkl")
        self.lift = self.index["lift"]
        self.setting = self.index["setting"]
        self.row_positions = np.asarray(self.index["row_positions"])
        self.vocabularies = self.index["vocabularies"]
        self.support = self.index["support"]
        self.widths = self.index["widths"]
        self.realisations = self.index["realisations"]

    def columns(self, block, min_support=None, max_columns=None):
        """Indices of the kept columns of a dict block under the support rule; every column when no rule is given"""
        support = np.asarray(self.support[block])
        kept = np.arange(len(support)) if min_support is None else np.where(support >= min_support)[0]
        if max_columns is not None and len(kept) > max_columns:
            kept = kept[np.argsort(-support[kept], kind="stable")[:max_columns]]
            kept = np.sort(kept)
        return kept

    def block(self, realisation, block, min_support=None, max_columns=None):
        assert realisation in self.realisations, f"{self.arm_name}: no realisation {realisation}"
        with np.load(f"{self.producer_dir}/iclr27_x9pcorrected_{self.arm_name}_{realisation}.npz") as stored:
            if f"{block}__dense" in stored:
                return stored[f"{block}__dense"].astype(float)
            assert f"{block}__rows" in stored, f"{self.arm_name}: block {block} not stored for {realisation}"
            rows, cols, vals = stored[f"{block}__rows"], stored[f"{block}__cols"], stored[f"{block}__vals"]
        kept = self.columns(block, min_support, max_columns)
        position_of = np.full(len(self.vocabularies[block]), -1, dtype=np.int64)
        position_of[kept] = np.arange(len(kept))
        matrix = np.zeros((len(self.row_positions), len(kept)))
        inside = position_of[cols] >= 0
        matrix[rows[inside], position_of[cols[inside]]] = vals[inside]
        return matrix

    def has_block(self, realisation, block):
        with np.load(f"{self.producer_dir}/iclr27_x9pcorrected_{self.arm_name}_{realisation}.npz") as stored:
            return f"{block}__dense" in stored or f"{block}__rows" in stored


# ---------------- provenance ----------------

def save_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def write_scores_csv(path, records, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def elapsed(start):
    return f"{time.time() - start:.0f}s"


# ---------------- shot-scaling predictions ----------------

def sorted_gap_shots(sorted_matrix, head, target_ratio=2.0):
    """Shots at which the median adjacent gap in the sorted head is target_ratio standard errors: N = ratio^2 p / gap^2, per row median"""
    head_matrix = np.asarray(sorted_matrix, dtype=float)[:, :head]
    gaps = np.diff(head_matrix, axis=1) * -1.0
    levels = head_matrix[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        shots = target_ratio ** 2 * levels / gaps ** 2
    shots = np.where(np.isfinite(shots) & (gaps > 0), shots, np.nan)
    return float(np.nanmedian(shots)), float(np.nanmedian(gaps))


def sector_mass_shots(sector_matrix, relative_error=0.05):
    """Shots at which the typical sector mass m has relative standard error relative_error: N = (1 - m) / (m e^2).

    Typical means the mass-weighted median: half of all probability mass sits in sectors at least this heavy, so the
    prediction concerns the sectors that carry the distribution rather than the many negligible ones.
    """
    masses = np.sort(np.asarray(sector_matrix, dtype=float).ravel())
    masses = masses[masses > 0]
    cumulative = np.cumsum(masses) / masses.sum()
    typical = float(masses[np.searchsorted(cumulative, 0.5)])
    return (1 - typical) / (typical * relative_error ** 2), typical
