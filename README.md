# ICLR27

## Notebook Summaries


### ICLR27-1-qm9-dataset Construction

Constructs a QM9 analysis panel for 130,831 molecules, combining chemical descriptors, graph invariants, geometry, equivalence metadata, and 19 molecular property targets. The panel contains 136 scalar fields and 25 structured fields. The recorded classification identifies constitutional-isomer alternatives for 130,737 molecules, duplicate matches for 311 molecules, one enantiomer pair, and no diastereomer pairs. The panel is exported to qm9_panel_columns.pkl.

## ICLR27-1-qm9-dataset Construction

Molecular construction and validation. Loads QM9 through PyTorch Geometric and constructs aligned RDKit molecules and NetworkX graphs, including both explicit-hydrogen and heavy-atom representations. The dataset contains molecules with 3–29 total atoms and 1–9 heavy atoms, using H, C, N, O, and F. Element-preserving graph isomorphism aligns the SMILES-derived atom ordering with the dataset coordinates. Assertions check atom counts, element ordering, bond connectivity, graph connectedness, and consistency between the supplied HOMO–LUMO gap and orbital energies. The run retains 1,819 records requiring partial RDKit sanitization and marks them with valence_error.

Descriptor construction. Builds a panel covering composition and bond types; typed radius-1 and radius-2 environments; 17 SMARTS-defined functional-group counts; ring identities, ring-system relationships, and aromatic substitution patterns; and conjugated-component structure. Graph descriptors include degree histograms, bridges, articulation points, cyclomatic number, girth, shortest-path statistics, spectral quantities, and simple-cycle counts of lengths three through six. Geometry descriptors include radii of gyration, pairwise-distance and bond-length spectra, heavy-atom angles, torsions, and ring planarity. The panel also retains all 19 QM9 target columns and structured records underlying the scalar summaries.

Equivalence classification. Groups molecules first by molecular formula and then by canonical SMILES with stereochemistry removed. Within each connectivity class, canonical stereochemical SMILES and mirrored tetrahedral assignments distinguish duplicates, enantiomers, diastereomers, and unresolved stereochemical comparisons. The assigned identifiers cover 616 formula classes and 130,672 connectivity classes. Summing the reported constitutional-isomer groups gives 522 formula groups containing multiple connectivity classes, encompassing 130,737 molecules; the largest contains 6,094 molecules. The per-molecule report identifies 311 molecules with duplicate matches, two molecules forming one enantiomer pair, and no diastereomer matches. Separately, 8,250 molecules have at least one unassigned stereocenter. These classifications use the supplied SMILES rather than coordinate-based stereochemical reassignment.

Recorded feature coverage. Reports nonzero counts, minima, maxima, and means for every scalar field. The saved results include 117,441 molecules with rings, 20,918 with aromatic rings, 48,992 with conjugated bonds, and 10,831 with spiro centers. RDKit ring descriptors and graph-theoretic simple-cycle counts are recorded separately.

Export and execution. Saves the panel as NumPy arrays for scalar fields and lists for structured fields, together with configuration information, target names, field definitions, and functional-group SMARTS. Reconstructible RDKit objects, graphs, and matrices are released after each molecule’s descriptors are built. The saved run reports 1,356 seconds for molecular construction and 60 seconds for isomer classification, followed by successful export to qm9_panel_columns.pkl. This notebook constructs and characterizes the dataset; it does not train models or evaluate quantum representations.
