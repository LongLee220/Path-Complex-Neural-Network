from rdkit import Chem
from rdkit.Chem import AllChem

# 创建一个分子对象
smiles = "[H]C1([H])C([H])([H])C2([H])C([H])([H])C([H])([H])C12[H]"
mol = Chem.MolFromSmiles(smiles)
print("Initial molecule:", Chem.MolToSmiles(mol))

mol = Chem.AddHs(mol)  # 加氢
print("Molecule after adding Hs:", Chem.MolToSmiles(mol))

# 获取原子数量
num_atoms = mol.GetNumAtoms()
print(f"分子中的原子数量: {num_atoms}")

# 尝试生成构象
# 尝试生成构象
AllChem.EmbedMolecule(mol, randomSeed=43, maxAttempts=100)

# 检查是否有构象
num_conformers = mol.GetNumConformers()
print(num_conformers)
if num_conformers > 0:
    # 尝试优化分子结构
    if AllChem.UFFOptimizeMolecule(mol, confId=0) != 0:
        print("Error: Unable to optimize the molecule.")
    else:
        print("Molecule optimized using UFF.")

        # 获取构象
        conformer = mol.GetConformer(0)
        # 获取构象的原子坐标
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(atom_idx)
            print(f"Atom {atom_idx + 1}: {pos.x}, {pos.y}, {pos.z}")
else:
    print("No conformers found. Please check if the molecule is embeddable.")
