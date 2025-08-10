import torch
from typing import Dict, List, Tuple, Optional
from torch import nn

from model.path_complex import PCNN
from utils.mol_to_path import path_complex_mol
from rdkit import Chem


def _select_scalar_output(out, task='binary', task_target_idx=None):
    """
    将模型输出变成标量 y：
    - binary:  out 是 [B] 或 [B,1]，取均值或特定样本
    - multi:   out 是 [B,C]，取指定类 logit/prob 的均值
    - regression: 直接 mean
    """
    if out.ndim == 0:
        return out
    if task == 'regression':
        return out.mean()
    if task == 'binary':
        # 你的 Readout 是 Sigmoid 后的概率；解释时建议用 logit 更线性，避免饱和
        eps = 1e-6
        prob = out.view(-1)
        logit = torch.log(prob.clamp(eps,1-eps) / (1 - prob.clamp(eps,1-eps)))
        return logit.mean()
    if task == 'multi':
        assert task_target_idx is not None, "multi-class 需指定 task_target_idx"
        return out[..., task_target_idx].mean()
    raise ValueError("unknown task type")

def forward_with_edge_gates(model: nn.Module,
                            order: int,
                            gates: torch.Tensor,
                            g_graph, lg_graph, fg_graph,
                            g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                            device='cpu',
                            resent=True,
                            pooling='avg',
                            task='binary',
                            task_target_idx=None):
    """
    在指定阶 (1/2/3) 的图的边特征上乘 gates（形状 [E]），其他不变。
    """
    if order == 1:
        e_mod = g_edge * gates.unsqueeze(-1)
        out = model(
            g_feats=None, g_graph=g_graph, lg_graph=lg_graph, fg_graph=fg_graph,
            g_graph_node_feat=g_node, g_graph_edge_feat=e_mod,
            lg_graph_node_feat=lg_node, lg_graph_edge_feat=lg_edge,
            fg_graph_node_feat=fg_node, fg_graph_edge_feat=fg_edge,
            device=device, resent=resent, pooling=pooling
        )
    elif order == 2:
        e_mod = lg_edge * gates.unsqueeze(-1)
        out = model(
            g_feats=None, g_graph=g_graph, lg_graph=lg_graph, fg_graph=fg_graph,
            g_graph_node_feat=g_node, g_graph_edge_feat=g_edge,
            lg_graph_node_feat=lg_node, lg_graph_edge_feat=e_mod,
            fg_graph_node_feat=fg_node, fg_graph_edge_feat=fg_edge,
            device=device, resent=resent, pooling=pooling
        )
    elif order == 3:
        e_mod = fg_edge * gates.unsqueeze(-1)
        out = model(
            g_feats=None, g_graph=g_graph, lg_graph=lg_graph, fg_graph=fg_graph,
            g_graph_node_feat=g_node, g_graph_edge_feat=g_edge,
            lg_graph_node_feat=lg_node, lg_graph_edge_feat=lg_edge,
            fg_graph_node_feat=fg_node, fg_graph_edge_feat=e_mod,
            device=device, resent=resent, pooling=pooling
        )
    else:
        raise ValueError("order must be 1, 2, or 3")
    return _select_scalar_output(out, task=task, task_target_idx=task_target_idx)

@torch.no_grad()
def _prep_eval(model):
    was_training = model.training
    model.eval()
    return was_training

def path_importance_grad(model,
                         order,
                         g_graph, lg_graph, fg_graph,
                         g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                         device='cpu', resent=True, pooling='avg',
                         task='binary', task_target_idx=None,
                         use_grad_times_input=False):
    """
    Grad×Mask: 重要性 ≈ |∂y/∂m_i| 或 |m_i * ∂y/∂m_i|（前者更稳）
    """
    was_training = _prep_eval(model)
    E = {1: g_graph.num_edges(), 2: lg_graph.num_edges(), 3: fg_graph.num_edges()}[order]
    gates = torch.ones(E, device=device, requires_grad=True)

    y = forward_with_edge_gates(model, order, gates,
                                g_graph, lg_graph, fg_graph,
                                g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                device=device, resent=resent, pooling=pooling,
                                task=task, task_target_idx=task_target_idx)
    model.zero_grad(set_to_none=True)
    if gates.grad is not None:
        gates.grad.zero_()
    y.backward()
    imp = gates.grad.detach().abs()
    if use_grad_times_input:
        imp = (imp * gates.detach()).abs()  # gates=1 等价
    if was_training: model.train()
    return imp  # shape [E]


def path_importance_ig(model,
                       order,
                       g_graph, lg_graph, fg_graph,
                       g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                       device='cpu', resent=True, pooling='avg',
                       task='binary', task_target_idx=None,
                       steps=32):
    was_training = _prep_eval(model)

    E = {1: g_graph.num_edges(), 2: lg_graph.num_edges(), 3: fg_graph.num_edges()}[order]
    ones = torch.ones(E, device=device)
    zeros = torch.zeros(E, device=device)

    total = torch.zeros(E, device=device)

    # 关键点：每一步用一个全新的 leaf（gates_step），只对它求梯度
    for s in range(1, steps + 1):
        alpha = s / steps
        gates_step = (zeros + alpha * (ones - zeros)).detach().requires_grad_(True)

        y = forward_with_edge_gates(model, order, gates_step,
                                    g_graph, lg_graph, fg_graph,
                                    g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                    device=device, resent=resent, pooling=pooling,
                                    task=task, task_target_idx=task_target_idx)

        # y 可能是标量，也可能是形如 [B]；确保是标量
        y_scalar = y if y.ndim == 0 else y.mean()

        grad, = torch.autograd.grad(y_scalar, gates_step, retain_graph=False, create_graph=False)
        total += grad.detach()

    ig = (ones - zeros) * total / steps
    if was_training:
        model.train()
    return ig.abs()

@torch.no_grad()
def path_importance_occlusion(model,
                              order,
                              g_graph, lg_graph, fg_graph,
                              g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                              device='cpu', resent=True, pooling='avg',
                              task='binary', task_target_idx=None,
                              batch_size=512):
    """
    留一遮蔽：把每条边的门置0，计算 Δy = y_full - y_masked。可分批评估。
    """
    was_training = _prep_eval(model)
    # full y
    E = {1: g_graph.num_edges(), 2: lg_graph.num_edges(), 3: fg_graph.num_edges()}[order]
    ones = torch.ones(E, device=device)
    y_full = forward_with_edge_gates(model, order, ones,
                                     g_graph, lg_graph, fg_graph,
                                     g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                     device=device, resent=resent, pooling=pooling,
                                     task=task, task_target_idx=task_target_idx)
    imps = torch.zeros(E, device=device)
    idx = torch.arange(E, device=device)
    # 分批把一些边置0（可矢量化：一次屏蔽一批）
    for start in range(0, E, batch_size):
        end = min(start+batch_size, E)
        m = ones.unsqueeze(0).repeat(end-start, 1)  # [B, E]
        m.scatter_(1, (idx[start:end]).unsqueeze(1), 0.0)  # 置0
        ys = []
        for b in range(m.size(0)):
            ys.append(
                forward_with_edge_gates(model, order, m[b],
                                        g_graph, lg_graph, fg_graph,
                                        g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                        device=device, resent=resent, pooling=pooling,
                                        task=task, task_target_idx=task_target_idx).item()
            )
        ys = torch.tensor(ys, device=device)
        imps[start:end] = (y_full - ys)
    if was_training: model.train()
    return imps.abs()


# ========= 归一化小工具 =========
def _normalize_scores(scores: torch.Tensor, how: Optional[str] = "minmax") -> torch.Tensor:
    if how is None:
        return scores
    if how == "minmax":
        smin, smax = scores.min(), scores.max()
        if (smax - smin) > 0:
            return (scores - smin) / (smax - smin)
        return torch.zeros_like(scores)
    if how == "zscore":
        mu, std = scores.mean(), scores.std(unbiased=False)
        return (scores - mu) / (std + 1e-8)
    if how == "softmax":
        return torch.softmax(scores, dim=0)
    raise ValueError("normalize must be one of {None,'minmax','zscore','softmax'}")

# ========= 路径元数据解析 =========
def _edge_atoms_from_bond_nodes(bond_a: Tuple[int,int], bond_b: Tuple[int,int]) -> Optional[Tuple[int,int,int]]:
    # bond_a=(i,j), bond_b=(j,k) or (i,k). 找到交点作为中间原子
    A, B = set(bond_a), set(bond_b)
    inter = list(A & B)
    if len(inter) != 1:
        return None
    center = inter[0]
    ends = list((A | B) - {center})
    if len(ends) != 2:
        return None
    # 角的顺序 i-center-k
    return (ends[0], center, ends[1])

def _dihedral_from_angles(ang_a: Tuple[int,int,int], ang_b: Tuple[int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    # 角 (i,j,k) 与 (j,k,l) 共享 j,k 两个原子 -> 二面角 (i,j,k,l)
    a, b = ang_a, ang_b
    inter = set(a) & set(b)
    if len(inter) != 2:
        return None
    # 找顺序：a = (i,j,k)，b = (j,k,l) 或其旋转
    # 以 a[1], a[2] 作为中心对
    j, k = a[1], a[2]
    if j in b and k in b:
        l_candidates = [x for x in b if x not in {j,k}]
        if len(l_candidates) == 1:
            i = a[0]
            l = l_candidates[0]
            return (i, j, k, l)
    return None

def _resolve_path_metadata(order: int,
                           g_graph, lg_graph, fg_graph,
                           edge_idx: int) -> Dict:
    """
    返回一个 dict，尽量给出可读的路径ID。
    需要你的构图/编码阶段在 (n)data 或 (e)data 里存过字段：
      - 1-path: g_graph.edata 可能有 'bond_atoms' 或回退到 (src,dst)
      - 2-path: lg_graph.ndata 可能有 'bond_atoms'（每个节点是一条 bond），
                lg_graph.edata 可能有 'angle_atoms'；否则尝试由相邻两条 bond 推断角
      - 3-path: fg_graph.ndata 可能有 'angle_atoms'（每个节点一个三元组），
                fg_graph.edata 可能有 'dihedral_atoms'；否则尝试由相邻两个角推断
    """
    meta = {"order": order, "edge_idx": int(edge_idx)}

    if order == 1:
        g = g_graph
        u, v = g.edges()
        u_i, v_i = int(u[edge_idx]), int(v[edge_idx])
        meta["edge_uv"] = (u_i, v_i)
        # 优先用显式字段
        if "bond_atoms" in g.edata:
            try:
                ba = g.edata["bond_atoms"][edge_idx]
                meta["bond_atoms"] = tuple(map(int, ba.tolist()))
            except Exception:
                meta["bond_atoms"] = (u_i, v_i)
        else:
            meta["bond_atoms"] = (u_i, v_i)
        return meta

    if order == 2:
        lg = lg_graph
        eu, ev = lg.edges()
        a, b = int(eu[edge_idx]), int(ev[edge_idx])  # 两个 1-path 节点索引
        meta["edge_uv"] = (a, b)
        # 若 edata 已存角
        if "angle_atoms" in lg.edata:
            try:
                ang = lg.edata["angle_atoms"][edge_idx]
                meta["angle_atoms"] = tuple(map(int, ang.tolist()))
                return meta
            except Exception:
                pass
        # 否则尝试由两条 bond 节点推断
        bondA = None
        bondB = None
        if "bond_atoms" in lg.ndata:  # 每个节点携带这条 1-path 的两个原子
            try:
                ba = tuple(map(int, lg.ndata["bond_atoms"][a].tolist()))
                bb = tuple(map(int, lg.ndata["bond_atoms"][b].tolist()))
                bondA, bondB = ba, bb
            except Exception:
                bondA = bondB = None
        if bondA is not None and bondB is not None:
            ang = _edge_atoms_from_bond_nodes(bondA, bondB)
            if ang is not None:
                meta["angle_atoms"] = ang
        return meta

    if order == 3:
        fg = fg_graph
        eu, ev = fg.edges()
        a, b = int(eu[edge_idx]), int(ev[edge_idx])  # 两个 2-path 节点索引（角）
        meta["edge_uv"] = (a, b)
        # 若 edata 已存二面角
        if "dihedral_atoms" in fg.edata:
            try:
                dih = fg.edata["dihedral_atoms"][edge_idx]
                meta["dihedral_atoms"] = tuple(map(int, dih.tolist()))
                return meta
            except Exception:
                pass
        # 否则尝试由两个角节点推断
        if "angle_atoms" in fg.ndata:
            try:
                angA = tuple(map(int, fg.ndata["angle_atoms"][a].tolist()))
                angB = tuple(map(int, fg.ndata["angle_atoms"][b].tolist()))
                dih = _dihedral_from_angles(angA, angB)
                if dih is not None:
                    meta["dihedral_atoms"] = dih
            except Exception:
                pass
        return meta

    raise ValueError("order must be 1, 2, or 3")

# ========= 统一入口 =========
def get_path_importance(model: torch.nn.Module,
                        order: int,
                        g_graph, lg_graph, fg_graph,
                        g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                        method: str = "ig",           # 'grad' | 'ig' | 'occlusion'
                        ig_steps: int = 32,
                        task: str = "binary",         # 'binary' | 'multi' | 'regression'
                        task_target_idx: Optional[int] = None,
                        normalize: Optional[str] = "minmax",  # None | 'minmax' | 'zscore' | 'softmax'
                        topk: Optional[int] = None,
                        device: str = "cpu",
                        resent: bool = True,
                        pooling: str = "avg") -> Dict:
    """
    返回：
      {
        'scores_raw': Tensor[E],
        'scores': Tensor[E],            # 归一化后
        'order': int,
        'topk_idx': LongTensor[k] or None,
        'topk_scores': Tensor[k] or None,
        'topk_meta': List[dict] or None
      }
    """
    assert method in {"grad", "ig", "occlusion"}
    if method == "grad":
        scores = path_importance_grad(model, order,
                                      g_graph, lg_graph, fg_graph,
                                      g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                      device=device, resent=resent, pooling=pooling,
                                      task=task, task_target_idx=task_target_idx)
    elif method == "ig":
        scores = path_importance_ig(model, order,
                                    g_graph, lg_graph, fg_graph,
                                    g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                    device=device, resent=resent, pooling=pooling,
                                    task=task, task_target_idx=task_target_idx,
                                    steps=ig_steps)
    else:
        scores = path_importance_occlusion(model, order,
                                           g_graph, lg_graph, fg_graph,
                                           g_node, g_edge, lg_node, lg_edge, fg_node, fg_edge,
                                           device=device, resent=resent, pooling=pooling,
                                           task=task, task_target_idx=task_target_idx)

    scores_raw = scores.detach()
    scores_norm = _normalize_scores(scores_raw, how=normalize)

    result = {
        "scores_raw": scores_raw,
        "scores": scores_norm,
        "order": order,
        "topk_idx": None,
        "topk_scores": None,
        "topk_meta": None
    }

    if topk is not None:
        k = min(topk, scores_norm.numel())
        vals, idx = torch.topk(scores_norm, k=k)
        result["topk_idx"] = idx
        result["topk_scores"] = vals
        # 生成可读元数据
        metas = []
        for i in idx.tolist():
            metas.append(_resolve_path_metadata(order, g_graph, lg_graph, fg_graph, i))
        result["topk_meta"] = metas

    return result



if __name__ == "__main__":
    

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    
    encode_dim = [92,21,9,7]
    out_dim = 1
    num_heads = 1
    num_layers = 2
    num_blcok= 1

    resent= False
    tras_med= 'hl'
    model = PCNN(in_feats=10, hidden_size = 32, out_feats=64, encode_dim=encode_dim, out_dim = out_dim,tras_med = tras_med,num_blcok=num_blcok,num_heads=num_heads,num_layers=num_layers)
    
    checkpoint_path = f'model.pth'  

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)


    encoder_atom= "cgcnn"
    encoder_bond= "dim_14"
    encode_two_path= "dim_8"

    encode_tree_path= "dim_6"
    

    smiles = "Sc1nc(Nc2ccc(Cl)cc2)c2c3c(sc2n1)CCCC3"
    mol = Chem.MolFromSmiles(smiles)   # RDKit 原子索引是 0 开始

    print("=== Atoms (idx, symbol, degree) ===")
    for a in mol.GetAtoms():
        print(a.GetIdx(), a.GetSymbol(), a.GetDegree())

    print("\n=== Bonds (idx, begin, end, type) ===")
    for b in mol.GetBonds():
        print(b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
        
    g_graph, lg_graph, fg_graph = path_complex_mol(smiles, encoder_atom,encoder_bond,encode_two_path,encode_tree_path)

    u, v, eid = lg_graph.edges(form='all')  # 'all' 会返回 (u, v, edge_id)
    for i in range(len(eid)):
        print(f"edge {eid[i].item()}: {u[i].item()} -> {v[i].item()}")
        
    g_graph, lg_graph, fg_graph = g_graph.to(device), lg_graph.to(device), fg_graph.to(device)
    g_node_feat = g_graph.ndata['feat'].to(device)
    g_edge_feat = g_graph.edata['feat'].to(device)

    lg_node_feat = lg_graph.ndata['feat'].to(device)
    lg_edge_feat = lg_graph.edata['feat'].to(device)

    fg_node_feat = fg_graph.ndata['feat'].to(device)
    fg_edge_feat = fg_graph.edata['feat'].to(device)



    graph_feats = torch.tensor([0])
    # 以 2-path（角）为例：先 Grad 预筛，再 IG 精炼
    res_grad = get_path_importance(model, order=2,
                                g_graph=g_graph, lg_graph=lg_graph, fg_graph=fg_graph,
                                g_node=g_node_feat, g_edge=g_edge_feat,
                                lg_node=lg_node_feat, lg_edge=lg_edge_feat,
                                fg_node=fg_node_feat, fg_edge=fg_edge_feat,
                                method="grad", normalize="minmax", topk=50, device=device)

    # 在 Grad Top-50 基础上（可选：实际做子集），再跑更稳的 IG 全量或仅对候选
    res_ig = get_path_importance(model, order=2,
                                g_graph=g_graph, lg_graph=lg_graph, fg_graph=fg_graph,
                                g_node=g_node_feat, g_edge=g_edge_feat,
                                lg_node=lg_node_feat, lg_edge=lg_edge_feat,
                                fg_node=fg_node_feat, fg_edge=fg_edge_feat,
                                method="ig", ig_steps=64, normalize="minmax", topk=20, device=device)

    print("Top-20 angle paths (order=2):")
    for score, meta in zip(res_ig["topk_scores"].tolist(), res_ig["topk_meta"]):
        print(f"score={score:.3f} -> {meta}")
