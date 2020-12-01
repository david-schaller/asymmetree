# -*- coding: utf-8 -*-


import itertools

import networkx as nx

from asymmetree.tools.TreeTools import LCA
from asymmetree.tools.GraphTools import symmetric_part


__author__ = 'David Schaller'


def true_transfer_edges(T):
    """Returns a set containing v if (u, v) is labeled as a transfer edge."""
    
    return {v for _, v in T.edges() if v.transferred}


def rs_transfer_edges(T, S, lca_S=None):
    """Transfer edges in T according to the relaxed scenario definition.
    
    An edge (u,v) in T is an (rs-)transfer edge if u and v are mapped to
    incomparable nodes/edges in the species tree S.
    """
    
    if not isinstance(lca_S, LCA):
        lca_S = LCA(S)
        
    transfer_edges = set()
    
    for u, v in T.edges():
        if not lca_S.are_comparable(u.color, v.color):
            transfer_edges.add(v)
    
    return transfer_edges


def fitch(tree, transfer_edges, supply_undirected=False, lca_T=None):
    """Returns the (directed) Fitch graph.
    
    Keyword arguments:
        supply_undirected - additionally return the undirected Fitch graph,
            default is False
        lca_T - instance of LCA corresponding to the tree, default is False,
            in which case a new instance is created and used
    """
    
    if not isinstance(lca_T, LCA):
        lca_T = LCA(tree)
    
    leaves = tree.supply_leaves()
    fitch = nx.DiGraph()
    
    # store for each leaf the first transfer edge on the way to the root
    first_transfer = {}
    
    for x in leaves:
        fitch.add_node(x.ID, label=x.label, color=x.color)
        
        current = x
        while current:
            if current.transferred:
                first_transfer[x] = current
                break
            current = current.parent
    
    for x, y in itertools.permutations(leaves, 2):
        
        if (y in first_transfer and
            lca_T.ancestor_not_equal(lca_T(x, y), first_transfer[y])):
            fitch.add_edge(x, y)
    
    if not supply_undirected:
        return fitch
    else:
        return fitch, symmetric_part(fitch)
    

def undirected_fitch(tree, transfer_edges, lca_T=None):
    """Returns the undirected Fitch graph.
    
    Keyword arguments:
        lca_T - instance of LCA corresponding to the tree, default is False,
            in which case a new instance is created and used
    """
    
    return fitch(tree, transfer_edges, supply_undirected=True, lca_T=lca_T)[1]
    