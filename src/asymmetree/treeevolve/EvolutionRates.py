# -*- coding: utf-8 -*-

"""
Evolution rate heterogeneity.

Introduce evolution rate asymmetries and autocorrelation.
"""

import numpy as np

from asymmetree.treeevolve.GeneTree import GeneTreeSimulator
from asymmetree.tools.Sampling import Sampler
from asymmetree.tools.PhyloTreeTools import sorted_nodes


__author__ = 'David Schaller'


# --------------------------------------------------------------------------
#                         USER INTERFACE FUNCTION
# --------------------------------------------------------------------------

def simulate_gene_trees(S, N=1,
                        dupl_rate=0.0,
                        loss_rate=0.0,
                        hgt_rate=0.0,
                        base_rate=1.0,
                        **kwargs):
    """Simulates dated gene trees with non-ultrametric edge lengths along a
    species tree.
    
    Parameters
    ----------
    S : Tree
        The species tree along which the gene trees are simulated.
    N : int, optional
        Number of gene trees to be simulated, default is 1, in which case a
        tree is returned, otherwise a list is returned.
    dupl_rate : float or tuple
        The (distribution for the) duplication rate, see documentation for
        available option. The default is constant 0.0.
    loss_rate : float or tuple
        The (distribution for the) loss rate, see documentation for
        available option. The default is constant 0.0.
    hgt_rate : float or tuple
        The (distribution for the) HGT rate, see documentation for
        available option. The default is constant 0.0.
    base_rate : float or tuple
        The (distribution for the) evolution rate at the roots of the gene
        trees, see documentation for available options. The default is
        constant 1.0.
    kwargs : optional
        See documentation or parameters of GeneTreeSimulator.simulate and
        assign_rates for additional parameters.
    
    Returns
    -------
    Tree or list
        Return a single gene tree if N = 1 and a list of gene tree if N > 1.
    """
    
    gene_trees = []
    simulator = GeneTreeSimulator(S)
    
    dupl_rate_sampler = Sampler(dupl_rate)
    loss_rate_sampler = Sampler(loss_rate)
    hgt_rate_sampler = Sampler(hgt_rate)
    base_rate_sampler = Sampler(dupl_rate)
    
    # autocorrelation between genes of the same or related species
    autocorr_variance = kwargs.pop('autocorr_variance', 0.0)
    _, autocorr_factors = autocorrelation_factors(S, autocorr_variance)
    
    # main simulation and imbalancing
    for i in range(N):
        TGT = simulator.simulate(dupl_rate=dupl_rate_sampler.draw(),
                                 loss_rate=loss_rate_sampler.draw(),
                                 hgt_rate=hgt_rate_sampler.draw(),
                                 **kwargs)
        assign_rates(TGT, S,
                     base_rate=base_rate_sampler.draw(),
                     autocorr_factors=autocorr_factors,
                     **kwargs)
        gene_trees.append(TGT)
    
    return gene_trees[0] if N == 1 else gene_trees
    

# --------------------------------------------------------------------------
#                      IMBALANCING OF THE GENE TREE
# --------------------------------------------------------------------------


def assign_rates(T, S, base_rate=1.0,
                 autocorr_factors=None,
                 autocorr_variance=0.0,
                 rate_increase=('gamma', 0.5, 2.2),
                 CSN_weights=(1, 1, 1),
                 inplace=True,
                 **kwargs):
    """Assigns realistic evolution rates to a TRUE gene tree.
    
    The assigned rates are used to modify the length ('dist') of the edges of
    the (originally ultrametric) dated gene tree.
    
    Parameters
    ----------
    T : Tree
        The gene tree.
    S : Tree
        The species tree.
    base_rate : float, optional
        Mean of substitution rate for conserved genes.
    autocorr_factors : dict, optional
        A dictonary containing autocorrelation rate factors for the edges of S
        (key = v.label for edge (v.parent, v); value = the rate as a float).
        The default is None, in which case autocorrelation factors are 
        generated if 'autocorr_variance' > 0.0, or no such modification is
        applied. See [1] for theoretical background.
    autocorr_variance : float, optional
        Autocorrelation variance factor for a lognormal distribution, only
        considered if 'autocorrelation_rates' are not supplied. See [1] for
        theoretical background.
    rate_increase : float or tuple, optional
        Distribution of the (relative) rate increase (w.r.t. the base rate)
        for divergent genes, i.e. to a factor 1 + x. The default is a Gamma
        distribution with shape 0.5 and scale 2.2, which was fitted to the
        data in [2].
    CSN_weights : tuple, optional
        Weights for choice between conservation, subfunctionalization and
        neofunctionalization. The default is (1, 1, ,1), i.e., all three modes
        are equally likely to be chosen at each duplication event.
    inplace : bool, optional
        If False, copy the tree before imbalancing. The deafault is True.
    
    Returns
    -------
    Tree
        The original instance of the gene tree (inplace=True) or a copy of
        the gene tree (inplace=False) with modified 'dist' attributes of the
        nodes.
    
    References
    ----------
    .. [1] H. Kishino, J. L. Thorne, and W. J. Bruno.
       Performance of a Divergence Time Estimation Method under a Probabilistic
       Model of Rate Evolution. 
       In: Molecular Biology and Evolution, 18(3):352-361, March 2001.
       doi: 10.1093/oxfordjournals.molbev.a003811.
    .. [2] K. P. Byrne and K. H. Wolfe.
       Consistent Patterns of Rate Asymmetry and Gene Loss Indicate Widespread
       Neofunctionalization of Yeast Genes After Whole-Genome Duplication.
       In: Genetics, 175(3):1341-1350, March 2007.
       doi: 10.1534/genetics.106.066951.
    """
    
    if not inplace:
        T = T.copy()
     
    # factors for subfunctionalization/neofunctionalization
    CSN_weights = np.asarray(CSN_weights) / sum(CSN_weights)
    sampler = Sampler(rate_increase, shift=1.0)
    _divergent_rates(T, S, sampler, CSN_weights)
    
    # autocorrelation
    if autocorr_factors:
        _apply_autocorrelation(T, autocorr_factors, inplace=True)
    elif autocorr_variance > 0.0:
        _, edge_rates = autocorrelation_factors(S, autocorr_variance)
        _apply_autocorrelation(T, edge_rates, inplace=True)
    
    # finally apply base rate
    for v in T.preorder():
        v.dist *= base_rate
    
    return T


# --------------------------------------------------------------------------
#                       EVOLUTION RATE ASYMMETRY
# --------------------------------------------------------------------------

def _adjust_distances(T, rates):
    
    for edge, rate_list in rates.items():
        time_points = np.asarray([tstamp for tstamp, _ in rate_list] + [edge[1].tstamp])
        rate_values = np.asarray([rate for _, rate in rate_list])
        edge[1].dist = np.dot(-np.diff(time_points), rate_values)
        

def _duplication_type(marked_as, CSN_weights):
    
    if marked_as == 'divergent':
        return 'divergent', 'divergent'
    else:
        r = np.random.choice(3, p=CSN_weights)
        if r == 0:                                  # conservation
            return 'conserved', 'conserved'
        elif r == 1:                                # subfunctionalization
            return 'divergent', 'divergent'
        else:                                       # neofunctionalization
            if np.random.uniform() < 0.5:
                return 'divergent', 'conserved'
            else:
                return 'conserved', 'divergent'


def _divergent_rates(T, S, sampler, CSN_weights):
    """
    Assign divergent genes and manipulate the distances in the gene tree.
    
    Parameters
    ----------
    T : Tree
        The gene tree.
    S : Tree
        The species tree.
    sampler : asymmetree.tools.Sampling.Sampler
        Sampler for rate increase for divergent genes.
    CSN_weights : tuple
         Weights for choice between conservation, subfunctionalization and
         neofunctionalization.
    
    Returns
    -------
    Tree
        The original gene tree instance with manipulated 'dist' attributes of
        its nodes.
    """
    
    T_nodes = sorted_nodes(T)
    rates = {edge: [] for edge in T.edges()}        # edge --> list of (tstamp, rate) tuples
    
    S_parents = {v.label: v.parent.label for v in S.preorder() if v.parent}
    gene_counter = {(e[0].label, e[1].label): [] for e in S.edges()}
    marked = {v: 'conserved' for v in T_nodes}      # marked as conserved or divergent
    
    for u in T_nodes:
        
        # ----------------- SPECIATION -----------------
        if u.event in ('S', '', None):
            for v in u.children:
                marked[v] = marked[u]
                S_u = u.color
                S_v = v.color if not isinstance(v.color, (tuple, list)) else v.color[1]
                gene_counter[(S_u, S_v)].append(v)
                new_rate = sampler.draw() if marked[v] == 'divergent' else 1.0
                rates[(u,v)].append((u.tstamp, new_rate))
            
        # ---------------- DUPLICATION -----------------
        elif u.event == 'D':
            marked[u.children[0]], marked[u.children[1]] = _duplication_type(marked[u],
                                                                             CSN_weights)
            gene_counter[u.color].remove(u)
            for v in u.children:
                gene_counter[u.color].append(v)
                new_rate = sampler.draw() if marked[v] == 'divergent' else 1.0
                rates[(u,v)].append((u.tstamp, new_rate))
        
        # ------------------- LOSS ---------------------
        elif u.event == 'L':
            gene_counter[u.color].remove(u)
            if len(gene_counter[u.color]) == 1:
                v = gene_counter[u.color][0]
                if marked[v] == 'divergent':
                    marked[v] = 'conserved'
                    rates[(v.parent,v)].append((u.tstamp, 1.0))
        
        # ---------- HORIZONTAL GENE TRANSFER ----------
        elif u.event == 'H':
            v1, v2 = u.children
            if v1.transferred:
                v1, v2 = v2, v1         # now v2 is the transferred copy
                
            # untransferred copy
            marked[v1] = marked[u]
            gene_counter[u.color].remove(u)
            gene_counter[u.color].append(v1)
            if u.parent:
                rates[(u,v1)].append((u.tstamp, rates[(u.parent,u)][-1][1]))
            else:
                new_rate = sampler.draw() if marked[v1] == 'divergent' else 1.0
                rates[(u,v1)].append((u.tstamp, new_rate))
            
            # transferred copy
            marked[v2] = 'divergent'
            if isinstance(v2.color, (tuple, list)):
                gene_counter[v2.color].append(v2)
            else:
                gene_counter[(S_parents[v2.color], v2.color)].append(v2)
            new_rate = sampler.draw() if marked[v2] == 'divergent' else 1.0
            rates[(u,v2)].append((u.tstamp, new_rate))
            
    _adjust_distances(T, rates)
    return T

# --------------------------------------------------------------------------
#                         AUTOCORRELATION
# --------------------------------------------------------------------------
    
def autocorrelation_factors(tree, variance):
    """Geometric Brownian motion process to assign rate factors to species tree.
    
    The parameter 'variance' is a hyperparameter for a log-normal distribution
    from which offspring rates are drawn. The overall variance of this
    distribution is 'variance' * divergence time.
    The rates are first computed for the nodes, the rates of the edges are
    assigned afterwards as the arithmetic mean of the rates of the two incident
    nodes.
    
    Parameters
    ----------
    tree : Tree
        The species tree.
    variance : float
        The hyperparameter for a log-normal distribution from which offspring
        rates are drawn at each node.
    
    Returns
    -------
    tuple of two dicts
        A dict mapping the labels of the nodel to their assigned rated, and a
        second dict mapping the labels of v of edges (v.parent, v) to the
        assigned rates of the edges.
    
    References
    ----------
    .. [1] H. Kishino, J. L. Thorne, and W. J. Bruno.
       Performance of a Divergence Time Estimation Method under a Probabilistic
       Model of Rate Evolution. 
       In: Molecular Biology and Evolution, 18(3):352-361, March 2001.
       doi: 10.1093/oxfordjournals.molbev.a003811.
    """
    
    node_rates = {}                 # maps node v --> rate of v
    edge_rates = {}                 # maps v of edge (u,v) --> rate of (u,v)
    
    for v in tree.preorder():
        if not v.parent:
            # assign factor 1.0 to root (= expected value for all other nodes
            # and edges)
            node_rates[v.label] = 1.0
            edge_rates[v.label] = 1.0
        else:
            var = variance * v.dist
            # ensure that exp. value is equal to parent's rate
            mu = np.log(node_rates[v.parent.label]) - var/2
            
            node_rates[v.label] = np.exp(np.random.normal(mu, np.sqrt(var)))
            
            # edge rate as arithmetic mean of u and v
            edge_rates[v.label] = (node_rates[v.parent.label] + node_rates[v.label]) / 2
            
    return node_rates, edge_rates


def _apply_autocorrelation(T, edge_rates, inplace=True):
    
    if not inplace:
        T = T.copy()
    
    for v in T.preorder():
        if v.parent:
            edge_ID = v.color[1] if isinstance(v.color, (tuple, list)) else v.color
            v.dist *= edge_rates[edge_ID]
    
    return T
