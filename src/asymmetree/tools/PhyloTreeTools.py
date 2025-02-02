# -*- coding: utf-8 -*-

"""
Auxiliary functions for phylogentic trees.

Phylogenetic trees are usually defined as trees where every inner node has
at least two children. However, the class `Tree` does not force this property.
Simulated trees are often 'planted' i.e. the root has only a single child,
and this edge represents the ancestral lineage.
"""

import collections, itertools, random, re

import numpy as np

from tralda.datastructures.Tree import Tree, TreeNode


__author__ = 'David Schaller'


default_attributes = {'label': '',
                      'event': None,
                      'reconc': None,
                      'dist': 1.0,
                      'tstamp': None,
                      'transferred': 0}


def node_to_str(node):
    """String representation of a node in a phylogenetic tree.
    
    Returns
    -------
    str
        String representation of the node including reconc (if defined)
        and dist.
    """
    
    if isinstance(node.reconc, (tuple, list)):
        return '{}<{}-{}>:{}'.format(node.label, *node.reconc, node.dist)
    elif node.reconc:
        return '{}<{}>:{}'.format(node.label, node.reconc, node.dist)
    else:
        return '{}:{}'.format(node.label, node.dist)


def sorted_nodes(tree, oldest_to_youngest=True):
    """List of nodes sorted by timestamp.
    
    Return a list of all nodes in the tree sorted by time stamp beginning
    with the oldest (highest time stamp). Optionally the order can be
    reversed.
    
    Parameters
    ----------
    tree : Tree
        A phylogenetic tree with nodes that have a `tstamp` attribute.
    oldest_to_youngest : bool, optional
        If True, the nodes are sorted from oldest to youngest, otherwise
        from youngest to oldest (the default is True).
        
    Returns
    -------
    list of TreeNode instances
        Nodes sorted by their timestamp.
    """
    
    return sorted(tree.preorder(),
                  key=lambda node: node.tstamp,
                  reverse=oldest_to_youngest)


def sorted_edges(tree):
    """List of edges (u,v) sorted by timestamp of u.
    
    Return a list of all edges (u,v) sorted by timestamp of u beginning
    with the oldest (highest time stamp).
    
    Parameters
    ----------
    tree : Tree
        A phylogenetic tree with nodes that have a `tstamp` attribute.
        
    Returns
    -------
    list of pairs of TreeNode instances
        Edges sorted by their timestamp.
    """

    return sorted(tree.edges(),
                  key=lambda e: (e[0].tstamp, e[1].tstamp),
                  reverse=True)

def distance_from_timing(tree):
    """Adjusts all distances according to the time stamp difference.
    
    Parameters
    ----------
    tree : Tree
        A phylogenetic tree with nodes that have a `tstamp` attribute.
    """
    
    if tree.root:
        tree.root.dist = 0.0
    
    for u, v in tree.edges():
        v.dist = abs(u.tstamp - v.tstamp)


def delete_and_reconnect(tree, node,
                         add_distances=True,
                         keep_transferred=True):
    """Delete a node from the tree and reconnect its parent and children.
    
    Parameters
    ----------
    node : TreeNode
        The node in the tree to be deleted.
    add_distances : bool, optional
        When the node v is deleted and its children are reconnected to its
        parent, add to the `dist` parameter of the children `v.dist`, i.e.
        the distance of v from its parent (the default is True).
    keep_transferred : bool, optional
        When the edge of the deleted node v from its parent u was a
        transfer edge, make all edges from u to the children of v transfer
        edges (the default is True).
        
    Returns
    -------
    TreeNode or bool
        The parent of the node, if it could be deleted, or False, if the
        node could not be deleted, i.e., it has no parent.
    """
    
    if not node.parent:
        return False
    
    for child in node.children:
        if add_distances and hasattr(node, 'dist') and hasattr(child, 'dist'):
            child.dist += node.dist
        if (keep_transferred and 
            hasattr(node, 'transferred') and 
            node.transferred):
            child.transferred = 1
    
    return tree.delete_and_reconnect(node)


def add_planted_root(tree):
    """Add a new root that has the original root as its single child.
    
    Parameters
    ----------
    tree : Tree
        A phylogenetic tree.
    
    Returns
    -------
    TreeNode
        The newly added root.
    """
    
    old_root = tree.root
    tree.root = TreeNode()
    for k, v in default_attributes.items():
        setattr(tree.root, k, v)
    
    if old_root:
        tree.root.add_child(old_root)
    
    return tree.root


def reconc_sorted_leaves(tree, return_list=False):
    """Sort leaves by their reconciliation attribute.
    
    Parameters
    ----------
    tree : Tree
        A tree with nodes that have a `reconc` attribute.
    return_list : bool, optional
        If True, also return a list of leaves such that leaves of the same
        reconciliation appear as a coherent sequence (the default is False).
    
    Returns
    -------
    dict
        A dictionary with reconciliations as keys and a list of corresponding
        nodes as values.
    list, optional
        List of leaves such that leaves of the same reconciliation appear as a
        coherent sequence.
    """
    
    reconc_dict = {}
    
    for leaf in tree.leaves():
        if leaf.reconc not in reconc_dict:
            reconc_dict[leaf.reconc] = []
        reconc_dict[leaf.reconc].append(leaf)
    
    if not return_list:
        return reconc_dict
    else:
        leaves = []
        for reconc, leaf_list in reconc_dict.items():
            for leaf in leaf_list:
                leaves.append(leaf)
        
        return reconc_dict, leaves
    
    
def distance_matrix(tree, leaf_order=None):
    """Distance matrix on the leaf set of the phylogenetic tree.
    
    Computes a distance matrix on the set of leaves of the tree where each
    distances is the sum of the distances (`dist`) on the path connecting
    the pair of leaves.
    
    Additionally a list of leaves corresponding to the indices in the
    matrix is returned.
    
    Parameters
    ----------
    tree : Tree
        A tree with nodes that have a `dist` attribute.
    leaf_order : list, optional
        A list of all leaves in the tree defining the indices for the
        matrix (the default is None, in which case leaves are indexed in
        sibling order).
    
    Returns
    -------
    list of TreeNode objects
        Represents the order for the lines/columns in the distance matrix.
    numpy.ndarray (dtype=numpy.float)
        The distance matrix.
    """
    
    distance_dict = distances_from_root(tree)
    leaves = tree.leaf_dict()
    
    if leaf_order:
        if set(leaf_order) != set(leaves[tree.root]):
            raise ValueError('ordered leaf list does not match with the '\
                             'leaves in the tree')
        L = leaf_order
    else:
        # leaves in sibling order
        L = leaves[tree.root]     
    
    leaf_index = {l: i for i, l in enumerate(L)}
    
    D = np.zeros((len(L),len(L)), dtype=np.float)
    
    for v in tree.preorder():
        if v.children:
            for c1, c2 in itertools.combinations(v.children, 2):
                for x in leaves[c1]:
                    x_index = leaf_index[x]
                    x_dist = distance_dict[x] - distance_dict[v]
                    for y in leaves[c2]:
                        y_index = leaf_index[y]
                        y_dist = distance_dict[y] - distance_dict[v]
                        D[x_index, y_index] = x_dist + y_dist
                        D[y_index, x_index] = x_dist + y_dist
    
    return L, D


def distances_from_root(tree):
    """The distances of each node to the root of the tree.
    
    Parameters
    ----------
    tree : Tree
        A tree with nodes that have a `dist` attribute.
    
    Returns
    -------
    dict
        The keys are TreeNode objects and the values their distances
        (sum of `dist`) to the root.
    """
    
    distance_dict = {}
    
    for v in tree.preorder():
        if not v.parent:
            distance_dict[v] = 0.0
        else:
            depth = distance_dict[v.parent] + v.dist
            distance_dict[v] = depth
            
    return distance_dict


def topology_only(tree, inplace=True):
    """Reset reconciliations, distances, time stamps, transfer status, and
    inner labels.
    
    Parameters
    ----------
    tree : Tree
        A tree.
    inplace : bool
        If True, reset the attributes of this tree instance, otherwise
        make a copy first and modify the copy.
    
    Returns
    -------
    Tree
        The original or a copy of the tree instance with reset attributes.
    """
    
    if not inplace:
        T = tree.copy()
    else:
        T = tree
    
    for v in T.preorder():
        if v.children:
            v.label = ''
            v.reconc = None
        v.dist = 1.0
        v.tstamp = None
        v.transferred = 0
    
    return T


def count_node_types(tree):
    """Count speciations, duplication, losses, HGTs and surviving genes.
    
    Parameters
    ----------
    tree : Tree
        A tree with nodes that have a `label` attribute.
    
    Returns
    -------
    dict
        With the event counts as values. Key are 'S' (speciations), 'D'
        (duplication), 'L' (losses), 'H' (HGTs) and 'extant' (surviving
        genes).
    """
    
    counts = {'S': 0, 'D': 0, 'L': 0, 'H': 0, 'extant': 0}
    
    for v in tree.preorder():
        
        if not v.children:
            if v.event == 'L':
                counts['L'] += 1
            else:
                counts['extant'] += 1
        
        elif v.event == 'S':
            counts['S'] += 1
        elif v.event == 'D':
            counts['D'] += 1
        elif v.event == 'H':
            counts['H'] += 1
    
    return counts


def random_colored_tree(n, colors, binary=False, force_all_colors=False):
     """Create a random colored tree.
     
     The number of leaves and the reconciliation labels are specified in the
     parameters n and colors, respectively. Each non-leaf node in the 
     resulting tree will have at least children (property of phylogenetic
     trees).
     
     Parameters
     ----------
     n : int
         The desired number of leaves.
     colors : int or list
         The list of recociliations, or the desired number in which case the
         reconciliations {1, ..., colors} are used.
     binary : bool, optional
         If True, forces the tree to be binary (the default is False).
     force_all_colors : bool
         If True, the resulting tree is guaranteed to have at least one leaf
         of each reconciliation (the default is False).
     
     Returns
     -------
     Tree
         A random tree with n leaves to which the `reconc` attribute is
         assigned at random.
     
     Raises
     ------
     TypeError
         If n is not an integer > 0.
     ValueError
         If the number of colors is greater than n and `force_all_colors`
         is true.
     """
     
     tree = Tree.random_tree(n, binary=binary)
         
     if isinstance(colors, int):
         colors = [i+1 for i in range(colors)]
     elif not isinstance(colors, collections.abc.Iterable):
         raise TypeError("'colors' must be of type 'int' or iterable")
         
     if len(colors) > n and force_all_colors:
         raise ValueError('cannot force all colors since #colors > n')
     
     leaves = [l for l in tree.leaves()]
     
     if force_all_colors:
         # use every color at least once
         permutation = np.random.permutation(len(leaves))
         for i in range(len(leaves)):
             if i < len(colors):
                 leaves[permutation[i]].reconc = colors[i]
             else:
                 # color the remaining leaves randomly
                 leaves[permutation[i]].reconc = random.choice(colors)
     else:
         # assign colors completely randomly
         for leaf in leaves:
             leaf.reconc = random.choice(colors)
             
     return tree
 

def random_ultrametric_timing(tree, inplace=False, adjust_distances=False):
    """Generate a random ultrametric timing for the tree.
    
    Parameters
    ----------
    tree : Tree
        The tree for which a random timing shall be generated.
    inplace : bool, optional
        If True, the input tree is modified, otherwise a copy is returned.
        The default is False.
    adjust_distances : bool, optional
        If True, also adjust the dist attribute of the tree nodes to match the
        differences of the tstamp values. The default is False.
    
    Returns
    -------
    Tree
        A random tree whose nodes have tstamp attributes that represent the
        generated random ultrametric timing.
    """
    
    if not inplace:
        tree = tree.copy()
        
    for v in tree.preorder():
        if not v.children:
            v.tstamp = 0.0
        elif not v.parent:
            v.tstamp = 1.0
        else:                               # random walk to a leaf
            pos = v                         # current position
            length = 0                      # path length |P|
            while pos.children:
                length += 1
                pos = pos.children[np.random.randint(len(pos.children))]
            v.tstamp = (
                v.parent.tstamp * (1 - 2 * np.random.uniform() / (length+1))
            )
    
    if adjust_distances:
        distance_from_timing(tree)
             
    return tree
 
    
def phylo_tree_attributes(tree, inplace=True):
    """Add the attributes for a phylogentic tree if not already set.
    
    Parameters
    ----------
    tree : Tree
    inplace : bool
        If True, the input tree is modified, otherwise a copy is returned.
        The default is True.
    
    Returns
    -------
    Tree
    """
    
    if not inplace:
        tree = tree.copy()
    
    for v in tree.preorder():
        for key, value in default_attributes.items():
            if not hasattr(v, key):
                setattr(tree.root, key, value)
    
    return tree


# --------------------------------------------------------------------------
#                    RECONSTRUCTION OF INFORMATION
# --------------------------------------------------------------------------

def assign_missing_labels(tree):
    """Assign integer label to nodes with no or missing label.
    
    Also assign the `number_of_species` attribute, i.e., the number of leaves,
    to the tree.
    
    Parameters
    ----------
    tree : Tree
    """
    
    tree.number_of_species = 0
    labels = set()
    
    for v in tree.preorder():
        if not v.children:
            tree.number_of_species += 1
            
        if hasattr(v, 'label') and v.label:
            labels.add(v.label)
    
    # assign new labels to remaining nodes
    current_label = 0
    for v in tree.preorder():
        if not hasattr(v, 'label') or not v.label:
            while current_label in labels:
                current_label += 1
            v.label = current_label
            labels.add(current_label)


def reconstruct_reconc_from_graph(tree, G):
    """Reconstruct the reconciliations from a NetworkX Graph.
    
    Parameters
    ----------
    tree : Tree
    G : networkx.Graph
        The graph from which labels and reconciliations shall be reconstructed.
    """
    
    for v in tree.preorder():
        if hasattr(v, 'label') and v.label in G:
            if 'reconc' in G.nodes[v.label]:
                if isinstance(G.nodes[v.label]['reconc'], list):
                    v.reconc = tuple(G.nodes[v.label]['reconc'])
                else:
                    v.reconc = G.nodes[v.label]['reconc']
            elif 'color' in G.nodes[v.label]:
                if isinstance(G.nodes[v.label]['color'], list):
                    v.reconc = tuple(G.nodes[v.label]['color'])
                else:
                    v.reconc = G.nodes[v.label]['color']
            
            
def reconstruct_timestamps(tree):
    """Reconstruct the timestamps.
    
    Make the time stamps matching with the distance attribute. The root
    obtains time stamp 1.0, and all other node smaller time stamps such
    that the difference to the parent's time stamp is exactly `dist`.
    
    Parameters
    ----------
    tree : Tree
        A tree with nodes that have a `dist` attribute.
    """
    
    tree.root.tstamp = 1.0
    for v in tree.preorder():
        if v.parent:
            v.tstamp = v.parent.tstamp - v.dist
    

# --------------------------------------------------------------------------
#                         TREE MANIPULATION
# -------------------------------------------------------------------------- 

def delete_losses_and_contract(tree, inplace=False):
    """Delete all branches leading to loss leaves only.
    
    Nodes that would have only a single child afterwards are suppressed (except
    possibly the planted root), i.e. their children are recursively reconnected
    to the parents. Distances are cumulated in this process and the transferred
    status is kept in the sense that an edge is a transfer edge if at least
    one edge on the contracted path to which it corresponds was a transfer edge.
    
    Parameters
    ----------
    tree : Tree
        The tree in which loss branches shall be removed.
    inplace : bool, optional
        If True, the tree is directly manipulated. The default is False in
        which case a copy of the tree is created which gets manipulated while
        the original tree remains untouched.
    
    Returns
    -------
    Tree
        The tree with all loss branches removed (original instance or a new one
        depending on the `inplace` parameter).
    """
    
    if not inplace:
        tree = tree.copy()
    
    loss_nodes = []
    for node in tree.postorder():
        if not node.children and node.event == 'L':
            loss_nodes.append(node)
    
    # traverse from loss node to root delete if degree <= 1
    for loss_node in loss_nodes:
        current = delete_and_reconnect(tree, loss_node,
                                       add_distances=True,
                                       keep_transferred=True)
        
        while len(current.children) < 2 and current.parent:
            current = delete_and_reconnect(tree, current,
                                           add_distances=True,
                                           keep_transferred=True)
    
    return tree


def remove_planted_root(tree, inplace=True):
    """Remove the planted root of the tree (if existent).
    
    Parameters
    ----------
    tree : Tree
        The tree in which the planted root shall be removed.
    inplace : bool, optional
        If True, the tree is directly manipulated, otherwise a copy is created
        and the original tree remains untouched (the default is True).
    
    Returns
    -------
    Tree
        The tree with the planted root removed (original instance or a new one
        depending on the `inplace` parameter).
    """
    
    if not inplace:
        tree = tree.copy()
        
    # delete the root if the tree is planted
    if len(tree.root.children) == 1:
        new_root = tree.root.children[0]
        new_root.detach()
        tree.root = new_root
        new_root.dist = 0.0
        
    if not tree.root.children and not tree.root.label:
        # no surviving genes --> return empty tree
        tree.root = None
    
    return tree


# --------------------------------------------------------------------------
#                          TREE  <--->  NEWICK
# --------------------------------------------------------------------------
        
def to_newick(tree, label=True, reconc=True, distance=True,
                    label_inner=True, reconc_inner=False):
    """Return a Newick representation of the tree.
    
    This function overrides the function of the parent class.
    
    Parameters
    ----------
    label : bool, optional
        If True, the Newick str contains the labels of the nodes (the
        default is True).
    reconc : bool, optional
        If True, the Newick str contains the reconciliations of the nodes in
        <[...]> brackets (the default is True).
    distance : bool, optional
        If True, the Newick str contains the distances of the nodes in
        standard :[...] notation (the default is True).
    label_inner : bool, optional
        If True, the Newick str also contains the labels of the inner 
        nodes (the default is True).
    reconc_inner : bool, optional
        If True, the Newick str contains the reconciliations of the inner nodes
        (the default is False).
    
    Returns
    -------
    str
        A Newick representation of the tree.
    """
    
    def _to_newick(node):
        
        if not node.children:
            token = ''
            if label and hasattr(node, 'label'):
                token += str(node.label)
            if reconc and hasattr(node, 'reconc') and node.reconc:
                token += '<{}-{}>'.format(*node.reconc) \
                    if isinstance(node.reconc, (tuple, list)) \
                    else '<{}>'.format(node.reconc)
            if distance and hasattr(node, 'dist'):
                token += ":{}".format(node.dist)
            return token
        else:
            s = ''
            for child in node.children:
                s += _to_newick(child) + ','
            token = ''
            if label and hasattr(node, 'label') and label_inner:
                token += str(node.label)
            if reconc_inner and hasattr(node, 'reconc') and node.reconc:
                token += '<{}-{}>'.format(*node.reconc) \
                    if isinstance(node.reconc, (tuple, list)) \
                    else '<{}>'.format(node.reconc)
            if distance and hasattr(node, 'dist'):
                token += ':{}'.format(node.dist)
            return '({}){}'.format(s[:-1], token)
    
    
    if tree.root:
        return _to_newick(tree.root) + ';'
    else:
        return ';'


def parse_newick(newick):
    """Parses trees in Newick format into object of type 'Tree'.
    
    Parameters
    ----------
    newick : str
        A tree in Newick format.
    
    Returns
    -------
    Tree
        The parsed tree.
    
    Raises
    ------
    TypeError
        If the input is not a string.
    ValueError
        If the input is not a valid Newick string.
    
    Notes
    -----
    Do not use this function for serialization and reloading Tree
    objects. Use the `serialize()` function instead.
    Labels and reconciliations that are integer numbers are converted to int.
    The reconciliations (if present in <...> in the string) are parsed as
    strings and need to be converted to integers afterwards if necessary.
    """
    
    def to_int(item):
        """Trys to convert the string into int."""
        
        return int(item) if item.isdigit() else item
    
    tree = Tree.parse_newick(newick)
    
    # regex for notation label<reconc>
    label_col_regex = re.compile(r"'?([a-zA-Z0-9_]*)'?<(.*)>")
    
    for node in tree.preorder():
        
        node.event = ''
        
        if not hasattr(node, 'dist'):
            node.dist = 1.0
        
        label = str(node.label)
        
        label_col = label_col_regex.match(label)
        if label_col:
            node.label = to_int(label_col.group(1))
            node.reconc = to_int(label_col.group(2))
        else:
            node.reconc = None
        
        # reconc is a tuple
        if (node.reconc and 
            isinstance(node.reconc, str) and 
            node.reconc.find('-') != -1):
            
            a, b = node.reconc.split('-')
            node.reconc = (to_int(a), to_int(b))
    
    return tree
