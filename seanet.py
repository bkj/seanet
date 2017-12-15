#!/usr/bin/env python

"""
    pytorch-dask.py
"""

from __future__ import print_function

import numpy as np
from dask import get
from pprint import pprint
from toposort import toposort

import torch
from torch import nn
from torch.nn import functional as F

# --
# Helpers

def to_set(x):
    return set(x) if isinstance(x, list) else set([x])

def cond_replace(x, src, dst):
    if isinstance(x, int):
        return dst if x == src else x
    elif isinstance(x, list):
        return [dst if xx == src else xx for xx in x]
    else:
        return x

class SeaNet(nn.Module):
    def __init__(self, graph):
        super(SeaNet, self).__init__()
        
        self.graph = graph
        self.top_layer = max(graph.keys())
        
        # Register parameters
        for k,(layer, inputs) in self.graph.items():
            if isinstance(layer, nn.Module):
                self.add_module(str(k), layer)
    
    def forward(self, x, layer=None):
        if layer is None:
            layer = self.top_layer
        
        self.graph['data'] = x
        output = get(self.graph, layer)
        del self.graph['data']
        return output
    
    def pprint(self):
        if 'data' in self.graph:
            del self.graph['data']
        
        pprint(self.graph)
    
    @classmethod
    def toposort(cls, model):
        graph      = dict([(k, to_set(inputs)) for k,(_,inputs) in model.graph.items() if inputs != 'data'])
        node_order = reduce(lambda a,b: list(a) + list(b), toposort(graph))
        lookup     = dict(zip(node_order, range(len(node_order))))
        
        out = {}
        for k, (layer, inputs) in model.graph.items():
            if isinstance(inputs, int):
                out[lookup[k]] = (layer, lookup[inputs])
            elif isinstance(inputs, list):
                out[lookup[k]] = (layer, [lookup[inp] for inp in inputs])
            else:
                out[lookup[k]] = (layer, inputs)
        
        return cls(out)
    
    def modify_edge(self, idx1, idx2, new_layer):
        """
            Add a layer on an edge
        """
        
        
        # Target node now points to tmp node
        layer, inputs = self.graph[idx2]
        self.graph[idx2] = (layer, cond_replace(inputs, idx1, -1))
        
        # Create tmp node
        self.graph[-1] = (new_layer, idx1)
        
        return SeaNet.toposort(self)
    
    def modify_node(self, idx, new_layer):
        """
            Modify a node
        """
        
        self.graph[idx] = (new_layer, self.graph[idx][1])
        return SeaNet.toposort(self)
    
    def add_skip(self, idx1, idx2, new_layer):
        """
            Add a skip connection that adds output of `idx1` and output of `idx2`
        """
        
        # All layers that take idx2 as input now take tmp node
        for k,(layer, inputs) in self.graph.items():
            self.graph[k] = (layer, cond_replace(inputs, idx2, -1))
        
        # Create tmp node
        self.graph[-1] = (new_layer, [idx2, idx1])
        
        return SeaNet.toposort(self)
