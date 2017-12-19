#!/usr/bin/env python

"""
    pytorch-dask.py
"""

from __future__ import print_function

import numpy as np
from dask import get
from dask.dot import dot_graph
from pprint import pprint
from toposort import toposort

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from morph_layers import MorphMixin

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
    def __init__(self, graph, input_shape=(1, 28, 28), input_data=None):
        assert 0 not in graph.keys(), "SeaNet: 0 in graph.keys() -- 0 is reserved for data"
        
        super(SeaNet, self).__init__()
        
        self.graph = graph
        self.top_layer = max(graph.keys())
        
        # Register parameters
        for k,(layer, inputs) in self.graph.items():
            if isinstance(layer, nn.Module):
                self.add_module(str(k), layer)
        
        if input_data is None:
            self._input_shape = input_shape
            self._input_data  = Variable(torch.randn((10,) + self._input_shape))
        else:
            self._input_shape = input_data.size()[1:]
            self._input_data  = input_data
    
    def forward(self, x=None, layer=None):
        if layer is None:
            layer = self.top_layer
        
        if x is None:
            self.graph[0] = self._input_data.clone()
        else:
            self.graph[0] = x
        
        output = get(self.graph, layer)
        del self.graph[0]
        return output
    
    def pprint(self):
        if 0 in self.graph:
            del self.graph[0]
        
        pprint(self.graph, width=120)
    
    def plot(self, *args, **kwargs):
        self.graph[0] = 'data'
        dot_graph(self.graph)
        del self.graph[0]
    
    def get_edgelist(self):
        for k, (_, inputs) in self.graph.items():
            if isinstance(inputs, int):
                if inputs != 0:
                    yield k, inputs
            elif isinstance(inputs, list):
                for inp in inputs:
                    if inp != 0:
                        yield k, inp
    
    def compile(self):
        adjlist    = dict([(k, to_set(inputs)) for k,(_,inputs) in self.graph.items()])
        node_order = reduce(lambda a,b: list(a) + list(b), toposort(adjlist))
        lookup     = dict(zip(node_order, range(len(node_order))))
        
        out = {}
        for k, (layer, inputs) in self.graph.items():
            if isinstance(inputs, int):
                out[lookup[k]] = (layer, lookup[inputs])
            elif isinstance(inputs, list):
                out[lookup[k]] = (layer, [lookup[inp] for inp in inputs])
            else:
                out[lookup[k]] = (layer, inputs)
        
        self.__init__(out, input_data=self._input_data)
        return lookup
    
    def fix_shapes(self):
        for k, (layer, inputs) in self.graph.items():
            try:
                _ = self(layer=k)
            except:
                layer.allow_morph = True
                _ = self(layer=k)
                layer.allow_morph = False
    
    def modify_edge(self, idx1, idx2, new_layer):
        tmp_idx = 1000 + len(self.graph)
        
        if idx2 is not None:
            layer, inputs = self.graph[idx2]
            self.graph[idx2] = (layer, cond_replace(inputs, idx1, tmp_idx))
        else:
            for idx2, (layer, inputs) in self.graph.items():
                self.graph[idx2] = (layer, cond_replace(inputs, idx1, tmp_idx))
        
        # Create tmp node
        self.graph[tmp_idx] = (new_layer, idx1)
        
        return tmp_idx
    
    def modify_node(self, idx, new_layer):
        self.graph[idx] = (new_layer, self.graph[idx][1])
        return idx
        
    def add_skip(self, idx1, idx2, new_layer):
        tmp_idx = 1000 + len(self.graph)
        
        for k, (layer, inputs) in self.graph.items():
            self.graph[k] = (layer, cond_replace(inputs, idx2, tmp_idx))
        
        # Create tmp node
        self.graph[tmp_idx] = (new_layer, [idx2, idx1])
        
        return tmp_idx