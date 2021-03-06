#!/usr/bin/env python

"""
    pytorch-dask.py
"""

from __future__ import print_function

import numpy as np
from dask import get
from hashlib import md5
import _pickle as cPickle
from pprint import pprint
from toposort import toposort
from dask.dot import dot_graph
from string import ascii_letters as letters

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
    if isinstance(x, list):
        return [dst if xx == src else xx for xx in x]
    else:
        return dst if x == src else x

def short_uuid(n=8):
    return ''.join(np.random.choice(list(letters), n))

# --
# SeaNet

class SeaNet(nn.Module):
    def __init__(self, graph, input_shape=(1, 28, 28), input_data=None, tags=None):
        assert 0 not in graph.keys(), "SeaNet: 0 in graph.keys() -- 0 is reserved for data"
        
        super(SeaNet, self).__init__()
        
        self._id = short_uuid()
        
        self.graph = graph
        self.output_layer = max(graph.keys())
        
        # Register parameters
        for k,(layer, inputs) in self.graph.items():
            if isinstance(inputs, tuple):
                raise Exception('!! inputs must be a list, not a tuple')
            
            if isinstance(layer, nn.Module):
                self.add_module(str(k), layer)
        
        if input_data is None:
            self._input_shape = input_shape
            self._input_data  = Variable(torch.randn((10,) + self._input_shape))
        else:
            self._input_shape = input_data.shape[1:]
            self._input_data  = input_data
        
        self.tags = tags
    
    def forward(self, x=None, layer=None):
        if layer is None:
            layer = self.output_layer
        
        if x is None:
            self.graph[0] = self._input_data.clone()
        else:
            self.graph[0] = x
        
        output = get(self.graph, layer)
        del self.graph[0]
        return output
    
    # --
    # Helpers
    
    def get_id(self):
        model_name = md5(str(self.graph).encode('utf-8')).hexdigest()
        model_name += '-' + self._id
        return model_name
    
    def pprint(self):
        if 0 in self.graph:
            del self.graph[0]
        
        pprint(self.graph, width=120)
    
    def plot(self, *args, **kwargs):
        self.graph[0] = 'data'
        dot_graph(self.graph)
        del self.graph[0]

    def save(self, filename):
        cPickle.dump({
            "graph" : self.graph,
            "input_data" : self._input_data
        }, open(filename, 'wb'))
    
    @classmethod
    def load(cls, filename):
        tmp = cPickle.load(open(filename, 'rb'))
        return cls(graph=tmp['graph'], input_data=tmp['input_data'])
    
    # --
    # Compilation
    
    def compile(self, reorder=True):
        adjlist    = dict([(k, to_set(inputs)) for k,(_,inputs) in self.graph.items()])
        node_order = [item for sublist in toposort(adjlist) for item in sublist]
        lookup     = dict(zip(node_order, range(len(node_order))))
        
        out = {}
        for k, (layer, inputs) in self.graph.items():
            if isinstance(inputs, list):
                out[lookup[k]] = (layer, [lookup[inp] for inp in inputs])
            else:
                out[lookup[k]] = (layer, lookup[inputs])
        
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
    
    # --
    # Sample from computational graph
    
    def get_node(self, idx):
        return self.graph[idx][0]
    
    def get_edgelist(self):
        for k, (_, inputs) in self.graph.items():
            if isinstance(inputs, list):
                for inp in inputs:
                    if inp != 0:
                        yield k, inp
            else:
                if inputs != 0:
                    yield k, inputs
                    
    def random_nodes(self, n=1, allow_input=True):
        nodes = set(self.graph.keys())
        nodes.remove(self.output_layer)
        if allow_input:
            nodes.add(0)
        
        return np.sort(np.random.choice(list(nodes), n, replace=False))
    
    def random_edge(self):
        edges = list(self.get_edgelist())
        return edges[np.random.choice(len(edges))]
    
    # --
    # Graph modifications
    
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