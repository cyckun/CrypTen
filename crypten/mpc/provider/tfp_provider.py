#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
import crypten
import numpy as np
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from .provider import TupleProvider


class TrustedFirstParty(TupleProvider):
    NAME = "TFP"

    def init_beaver(self):
        file_name = ""
        if comm.get().get_rank() == 0:
            file_name = "./alice.csv"
        else:
            file_name = "./bob.csv"
        with open(file_name, 'r') as file:
                beaver_data = file.readlines()
        print("beaver data ", beaver_data)
        return beaver_data
       

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        beaver_data = self.init_beaver()
        size = (size0[0],)
   
        list_a = []
        list_b = []
        list_c = []
        for count in range(0, size0[0]):
            data_split = beaver_data[count].split(',')
            a = data_split[0]
            b = data_split[1]
            c = data_split[2]
            list_a.append(int(a))
            list_b.append(int(b))
            list_c.append(int(c))

        aa = torch.tensor(list_a, dtype=torch.long)
        bb = torch.tensor(list_b, dtype=torch.long)
        cc = torch.tensor(list_c, dtype=torch.long)
        a = ArithmeticSharedTensor.from_shares(aa, precision=0)
        b = ArithmeticSharedTensor.from_shares(bb, precision=0)
        c = ArithmeticSharedTensor.from_shares(cc, precision=0)
     
        ############################################################origin code
        # a = generate_random_ring_element(size0, device=device)
        # b = generate_random_ring_element(size1, device=device)

        # c = getattr(torch, op)(a, b, *args, **kwargs)

        # a = ArithmeticSharedTensor(a, precision=0, src=0)
        # b = ArithmeticSharedTensor(b, precision=0, src=0)
        # c = ArithmeticSharedTensor(c, precision=0, src=0)
        #############################################################

        return a, b, c
        

    def square(self, size, device=None):
        """Generate square double of given size"""
        r = generate_random_ring_element(size, device=device)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        stacked = torch_stack([r, r2])
        stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        a = generate_kbit_random_tensor(size0, device=device)
        b = generate_kbit_random_tensor(size1, device=device)
        c = a & b

        a = BinarySharedTensor(a, src=0)
        b = BinarySharedTensor(b, src=0)
        c = BinarySharedTensor(c, src=0)

        return a, b, c

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        num_parties = comm.get().get_world_size()
        r = [
            generate_random_ring_element(size, device=device)
            for _ in range(num_parties)
        ]
        theta_r = count_wraps(r)

        shares = comm.get().scatter(r, 0)
        r = ArithmeticSharedTensor.from_shares(shares, precision=0)
        theta_r = ArithmeticSharedTensor(theta_r, precision=0, src=0)

        return r, theta_r

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        # generate random bit
        r = generate_kbit_random_tensor(size, bitlength=1, device=device)

        rA = ArithmeticSharedTensor(r, precision=0, src=0)
        rB = BinarySharedTensor(r, src=0)

        return rA, rB
