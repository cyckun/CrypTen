#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from grpc import server
import crypten.communicator as comm
import torch
import crypten
import tenseal as ts
import random
import numpy as np
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from .provider import TupleProvider



class TrustedFirstParty(TupleProvider):
    NAME = "TFP"

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        plain_vector = []
        width = size0[0]
        random_scale = 1000  # = sqrt(plain_modulus)
        # Send forward, receive backward
        # dst = (self.rank + 1) % self.world_size
        # src = (self.rank - 1) % self.world_size
        dst = 1
        src = 0

        if comm.get().get_rank() == 0:
            # Setup TenSEAL context
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=4096,
                plain_modulus=1032193  # should update, more bigger
            )

            # step 2: prepare F(a), a1:
            plain_share0 = []
            plain_share1 = []
            for i in range(0, width):
                plain_vector.append(random.randint(2, random_scale))  # alice random generate a; 1000 is associated with plain_modulus
                plain_share0.append(random.randint(1, plain_vector[i] - 1))  # random select a0, a1 = a - a0;
                plain_share1.append(plain_vector[i] - plain_share0[i])
            try:
                encrypted_vector = ts.bfv_vector(context, plain_vector)  # F(a), a1 send to client.
                temp = encrypted_vector.serialize()
            except ValueError as value_err:  # 可以写多个捕获异常
                return value_err

            first_package = {"context":context.serialize(), "fa":temp, "a1":plain_share1}
            comm.get().send_obj(first_package, 1)
            second_package = {
                "fb": "",
                "b0": ""
            }
            # get F(a*b-r), b0 from bob.
            second_package = comm.get().recv_obj(dst)
            cipher = ts.BFVVector.load(context, second_package.get("fb"))
            c = cipher.decrypt()
            
            for i in range(0, len(c)):
                if c[i] < 0 and abs(c[i]) > 1000:  # 不科学，需要推导.
                    c[i] = c[i] + 1032193

           
            b = second_package.get("b0")
            a = plain_share0

            # print("a = ", a)
            # print("b = ", b)
            # print("c = ", c)

        else:
            first_package = {
                "context": "",
                "fa": "",
                "a1": ""
            }
       
            first_package = comm.get().recv_obj(0)
            a1 = first_package.get("a1")
            fa = first_package.get("fa")
            context_client = ts.Context.load(first_package.get("context"))
            
            Fa = ts.BFVVector.load(context_client, fa)

            b = []
            b0 = []
            b1 = [] 
            bob_r = []  # c1
            for i in range(0, width):
                b.append(random.randint(2, 1000))  # 2 is for b1 > 0
                bob_r.append(random.randint(1, 1000))
                b0.append(random.randint(1, b[i]-1))  # random generate b, b0, b1, r
                b1.append(b[i] - b0[i])

            cipher = b * Fa - bob_r  # F(a)*b -r = F(a*b-r), this puzzle me, tenseal wrap it.
            second_package = {"fb": cipher.serialize(), "b0": b0}
            comm.get().send_obj(second_package, dst=src)
            # a = server_share1
            a = a1
            b = b1
            c = bob_r
            # print("a = ", a)
            # print("b = ", b)
            # print("c = ", c, type(c[0]))


        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        cc = torch.tensor(c, dtype=torch.long)
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
