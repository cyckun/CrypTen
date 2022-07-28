import argparse
from ast import Num
import crypten
import torch
import os
import crypten.communicator as comm
import numpy

torch.set_num_threads(1)

custom_env = {
    "WORLD_SIZE": "2",
    "RANK": "1",  # 1, 2
    "RENDEZVOUS": "env://",
    "MASTER_ADDR": "0.0.0.0",
    "MASTER_PORT": "9003",
    "BACKEND": "gloo",
}

for k, v in custom_env.items():
    os.environ[k] = v
print("before init")
crypten.init()
print("inited 1")


def run_code():
    print(comm.get().get_rank())


if __name__ == "__main__":
    run_code()
    parser = argparse.ArgumentParser('传入参数：***.py')
    parser.add_argument('-d', '--data', default='1')
    parser.add_argument('-r', '--role', default='0')

    args = parser.parse_args()
    print(args)
    print(type(args))
    role = args.role
    data = args.data

    test_list0  = []
    test_list1  = []
    for i in range(1,10):
        test_list0.append(1)
        test_list1.append(1)
    
    array0 = numpy.array(test_list0)
    array1 = numpy.array(test_list1)
    x = torch.tensor(array0)
    y = torch.tensor(array1)
    print("y = ", y)
    x_enc = crypten.cryptensor(x, src=0)  # encrypt
    y_enc = crypten.cryptensor(y, src=1)
    #z_enc = x_enc * y_enc
    for j in range(0, 20000):
        z_enc = x_enc.matmul(y_enc)
    # z_enc = torch.dot(x_enc, y_enc)
    z = z_enc.get_plain_text(dst=0)

    print("result = ", z)
