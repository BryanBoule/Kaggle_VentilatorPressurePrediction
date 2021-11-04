import torch

if __name__ == '__main__':
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))