import torch
import torch.nn as nn


# print inforamtion of torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

x = torch.randn(16, 8, 800, 1056)

y_cpu = conv1(x)
print(y_cpu.shape)

# to cuda
x=x.cuda()
conv1.cuda()

try:
    y = conv1(x)
    print(y.shape)
except Exception as e:
    print(f"Error: {e}")