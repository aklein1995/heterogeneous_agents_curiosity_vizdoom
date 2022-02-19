import torch

print('torch version:',torch.__version__)
print('cuda (torch) version:',torch.version.cuda)
print('cuda available:',torch.cuda.is_available())
print('# of devices:',torch.cuda.device_count())

# select device
torch.cuda.set_device(3)
print()
print('selected device:',torch.cuda.current_device())
print('name:',torch.cuda.get_device_name(0))

# test
print('init testing...')
for i in range(1000):
    x = torch.tensor(1.0).to('cuda')
    y = torch.tensor(2.0).to('cuda')
    z = x+y
    if i % 100 == 0:
        print('*** Memory Usage ***')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
