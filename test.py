import torch

def test_cuda():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Device Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
        print("Memory Allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
        print("Memory Cached (MB):", torch.cuda.memory_reserved(0) / 1024**2)
    else:
        print("CUDA is not available. Make sure you have a supported GPU and the correct drivers and CUDA toolkit installed.")

test_cuda()
