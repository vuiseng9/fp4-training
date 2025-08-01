import torch

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available with {num_gpus} GPU(s).")

    for i in range(num_gpus):
        # Get the name of the GPU
        gpu_name = torch.cuda.get_device_name(i)

        # Get the compute capability (major, minor)
        compute_capability = torch.cuda.get_device_capability(i)

        print(f"GPU {i}: {gpu_name}")
        print(f"  Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

        # You can also get other properties like total memory
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
else:
    print("CUDA is not available. PyTorch will use CPU.")