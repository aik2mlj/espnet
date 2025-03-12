import torch
import argparse


def print_recursive(state_dict, prefix=""):
    """
    Recursively prints the keys and shapes of tensors in a checkpoint or state_dict.
    """
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if isinstance(value, dict):  # Recursively print nested dictionaries
                print(f"{prefix}{key}/")
                print_recursive(value, prefix=prefix + "  ")
            elif isinstance(value, torch.Tensor):
                print(f"{prefix}{key}: {value.shape}")
            else:
                print(f"{prefix}{key}: {type(value)}")
    else:
        print(f"{prefix}{type(state_dict)} (Unexpected Type)")


def load_and_print_checkpoint(file_path):
    """
    Loads a .pth checkpoint file and recursively prints its components.
    """
    try:
        checkpoint = torch.load(file_path, map_location="cpu")

        print(f"\n--- Model Checkpoint Components ({file_path}) ---\n")

        # If it's a full checkpoint (with 'model_state_dict'), print only model weights
        if "model_state_dict" in checkpoint:
            print("[MODEL WEIGHTS]")
            print_recursive(checkpoint["model_state_dict"])

        # If optimizer states are present, print them too
        if "optimizer_state_dict" in checkpoint:
            print("\n[OPTIMIZER STATE]")
            print_recursive(checkpoint["optimizer_state_dict"])

        # Print other metadata (epoch, loss, etc.)
        print("\n[OTHER METADATA]")
        for key, value in checkpoint.items():
            if key not in ["model_state_dict", "optimizer_state_dict"]:
                print(f"{key}: {type(value)}")

        print("\n--- End of Checkpoint ---\n")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint (.pth) file.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint (.pth) file")
    args = parser.parse_args()

    # Run the function with the provided checkpoint path
    load_and_print_checkpoint(args.checkpoint_path)
