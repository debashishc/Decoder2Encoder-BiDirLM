import torch


class ModelTools:
    """Tools for Model."""

    @staticmethod
    def model_summary(model, model_layers:bool=False) -> None:
        """Prints a summary of the model."""
        name_or_path = getattr(model, "name_or_path", "Unknown")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Model {name_or_path} loaded with {total_params / 1e6:.2f} Million of trainable parameters."
        )
        if model_layers:
            print(model)

    @staticmethod
    def clear_gpu_cache():
        """Clear the GPU cache."""
        torch.cuda.empty_cache()
