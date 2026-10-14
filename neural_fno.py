import warp as wp

class HybridFNOPredictor:
    """Zero-copy DLPack bridge between Warp 1.16+ SoA buffers and 3D FNO PyTorch surrogates."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def to_torch_zero_copy(self, wp_array: wp.array):
        """Converts a Warp GPU array into a PyTorch tensor without copying GPU memory."""
        return wp.to_torch(wp_array)
