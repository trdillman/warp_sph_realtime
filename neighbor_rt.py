import warp as wp

class NeighborSearchBackend:
    """Unified wrapper supporting O(n) HashGrid and optional BVH / RT-Core range queries."""
    def __init__(self, grid_res: int = 128, device: str = "cuda:0"):
        self.device = device
        self.mode = "HASH_GRID"
        self.hash_grid = wp.HashGrid(grid_res, grid_res, grid_res, device=device)

    def build(self, points: wp.array, radius: float):
        self.hash_grid.build(points=points, radius=radius)

    @property
    def id(self):
        return self.hash_grid.id
