import open3d as o3d
from pathlib import Path

def load_mesh(mesh_path: Path) -> o3d.t.geometry.TriangleMesh:

        if not mesh_path.exists():
            print(f"Mesh not found at {mesh_path}.")
            return None

        return o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(mesh_path)))
