import coacd
import trimesh
import numpy as np  

def convex_decomposition(mesh, threshold=0.03, mcts_max_depth=10, 
                         mcts_nodes=100, **kwargs):
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh, threshold=threshold, 
                            mcts_max_depth=mcts_max_depth, 
                            mcts_nodes=mcts_nodes, **kwargs)
    
    return parts