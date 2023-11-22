from lgf.utils.hierarchical_object import HierarchicalObject
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-I', type=str, default='input.obj')
parser.add_argument('--threshold', '-T', type=float, default=0.06)
parser.add_argument('--mcts_max_depth', '-MMD', type=int, default=20)
parser.add_argument('--mcts_iterations', '-MI', type=int, default=100)
parser.add_argument('--preprocess_resolution', '-PR', type=int, default=80)
parser.add_argument('--merge', '-ME', type=bool, default=True)
parser.add_argument('--mcts_nodes', '-MN', type=int, default=300)
parser.add_argument('--pca', '-PCA', type=bool, default=False)
parser.add_argument('--display', '-D', type=bool, default=False)
args = parser.parse_args()

h = HierarchicalObject()

h.load_mesh(args.input, threshold=args.threshold, 
            mcts_max_depth=args.mcts_max_depth, 
            mcts_iterations=args.mcts_iterations, 
            preprocess_resolution=args.preprocess_resolution, 
            merge=args.merge,
            mcts_nodes=args.mcts_nodes,
            pca=args.pca,
            display=args.display)

h.compute_bounding_box()
