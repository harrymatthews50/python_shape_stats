"""
This is a demo script
"""
from python_shape_stats import helpers
import pathlib
from python_shape_stats.procrustes import do_generalized_procrustes_analysis
path = helpers._get_path_to_simulated_population()
obj_paths = [item for item in pathlib.Path(path).glob('*.obj')]
r = helpers.load_shapes_to_array(obj_paths, n_jobs=1)
ref_polydata,ref_vertices,_ = helpers.load_shape(obj_paths[0])
r = do_generalized_procrustes_analysis(r,scale=True,init_landmarks=ref_vertices,max_iter=100)

helpers.plot_shape(ref_polydata)