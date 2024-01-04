import copy
import os
import tkinter.filedialog
import pandas as pd
from scipy.stats import ortho_group
import pyvista as pv
import time
import numpy as np
import csv
import importlib
from collections.abc import Callable
from sklearn.model_selection import KFold
import joblib
from joblib_progress import joblib_progress
from tqdm import tqdm
import pickle
import PIL
def get_camera_properties(cam):
    keys = [item for item in dir(cam) if item[0].islower()]
    out = dict()
    for key in keys:
        val = getattr(cam,key)
        if not callable(val):
            out[key] = val
    return out

def get_user_input_for_cam_view(mesh,link_views=True):
    if not my_is_iterable(mesh):
        mesh = [mesh]
    n_plots = len(mesh)
    nearest_sq = np.ceil(np.sqrt(n_plots))
    n_cols = int(nearest_sq)
    n_rows = int(np.ceil(n_plots / n_cols))
    pl = make_plotter(shape=(n_rows, n_cols))
    for x in range(n_plots):
        r, c = np.unravel_index(x, [n_rows, n_cols])
        pl.subplot(r, c)
        add_shape(mesh[x], pl)
        pl.add_text('Click and drag to edit view and close when finished',color=[0,0,0])
    if link_views:
        pl.link_views()
    pl.show()
    # after it is closed record the camera views
    cam_views = []
    for x in range(n_plots):
        r, c = np.unravel_index(x, [n_rows, n_cols])
        pl.subplot(r, c)
        cam_views.append(get_camera_properties(pl.camera))

    return cam_views

def _generate_random_cov_matrix(sz,rank,eig_val_scale,seed=None):
    """
    Generate a random covariance matrix with known rank by building it from a specified number of eignevectors and eignevalues

    :param sz: the returned covariance matrix with be shaped sz x sz
    :param rank: the desired rank of the returned matrix
    :param eig_val_scale: eigenvalues will be randomly sampled
    :param seed:
    :return: the covariance matrix
    """
    if rank > sz:
        raise ValueError('rank cannot be greater than size')
    eig_vecs = ortho_group.rvs(dim=sz)
    eig_vecs = eig_vecs[:,0:rank]
    # randomly sample eigenvalues from chi square (df=1) just so that  smaller eigenvalues are more likely
    eig_vals = np.random.chisquare(1,rank)*eig_val_scale
    # sort in descending order
    eig_vals = np.sort(eig_vals)
    eig_vals = eig_vals[::-1]
    cov = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
    return cov, eig_vecs, eig_vals

def set_camera_view(cam,view : dict | Callable ):
    if isinstance(view,dict):
        for key, item in view.items():
            try:
                setattr(cam,key,item)
            except:
                pass
    elif callable(view):
        view()
    else:
        raise TypeError('View should be a dictionary or a callable')

def _generate_circular_sequence(maxi,mini,origin=0,n_in_sequence=20):
    if n_in_sequence % 2 == 0:
        n_in_sequence += 1
    a = np.linspace(origin,maxi,int(np.ceil(n_in_sequence/4)))
    b = a[1:-1]
    b = b[::-1]
    c = np.linspace(origin,mini,int(np.ceil(n_in_sequence/4)))
    d = c[1:-1]
    d = d[::-1]
    return np.concatenate([a,b,c,d])

def broken_stick_expectation(N):
    x = 1/np.linspace(1,N,N)
    summation = lambda k0 : np.sum(x[k0:])
    return np.array([summation(i) for i in range(N)])*1/N
def _trim_arrays_to_min_size(matrices,axis=0):
    shp = [item.shape[axis] for item in matrices]
    min_shp = min(shp)
    inds = [i for i in range(min_shp)]
    return [np.take(x,inds,axis=axis) for x in matrices]

def _rng_kfold_split(k,seed=None):
    # to use KFold with numpy.random.Generator is not currrently supported (it is a WIP: https://github.com/scikit-learn/scikit-learn/pull/23962 )
    # so this function is (as a temporary solution) forces KFold to listen to an numpy.randon.Generator will use np.random.Generator to generate an integer seed for KFold
    # TODO Fix when scikit supports numpy.random.Generator
    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=k, shuffle=True, random_state=rng.integers(0, 2**32-1))
    return kf

def broken_stick_empirical(N,n_reps=100,seed=None):
    """
    Empirically estimate the broken stick distribution when random splitting a stick of length 1 into N segments
    :param n_reps: number of repetitions
    :param N: number of segments to break the stick into
    :return:
    """

    rng = np.random.default_rng(seed)
    # randomly sample break points from the uniform distribution 0-1
    bp = rng.uniform(0, 1, [n_reps,N - 1])
    # add the start and end of the stick
    bp = np.concatenate([np.zeros([n_reps,1]),bp,np.ones([n_reps,1])],axis=1)
    # sort ascending for each row
    bp = np.sort(bp,axis=1)
    # compute length between the i-1th breakpoint to the ith
    segment_lengths = bp[:,1:]-bp[:,0:-1]

    # sort descending for each row
    segment_lengths = np.sort(segment_lengths,axis=1)
    segment_lengths = segment_lengths[:,::-1]

    return segment_lengths
def get_dummy(x : pd.DataFrame,dtype_obj : pd.CategoricalDtype) -> pd.DataFrame:
    """turns x into dummy variables, listening to the categories listed in dtype_obj
    """
    cats=dtype_obj.categories
    if len(cats)==1:
        raise ValueError('Only one category is specified in dtype_obj.categories')
    dum_vars = pd.DataFrame(data=np.zeros([x.shape[0],len(cats)-1]),index=x.index)
    for i in range(1,len(cats)):
        dum_vars.loc[:,i-1] = (x == cats[i]).astype('float')
    # assemble column names for the dummy_variables
    if isinstance(x,pd.DataFrame):
        prefix = x.columns[0]
    elif isinstance(x, pd.Series):
        prefix=x.name
    col_names = [str(prefix)+'_'+str(item) for item in cats[1:]]
    dum_vars.columns = col_names
    return dum_vars
def squeeze_categorical_dtypes(x):
    dts = copy.deepcopy(x.dtypes)
    is_cat = [str(item) == 'category' for item in x.dtypes]
    if np.all(np.equal(is_cat,False)):
        return dts
    for i in range(x.shape[1]):
        if str(x.dtypes.iloc[i])=='category':
            # check all the categories specified in the dtype are actually there!
            dt = x.dtypes.iloc[i]
            new_dt = pd.CategoricalDtype(categories=[item for item in dt.categories if item in x.iloc[:, i].to_list()],
                                ordered=dt.ordered)
            dts.iloc[i] = new_dt
    return dts

def weighted_column_mean(x,w):

    x = x*w[:,np.newaxis]
    return np.sum(x,axis=0) / np.sum(w)

def weighted_rms(x,w):
    xsq = x**2
    xsq *= w[:,np.newaxis]
    return np.sqrt(np.sum(xsq,axis=0) / np.sum(w))

def randomize_matrix(x,seed=None):
    rng = np.random.default_rng(seed)
    n_rows,n_cols = x.shape
    x = x.copy()
    for r in range(n_rows):
        x[r,:] = rng.permutation(x[r,:])
    for c in range(n_cols):
        x[:,c] = rng.permutation(x[:,c])
    return x


def animate_vectors(base_polydata,point_vectors,frame_scalars,mode='write_gif',file_name='animation.gif',fps=10,cam_view=None,off_screen=True,link_views=True,title=None,same_coordinate_system=True):
    def _morph_shape():
        [item.VisibilityOff() for item in prompt_text]
        [item.VisibilityOn() for item in title_text]
        if mode == 'write_gif':
            writing = True
            pl.open_gif(filename=file_name,fps = fps,loop=0)
        n_frames = len(frame_scalars[0])
        for f in tqdm (range(n_frames),desc="Animating…",
               ascii=False, ncols=75):
            # update all the viewers
            for x in range(n_meshes):
                if not same_coordinate_system:
                    r, c = np.unravel_index(x, [n_rows, n_cols])
                    pl.subplot(r, c)
                pl.update_coordinates(init_vertices[x]+point_vectors[x]*frame_scalars[x][f],mesh=base_polydata[x])

            time.sleep(1/fps)
            if writing:
                pl.write_frame()
        if writing:
            print('Written to '+os.path.abspath(file_name))
        [item.VisibilityOff() for item in title_text]
        [item.VisibilityOn() for item in prompt_text]
        pl.update()

    if not my_is_iterable(base_polydata):
        base_polydata = [base_polydata]
    if not my_is_iterable(point_vectors):
        point_vectors = [point_vectors]
    if not my_is_iterable(frame_scalars):
        frame_scalars = [frame_scalars]
    if cam_view is not None:
        if not my_is_iterable(cam_view):
            cam_view = [cam_view]


    if mode == 'write_gif':
        ext = '.gif'
        file_name = os.path.splitext(file_name)[0] + ext
    elif mode is not None:
        raise ValueError('mode must be \'write gif\' or None')

    if off_screen:
        if cam_view is None:
            cam_view = get_user_input_for_cam_view(base_polydata,link_views=link_views)
    ## open plotter
    # determine how many and shape of the subplots
    n_meshes = len(base_polydata)
    if not same_coordinate_system:
        n_plots = n_meshes
    else:
        n_plots = 1
    if title is None:
        title = ['']*n_plots

    if n_plots>1:
        n_rows, n_cols = _determine_multiplot_layout(n_plots)
        pl = make_plotter(off_screen=off_screen,shape=(n_rows,n_cols))
    else:
        pl = make_plotter(off_screen=off_screen)
    # make deep copies of all the polydata
    base_polydata = copy.deepcopy(base_polydata)
    # record the inital locations of the vertices
    init_vertices = [item.points for item in base_polydata]
    # add each shape in a window
    prompt_text = []
    title_text = []
    for x in range(n_meshes):
        if not same_coordinate_system:
            r,c = np.unravel_index(x,[n_rows,n_cols])
            pl.subplot(r,c)
           # pl.add_key_event('k', _morph_shape) # add key event to each subplot
        if n_plots > 1:
            prompt_text.append(pl.add_text('Press k to begin',color=[0,0,0]))
            title_text.append(pl.add_text(title[x],color=[0,0,0],position='upper_edge'))
        add_shape(base_polydata[x], pl)

        if cam_view is not None:
            if not same_coordinate_system:
                    set_camera_view(pl.camera, cam_view[x])
            else:
                    set_camera_view(pl.camera, cam_view[0])

    # set visible off
    [item.VisibilityOff() for item in title_text]

    if n_plots==1:
        prompt_text.append(pl.add_text('Press k to begin', color=[0, 0, 0]))
        title_text.append(pl.add_text(title[0], color=[0, 0, 0], position='upper_edge'))

    pl.add_key_event('k', _morph_shape)  # add key event once to the plotter

    if link_views:
        pl.link_views()

    if off_screen:
        _morph_shape()
    else:
        pl.show()

def plot_colormaps(base_polydata,point_scalars,title=None,file_name ='colormap.pdf',clim=None,cmap=None,link_cmaps=False,cam_view=None,off_screen=True,link_views=True,same_coordinate_system=True):
    def _print_to_file():
        [item.VisibilityOff() for item in prompt_text]
        [item.VisibilityOn() for item in title_text]
        pl.update()
        fn, ext = os.path.splitext(file_name)
        if ext == '':
            ext='.pdf'
        file_name_ = fn+ext

        if ext in ['.svg','.eps','.ps','.pdf','.tex']:
            pl.save_graphic(filename=file_name_)
        else:
            # try saving a screenshot via pillow
            im = PIL.Image.fromarray(pl.image)
            im.save(file_name_)
        print('Written to '+os.path.abspath(file_name_))
        [item.VisibilityOn() for item in prompt_text]
        [item.VisibilityOff() for item in title_text]
        pl.update()

    if same_coordinate_system == True:
        if not link_cmaps:
            Warning('Plotting multiple meshes in the same plotter (same_coordinate_system==True) '
                    'with different colormaps and limits (link_cmaps==False) will result in a misleading figure and'
                    'and is not recommended')


    if not my_is_iterable(base_polydata):
        base_polydata = [base_polydata]
    if not my_is_iterable(point_scalars):
        point_scalars = [point_scalars]
    if (not my_is_iterable(cam_view)) & (cam_view is not None):
            cam_view = [cam_view]
    if (not my_is_iterable(cmap)) & (cmap is not None):
        cmap =[cmap]
    if (not my_is_iterable(clim)) & (clim is not None):
        clim = [clim]

    if title is None:
        title = ['']
    if not my_is_iterable(title):
        title = [title]


    if off_screen:
        if cam_view is None:
            cam_view = get_user_input_for_cam_view(base_polydata,link_views=link_views)
    n_meshes = len(point_scalars)

    if not same_coordinate_system:
        n_plots = n_meshes
    else:
        n_plots = 1

    if n_plots>1:
        n_rows, n_cols = _determine_multiplot_layout(n_plots)
        pl = make_plotter(off_screen=off_screen,shape=(n_rows,n_cols))
    else:
        pl = make_plotter(off_screen=off_screen)

    # make deep copies of all the polydata
    if len(base_polydata)==1:
        base_polydata = [base_polydata[0].copy(True) for i in range(n_meshes)]
    else:
        base_polydata = [base_polydata[i].copy(True) for i in range(n_meshes)]

    if len(title) == 1:
        title = title*n_plots
    # work out the colormaps and color limits should be depending on the settings
    if link_cmaps:
        if clim is not None:
            if len(clim)>1:
                raise ValueError('Only specify one clim if you want to link colormaps across multiple plots (link_cmaps==True)')
        else:
            # estimate from the scalars
             _,clim= _set_colormap(np.concatenate(point_scalars), None, None)
             clim = [clim]
        # expand so there is an entry for every mesh
        clim=clim*n_meshes
        if cmap is not None:
            if len(cmap)>1:
                raise ValueError('Only specify one cmap if you want to link colormaps across multiple plots (link_cmaps==True)')
            else:
                cmap = cmap * n_meshes
        else:
            cmap,_= _set_colormap(np.concatenate(point_scalars), None, None)
            cmap = [cmap]
            cmap = cmap * n_meshes
    else: # estimate them separately if not specified
        if cmap is None:
            cmap=[_set_colormap(sc,None,None)[0] for sc in point_scalars]
        if clim is None:
            clim = [_set_colormap(sc,None,None)[1] for sc in point_scalars]


    # check there are as many cmaps and clims as plots
    if (len(clim) != n_meshes) | (len(cmap) != n_meshes):
        raise ValueError('clim and cmap should have as many entries as there are meshes, or one entry if link_cmaps==True')

    # add each shape in a window
    prompt_text = []
    title_text = []
    for x in range(n_meshes):
        if n_plots>1:
            r,c = np.unravel_index(x,[n_rows,n_cols])
            pl.subplot(r,c)
        add_shape(base_polydata[x],pl,scalars=point_scalars[x],clim=clim[x],cmap=cmap[x])

        if n_plots>1:
            add_scalar_bar(pl,title=str(x)) # title must be specified (otherwise pyvista won't make separate colorbars), the title will not be visibly rendered
            title_text.append(pl.add_text(title[x], color=[0, 0, 0], position='upper_edge'))
        pl.update_scalar_bar_range(clim[x])
        prompt_text.append(pl.add_text('Press k to save to file',color=[0,0,0]))
        if cam_view is not None:
            if not same_coordinate_system:
                    set_camera_view(pl.camera, cam_view[x])
            else:
                    set_camera_view(pl.camera, cam_view[0])
    if n_plots==1:
        add_scalar_bar(pl)
        title_text.append(pl.add_text(title[0], color=[0, 0, 0], position='upper_edge'))

    pl.add_key_event('k',_print_to_file)
    [item.VisibilityOff() for item in title_text]
    if link_views:
        pl.link_views()

    if off_screen:
        _print_to_file()
    else:
        pl.show()

def make_plotter(background_color =[255,255,255] ,**kwargs):
    kwkeys = kwargs.keys()
    pl = pv.Plotter(**kwargs)
    pl.background_color = background_color
    pl.enable_parallel_projection()
    return pl

def _determine_multiplot_layout(n_plots):
    nearest_sq = np.ceil(np.sqrt(n_plots))
    n_cols = int(nearest_sq)
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols



def add_vectors(mesh,vectors : np.ndarray | str =None,vectors_name=None,plotter=None,color_by_length=True,clim=None,cmap=None,**kwargs):
    if not isinstance(vectors,str): # then add the vectors to the polydata
        if vectors_name is None:
            appendix = 0
            while True:
                vectors_name = 'vectors_'+str(appendix)
                if vectors_name not in mesh.array_names:
                    break
                appendix +=1
        mesh[vectors_name] = vectors
    else:
        vectors = mesh[vectors]
    if color_by_length:
        scalars = np.linalg.norm(vectors, axis=1)

    # for safety make copy
    kwargs = copy.deepcopy(kwargs)
    clim = kwargs.pop('clim', None)
    cmap = kwargs.pop('cmap', None)
    if scalars is not None:
        if (clim is None) | (cmap is None):
            cmap, clim = _set_colormap(scalars, cmap, clim)  # work out some sensible default values
    mesh.set_active_vectors(vectors_name)
    if plotter is None:
        plotter = make_plotter()
    plotter.add_mesh(mesh.arrows,clim=clim,cmap=cmap,scalars=scalars,**kwargs)
    return plotter

def add_shape(mesh,plotter=None,**kwargs):
    kwargs = copy.deepcopy(kwargs)
    scalars = kwargs.pop('scalars',None)
    clim = kwargs.pop('clim',None)
    cmap = kwargs.pop('cmap',None)
   # lighting = kwargs.pop('lighting', False)
    smooth_shading = kwargs.pop('smooth_shading',True)
    if plotter is None:
        plotter = make_plotter()
    if scalars is not None:
        if (clim is None) | (cmap is None):
            cmap,clim = _set_colormap(scalars,cmap,clim) # work out some sensible default values
        plotter.add_mesh(mesh=mesh,clim=clim,cmap=cmap,show_scalar_bar=False,scalars=scalars,smooth_shading=smooth_shading,**kwargs)
    else:
        plotter.add_mesh(mesh=mesh,color = [178,178,178],smooth_shading=smooth_shading,**kwargs)

    plotter.view_xy()

    return plotter

def add_scalar_bar(plotter,**kwargs):
    kwargs = copy.deepcopy(kwargs)
    color = kwargs.pop('color',[0.,0.,.0])
    vertical = kwargs.pop('vertical',True)
    #TODO dynamically set the tick label format
    position_x = kwargs.pop('position_x',None)
    position_y = kwargs.pop('position_y',None)
    height = kwargs.pop('height',None)
    width = kwargs.pop('width',None)
    font_family = kwargs.pop('font_family','times')
    label_font_size = kwargs.pop('label_font_size',15)
    title_font_size = kwargs.pop('title_font_size',1) # make invisible
    if vertical:
        if position_x is None:
            position_x = 0.8
        if position_y is None:
            position_y = 0.1
        if height is None:
            height = 1-position_y*2
        if width is None:
            width=0.1

    if vertical is False:
        if (position_y is None):
            position_y = 0.05
        if position_x is None:
            position_x = 0.1
        if width is None:
            width = 1-position_x*2
        if height is None:
            height = 0.1

    a = plotter.add_scalar_bar(color=color, font_family=font_family, label_font_size=label_font_size,
                           title_font_size=title_font_size, vertical=vertical, position_x=position_x,
                           position_y=position_y, width=width, height=height, **kwargs)

    return plotter.add_scalar_bar(color=color,font_family=font_family,label_font_size=label_font_size,title_font_size=title_font_size,vertical=vertical,position_x=position_x,position_y=position_y,width=width,height=height,**kwargs)

def get_path_to_pinnochio_demo_face():
    """
    Get full path to the location of the Pinnochio demo face

    :return: the full path to the face
    :meta: private
    """
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data', 'pinnochio_data', 'HM_Pinnochio.obj')

def get_path_to_demo_face():
    """
    Get full path to the location of the non-Pinnochio demo face

    :return: the full path to the face
    :meta: private
    """
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data', 'pinnochio_data', 'HM.obj')

def get_path_to_simulated_faces():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data','sim_faces','faces')

def get_path_to_simulated_noses():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data','sim_faces','nose')

def get_path_to_simulated_foreheads():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data','sim_faces','forehead_orbits')
def get_path_to_simulated_metadata():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data','sim_faces','SIMPOP_Metadata.xlsx')

def get_path_to_pickled_shape_pca():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data', 'pickled_models', 'ShapePCA.p')

def get_path_to_pickled_shape_hypthesis_test():
    with importlib.resources.as_file(importlib.resources.files(__package__)) as r:
        return os.path.join(str(r), 'data', 'pickled_models', 'ShapeHypothesisTest.p')


def load_shape_hypothesis_test():
    with open(get_path_to_pickled_shape_hypthesis_test(),'rb') as f:
        obj = pickle.load(f)
    return obj
def load_shape_pca():
    with open(get_path_to_pickled_shape_pca(), 'rb') as f:
        obj = pickle.load(f)
    return obj

def _random_transformation(translate_sigma=[10, 10, 10], include_scaling=True, scale_sigma=2,
                           include_reflection=False) -> dict:
    """
    Generates a random rigid scaled transformation. The translation component will be sampled from independent Gaussian
    distributions with 0 means and user specified sigmas. The scaling component will be sampled from a lognormal distribution
    where the normal distribution has 0 mean and user-specified sigma. The rotation matrix will be an orthonormal rotation and optionally reflection matrix

    :param translate_sigma: a vector of sigmas for sampling the translation component
    :param include_scaling: if True scaling will be calculated, otherwise it will be one
    :param scale_sigma: sigma of the underlying normal distribution from which to sample the scaling factor
    :param include_reflection: if True the transformation will include a reflection component, otherwise it will be only a rotation component
    :return: a dictionary with entries:
        1.'matrix' - a 4 x 4 transformation matrix
        2.'rotation' - 3x3 rotation (and reflection) matrix
        3.'scaling' - The scaling component
        4.'translation' the translation component
    """

    # compute rotation (and reflection matrix)
    while True:
        r = ortho_group.rvs(dim=3)  # sample a 3 x 3 orthonormal matrix
        # check it satisfies the specifications
        d = np.linalg.det(r)
        if include_reflection:
            if np.allclose(d, -1.):
                break
        else:
            if np.allclose(d, 1):
                break

    # get scaling component
    if include_scaling:
        s = np.random.lognormal(mean=0, sigma=scale_sigma)
    else:
        s = 1.

    # get translation component
    t = np.random.normal(loc=[0, 0, 0], scale=translate_sigma)

    # put together transformation matrix
    m = np.identity(4)
    m[0:3, 0:3] = r * s
    m[3, 0:3] = t

    return {'matrix': m, 'rotation': r, 'translation': t, 'scaling': s}


def landmark_3d_to_2d(a: np.ndarray) -> np.ndarray:
    """Converts the array of shapes (landmark coordinates) in 'a' from n (vertices) x 3 dimensions x k observations)
     into k x 3n

     :param a: the array of landmark coordinates in the shape  n (vertices) x 3 (dimensions) x k (observations)
     if the shape is n x 3 it will be converted to n x 3 x 1
     :return: a 2d array of size k x 3n (each row contains the landmarks of observation)

     """
    if len(a.shape) not in [2, 3]:
        raise ValueError('a should be 2d or 3d')
    if a.shape[1] != 3:
        raise ValueError('Dimension 1 should be of length 3')
    if len(a.shape) == 2:
        a = a[:, :, np.newaxis]
    n = a.shape[0]
    k = a.shape[2]
    return np.reshape(a, [3 * n,k],order='F').T

def landmark_2d_to_3d(a: np.ndarray) -> np.ndarray:
    """
    Converts landmark configurations from  representation as a 2d array (k (observations) x 3n (n = number of vertices)
    to a three-dimensional array n x 3 x k. If k==1 the result will be n x 3.

    :param a: a k x 3n matrix, if array is one dimensional it will be treated as 1 x 3n
    :return: an n x 3 x k matrix (if k==1) the result will be n x 3
    """
    if len(a.shape) not in [1, 2]:
        raise ValueError('a should be 1 or 2 dimensional')
    if len(a.shape) == 1:
        a = a[np.newaxis, :]
    n3 = a.shape[1]
    k = a.shape[0]
    if n3 % 3 != 0:
        raise ValueError('Dimension 1 (or 0 for 1-dimensional arrays) should be divisible by 3')

    return np.squeeze(np.reshape(a.T, [int(n3 / 3), 3, k],order='F'))

def load_shape(path: str = None, **kwargs) -> tuple:
    """
    Loads a file containing shape data

    :param path: path to the shape to be loaded if str. If None a file dialog will be opened to select the files
    :param kwargs: keyword arguments to be passed to helpers.TriPolyData.__init__
    :return: a tuple containing an instance of helpers.TriPolyData,
    """
    if path is None:  # lauch file dia
        path = tkinter.filedialog.askopenfilename(initialdir=os.getcwd(), title='Select the mesh file to open')
    pd = TriPolyData(path, **kwargs)
    return pd, np.array(pd.points), np.array(pd.faces_array)

def load_shapes_to_array(paths: iter, n_jobs: int = 1) -> tuple:
    """
    Loads the meshes specified in 'paths' and returns their vertices in an n (vertices) x 3 x k observations array, as well as a list of corresponding helpers.TriPolyData objects
    All meshes must have the same number of vertices
    :param paths: an iterable containing the paths to the mehses to load
    :param n_jobs: the number of jobs to run concurrently. This uses joblib.Parallel rules. i.e. a positive integer specifies the number of jobs and negative integers indicates to run j + 1 + n_jobs where j is the maximum possible (e.g. -1 indictaes to use all available cores)
    :return: a tuple containing 1. the vertices in  n (vertices) x 3 x k observations array, 2. a list of helpers.TriPolyData objects
    """
    paths = [item for item in paths]
    with joblib_progress('Loading shapes',len(paths)):
        r = joblib.Parallel(n_jobs=n_jobs, )(joblib.delayed(load_shape)(path) for path in paths)
    poly, vertices, _ = zip(*r)
    try:
        vertices = np.stack(vertices, axis=2)
    except:
        Warning('Was unable to stack all shapes into a single array, please check that they all have the same number of vertices...a list of separate arrays will be returned')
    return vertices,poly


def ismember(A,B):
    isIn = [item in B for item in A]
    arB = np.array(B)
    locB = []
    for i in range(len(A)):
        if isIn[i]:
            locB.append(np.nonzero(arB==A[i])[0][0])
        else:
            locB.append(False)
    return isIn, locB

def _set_colormap(scalars,colormap,clim):
    if scalars is not None:
        if (colormap is None) | (clim is None):
            maxi = np.max(scalars)
            mini = np.min(scalars)
            if (maxi >= 0) & (mini < 0):
                if colormap is None:
                    # use a red/blue diverging colormap as default
                    colormap = 'bwr'
                if clim is None:
                    # set a sensible symmetrical clim
                    extr = np.percentile(np.abs(scalars), 97.5)
                    clim = [-extr, extr]
            else:
                if colormap is None:
                    colormap = 'plasma'
                if clim is None:
                    extr = np.percentile(np.abs(scalars), 97.5)
                    clim = [0,extr]
        return colormap, clim

def _vecs_to_scalars(vecs,direction = 'normal',poly=None):
    if direction.lower() == 'normal':
        if poly is None:
            raise ValueError('A poly data with normals is required to calculate displacements along the surface normals')
        normals = poly.point_normals
        sc = [np.sum(v * normals, axis=1) for v in vecs]
    elif direction.lower() == 'total':
        sc = [np.linalg.norm(v, axis=1) for v in vecs]
    else:
        raise ValueError('Direction should be \'normal\' or \'total\'')
    return sc
class TriPolyData(pv.PolyData):
    """
    This class adds some extra functionality and convenience to the pyvista.PolyData
    class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # triangulate the mesh if not already
        if not self.is_all_triangles:
            self.triangulate(inplace=True)

    @property
    def faces_array(self):
        """
        The faces of the polydata in shape n(faces) x 3
        """
        if len(self.faces) == 0:
            return self.faces
        else:
            f = np.reshape(self.faces, (int(len(self.faces) / 4), 4))
            return f[1, :]

    def save(self, filename, **kwargs):
        """
        Overrides the save method of pyvista.PolyData, to handle some additional formats. When the requested filetype
        is supported by the super class ('.ply', '.vtp', '.stl', '.vtk']) the super class 'save' method is used, otherwise a custom writer is searched for
        (for now the only additional format supported is obj)

        :param filename: the filename of the output file
        :param kwargs: keyword arguments to be passed to pyvista.PolyData.save is it is used.

        """
        # check if the filetype is supported by the writers in the super class
        ext = os.path.splitext(filename)[1]
        if ext in self._WRITERS.keys():
            super().save(filename, **kwargs)
        elif ext in ['.obj']:  # if not, and there is a writer defined for the sub class use it
            if ext == '.obj':
                self.write_obj(filename)
        else:
            raise ValueError('File extension ' + ext + ' is not supported')

    def write_obj(self, filename):
        # basic obj exporter for pyvista polydata since obj is not supported yet in pyvista.save
        if os.path.splitext(filename)[1] != '.obj':
            raise ValueError('Filename does not have \'.obj\' extension')
        verts = self.points.astype(str)
        verts = np.concatenate((np.tile('v', [verts.shape[0], 1]), verts), axis=1)
        faces = np.reshape(self.faces, (int(len(self.faces) / 4), 4))
        faces = faces[:, 1:] + 1  # remove first column and add 1 to the index
        faces = np.concatenate((np.tile('f', [faces.shape[0], 1]), faces.astype(str)), axis=1)
        with open(filename, 'w') as csvfile:
            writerobj = csv.writer(csvfile, delimiter=' ')
            writerobj.writerows(np.concatenate((verts, faces), axis=0))

def my_is_iterable(obj):
    """Custom check for iterability of objects, but ignriung some types"""
    try: # if no iteration is possible return False
        iter(obj)
    except:
        return False
    # ignore some custom types
    if isinstance(obj,(str,pv.DataSet,np.ndarray)):
        return False
    return True







# class ShapeReaderSaver:
#     """The class is for saving and loading meshes
#      essentially this wraps pyvista's read and save function but also allows extension to other unsupported filetypes """
#     def __init__(self):
#         self._in_file_fullpath = None
#     # property setters and getters for non-dependent properties
#     @property
#     def in_file_fullpath(self):
#         if self._in_file_fullpath is not None:
#             return self._in_file_fullpath
#
#     @in_file_fullpath.setter
#     def in_file_fullpath(self,value : str):
#         if os.path.isdir(value):
#             self._in_file_fullpath = value
#         else:
#             raise ValueError('file not found')
#
#     # property getters for dependent properties
#     @property
#     def in_file_path(self):
#         if self.in_file_fullpath is None:
#             return None
#         return os.path.split(self.in_file_fullpath)[0]
#
#     @property
#     def in_file_name(self):
#         if self.in_file_fullpath is None:
#             return None
#         return os.path.split(self.in_file_fullpath)[1]
#
#     @property
#     def in_file_ext(self):
#         if self.in_file_fullpath is None:
#             return None
#         return os.path.splitext(os.path.split(self.in_file_fullpath)[1])[1]


def validate_landmark_configuration_and_weights(p, q=None, w=None):
    """
    Implements some reusable checks of the inputs

    """
    if (len(p.shape) != 2) | (p.shape[1] != 3):
        raise ValueError("p should be n x 3")

    if q is not None:
        if len(q.shape) != 2 | q.shape[1] != 3:
            raise ValueError("q should be n x 3")
        if q.shape[0] != p.shape[0]:
            raise ValueError("p and q should be the same size")
    if w is not None:
        if len(w.shape) == 1:
            w = w[np.newaxis, :]
        if not np.all((np.array(w.shape) - [p.shape[0], 1]) == 0):
            raise ValueError("w should be the same length as the first dimension of p (i.e. the number of landmarks")
    else:
        w = np.ones([p.shape[0], 1])

    return w


def validate_vector(x, var_name, output_type='flat'):
    if len(x.shape) > 2:
        raise ValueError(var_name + ' should be at most 2-dimensional')
    if len(x.shape) == 2:
        if not any(x.shape == 1):
            raise ValueError(var_name + ' should be one dimensional or 2 dimensional with one dimension of length 1')
    return reshape_vector(x, output_type)


def reshape_vector(x, output_type):
    if len(x.shape) == 2:
        if not any(x.shape == 1):
            raise ValueError('x should be one dimensional or 2 dimensional with one dimension of length 1')
    if output_type in ['flat', 'f']:
        return np.squeeze(x)
    elif output_type in ['row', 'r']:
        return np.reshape(x, [1, len(x)])
    elif output_type in ['column', 'col', 'c']:
        return np.reshape(x, [len(x), 1])
    else:
        raise ValueError('Invalid output type')
