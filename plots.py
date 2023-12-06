import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metrics import select_threshold

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


mat_props = {'square':True, 'annot':True, 'cbar':False, 'lw':1, 
             'cmap':'Reds', 'fmt':'2.0f', 'annot_kws':{'fontsize':24}, 
             'xticklabels': ['Positivo', 'Negativo'], 
             'yticklabels': ['Positivo', 'Negativo']}


def plot_confusion_matrix(mat, ax=None, title=None, save_as=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.heatmap(mat, **mat_props)
    ax.tick_params(labelsize=19)
    ax.set_xlabel('Valor predecido', fontsize=20, fontweight='bold')
    ax.set_ylabel('Valor real', fontsize=20, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold')
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
        
    return ax

def plot_kde_predictions(ytrue, ypred, ax=None, draw_r=False, save_as=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        
    sns.kdeplot(np.array(ypred)[np.array(1-np.array(ytrue), dtype=bool)], 
                color='darkblue', fill=True, label='Negativo', 
                clim=(0, 1), ax=ax)
    sns.kdeplot(np.array(ypred)[np.array(ytrue, dtype=bool)], 
                color='darkred', fill=True, label='Positivo', 
                clim=(0, 1), ax=ax)
    
    ax.set_ylim(ax.get_ylim())
    
    if draw_r:
        ax.vlines(select_threshold(ytrue, ypred), *ax.get_ylim(), 
                  color='gray', ls='--', alpha=0.35)
    
    ax.set_xlabel('Predicción', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(0, 1)
    
    ax.legend(loc='upper left', fontsize=13)
    
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
        
        
# Grafica de aranas
def radar_factory(num_vars, frame='polygon'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta