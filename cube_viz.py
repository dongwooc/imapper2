"""
    Script for quickly visualizing 3D brightness temperature cubes.  Requires PyQt4(?).
    
    Author: Dongwoo Chung
"""

import argparse
import numpy as np
# at some point this stuff should be ported over to VisPy
# however, VisPy doesn't actually handle transparency for overlapping volumes yet
#     so when they can actually do that, we'll start using VisPy
#     (see wiki page titled 'Tech. Transparency')
_use_vispy = False

if _use_vispy:
    import vispy.scene as vpsc
    from vispy.color import BaseColormap
else:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl

parser = argparse.ArgumentParser()
parser.add_argument('cubefiles',nargs='+') # .npz files to parse
parser.add_argument('--tmax',nargs='?',type=float,default=10.) # temperature cutoff

args = parser.parse_args()

if _use_vispy:
    class magentaCmap(BaseColormap):
        glsl_map = """
        vec4 magenta_cmap(float t) {
            return vec4(t, 0, t, t/4.2);
        }
        """
    class cyanCmap(BaseColormap):
        glsl_map = """
        vec4 cyan_cmap(float t) {
            return vec4(0, t, t, t/4.2);
        }
        """
    class yellowCmap(BaseColormap):
        glsl_map = """
        vec4 yellow_cmap(float t) {
            return vec4(t, t, 0, t/4.2);
        }
        """
    colours = (magentaCmap(),cyanCmap(),yellowCmap())
    w = vpsc.SceneCanvas('cube_viz',keys='interactive',show=True)
    v = w.central_widget.add_view()
else:
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    colours = ((255,0,255),(0,255,255),(255,255,0))

colour_idx = -1;
t_max = args.tmax;

for filename in args.cubefiles:
    colour_idx = (colour_idx + 1) % len(colours)
    print('plotting: ',filename) #,colours[colour_idx])
    cube = np.load(filename)
    x,y,z,t = [cube[i] for i in 'xyzt']
    if not(_use_vispy):
        w.opts['distance'] = (len(x)**2+len(y)**2+len(z)**2)**0.5
        t = np.rollaxis(t,-1)
    t_norm = np.clip(t,0,t_max)/t_max
    if _use_vispy:
        viz = vpsc.visuals.Volume(t_norm,parent=v.scene,method='additive')
        viz.cmap = colours[colour_idx]
    else:
        t_gl = np.empty(t.shape+(4,),dtype=np.ubyte)
        t_gl[...,0] = t_norm*colours[colour_idx][0]
        t_gl[...,1] = t_norm*colours[colour_idx][1]
        t_gl[...,2] = t_norm*colours[colour_idx][2]
        t_gl[...,3] = np.max(t_gl[...,0:3],axis=-1)/4.2
        t_gl[:,0,0] = [255,0,0,255]
        t_gl[0,:,0] = [0,255,0,255]
        t_gl[0,0,:] = [0,0,255,255]
        t_gl[:,0,-1] = [255,0,0,255]
        t_gl[0,:,-1] = [0,255,0,255]
        t_gl[0,-1,:] = [0,0,255,255]
        t_gl[:,-1,0] = [255,0,0,255]
        t_gl[-1,:,0] = [0,255,0,255]
        t_gl[-1,0,:] = [0,0,255,255]
        t_gl[:,-1,-1] = [255,0,0,255]
        t_gl[-1,:,-1] = [0,255,0,255]
        t_gl[-1,-1,:] = [0,0,255,255]
        
        v = gl.GLVolumeItem(t_gl,smooth=True,glOptions='additive')
        v.translate(-len(z)//2,-len(x)//2,-len(y)//2)
        w.addItem(v)

if _use_vispy:
    fov = 60.
    v.camera = vpsc.cameras.TurntableCamera(parent=w.scene,fov=fov)
    w.app.run()
else:
    w.show()
    app.exec_()
