# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:38:35 2023

@author: devoi
"""
from datetime import datetime
import os
import vpython as vpy
from drop_test_sim import DropTestModelCase, CustomResult
import numpy as np

vpy.set_browser(type='pyqt')

test_cond = {"path_to_params": r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data",
            "lg_type": "MLG",
            "impact_energy_kJ": 242,
             "cage_mass_t": 33.9,
             "angle_deg": 2.5}

path_to_results = os.getcwd() + r"\results_data"

dtc = DropTestModel(test_cond, 8)

result = dtc.get_result()

tt = result.t
axle_position = result.axle_position
lg_model = dtc.landing_gear

###############################################################################

SCENE_HEIGHT = 600

PLANE_LEN = 3000.0
PLANE_WIDTH = 3000.0
PLANE_HEIGHT = 2.0
WHEEL_OFFSET = 10.0
DISK_RADIUS_COEFF = 1.1
AXLE_RADIUS_COEFF = 0.5
CYL_RADIUS_COEFF = 1.5
CAGE_LENGTH = 1000.0
CAGE_WIDTH = 1000.0
CAGE_HEIGHT = 1000.0
TYRE_COLOR_GRAY_RATIO = 0.05
PLANE_COLOR_GRAY_RATIO = 0.5
TYRE_SHININESS = 0.03
METAL_SHININESS = 0.95
ARROW_LENGTH = 500.0
ARROW_WIDTH = 20.0
ORIGIN_SPHERE_RADIUS = 50.0

class Tyre3DModel():
    def __init__(self, pos, axis, param):
        self.model_3d = vpy.ring()



###############################################################################
tyre_radius = (lg_model.tyre_model.param["forcing"]["R_t"] -
               lg_model.tyre_model.param["forcing"]["R_w"])
tyre_thickness = lg_model.tyre_model.param["forcing"]["R_w"]

disk_radius = (DISK_RADIUS_COEFF *
               (tyre_radius - lg_model.tyre_model.param["forcing"]["R_w"]))
axle_radius = AXLE_RADIUS_COEFF * disk_radius

angle = lg_model.strut_model.strut.param["explicit"]["alpha"]

piston_radius = lg_model.strut_model.strut.param["forcing"]["d1"] / 2
piston_thickness = ((lg_model.strut_model.strut.param["forcing"]["d1"] -
                     lg_model.strut_model.strut.param["forcing"]["d2"]) /
                    (2 * piston_radius))
piston_length = lg_model.strut_model.strut.param["forcing"]["a"]

wheel_offset = (piston_radius + tyre_thickness + WHEEL_OFFSET)

cylinder_radius = CYL_RADIUS_COEFF * piston_radius
cylinder_thickness = 1 - 1 / CYL_RADIUS_COEFF
cylinder_length = (lg_model.strut_model.strut.param["explicit"]["L"] -
                   piston_length)

###############################################################################
for obj in vpy.scene.objects:
    obj.visible = False

vpy.scene.background = vpy.color.white
vpy.scene.center = vpy.vector(0.0, 0.5 * (cylinder_length +
                                               piston_length +
                                               tyre_radius +
                                               tyre_thickness), -200.0)
vpy.scene.forward = vpy.vector(-0.6, -0.6, -1.2)
vpy.scene.up = vpy.vector(0.0, 1.0, 0.0)
vpy.scene.fov = 0.01

vpy.scene.autoscale = True
vpy.scene.height = SCENE_HEIGHT

###############################################################################
plane = vpy.box(pos=vpy.vector(0.0, 0.0, 0.0),
                up=vpy.vector(0.0 , 1.0, 0.0),
                length=PLANE_LEN,
                height=PLANE_HEIGHT,
                width=PLANE_WIDTH,
                texture="tsupii.png")

tyre_1 = vpy.ring(pos=vpy.vector(0.0, axle_position[0], wheel_offset),
                  axis=vpy.vector(0.0, 0.0, 1.0),
                  radius=tyre_radius,
                  thickness=tyre_thickness,
                  color=vpy.color.gray(TYRE_COLOR_GRAY_RATIO),
                  shininess=TYRE_SHININESS)

tyre_2 = vpy.ring(pos=vpy.vector(0.0, axle_position[0], -wheel_offset),
                  axis=vpy.vector(0.0, 0.0, 1.0),
                  radius=tyre_radius,
                  thickness=tyre_thickness,
                  color=vpy.color.gray(TYRE_COLOR_GRAY_RATIO),
                  shininess=TYRE_SHININESS)

disk_1 = vpy.cylinder(pos=vpy.vector(0.0,
                                     axle_position[0],
                                     wheel_offset - 0.5 * tyre_thickness),
                      axis=vpy.vector(0.0, 0.0, tyre_thickness),
                      radius=disk_radius,
                      texture=vpy.textures.metal)

disk_2 = vpy.cylinder(pos=vpy.vector(0.0,
                                     axle_position[0],
                                     -wheel_offset - 0.5 * tyre_thickness),
                      axis=vpy.vector(0.0, 0.0, tyre_thickness),
                      radius=disk_radius,
                      texture=vpy.textures.metal)

axle = vpy.cylinder(pos=vpy.vector(0.0, axle_position[0], -wheel_offset),
                    axis=vpy.vector(0.0, 0.0, 2 * wheel_offset),
                    radius=piston_radius,
                    color=vpy.color.white,
                    shininess=METAL_SHININESS)

piston = vpy.extrusion(path=[vpy.vec(0.0, 0.0, 0.0),
                             vpy.vec(piston_length * np.sin(angle),
                                     piston_length * np.cos(angle),
                                     0.0)],
                       shape=vpy.shapes.circle(radius=piston_radius,
                                               thickness=piston_thickness),
                       pos=vpy.vec(0.5 * piston_length * np.sin(angle),
                                   (axle_position[0] +
                                    0.5 * piston_length * np.cos(angle)),
                                   0.0),
                       color=vpy.color.white,
                       shininess=METAL_SHININESS)

cylinder = vpy.extrusion(path=[vpy.vec(0.0, 0.0, 0.0),
                               vpy.vec(cylinder_length * np.sin(angle),
                                       cylinder_length * np.cos(angle),
                                       0.0)],
                         shape=vpy.shapes.circle(radius=cylinder_radius,
                                               thickness=cylinder_thickness),
                         pos=vpy.vec(((0.5 * cylinder_length + piston_length) *
                                    np.sin(angle)),
                                   (axle_position[0] +
                                    (0.5 * cylinder_length + piston_length) *
                                    np.cos(angle)),
                                   0.0),
                         color=vpy.color.blue,
                         shininess=METAL_SHININESS)

cage = vpy.extrusion(path=vpy.paths.arc(radius=2000,
                                        angle1=-vpy.pi/3,
                                        angle2=vpy.pi/3),
                     shape=vpy.shapes.trapezoid(pos=[0.0, 0.0],
                                                width=500, height=1500),
                     pos=vpy.vector(((cylinder_length + piston_length) *
                                     np.sin(angle)),
                                    result.y[0][0] + 0.5 * CAGE_HEIGHT,
                                    0.0),
                     axis=vpy.vector(0, -1.0, 0.0),
                     color=vpy.color.white,
                     shininess=1.0)


# vpy.box(pos=vpy.vec(((cylinder_length + piston_length) *
#                             np.sin(angle)),
#                             result[0][0] + 0.5 * CAGE_HEIGHT,
#                            0.0),
#                 up=vpy.vector(0.0, 1.0, 0.0),
#                 length=CAGE_LENGTH,
#                 height=CAGE_HEIGHT,
#                 width=CAGE_WIDTH,
#                 texture=vpy.textures.stucco,
#                 shininess=0.0)

pointer_x = vpy.arrow(pos=vpy.vector(0.5 * PLANE_LEN, 0.0, -0.5 * PLANE_WIDTH),
                      axis=vpy.vector(-ARROW_LENGTH,0.0, 0.0),
                      shaftwidth=ARROW_WIDTH,
                      color=vpy.color.red)
pointer_y = vpy.arrow(pos=vpy.vector(0.5 * PLANE_LEN, 0.0, -0.5 * PLANE_WIDTH),
                      axis=vpy.vector(0.0, ARROW_LENGTH, 0.0),
                      shaftwidth=ARROW_WIDTH,
                      color=vpy.color.green)
pointer_z = vpy.arrow(pos=vpy.vector(0.5 * PLANE_LEN, 0.0, -0.5 * PLANE_WIDTH),
                      axis=vpy.vector(0.0 ,0.0, ARROW_LENGTH),
                      shaftwidth=ARROW_WIDTH,
                      color=vpy.color.blue)
origin = vpy.sphere(pos=vpy.vector(0.5 * PLANE_LEN, 0.0, -0.5 * PLANE_WIDTH),
                    radius=ORIGIN_SPHERE_RADIUS,
                    color=vpy.color.magenta)

Lx = vpy.label(pos=pointer_x.pos + pointer_x.axis,
               text='X', xoffset=3,
               yoffset=3, space=30,
               height=16, border=4,
               box=False, line=False, opacity=0,
               font='sans')

Ly = vpy.label(pos=pointer_y.pos + pointer_y.axis,
               text='Y', xoffset=3,
               yoffset=3, space=30,
               height=16, border=4,
               box=False, line=False, opacity=0,
               font='sans')

Lz = vpy.label(pos=pointer_z.pos + pointer_z.axis,
               text='Z', xoffset=5,
               yoffset=-3, space=30,
               height=16, border=4,
               box=False, line=False, opacity=0,
               font='sans')

Lo = vpy.label(pos=origin.pos,
               text='0', xoffset=7,
               yoffset=-7, space=30,
               height=16, border=4,
               box=False, line=False, opacity=0,
               font='sans')

Li = vpy.label(pos=vpy.vector(-50, 0, 0),
               pixel_pos=True, align='left',
               text='', xoffset=50,
               yoffset=-50, space=0,
               height=14, border=10,
               box=False, line=False, opacity=0,
               font='sans')

def display_state_label(time):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    Li.text = f"""Date: {dt_string}
Type: {test_cond["lg_type"]}
Impact energy: {test_cond["impact_energy_kJ"]:3.2f} kJ
Cage Mass: {test_cond["cage_mass_t"]:3.2f} tonn
Installation angle: {test_cond["angle_deg"]:3.2f}°
State time: {time:2.2f} sec
"""

display_state_label(0.0)

#cage.visible = False
piston_compound = [tyre_1, tyre_2, disk_1, disk_2, piston, axle]
cage_compound = [cage, cylinder]
###############################################################################

piston_compound_init_posy = [body.pos.y for body in piston_compound]
cage_compound_init_posy = [body.pos.y for body in cage_compound]
running = False

num_frames = tt.shape[0]

def play_simulation(b):
    for i,t in enumerate(tt):
        if running:
            if i < tt.shape[0] - 1:
                vpy.rate(np.round(1 / (tt[i + 1] - tt[i])))
            for body, init_pos in zip(piston_compound, piston_compound_init_posy):
                body.pos.y = init_pos - axle_position[0] + axle_position[i]
            for body, init_pos in zip(cage_compound, cage_compound_init_posy):
                body.pos.y = init_pos - result.y[0][0] + result.y[0][i]
            display_state_label(t)
    b.text = "Run"

def run(b):
    global running
    running = not running
    if running:
        b.text = "Stop"
        play_simulation(b)
    else: b.text = "Run"

def set_state(slider):
    t = slider.value
    axle_pos_t = np.interp(t, tt, axle_position)
    cage_pos_t = np.interp(t, tt, result.y[0])
    for body, init_pos in zip(piston_compound, piston_compound_init_posy):
        body.pos.y = init_pos - axle_position[0] + axle_pos_t
    for body, init_pos in zip(cage_compound, cage_compound_init_posy):
        body.pos.y = init_pos - result.y[0][0] + cage_pos_t
    display_state_label(t)

button_start = vpy.button(text="Run", pos=vpy.scene.title_anchor, bind=run)
vpy.wtext(text="    State   ", pos=vpy.scene.title_anchor)
slider_time = vpy.slider(length=200, left=10, min=0, max=tt[-1],
                         pos=vpy.scene.title_anchor, bind=set_state)
text_caption = vpy.wtext(text=f"  {tt[-1]}   ", pos=vpy.scene.title_anchor)
