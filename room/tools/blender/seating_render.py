"""Render a GLB (3 copies: 3/4-front, side, 3/4-back) to PNG for visual QA.
Run: blender -b --python seating_render.py -- model.glb out.png
"""
import bpy, sys, math
from mathutils import Vector

glb, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

views = [(math.radians(-35), -0.95), (math.radians(-90), 0.0), (math.radians(180 - 35), 0.95)]
for ang, x in views:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb)
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new("root", None)
    bpy.context.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    root.location = (x * 1.0, 0, 0)
    root.rotation_euler = (0, 0, ang)

# floor
bpy.ops.mesh.primitive_plane_add(size=8)
fl = bpy.context.object
m = bpy.data.materials.new("floor"); m.use_nodes = True
m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.55, 0.52, 0.48, 1)
m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
fl.data.materials.append(m)

w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.78, 0.85, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8

def light(loc, energy, size=2.0):
    d = bpy.data.lights.new("l", "AREA"); d.energy = energy; d.size = size
    o = bpy.data.objects.new("l", d); bpy.context.collection.objects.link(o)
    o.location = loc
    direction = Vector((0, 0, 0.5)) - Vector(loc)
    o.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
light((2.5, 3, 3.5), 350)
light((-3, -1.5, 2.5), 150)

cam_d = bpy.data.cameras.new("c"); cam_d.lens = 45
cam = bpy.data.objects.new("c", cam_d); bpy.context.collection.objects.link(cam)
cam.location = (0.0, 4.6, 1.7)
cam.rotation_euler = (Vector((0, 0, 0.5)) - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y = 1500, 700
sc.render.filepath = out
sc.render.engine = "CYCLES"
sc.cycles.samples = 48
sc.cycles.use_denoising = False
sc.view_settings.view_transform = "Standard"
bpy.ops.render.render(write_still=True)
print("RENDERED", out)
