#!/usr/bin/env python3
"""Blender Cycles GPU rendering workload.

Downloads the BMW benchmark scene and renders it with CUDA. Produces a
rendering-specific telemetry signature: sustained high GPU utilization,
no tensor cores, different memory pattern than ML workloads.

Usage:
    python blender_render.py --samples 128
    python blender_render.py --samples 128 --loops 3
"""

import argparse
import os
import signal
import subprocess
import sys
import time


SCENE_BLEND = "/tmp/blender_benchmark/complex_scene.blend"


def check_blender():
    """Check that blender is available."""
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  Blender: {result.stdout.strip().splitlines()[0]}")
    except FileNotFoundError:
        print("ERROR: blender not found. Install with: apt-get install blender")
        sys.exit(1)


def create_complex_scene():
    """Create a procedurally complex scene for GPU rendering."""
    if os.path.exists(SCENE_BLEND):
        print(f"  Scene already exists: {SCENE_BLEND}")
        return SCENE_BLEND

    os.makedirs(os.path.dirname(SCENE_BLEND), exist_ok=True)

    # Blender script to build a complex scene with many objects,
    # materials, and volumetrics to stress the GPU renderer.
    build_script = r"""
import bpy, math, random
random.seed(42)
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene

# Ground plane with glossy material
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
ground = bpy.context.object
mat_ground = bpy.data.materials.new('Ground')
mat_ground.use_nodes = True
nodes = mat_ground.node_tree.nodes
nodes.clear()
output = nodes.new('ShaderNodeOutputMaterial')
glossy = nodes.new('ShaderNodeBsdfGlossy')
glossy.inputs['Roughness'].default_value = 0.1
glossy.inputs['Color'].default_value = (0.8, 0.8, 0.9, 1)
mat_ground.node_tree.links.new(glossy.outputs[0], output.inputs[0])
ground.data.materials.append(mat_ground)

# Array of Suzanne heads with different glass/metal materials
for i in range(80):
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    z = random.uniform(0.5, 4)
    bpy.ops.mesh.primitive_monkey_add(size=0.6, location=(x, y, z))
    obj = bpy.context.object
    obj.rotation_euler = (random.uniform(0, 6.28), random.uniform(0, 6.28), random.uniform(0, 6.28))
    # Subdivision for more geometry
    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers['Subdivision'].levels = 2
    obj.modifiers['Subdivision'].render_levels = 2

    mat = bpy.data.materials.new(f'Mat_{i}')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if random.random() < 0.4:
        # Glass
        shader = nodes.new('ShaderNodeBsdfGlass')
        shader.inputs['IOR'].default_value = random.uniform(1.3, 2.0)
        shader.inputs['Color'].default_value = (random.random(), random.random(), random.random(), 1)
    else:
        # Glossy metal
        shader = nodes.new('ShaderNodeBsdfGlossy')
        shader.inputs['Roughness'].default_value = random.uniform(0.0, 0.3)
        shader.inputs['Color'].default_value = (random.random(), random.random(), random.random(), 1)
    mat.node_tree.links.new(shader.outputs[0], output.inputs[0])
    obj.data.materials.append(mat)

# Sphere array with emission
for i in range(20):
    x = random.uniform(-8, 8)
    y = random.uniform(-8, 8)
    z = random.uniform(0.3, 2)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(x, y, z))
    obj = bpy.context.object
    mat = bpy.data.materials.new(f'Emissive_{i}')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Strength'].default_value = random.uniform(5, 20)
    emit.inputs['Color'].default_value = (random.random(), random.random(), random.random(), 1)
    mat.node_tree.links.new(emit.outputs[0], output.inputs[0])
    obj.data.materials.append(mat)

# Volume cube for atmospheric scattering
bpy.ops.mesh.primitive_cube_add(size=25, location=(0, 0, 5))
vol_obj = bpy.context.object
vol_obj.display_type = 'WIRE'
mat_vol = bpy.data.materials.new('Volume')
mat_vol.use_nodes = True
nodes = mat_vol.node_tree.nodes
nodes.clear()
output = nodes.new('ShaderNodeOutputMaterial')
vol_scatter = nodes.new('ShaderNodeVolumeScatter')
vol_scatter.inputs['Density'].default_value = 0.02
vol_scatter.inputs['Color'].default_value = (0.8, 0.85, 1.0, 1)
mat_vol.node_tree.links.new(vol_scatter.outputs[0], output.inputs['Volume'])
vol_obj.data.materials.append(mat_vol)

# Camera
bpy.ops.object.camera_add(location=(12, -12, 8))
cam = bpy.context.object
cam.rotation_euler = (1.1, 0, 0.8)
scene.camera = cam

# Sun light
bpy.ops.object.light_add(type='SUN', location=(10, 10, 15))
sun = bpy.context.object
sun.data.energy = 3

# HDRI-like world (sky texture)
world = bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
output = nodes.new('ShaderNodeOutputWorld')
bg = nodes.new('ShaderNodeBackground')
sky = nodes.new('ShaderNodeTexSky')
sky.sky_type = 'NISHITA'
world.node_tree.links.new(sky.outputs[0], bg.inputs['Color'])
world.node_tree.links.new(bg.outputs[0], output.inputs['Surface'])

# Render settings
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

bpy.ops.wm.save_as_mainfile(filepath='""" + SCENE_BLEND + """')
print('SCENE CREATED')
"""

    print("  Creating complex procedural scene...")
    result = subprocess.run(
        ["blender", "--background", "--python-expr", build_script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not os.path.exists(SCENE_BLEND):
        print(f"  Scene creation failed: {result.stderr[-300:]}")
        sys.exit(1)
    print(f"  Scene saved: {SCENE_BLEND}")
    return SCENE_BLEND


def render(blend_file, samples, output_prefix="/tmp/blender_render_"):
    """Render one frame with Blender Cycles CUDA."""
    setup_script = (
        "import bpy\n"
        f"bpy.context.scene.cycles.samples = {samples}\n"
        "bpy.context.scene.cycles.use_denoising = False\n"
        "prefs = bpy.context.preferences.addons['cycles'].preferences\n"
        "prefs.compute_device_type = 'CUDA'\n"
        "prefs.get_devices('CUDA')\n"
        "for d in prefs.devices:\n"
        "    d.use = (d.type == 'CUDA')\n"
        "bpy.context.scene.cycles.device = 'GPU'\n"
    )

    cmd = [
        "blender", "-b", blend_file,
        "--python-expr", setup_script,
        "-o", output_prefix,
        "-f", "1",
    ]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def main():
    parser = argparse.ArgumentParser(description="Blender Cycles rendering workload")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument("--loops", type=int, default=3, help="Number of render passes")
    args = parser.parse_args()

    print(f"Blender Cycles rendering workload")
    print(f"  Samples: {args.samples}, Loops: {args.loops}")

    check_blender()
    blend_file = create_complex_scene()

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    wall_start = time.time()
    for i in range(args.loops):
        if stopped:
            break
        iter_start = time.time()
        print(f"  Render {i + 1}/{args.loops}...")
        result = render(blend_file, args.samples)
        iter_time = time.time() - iter_start
        if result.returncode != 0:
            print(f"  Render failed: {result.stderr[-300:]}")
            break
        print(f"  Render {i + 1} complete: {iter_time:.1f}s")

    wall_time = time.time() - wall_start
    print(f"Blender rendering complete. Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
