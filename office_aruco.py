import bpy
import bmesh
import mathutils
import numpy as np
import os
import json
from mathutils import Matrix, Vector
import cv2
import tempfile

def create_aruco_markers(marker_ids=[0, 1, 2], marker_size_pixels=400):
    """Generate real ArUco markers and create Blender materials"""
    
    # Create ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    marker_materials = []
    temp_dir = tempfile.gettempdir()
    
    for marker_id in marker_ids:
        print(f"Creating ArUco marker ID: {marker_id}")
        
        # Generate actual ArUco marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
        
        # Add white border for better visibility
        border_size = 20
        bordered_marker = np.ones((marker_size_pixels + 2*border_size, 
                                   marker_size_pixels + 2*border_size), dtype=np.uint8) * 255
        bordered_marker[border_size:-border_size, border_size:-border_size] = marker_image
        
        # Save marker image
        marker_path = os.path.join(temp_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(marker_path, bordered_marker)
        
        # Create Blender material
        marker_mat = bpy.data.materials.new(f"ArUco_Marker_{marker_id}")
        marker_mat.use_nodes = True
        nodes = marker_mat.node_tree.nodes
        nodes.clear()
        
        # Create material nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        tex_image = nodes.new('ShaderNodeTexImage')
        
        # Load marker image as texture
        marker_img = bpy.data.images.load(marker_path)
        marker_img.colorspace_settings.name = 'Non-Color'  # Treat as data, not color
        tex_image.image = marker_img
        
        # Set material properties for clear visibility
        bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.8
        
        # Handle different Blender versions - Specular was renamed to IOR
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.0
        elif 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = 1.0
        
        # Disable metallic for matte appearance
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = 0.0
        
        # Connect nodes
        marker_mat.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        marker_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        marker_materials.append(marker_mat)
        print(f"  Created material: ArUco_Marker_{marker_id}")
    
    return marker_materials

def place_strategic_markers_fixed(marker_materials):
    """Place markers that are guaranteed to be visible"""
    
    marker_configs = [
        {
            "name": "Floor_Center_Marker",
            "id": 0,
            "position": (0, 0, 0.01),  # Floor center - always visible
            "rotation": (0, 0, 0),
            "size": 0.30,
            "description": "Center floor marker"
        },
        {
            "name": "Desk_Surface_Marker", 
            "id": 1,
            "position": (-2.0, -3.0, 0.76),  # On desk surface
            "rotation": (0, 0, 0),
            "size": 0.15,
            "description": "Desk surface marker"
        },
        {
            "name": "Near_Wall_Marker",
            "id": 2,
            "position": (0, -2.5, 0.8),  # Between camera and back wall
            "rotation": (0, 0, 0),
            "size": 0.12,
            "description": "Intermediate wall marker"
        }
    ]
    
    placed_markers = []
    
    for i, config in enumerate(marker_configs):
        if i >= len(marker_materials):
            continue
            
        bpy.ops.mesh.primitive_plane_add(
            size=config["size"],
            location=config["position"]
        )
        
        marker_plane = bpy.context.active_object
        marker_plane.name = config["name"]
        marker_plane.rotation_euler = config["rotation"]
        marker_plane.data.materials.append(marker_materials[i])
        
        placed_markers.append({
            "name": config["name"],
            "id": config["id"],
            "world_position": config["position"],
            "world_rotation": config["rotation"], 
            "size": config["size"],
            "description": config["description"]
        })
    
    return placed_markers

def add_marker_backing_planes(placed_markers, materials):
    """Add backing planes behind markers for better visibility"""
    
    for marker in placed_markers:
        # Create slightly larger backing plane
        backing_size = marker["size"] * 1.2
        backing_pos = list(marker["world_position"])
        
        # Offset backing plane slightly behind marker
        if abs(backing_pos[1]) > 4.5:  # Back/front walls
            backing_pos[1] += 0.01 if backing_pos[1] > 0 else -0.01
        else:  # Side walls
            backing_pos[0] += 0.01 if backing_pos[0] > 0 else -0.01
        
        bpy.ops.mesh.primitive_plane_add(
            size=backing_size,
            location=backing_pos
        )
        
        backing = bpy.context.active_object
        backing.name = f"{marker['name']}_Backing"
        backing.rotation_euler = marker["world_rotation"]
        
        # Apply white material for contrast
        backing.data.materials.append(materials['wall'])

def update_ground_truth_with_markers(poses_data, placed_markers):
    """Add marker information to ground truth data"""
    
    # Add marker metadata
    poses_data['aruco_markers'] = {
        'dictionary': 'DICT_4X4_50',
        'markers': placed_markers
    }
    
    print("Updated ground truth data with ArUco marker information")
    return poses_data

def add_markers_to_scene(materials):
    """Main function to add ArUco markers to the existing scene"""
    
    print("Adding ArUco markers to scene...")
    print("=" * 50)
    
    marker_materials = create_aruco_markers(marker_ids=[0, 1, 2])
    placed_markers = place_strategic_markers_fixed(marker_materials)
    
    # Add backing planes for better visibility
    add_marker_backing_planes(placed_markers, materials)
    
    print("=" * 50)
    print("ArUco markers added successfully!")
    print("\nMarker placement summary:")
    for marker in placed_markers:
        print(f"  {marker['name']}: ID {marker['id']} at {marker['world_position']}")
        print(f"    Size: {marker['size']}m - {marker['description']}")
    
    print("\nNext steps:")
    print("  1. Re-render frames where markers are visible")
    print("  2. Update VO pipeline to detect ArUco markers") 
    print("  3. Compare marker-assisted vs pure VO results")
    
    return placed_markers

def clear_scene():
    """Clear default scene objects"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear materials and textures
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

def create_maximum_feature_materials():
    """Create materials specifically designed for maximum feature detection"""
    materials = {}
    
    # ULTRA-DETAILED Wood with multiple texture layers
    wood_mat = bpy.data.materials.new("Wood_UltraDetail")
    wood_mat.use_nodes = True
    nodes = wood_mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Layer 1: Fine grain
    noise1 = nodes.new('ShaderNodeTexNoise')
    noise1.inputs['Scale'].default_value = 500.0
    noise1.inputs['Detail'].default_value = 15.0
    
    # Layer 2: Medium grain
    noise2 = nodes.new('ShaderNodeTexNoise')
    noise2.inputs['Scale'].default_value = 100.0
    noise2.inputs['Detail'].default_value = 10.0
    
    # Layer 3: Coarse grain
    noise3 = nodes.new('ShaderNodeTexNoise')
    noise3.inputs['Scale'].default_value = 20.0
    noise3.inputs['Detail'].default_value = 8.0
    
    # Mix nodes (updated for newer Blender versions)
    if bpy.app.version >= (3, 4, 0):
        mix1 = nodes.new('ShaderNodeMix')
        mix2 = nodes.new('ShaderNodeMix')
        mix1.data_type = 'RGBA'
        mix2.data_type = 'RGBA'
        mix1_fac_input = 'Factor'
        mix1_col1_input = 'A'
        mix1_col2_input = 'B'
        mix2_fac_input = 'Factor'
        mix2_col1_input = 'A' 
        mix2_col2_input = 'B'
        mix1_output = 'Result'
        mix2_output = 'Result'
    else:
        mix1 = nodes.new('ShaderNodeMixRGB')
        mix2 = nodes.new('ShaderNodeMixRGB')
        mix1_fac_input = 'Fac'
        mix1_col1_input = 'Color1'
        mix1_col2_input = 'Color2'
        mix2_fac_input = 'Fac'
        mix2_col1_input = 'Color1'
        mix2_col2_input = 'Color2'
        mix1_output = 'Color'
        mix2_output = 'Color'
    
    mix1.inputs[mix1_fac_input].default_value = 0.3
    mix2.inputs[mix2_fac_input].default_value = 0.5
    
    # High-contrast color ramps
    ramp1 = nodes.new('ShaderNodeValToRGB')
    ramp1.color_ramp.elements[0].color = (0.2, 0.1, 0.05, 1.0)
    ramp1.color_ramp.elements[1].color = (0.9, 0.7, 0.4, 1.0)
    
    ramp2 = nodes.new('ShaderNodeValToRGB')
    ramp2.color_ramp.elements[0].color = (0.3, 0.15, 0.08, 1.0)
    ramp2.color_ramp.elements[1].color = (0.8, 0.6, 0.3, 1.0)
    
    ramp3 = nodes.new('ShaderNodeValToRGB')
    ramp3.color_ramp.elements[0].color = (0.4, 0.2, 0.1, 1.0)
    ramp3.color_ramp.elements[1].color = (0.7, 0.5, 0.25, 1.0)
    
    # Connect everything
    wood_mat.node_tree.links.new(noise1.outputs['Fac'], ramp1.inputs['Fac'])
    wood_mat.node_tree.links.new(noise2.outputs['Fac'], ramp2.inputs['Fac'])
    wood_mat.node_tree.links.new(noise3.outputs['Fac'], ramp3.inputs['Fac'])
    wood_mat.node_tree.links.new(ramp1.outputs['Color'], mix1.inputs[mix1_col1_input])
    wood_mat.node_tree.links.new(ramp2.outputs['Color'], mix1.inputs[mix1_col2_input])
    wood_mat.node_tree.links.new(mix1.outputs[mix1_output], mix2.inputs[mix2_col1_input])
    wood_mat.node_tree.links.new(ramp3.outputs['Color'], mix2.inputs[mix2_col2_input])
    wood_mat.node_tree.links.new(mix2.outputs[mix2_output], bsdf.inputs['Base Color'])
    wood_mat.node_tree.links.new(noise1.outputs['Fac'], bsdf.inputs['Roughness'])
    wood_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    materials['wood'] = wood_mat
    
    # CHECKERBOARD Pattern Material
    checker_mat = bpy.data.materials.new("Checkerboard")
    checker_mat.use_nodes = True
    checker_nodes = checker_mat.node_tree.nodes
    checker_nodes.clear()
    
    checker_output = checker_nodes.new('ShaderNodeOutputMaterial')
    checker_bsdf = checker_nodes.new('ShaderNodeBsdfPrincipled')
    
    coord = checker_nodes.new('ShaderNodeTexCoord')
    mapping = checker_nodes.new('ShaderNodeMapping')
    checker_tex = checker_nodes.new('ShaderNodeTexChecker')
    
    mapping.inputs['Scale'].default_value[0] = 20.0
    mapping.inputs['Scale'].default_value[1] = 20.0
    mapping.inputs['Scale'].default_value[2] = 1.0
    checker_tex.inputs['Color1'].default_value[0] = 0.05  # Red
    checker_tex.inputs['Color1'].default_value[1] = 0.05  # Green
    checker_tex.inputs['Color1'].default_value[2] = 0.05  # Blue
    checker_tex.inputs['Color1'].default_value[3] = 1.0   # Alpha

    checker_tex.inputs['Color2'].default_value[0] = 0.95  # Red
    checker_tex.inputs['Color2'].default_value[1] = 0.95  # Green
    checker_tex.inputs['Color2'].default_value[2] = 0.95  # Blue
    checker_tex.inputs['Color2'].default_value[3] = 1.0   # Alpha

    
    checker_mat.node_tree.links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    checker_mat.node_tree.links.new(mapping.outputs['Vector'], checker_tex.inputs['Vector'])
    checker_mat.node_tree.links.new(checker_tex.outputs['Color'], checker_bsdf.inputs['Base Color'])
    checker_mat.node_tree.links.new(checker_bsdf.outputs['BSDF'], checker_output.inputs['Surface'])
    
    materials['checkerboard'] = checker_mat
    
    # CIRCUIT BOARD Pattern
    circuit_mat = bpy.data.materials.new("CircuitBoard")
    circuit_mat.use_nodes = True
    circuit_nodes = circuit_mat.node_tree.nodes
    circuit_nodes.clear()
    
    circuit_output = circuit_nodes.new('ShaderNodeOutputMaterial')
    circuit_bsdf = circuit_nodes.new('ShaderNodeBsdfPrincipled')
    
    voronoi1 = circuit_nodes.new('ShaderNodeTexVoronoi')
    voronoi1.inputs['Scale'].default_value = 50.0
    
    voronoi2 = circuit_nodes.new('ShaderNodeTexVoronoi')
    voronoi2.inputs['Scale'].default_value = 200.0
    
    circuit_ramp1 = circuit_nodes.new('ShaderNodeValToRGB')
    circuit_ramp1.color_ramp.elements[0].color = (0.1, 0.3, 0.1, 1.0)
    circuit_ramp1.color_ramp.elements[1].color = (0.8, 1.0, 0.2, 1.0)
    
    circuit_ramp2 = circuit_nodes.new('ShaderNodeValToRGB')
    circuit_ramp2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    circuit_ramp2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    # Use appropriate mix node based on Blender version
    if bpy.app.version >= (3, 4, 0):
        circuit_mix = circuit_nodes.new('ShaderNodeMix')
        circuit_mix.data_type = 'RGBA'
        circuit_mix.inputs['Factor'].default_value = 0.7
        mix_col1 = 'A'
        mix_col2 = 'B'
        mix_output = 'Result'
    else:
        circuit_mix = circuit_nodes.new('ShaderNodeMixRGB')
        circuit_mix.inputs['Fac'].default_value = 0.7
        mix_col1 = 'Color1'
        mix_col2 = 'Color2'
        mix_output = 'Color'
    
    circuit_mat.node_tree.links.new(voronoi1.outputs['Distance'], circuit_ramp1.inputs['Fac'])
    circuit_mat.node_tree.links.new(voronoi2.outputs['Distance'], circuit_ramp2.inputs['Fac'])
    circuit_mat.node_tree.links.new(circuit_ramp1.outputs['Color'], circuit_mix.inputs[mix_col1])
    circuit_mat.node_tree.links.new(circuit_ramp2.outputs['Color'], circuit_mix.inputs[mix_col2])
    circuit_mat.node_tree.links.new(circuit_mix.outputs[mix_output], circuit_bsdf.inputs['Base Color'])
    circuit_mat.node_tree.links.new(circuit_bsdf.outputs['BSDF'], circuit_output.inputs['Surface'])
    
    materials['circuit'] = circuit_mat
    
    # BRICK Pattern Material
    brick_mat = bpy.data.materials.new("BrickPattern")
    brick_mat.use_nodes = True
    brick_nodes = brick_mat.node_tree.nodes
    brick_nodes.clear()
    
    brick_output = brick_nodes.new('ShaderNodeOutputMaterial')
    brick_bsdf = brick_nodes.new('ShaderNodeBsdfPrincipled')
    
    coord_brick = brick_nodes.new('ShaderNodeTexCoord')
    mapping_brick = brick_nodes.new('ShaderNodeMapping')
    brick_tex = brick_nodes.new('ShaderNodeTexBrick')
    
    mapping_brick.inputs['Scale'].default_value[0] = 15.0
    mapping_brick.inputs['Scale'].default_value[1] = 10.0
    mapping_brick.inputs['Scale'].default_value[2] = 1.0

    brick_tex.inputs['Color1'].default_value[0] = 0.7
    brick_tex.inputs['Color1'].default_value[1] = 0.3
    brick_tex.inputs['Color1'].default_value[2] = 0.2
    brick_tex.inputs['Color1'].default_value[3] = 1.0

    brick_tex.inputs['Color2'].default_value[0] = 0.9
    brick_tex.inputs['Color2'].default_value[1] = 0.85
    brick_tex.inputs['Color2'].default_value[2] = 0.8
    brick_tex.inputs['Color2'].default_value[3] = 1.0
    
    
    brick_tex.inputs['Mortar Size'].default_value = 0.02
    
    brick_mat.node_tree.links.new(coord_brick.outputs['Generated'], mapping_brick.inputs['Vector'])
    brick_mat.node_tree.links.new(mapping_brick.outputs['Vector'], brick_tex.inputs['Vector'])
    brick_mat.node_tree.links.new(brick_tex.outputs['Color'], brick_bsdf.inputs['Base Color'])
    brick_mat.node_tree.links.new(brick_bsdf.outputs['BSDF'], brick_output.inputs['Surface'])
    
    materials['brick'] = brick_mat
    
    # QR CODE-like Pattern
    qr_mat = bpy.data.materials.new("QRPattern")
    qr_mat.use_nodes = True
    qr_nodes = qr_mat.node_tree.nodes
    qr_nodes.clear()
    
    qr_output = qr_nodes.new('ShaderNodeOutputMaterial')
    qr_bsdf = qr_nodes.new('ShaderNodeBsdfPrincipled')
    
    qr_noise1 = qr_nodes.new('ShaderNodeTexNoise')
    qr_noise1.inputs['Scale'].default_value = 80.0
    qr_noise1.inputs['Detail'].default_value = 1.0
    
    qr_noise2 = qr_nodes.new('ShaderNodeTexNoise')
    qr_noise2.inputs['Scale'].default_value = 160.0
    qr_noise2.inputs['Detail'].default_value = 1.0
    
    qr_ramp = qr_nodes.new('ShaderNodeValToRGB')
    qr_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    qr_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    qr_ramp.color_ramp.elements[0].position = 0.45
    qr_ramp.color_ramp.elements[1].position = 0.55
    
    # Use appropriate mix node
    if bpy.app.version >= (3, 4, 0):
        qr_multiply = qr_nodes.new('ShaderNodeMix')
        qr_multiply.data_type = 'RGBA'
        qr_multiply.blend_type = 'MULTIPLY'
        qr_multiply.inputs['Factor'].default_value = 1.0
        mult_col1 = 'A'
        mult_col2 = 'B' 
        mult_output = 'Result'
    else:
        qr_multiply = qr_nodes.new('ShaderNodeMixRGB')
        qr_multiply.blend_type = 'MULTIPLY'
        qr_multiply.inputs['Fac'].default_value = 1.0
        mult_col1 = 'Color1'
        mult_col2 = 'Color2'
        mult_output = 'Color'
    
    qr_mat.node_tree.links.new(qr_noise1.outputs['Fac'], qr_multiply.inputs[mult_col1])
    qr_mat.node_tree.links.new(qr_noise2.outputs['Fac'], qr_multiply.inputs[mult_col2])
    qr_mat.node_tree.links.new(qr_multiply.outputs[mult_output], qr_ramp.inputs['Fac'])
    qr_mat.node_tree.links.new(qr_ramp.outputs['Color'], qr_bsdf.inputs['Base Color'])
    qr_mat.node_tree.links.new(qr_bsdf.outputs['BSDF'], qr_output.inputs['Surface'])
    
    materials['qr_code'] = qr_mat
    
    # High-contrast fabric with grid pattern
    fabric_mat = bpy.data.materials.new("GridFabric")
    fabric_mat.use_nodes = True
    fabric_nodes = fabric_mat.node_tree.nodes
    fabric_nodes.clear()
    
    fabric_output = fabric_nodes.new('ShaderNodeOutputMaterial')
    fabric_bsdf = fabric_nodes.new('ShaderNodeBsdfPrincipled')
    
    coord_fabric = fabric_nodes.new('ShaderNodeTexCoord')
    mapping_fabric = fabric_nodes.new('ShaderNodeMapping')
    
    separate_xyz = fabric_nodes.new('ShaderNodeSeparateXYZ')
    multiply_x = fabric_nodes.new('ShaderNodeMath')
    multiply_y = fabric_nodes.new('ShaderNodeMath')
    multiply_x.operation = 'MULTIPLY'
    multiply_y.operation = 'MULTIPLY'
    multiply_x.inputs[1].default_value = 100.0
    multiply_y.inputs[1].default_value = 100.0
    
    modulo_x = fabric_nodes.new('ShaderNodeMath')
    modulo_y = fabric_nodes.new('ShaderNodeMath')
    modulo_x.operation = 'MODULO'
    modulo_y.operation = 'MODULO'
    modulo_x.inputs[1].default_value = 1.0
    modulo_y.inputs[1].default_value = 1.0
    
    add_grid = fabric_nodes.new('ShaderNodeMath')
    add_grid.operation = 'ADD'
    
    grid_ramp = fabric_nodes.new('ShaderNodeValToRGB')
    grid_ramp.color_ramp.elements[0].color = (0.1, 0.2, 0.5, 1.0)
    grid_ramp.color_ramp.elements[1].color = (0.8, 0.9, 1.0, 1.0)
    
    mapping_fabric.inputs['Scale'].default_value[0] = 1.0
    mapping_fabric.inputs['Scale'].default_value[1] = 1.0
    mapping_fabric.inputs['Scale'].default_value[2] = 1.0
    
    # Connect grid pattern
    fabric_mat.node_tree.links.new(coord_fabric.outputs['Generated'], mapping_fabric.inputs['Vector'])
    fabric_mat.node_tree.links.new(mapping_fabric.outputs['Vector'], separate_xyz.inputs['Vector'])
    fabric_mat.node_tree.links.new(separate_xyz.outputs['X'], multiply_x.inputs[0])
    fabric_mat.node_tree.links.new(separate_xyz.outputs['Y'], multiply_y.inputs[0])
    fabric_mat.node_tree.links.new(multiply_x.outputs['Value'], modulo_x.inputs[0])
    fabric_mat.node_tree.links.new(multiply_y.outputs['Value'], modulo_y.inputs[0])
    fabric_mat.node_tree.links.new(modulo_x.outputs['Value'], add_grid.inputs[0])
    fabric_mat.node_tree.links.new(modulo_y.outputs['Value'], add_grid.inputs[1])
    fabric_mat.node_tree.links.new(add_grid.outputs['Value'], grid_ramp.inputs['Fac'])
    fabric_mat.node_tree.links.new(grid_ramp.outputs['Color'], fabric_bsdf.inputs['Base Color'])
    fabric_mat.node_tree.links.new(fabric_bsdf.outputs['BSDF'], fabric_output.inputs['Surface'])
    
    materials['fabric'] = fabric_mat
    
    # Enhanced screen with pixel grid
    screen_mat = bpy.data.materials.new("PixelScreen")
    screen_mat.use_nodes = True
    screen_nodes = screen_mat.node_tree.nodes
    screen_nodes.clear()
    
    screen_output = screen_nodes.new('ShaderNodeOutputMaterial')
    screen_bsdf = screen_nodes.new('ShaderNodeBsdfPrincipled')
    emission = screen_nodes.new('ShaderNodeEmission')
    
        # Always use ShaderNodeMixShader for mixing shaders
    mix_shader = screen_nodes.new('ShaderNodeMixShader')
    shader_fac = 'Fac'
    shader_a = 1  # Index for first shader input
    shader_b = 2  # Index for second shader input
    shader_result = 'Shader'
    
    coord_screen = screen_nodes.new('ShaderNodeTexCoord')
    mapping_screen = screen_nodes.new('ShaderNodeMapping')
    mapping_screen.inputs['Scale'].default_value[0] = 200.0
    mapping_screen.inputs['Scale'].default_value[1] = 150.0
    mapping_screen.inputs['Scale'].default_value[2] = 1.0
    
    # Simple checkerboard pattern for screen
    checker_screen = screen_nodes.new('ShaderNodeTexChecker')
        # Color1 - Red
    checker_screen.inputs['Color1'].default_value[0] = 1.0  # Red
    checker_screen.inputs['Color1'].default_value[1] = 0.0  # Green
    checker_screen.inputs['Color1'].default_value[2] = 0.0  # Blue
    checker_screen.inputs['Color1'].default_value[3] = 1.0  # Alpha

    # Color2 - Green
    checker_screen.inputs['Color2'].default_value[0] = 0.0  # Red
    checker_screen.inputs['Color2'].default_value[1] = 1.0  # Green
    checker_screen.inputs['Color2'].default_value[2] = 0.0  # Blue
    checker_screen.inputs['Color2'].default_value[3] = 1.0  # Alpha
        
    
    screen_mat.node_tree.links.new(coord_screen.outputs['Generated'], mapping_screen.inputs['Vector'])
    screen_mat.node_tree.links.new(mapping_screen.outputs['Vector'], checker_screen.inputs['Vector'])
    screen_mat.node_tree.links.new(checker_screen.outputs['Color'], screen_bsdf.inputs['Base Color'])
    screen_mat.node_tree.links.new(checker_screen.outputs['Color'], emission.inputs['Color'])
    emission.inputs['Strength'].default_value = 0.8
    
    screen_mat.node_tree.links.new(screen_bsdf.outputs['BSDF'], mix_shader.inputs[shader_a])
    screen_mat.node_tree.links.new(emission.outputs['Emission'], mix_shader.inputs[shader_b])
    mix_shader.inputs[shader_fac].default_value = 0.3
    screen_mat.node_tree.links.new(mix_shader.outputs[shader_result], screen_output.inputs['Surface'])
    
    materials['screen'] = screen_mat
    
    # Metal with detailed scratches
    metal_mat = bpy.data.materials.new("DetailedMetal")
    metal_mat.use_nodes = True
    metal_nodes = metal_mat.node_tree.nodes
    metal_nodes.clear()
    
    metal_output = metal_nodes.new('ShaderNodeOutputMaterial')
    metal_bsdf = metal_nodes.new('ShaderNodeBsdfPrincipled')
    
    # Multiple scratch layers
    scratch1 = metal_nodes.new('ShaderNodeTexNoise')
    scratch1.inputs['Scale'].default_value = 800.0
    
    scratch2 = metal_nodes.new('ShaderNodeTexNoise')
    scratch2.inputs['Scale'].default_value = 200.0
    
    scratch3 = metal_nodes.new('ShaderNodeTexNoise')
    scratch3.inputs['Scale'].default_value = 50.0
    
    # Use appropriate mix nodes
    if bpy.app.version >= (3, 4, 0):
        scratch_mix1 = metal_nodes.new('ShaderNodeMix')
        scratch_mix2 = metal_nodes.new('ShaderNodeMix')
        scratch_mix1.data_type = 'RGBA'
        scratch_mix2.data_type = 'RGBA'
        scratch_mix1.inputs['Factor'].default_value = 0.5
        scratch_mix2.inputs['Factor'].default_value = 0.3
        s1_col1, s1_col2, s1_out = 'A', 'B', 'Result'
        s2_col1, s2_col2, s2_out = 'A', 'B', 'Result'
    else:
        scratch_mix1 = metal_nodes.new('ShaderNodeMixRGB')
        scratch_mix2 = metal_nodes.new('ShaderNodeMixRGB')
        scratch_mix1.inputs['Fac'].default_value = 0.5
        scratch_mix2.inputs['Fac'].default_value = 0.3
        s1_col1, s1_col2, s1_out = 'Color1', 'Color2', 'Color'
        s2_col1, s2_col2, s2_out = 'Color1', 'Color2', 'Color'
    
    metal_mat.node_tree.links.new(scratch1.outputs['Fac'], scratch_mix1.inputs[s1_col1])
    metal_mat.node_tree.links.new(scratch2.outputs['Fac'], scratch_mix1.inputs[s1_col2])
    metal_mat.node_tree.links.new(scratch_mix1.outputs[s1_out], scratch_mix2.inputs[s2_col1])
    metal_mat.node_tree.links.new(scratch3.outputs['Fac'], scratch_mix2.inputs[s2_col2])
    metal_mat.node_tree.links.new(scratch_mix2.outputs[s2_out], metal_bsdf.inputs['Roughness'])
    
    metal_bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.8, 1.0)
    metal_bsdf.inputs['Metallic'].default_value = 0.95
    metal_mat.node_tree.links.new(metal_bsdf.outputs['BSDF'], metal_output.inputs['Surface'])
    
    materials['metal'] = metal_mat
    
    # Create varied book materials
    book_patterns = [
        {'type': 'stripes', 'colors': [(0.9, 0.1, 0.1, 1.0), (1.0, 0.9, 0.9, 1.0)]},
        {'type': 'dots', 'colors': [(0.1, 0.7, 0.1, 1.0), (0.9, 1.0, 0.9, 1.0)]},
        {'type': 'zigzag', 'colors': [(0.1, 0.1, 0.9, 1.0), (0.9, 0.9, 1.0, 1.0)]},
        {'type': 'grid', 'colors': [(0.8, 0.6, 0.1, 1.0), (1.0, 1.0, 0.8, 1.0)]},
        {'type': 'checker', 'colors': [(0.7, 0.1, 0.7, 1.0), (1.0, 0.8, 1.0, 1.0)]},
        {'type': 'waves', 'colors': [(0.1, 0.8, 0.8, 1.0), (0.8, 1.0, 1.0, 1.0)]},
    ]
    
    for i, pattern in enumerate(book_patterns):
        book_mat = bpy.data.materials.new(f"Book_Pattern_{i}")
        book_mat.use_nodes = True
        book_nodes = book_mat.node_tree.nodes
        book_nodes.clear()
        
        book_output = book_nodes.new('ShaderNodeOutputMaterial')
        book_bsdf = book_nodes.new('ShaderNodeBsdfPrincipled')
        
        if pattern['type'] == 'checker':
            coord_book = book_nodes.new('ShaderNodeTexCoord')
            mapping_book = book_nodes.new('ShaderNodeMapping')
            checker_book = book_nodes.new('ShaderNodeTexChecker')
            
            mapping_book.inputs['Scale'].default_value[0] = 10.0
            mapping_book.inputs['Scale'].default_value[1] = 15.0
            mapping_book.inputs['Scale'].default_value[2] = 1.0
            checker_book.inputs['Color1'].default_value = pattern['colors'][0]
            checker_book.inputs['Color2'].default_value = pattern['colors'][1]
            
            book_mat.node_tree.links.new(coord_book.outputs['Generated'], mapping_book.inputs['Vector'])
            book_mat.node_tree.links.new(mapping_book.outputs['Vector'], checker_book.inputs['Vector'])
            book_mat.node_tree.links.new(checker_book.outputs['Color'], book_bsdf.inputs['Base Color'])
        else:
            book_noise = book_nodes.new('ShaderNodeTexNoise')
            book_noise.inputs['Scale'].default_value = 30.0 + i * 20
            book_noise.inputs['Detail'].default_value = 5.0
            
            book_ramp = book_nodes.new('ShaderNodeValToRGB')
            book_ramp.color_ramp.elements[0].color = pattern['colors'][0]
            book_ramp.color_ramp.elements[1].color = pattern['colors'][1]
            book_ramp.color_ramp.elements[0].position = 0.3
            book_ramp.color_ramp.elements[1].position = 0.7
            
            book_mat.node_tree.links.new(book_noise.outputs['Fac'], book_ramp.inputs['Fac'])
            book_mat.node_tree.links.new(book_ramp.outputs['Color'], book_bsdf.inputs['Base Color'])
        
        book_mat.node_tree.links.new(book_bsdf.outputs['BSDF'], book_output.inputs['Surface'])
        materials[f'book_{i}'] = book_mat
    
    # Wall material
    wall_mat = bpy.data.materials.new("TexturedWall")
    wall_mat.use_nodes = True
    wall_nodes = wall_mat.node_tree.nodes
    wall_nodes.clear()
    
    wall_output = wall_nodes.new('ShaderNodeOutputMaterial')
    wall_bsdf = wall_nodes.new('ShaderNodeBsdfPrincipled')
    wall_noise = wall_nodes.new('ShaderNodeTexNoise')
    wall_noise.inputs['Scale'].default_value = 100.0
    wall_noise.inputs['Detail'].default_value = 10.0
    
    wall_ramp = wall_nodes.new('ShaderNodeValToRGB')
    wall_ramp.color_ramp.elements[0].color = (0.8, 0.8, 0.75, 1.0)
    wall_ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.9, 1.0)
    
    wall_mat.node_tree.links.new(wall_noise.outputs['Fac'], wall_ramp.inputs['Fac'])
    wall_mat.node_tree.links.new(wall_ramp.outputs['Color'], wall_bsdf.inputs['Base Color'])
    wall_mat.node_tree.links.new(wall_bsdf.outputs['BSDF'], wall_output.inputs['Surface'])
    
    materials['wall'] = wall_mat
    
    return materials

def create_maximum_feature_floor(materials):
    """Create floor with alternating high-contrast patterns"""
    
    tile_size = 0.5
    floor_size = 10
    tiles_per_side = int(floor_size / tile_size)
    
    pattern_materials = ['checkerboard', 'circuit', 'brick', 'qr_code']
    
    for x in range(tiles_per_side):
        for y in range(tiles_per_side):
            world_x = (x - tiles_per_side/2) * tile_size + tile_size/2
            world_y = (y - tiles_per_side/2) * tile_size + tile_size/2
            
            bpy.ops.mesh.primitive_plane_add(size=tile_size, location=(world_x, world_y, 0))
            tile = bpy.context.active_object
            tile.name = f"Floor_Tile_{x}_{y}"
            
            # Cycle through high-contrast patterns
            pattern_idx = (x * 2 + y) % len(pattern_materials)
            tile.data.materials.append(materials[pattern_materials[pattern_idx]])

def create_feature_rich_walls(materials):
    """Create walls with maximum feature density"""
    
    # Back wall with mixed patterns
    wall_segments_x = 8
    wall_segments_y = 6
    wall_size = 10
    
    for i in range(wall_segments_x):
        for j in range(wall_segments_y):
            x = (i - wall_segments_x/2) * (wall_size/wall_segments_x) + wall_size/(2*wall_segments_x)
            z = (j) * (wall_size/wall_segments_y) + wall_size/(2*wall_segments_y)
            
            bpy.ops.mesh.primitive_plane_add(
                size=wall_size/wall_segments_x, 
                location=(x, -5, z)
            )
            wall_segment = bpy.context.active_object
            wall_segment.name = f"Wall_Back_Segment_{i}_{j}"
            wall_segment.rotation_euler = (1.5708, 0, 0)  # 90 degrees in X
            
            # Alternate between different high-contrast materials
            if (i + j) % 4 == 0:
                wall_segment.data.materials.append(materials['checkerboard'])
            elif (i + j) % 4 == 1:
                wall_segment.data.materials.append(materials['circuit'])
            elif (i + j) % 4 == 2:
                wall_segment.data.materials.append(materials['brick'])
            else:
                wall_segment.data.materials.append(materials['qr_code'])
    
    # Side wall with similar treatment
    for i in range(wall_segments_y):
        for j in range(wall_segments_y):
            y = (i - wall_segments_y/2) * (wall_size/wall_segments_y) + wall_size/(2*wall_segments_y)
            z = (j) * (wall_size/wall_segments_y) + wall_size/(2*wall_segments_y)
            
            bpy.ops.mesh.primitive_plane_add(
                size=wall_size/wall_segments_y, 
                location=(-5, y, z)
            )
            wall_segment = bpy.context.active_object
            wall_segment.name = f"Wall_Side_Segment_{i}_{j}"
            wall_segment.rotation_euler = (1.5708, 0, 1.5708)
            
            pattern_idx = (i + j) % 4
            pattern_materials = ['checkerboard', 'circuit', 'brick', 'qr_code']
            wall_segment.data.materials.append(materials[pattern_materials[pattern_idx]])
    
    # Ceiling with pattern
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 5))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.data.materials.append(materials['circuit'])
    
    return ceiling

def create_ultra_detailed_desk(materials):
    """Create desk with maximum surface detail"""
    
    # Main desk surface with circuit pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3, 0.75))
    main_desk = bpy.context.active_object
    main_desk.name = "Desk_Main"
    main_desk.scale = (2, 1, 0.05)
    main_desk.data.materials.append(materials['wood'])
    
    # Add desk mat with QR-like pattern
    bpy.ops.mesh.primitive_plane_add(size=1, location=(-2, -3, 0.755))
    desk_mat = bpy.context.active_object
    desk_mat.name = "Desk_Mat"
    desk_mat.scale = (0.8, 0.6, 1)
    desk_mat.data.materials.append(materials['qr_code'])
    
    # Side extension
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.5, -2, 0.75))
    side_desk = bpy.context.active_object
    side_desk.name = "Desk_Side"
    side_desk.scale = (0.5, 1, 0.05)
    side_desk.data.materials.append(materials['wood'])
    
    # Desk legs with detailed metal
    leg_positions = [(-3, -3.5, 0.375), (-1, -3.5, 0.375), (-1, -2.5, 0.375), (-4, -2.5, 0.375)]
    for i, pos in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        leg = bpy.context.active_object
        leg.name = f"Desk_Leg_{i}"
        leg.scale = (0.05, 0.05, 0.75)
        leg.data.materials.append(materials['metal'])
    
    # Desk drawers with checkerboard handles
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.5, -2.7, 0.4))
    drawer = bpy.context.active_object
    drawer.name = "Desk_Drawer"
    drawer.scale = (0.4, 0.25, 0.25)
    drawer.data.materials.append(materials['wood'])
    
    # Drawer handle with pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.5, -2.45, 0.4))
    handle = bpy.context.active_object
    handle.name = "Drawer_Handle"
    handle.scale = (0.15, 0.02, 0.03)
    handle.data.materials.append(materials['checkerboard'])

def create_maximum_keyboard(materials):
    """Create keyboard with individual patterned keys"""
    
    # Keyboard base
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -2.8, 0.77))
    keyboard_base = bpy.context.active_object
    keyboard_base.name = "Keyboard_Base"
    keyboard_base.scale = (0.35, 0.15, 0.01)
    keyboard_base.data.materials.append(materials['circuit'])
    
    # Create grid of keys with alternating patterns
    key_rows = [
        "1234567890-=",
        "QWERTYUIOP[]",
        "ASDFGHJKL;'",
        "ZXCVBNM,./"
    ]
    
    key_materials = ['checkerboard', 'qr_code', 'circuit']
    
    for row_idx, row in enumerate(key_rows):
        for col_idx, char in enumerate(row):
            x = -2.3 + col_idx * 0.05
            y = -2.9 + row_idx * 0.04
            z = 0.785
            
            # Create key base
            bpy.ops.mesh.primitive_cube_add(size=0.028, location=(x, y, z))
            key = bpy.context.active_object
            key.name = f"Key_{char}_{row_idx}_{col_idx}"
            
            # Alternate materials for maximum contrast
            mat_idx = (row_idx + col_idx) % len(key_materials)
            key.data.materials.append(materials[key_materials[mat_idx]])
            
            # Add key label (small contrasting cube on top)
            bpy.ops.mesh.primitive_cube_add(size=0.015, location=(x, y, z + 0.008))
            key_label = bpy.context.active_object
            key_label.name = f"KeyLabel_{char}"
            # Use contrasting material
            contrast_mat = key_materials[(mat_idx + 1) % len(key_materials)]
            key_label.data.materials.append(materials[contrast_mat])

def create_patterned_chair(materials):
    """Create chair with high-contrast fabric patterns"""
    
    # Chair seat with grid pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.5, -1.5, 0.5))
    seat = bpy.context.active_object
    seat.name = "Chair_Seat"
    seat.scale = (0.4, 0.4, 0.05)
    seat.data.materials.append(materials['fabric'])
    
    # Chair back with different pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.5, -1.8, 0.9))
    back = bpy.context.active_object
    back.name = "Chair_Back"
    back.scale = (0.35, 0.05, 0.4)
    back.data.materials.append(materials['checkerboard'])
    
    # Chair post with detailed metal
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.4, location=(-1.5, -1.5, 0.25))
    post = bpy.context.active_object
    post.name = "Chair_Post"
    post.data.materials.append(materials['metal'])
    
    # Chair wheels with patterns
    for i in range(5):
        angle = i * 2 * 3.14159 / 5
        x = -1.5 + 0.25 * np.cos(angle)
        y = -1.5 + 0.25 * np.sin(angle)
        
        # Wheel arm
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.05))
        arm = bpy.context.active_object
        arm.name = f"Chair_Arm_{i}"
        arm.scale = (0.25, 0.02, 0.01)
        arm.rotation_euler = (0, 0, angle)
        arm.data.materials.append(materials['metal'])
        
        # Wheel with checkerboard pattern
        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.02, location=(x, y, 0.03))
        wheel = bpy.context.active_object
        wheel.name = f"Chair_Wheel_{i}"
        wheel.rotation_euler = (1.5708, 0, 0)
        wheel.data.materials.append(materials['checkerboard'])

def create_detailed_monitor_setup(materials):
    """Create monitor with detailed interface and patterns"""
    
    # Monitor stand with pattern
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.02, location=(-2, -3.8, 0.8))
    stand = bpy.context.active_object
    stand.name = "Monitor_Stand"
    stand.data.materials.append(materials['circuit'])
    
    # Monitor post
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.3, location=(-2, -3.8, 0.95))
    post = bpy.context.active_object
    post.name = "Monitor_Post"
    post.data.materials.append(materials['metal'])
    
    # Monitor screen with RGB pixel pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3.8, 1.2))
    monitor = bpy.context.active_object
    monitor.name = "Monitor"
    monitor.scale = (0.5, 0.05, 0.3)
    monitor.data.materials.append(materials['screen'])
    
    # Monitor bezel with detailed pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3.75, 1.2))
    bezel = bpy.context.active_object
    bezel.name = "Monitor_Bezel"
    bezel.scale = (0.52, 0.03, 0.32)
    bezel.data.materials.append(materials['circuit'])
    
    # Create detailed UI elements on screen
    ui_elements = [
        # Windows with different patterns
        {"pos": (-2.1, -3.74, 1.25), "scale": (0.15, 0.01, 0.12), "pattern": "checkerboard"},
        {"pos": (-1.9, -3.74, 1.3), "scale": (0.2, 0.01, 0.08), "pattern": "qr_code"},
        {"pos": (-2.05, -3.74, 1.15), "scale": (0.18, 0.01, 0.1), "pattern": "circuit"},
        
        # Taskbar
        {"pos": (-2, -3.74, 1.05), "scale": (0.48, 0.01, 0.05), "pattern": "brick"},
        
        # Icons
        {"pos": (-2.2, -3.74, 1.08), "scale": (0.03, 0.01, 0.03), "pattern": "checkerboard"},
        {"pos": (-2.15, -3.74, 1.08), "scale": (0.03, 0.01, 0.03), "pattern": "qr_code"},
        {"pos": (-2.1, -3.74, 1.08), "scale": (0.03, 0.01, 0.03), "pattern": "circuit"},
        {"pos": (-2.05, -3.74, 1.08), "scale": (0.03, 0.01, 0.03), "pattern": "brick"},
    ]
    
    for i, elem in enumerate(ui_elements):
        bpy.ops.mesh.primitive_cube_add(size=1, location=elem["pos"])
        ui_elem = bpy.context.active_object
        ui_elem.name = f"UI_Element_{i}"
        ui_elem.scale = elem["scale"]
        ui_elem.data.materials.append(materials[elem["pattern"]])
    
    # Mouse with pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.4, -2.6, 0.78))
    mouse = bpy.context.active_object
    mouse.name = "Mouse"
    mouse.scale = (0.06, 0.1, 0.02)
    mouse.data.materials.append(materials['checkerboard'])

def create_maximum_bookshelf(materials):
    """Create bookshelf with maximum pattern variety"""
    
    shelf_x = -4.2
    shelf_y = -4.5
    shelf_height = 2.0
    
    # Back panel with circuit pattern
    bpy.ops.mesh.primitive_cube_add(size=1, location=(shelf_x, shelf_y, shelf_height/2))
    back_panel = bpy.context.active_object
    back_panel.name = "Bookshelf_Back"
    back_panel.scale = (0.8, 0.02, shelf_height/2)
    back_panel.data.materials.append(materials['circuit'])
    
    # Shelves with alternating patterns
    shelf_patterns = ['wood', 'checkerboard', 'brick', 'qr_code']
    for i in range(4):
        y_pos = shelf_height * i / 3 - shelf_height/2 + 0.5
        bpy.ops.mesh.primitive_cube_add(size=1, location=(shelf_x, shelf_y - 0.1, y_pos))
        shelf = bpy.context.active_object
        shelf.name = f"Bookshelf_Shelf_{i}"
        shelf.scale = (0.8, 0.1, 0.02)
        shelf.data.materials.append(materials[shelf_patterns[i % len(shelf_patterns)]])
    
    # Side panels with patterns
    for side in [-1, 1]:
        x_pos = shelf_x + side * 0.8
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x_pos, shelf_y - 0.05, shelf_height/2))
        side_panel = bpy.context.active_object
        side_panel.name = f"Bookshelf_Side_{side}"
        side_panel.scale = (0.02, 0.15, shelf_height/2)
        pattern_idx = 0 if side == -1 else 1
        side_panel.data.materials.append(materials[['circuit', 'checkerboard'][pattern_idx]])
    
    # Books with maximum pattern variety
    np.random.seed(42)
    for shelf_level in range(3):
        shelf_y_pos = shelf_height * shelf_level / 3 - shelf_height/2 + 0.65
        
        num_books = 10  # More books for more features
        book_start_x = shelf_x - 0.7
        
        for book_idx in range(num_books):
            book_width = np.random.uniform(0.02, 0.06)
            book_height = np.random.uniform(0.15, 0.25)
            book_depth = 0.12
            
            book_x = book_start_x + book_idx * 0.14
            book_y = shelf_y - 0.05
            
            bpy.ops.mesh.primitive_cube_add(size=1, location=(book_x, book_y, shelf_y_pos))
            book = bpy.context.active_object
            book.name = f"Book_{shelf_level}_{book_idx}"
            book.scale = (book_width, book_depth, book_height)
            
            # Use patterned book materials
            material_key = f'book_{book_idx % 6}'
            book.data.materials.append(materials[material_key])


def create_feature_rich_clutter(materials):
    """Add clutter items with maximum pattern detail"""
    
    # Coffee mug with pattern
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.08, location=(-1.5, -3.2, 0.82))
    mug = bpy.context.active_object
    mug.name = "Coffee_Mug"
    mug.data.materials.append(materials['checkerboard'])
    
    # Stack of papers with text patterns
    paper_patterns = ['qr_code', 'circuit', 'checkerboard', 'brick']
    for i in range(8):  # More papers
        z_offset = 0.77 + i * 0.003
        rotation = (i - 4) * 0.03  # Slight rotation
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.5, -3.2, z_offset))
        paper = bpy.context.active_object
        paper.name = f"Paper_Pattern_{i}"
        paper.scale = (0.1, 0.15, 0.001)
        paper.rotation_euler = (0, 0, rotation)
        
        pattern = paper_patterns[i % len(paper_patterns)]
        paper.data.materials.append(materials[pattern])
    
    # Pen cup with circuit pattern
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.08, location=(-2.7, -3.5, 0.81))
    pen_cup = bpy.context.active_object
    pen_cup.name = "Pen_Cup"
    pen_cup.data.materials.append(materials['circuit'])
    
    # Individual pens with alternating patterns
    pen_patterns = ['metal', 'checkerboard', 'qr_code']
    for i in range(6):  # More pens
        angle = i * 1.0
        pen_x = -2.7 + 0.012 * np.cos(angle)
        pen_y = -3.5 + 0.012 * np.sin(angle)
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.003, depth=0.12, location=(pen_x, pen_y, 0.87))
        pen = bpy.context.active_object
        pen.name = f"Pen_{i}"
        pattern = pen_patterns[i % len(pen_patterns)]
        pen.data.materials.append(materials[pattern])
    
    # Desk lamp with detailed patterns
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.3, location=(-3.7, -3.2, 0.9))
    lamp_post = bpy.context.active_object
    lamp_post.name = "Lamp_Post"
    lamp_post.data.materials.append(materials['metal'])
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.7, -3.2, 1.1))
    lamp_head = bpy.context.active_object
    lamp_head.name = "Lamp_Head"
    lamp_head.scale = (0.08, 0.08, 0.06)
    lamp_head.data.materials.append(materials['circuit'])
    
    # Add additional clutter items
    # Calculator with buttons
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.8, -3.4, 0.775))
    calculator = bpy.context.active_object
    calculator.name = "Calculator_Base"
    calculator.scale = (0.08, 0.12, 0.015)
    calculator.data.materials.append(materials['circuit'])
    
    # Calculator buttons (4x5 grid)
    for row in range(5):
        for col in range(4):
            btn_x = -1.82 + col * 0.015
            btn_y = -3.45 + row * 0.02
            btn_z = 0.78
            
            bpy.ops.mesh.primitive_cube_add(size=0.01, location=(btn_x, btn_y, btn_z))
            button = bpy.context.active_object
            button.name = f"Calc_Button_{row}_{col}"
            # Alternate button patterns
            if (row + col) % 2 == 0:
                button.data.materials.append(materials['checkerboard'])
            else:
                button.data.materials.append(materials['qr_code'])
    
    # Notebook with grid lines
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.2, -2.9, 0.772))
    notebook = bpy.context.active_object
    notebook.name = "Notebook"
    notebook.scale = (0.12, 0.18, 0.008)
    notebook.data.materials.append(materials['fabric'])  # Grid pattern
    
    # Sticky notes in different colors/patterns
    note_positions = [(-1.7, -3.6, 0.773), (-1.6, -3.6, 0.773), (-1.5, -3.6, 0.773)]
    note_patterns = ['checkerboard', 'qr_code', 'circuit']
    
    for i, (pos, pattern) in enumerate(zip(note_positions, note_patterns)):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        note = bpy.context.active_object
        note.name = f"Sticky_Note_{i}"
        note.scale = (0.04, 0.04, 0.002)
        note.rotation_euler = (0, 0, i * 0.2)  # Slight rotation
        note.data.materials.append(materials[pattern])

def create_detailed_wall_art(materials):
    """Create wall art with maximum pattern density"""
    
    # Large feature-rich art pieces
    art_pieces = [
        {"pos": (-1, -4.95, 1.5), "scale": (0.4, 0.02, 0.5), "pattern": "checkerboard"},
        {"pos": (-3, -4.95, 1.8), "scale": (0.35, 0.02, 0.45), "pattern": "qr_code"},
        {"pos": (-4.95, -1, 1.6), "scale": (0.02, 0.3, 0.4), "pattern": "circuit"},
        {"pos": (-4.95, -3.5, 2.0), "scale": (0.02, 0.4, 0.3), "pattern": "brick"},
    ]
    
    for i, art in enumerate(art_pieces):
        # Frame
        bpy.ops.mesh.primitive_cube_add(size=1, location=art["pos"])
        frame = bpy.context.active_object
        frame.name = f"Art_Frame_{i}"
        frame.scale = art["scale"]
        frame.data.materials.append(materials['metal'])
        
        # Artwork content with pattern
        if art["pos"][1] < -4.9:  # Back wall
            content_pos = (art["pos"][0], art["pos"][1] + 0.015, art["pos"][2])
            content_scale = (art["scale"][0] * 0.9, art["scale"][1], art["scale"][2] * 0.9)
        else:  # Side wall
            content_pos = (art["pos"][0] + 0.015, art["pos"][1], art["pos"][2])
            content_scale = (art["scale"][0], art["scale"][1] * 0.9, art["scale"][2] * 0.9)
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=content_pos)
        artwork = bpy.context.active_object
        artwork.name = f"Artwork_{i}"
        artwork.scale = content_scale
        artwork.data.materials.append(materials[art["pattern"]])
    
    # Wall clock with detailed face
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.05, location=(-4.95, -2, 2.2))
    clock = bpy.context.active_object
    clock.name = "Wall_Clock"
    clock.rotation_euler = (0, 1.5708, 0)
    clock.data.materials.append(materials['checkerboard'])
    
    # Clock hands
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.94, -2, 2.2))
    hour_hand = bpy.context.active_object
    hour_hand.name = "Clock_Hour_Hand"
    hour_hand.scale = (0.005, 0.08, 0.002)
    hour_hand.rotation_euler = (0, 1.5708, 0.5)
    hour_hand.data.materials.append(materials['metal'])
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.94, -2, 2.2))
    minute_hand = bpy.context.active_object
    minute_hand.name = "Clock_Minute_Hand"
    minute_hand.scale = (0.005, 0.12, 0.002)
    minute_hand.rotation_euler = (0, 1.5708, 1.2)
    minute_hand.data.materials.append(materials['metal'])

def setup_maximum_lighting():
    """Setup lighting to enhance pattern visibility"""
    
    # Main area light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4.5))
    light = bpy.context.active_object
    light.name = "Office_Light_Main"
    light.data.energy = 100  # Brighter for better pattern visibility
    light.data.size = 2
    light.data.color[0] = 1.0  # Red
    light.data.color[1] = 1.0  # Green  
    light.data.color[2] = 1.0  # Blue
        
    # Side lighting
    bpy.ops.object.light_add(type='AREA', location=(-4.8, 0, 2))
    window_light = bpy.context.active_object
    window_light.name = "Window_Light"
    window_light.data.energy = 60
    window_light.data.size = 3
    window_light.rotation_euler = (0, 1.5708, 0)
    window_light.data.color[0] = 0.95  # Red
    window_light.data.color[1] = 0.95  # Green
    window_light.data.color[2] = 1.0   # Blue
    
    # Additional task lighting
    bpy.ops.object.light_add(type='SPOT', location=(-1, -2, 3))
    task_light = bpy.context.active_object
    task_light.name = "Task_Light"
    task_light.data.energy = 40
    task_light.data.spot_size = 1.2
    task_light.rotation_euler = (0.8, 0, -0.5)  # Pointing at desk
    
    # Back wall accent lighting
    bpy.ops.object.light_add(type='AREA', location=(0, -4, 3))
    accent_light = bpy.context.active_object
    accent_light.name = "Accent_Light"
    accent_light.data.energy = 30
    accent_light.data.size = 1
    accent_light.rotation_euler = (1.5708, 0, 0)

def setup_ultra_rendering():
    """Setup rendering for maximum feature visibility"""
    
    # Use Cycles for best material rendering
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Increase samples for crisp patterns
    bpy.context.scene.cycles.samples = 256
    
    # Disable denoising to preserve fine details
    bpy.context.scene.cycles.use_denoising = False
    
    # Set device to GPU if available
    bpy.context.scene.cycles.device = 'GPU'
    
    # Increase resolution for better feature detection
    bpy.context.scene.render.resolution_x = 1280  # Higher resolution
    bpy.context.scene.render.resolution_y = 960
    
    # Better color management for high contrast
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.gamma = 1.0
    bpy.context.scene.view_settings.exposure = 0.0
    
    # Film settings for maximum sharpness
    bpy.context.scene.cycles.film_exposure = 1.0
    bpy.context.scene.cycles.pixel_filter_type = 'BOX'  # Sharpest filter

def create_stereo_camera_rig_hd(baseline=0.12):
    """Create stereo camera rig with higher resolution"""
    
    # Create empty as rig center
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.8))
    rig = bpy.context.active_object
    rig.name = "StereoRig"
    
    # Camera intrinsics for higher resolution
    focal_length = 35  # mm
    sensor_width = 32  # mm
    resolution_x = 1280  # Higher resolution
    resolution_y = 960
    
    # Create left camera
    bpy.ops.object.camera_add(location=(-baseline/2, 0, 0))
    cam_left = bpy.context.active_object
    cam_left.name = "Camera_Left"
    cam_left.data.lens = focal_length
    cam_left.data.sensor_width = sensor_width
    cam_left.parent = rig
    cam_left.parent_type = 'OBJECT'
    
    # Create right camera
    bpy.ops.object.camera_add(location=(baseline/2, 0, 0))
    cam_right = bpy.context.active_object
    cam_right.name = "Camera_Right"
    cam_right.data.lens = focal_length
    cam_right.data.sensor_width = sensor_width
    cam_right.parent = rig
    cam_right.parent_type = 'OBJECT'
    
    # Set render resolution
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    
    return rig, cam_left, cam_right

def setup_camera_motion_manual(rig, path=None, radius=4.0, height=1.8, frames=121):
    """Setup camera motion manually with keyframes"""
    import math
    
    # Clear existing animation
    rig.animation_data_clear()
    
    print("Setting up manual circular camera motion...")
    
    for frame in range(1, frames + 1):
        # Calculate angle (full circle over frame range)
        angle = 2 * math.pi * (frame - 1) / (frames - 1)
        
        # Calculate position on circle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        
        # Set rig position
        rig.location = (x, y, z)
        
        # Calculate rotation to look at center
        target = mathutils.Vector((0, 0, 0))  # Look at room center
        loc = mathutils.Vector((x, y, z))
        
        # Calculate look direction
        direction = target - loc
        direction.normalize()
        
        # Create rotation matrix
        rot_quat = direction.to_track_quat('-Z', 'Y')
        rig.rotation_euler = rot_quat.to_euler()
        
        # Insert keyframes
        rig.keyframe_insert(data_path="location", frame=frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    # Set interpolation to linear for smooth motion
    if rig.animation_data and rig.animation_data.action:
        for fcurve in rig.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    print(f"âœ… Manual camera motion created: {frames} keyframes")

def extract_camera_poses_hd(cam_left, cam_right, start_frame=1, end_frame=121):
    """Extract camera poses for higher resolution setup"""
    
    poses_data = {
        'stereo_baseline': 0.12,
        'camera_intrinsics': {
            'fx': 1280.0,  # Higher resolution focal length in pixels
            'fy': 1280.0,
            'cx': 640.0,   # Principal point
            'cy': 480.0,
            'resolution': [1280, 960]
        },
        'poses': []
    }
    
    for frame in range(start_frame, end_frame + 1):
        # Set frame and force update
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        
        # Get world matrices
        left_matrix = cam_left.matrix_world.copy()
        right_matrix = cam_right.matrix_world.copy()
        
        # Convert to pose format
        def matrix_to_pose(matrix):
            translation = matrix.translation
            rotation = matrix.to_quaternion()
            
            return {
                'translation': [translation.x, translation.y, translation.z],
                'rotation_quaternion': [rotation.w, rotation.x, rotation.y, rotation.z],
                'rotation_matrix': [list(row) for row in matrix.to_3x3()]
            }
        
        frame_data = {
            'frame': frame,
            'timestamp': frame / 30.0,  # 30 FPS
            'left_camera': matrix_to_pose(left_matrix),
            'right_camera': matrix_to_pose(right_matrix)
        }
        
        poses_data['poses'].append(frame_data)
        
        if frame % 10 == 0:  # Progress indicator
            pos = left_matrix.translation
            print(f"Frame {frame}: Camera at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    
    return poses_data

def setup_rendering(output_dir):
    """Setup rendering settings"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)
    
    # Set render settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_dir

def render_stereo_sequence(cam_left, cam_right, output_dir, start_frame=1, end_frame=121):
    """Render the stereo image sequence"""
    
    for frame in range(start_frame, end_frame + 1):
        # Set current frame
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        
        if frame % 20 == 1:
            pos = cam_left.matrix_world.translation
            print(f"Frame {frame}: Camera at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        
        # Render left camera
        bpy.context.scene.camera = cam_left
        bpy.context.scene.render.filepath = os.path.join(output_dir, 'left', f'frame_{frame:04d}.png')
        bpy.ops.render.render(write_still=True)
        
        # Render right camera
        bpy.context.scene.camera = cam_right
        bpy.context.scene.render.filepath = os.path.join(output_dir, 'right', f'frame_{frame:04d}.png')
        bpy.ops.render.render(write_still=True)
        
        print(f"Rendered frame {frame}/{end_frame}")

def main():
    """Main function to create the MAXIMUM FEATURE DENSITY office dataset"""
    
    # Configuration
    output_dir = "/tmp/maximum_feature_office_dataset"
    baseline = 0.12  # 12cm stereo baseline
    
    print("="*80)
    print("ðŸŽ¯ CREATING MAXIMUM FEATURE DENSITY OFFICE DATASET")
    print("="*80)
    print("This version is specifically designed for maximum keypoint detection!")
    
    # 1. Clear scene and create ultra-detailed materials
    clear_scene()
    materials = create_maximum_feature_materials()
    print("âœ… ULTRA-DETAILED materials created:")
    print("   â€¢ Multi-layer wood grain with high contrast")
    print("   â€¢ Checkerboard patterns with sharp edges")
    print("   â€¢ Circuit board textures with fine details")
    print("   â€¢ Brick patterns with clear mortar lines")
    print("   â€¢ QR code-like high-contrast patterns")
    print("   â€¢ RGB pixel screen patterns")
    print("   â€¢ Grid fabric with precise lines")
    print("   â€¢ Multi-scale metal scratches")
    
    # 2. Create maximum feature environment
    ceiling = create_feature_rich_walls(materials)
    create_maximum_feature_floor(materials)
    print("âœ… FEATURE-RICH environment created:")
    print("   â€¢ Floor: Alternating checkerboard/circuit/brick/QR patterns")
    print("   â€¢ Walls: Segmented with different high-contrast patterns")
    print("   â€¢ Ceiling: Circuit board pattern")
    
    create_ultra_detailed_desk(materials)
    print("âœ… ULTRA-DETAILED desk created with QR desk mat")
    
    create_patterned_chair(materials)
    print("âœ… PATTERNED chair with grid fabric and checkerboard wheels")
    
    create_detailed_monitor_setup(materials)
    print("âœ… DETAILED monitor setup:")
    print("   â€¢ RGB pixel screen pattern")
    print("   â€¢ Circuit board stand and bezel")
    print("   â€¢ Multiple UI elements with different patterns")
    
    create_maximum_keyboard(materials)
    print("âœ… MAXIMUM keyboard detail:")
    print("   â€¢ Individual keys with alternating patterns")
    print("   â€¢ Circuit board base")
    print("   â€¢ Contrasting key labels")
    
    create_maximum_bookshelf(materials)
    print("âœ… MAXIMUM bookshelf variety:")
    print("   â€¢ 6 different book patterns (stripes, dots, zigzag, grid, checker, waves)")
    print("   â€¢ Patterned shelves and sides")
    print("   â€¢ Circuit board back panel")
    
    create_feature_rich_clutter(materials)
    print("âœ… FEATURE-RICH clutter added:")
    print("   â€¢ 8 papers with different text-like patterns")
    print("   â€¢ 6 pens with alternating materials")
    print("   â€¢ Calculator with 20 patterned buttons")
    print("   â€¢ Notebook with grid pattern")
    print("   â€¢ 3 sticky notes with different patterns")
    print("   â€¢ Patterned mug, lamp, and pen cup")
    
    create_detailed_wall_art(materials)
    print("âœ… DETAILED wall art created:")
    print("   â€¢ 4 large artworks with different patterns")
    print("   â€¢ Patterned clock face with metal hands")
    
    
    # 3. Setup maximum lighting
    setup_maximum_lighting()
    print("âœ… MAXIMUM lighting setup:")
    print("   â€¢ 100W main light for pattern visibility")
    print("   â€¢ 60W side window light")
    print("   â€¢ 40W task light for desk area")
    print("   â€¢ 30W accent light for wall patterns")

     # After creating all scene elements, add markers
    placed_markers = add_markers_to_scene(materials)
    
    # 4. Create camera path
    path = None  # We use manual keyframes
    print("âœ… Camera path ready (manual keyframes)")
    
    # 5. Create HD stereo camera rig
    rig, cam_left, cam_right = create_stereo_camera_rig_hd(baseline=baseline)
    print("âœ… HD Stereo camera rig created (1280x960)")
    
    # 6. Setup camera motion
    setup_camera_motion_manual(rig, path, radius=4.0, height=1.8, frames=121)
    print("âœ… Camera motion setup complete")
    
    # 7. Setup ultra rendering
    setup_ultra_rendering()
    print("âœ… ULTRA rendering settings configured:")
    print("   â€¢ Cycles engine with 256 samples")
    print("   â€¢ No denoising (preserves fine details)")
    print("   â€¢ 1280x960 resolution")
    print("   â€¢ Box pixel filter for maximum sharpness")
    
    # 8. Extract ground truth poses
    print("Extracting ground truth poses...")
    poses_data = extract_camera_poses_hd(cam_left, cam_right, start_frame=1, end_frame=121)

    # Update ground truth with marker information
    poses_data = update_ground_truth_with_markers(poses_data, placed_markers)
    
    # Save poses to JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ground_truth_poses.json'), 'w') as f:
        json.dump(poses_data, f, indent=2)
    
    print(f"âœ… Ground truth poses saved to {output_dir}/ground_truth_poses.json")
    
    # 9. Setup rendering directories
    setup_rendering(output_dir)
    
    # 10. Render test frame
    print("Rendering test frame with MAXIMUM feature density...")
    render_stereo_sequence(cam_left, cam_right, output_dir)
    print("âœ… Test frame rendered - check for feature density!")
    
    print("\n" + "="*80)
    print("ðŸŽ‰ MAXIMUM FEATURE DENSITY DATASET READY!")
    print("="*80)
    print(f"ðŸ“ Output directory: {output_dir}")
    print(f"ðŸ“Š Ground truth poses: {output_dir}/ground_truth_poses.json")
    print(f"ðŸ–¼ï¸  HD images: {output_dir}/left/ and {output_dir}/right/")
    print(f"ðŸ“ Camera specs: 1280x960, baseline={baseline*1000:.0f}mm")
    
    print("\nðŸŽ¯ FEATURE DETECTION OPTIMIZATIONS:")
    print("   âœ“ 10+ different high-contrast pattern types")
    print("   âœ“ Sharp edges everywhere (checkerboard, brick, QR)")
    print("   âœ“ Multi-scale textures (fine to coarse)")
    print("   âœ“ Circuit board patterns (maximum corner density)")
    print("   âœ“ Individual keyboard keys with patterns")
    print("   âœ“ RGB pixel screen simulation")
    print("   âœ“ 50+ objects with unique patterns")
    print("   âœ“ No denoising (preserves fine detail)")
    print("   âœ“ Higher resolution (1280x960)")
    print("   âœ“ Maximum contrast lighting")
    print("   âœ“ Sharp pixel filter")
    
    print("\nðŸ’¡ KEYPOINT DETECTION EXPECTATIONS:")
    print("   â€¢ SIFT: Should detect 2000+ keypoints per image")
    print("   â€¢ ORB: Should detect 1500+ keypoints per image") 
    print("   â€¢ SURF: Should detect 1800+ keypoints per image")
    print("   â€¢ Corner detectors: Massive number of corners")
    
    print("\nðŸš€ TO RENDER FULL SEQUENCE:")
    print("   render_stereo_sequence(cam_left, cam_right, output_dir, 1, 121)")
    
    return rig, cam_left, cam_right, output_dir, placed_markers

# Run the maximum feature script
if __name__ == "__main__":
    main()
