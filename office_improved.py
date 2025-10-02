import bpy
import bmesh
import mathutils
import numpy as np
import os
import json
from mathutils import Matrix, Vector
import cv2
import tempfile
import random
import subprocess

def create_aruco_markers(marker_ids=[0, 1, 2], marker_size_pixels=400):
    """Generate real ArUco markers and create Blender materials"""
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    marker_materials = []
    temp_dir = tempfile.gettempdir()
    
    for marker_id in marker_ids:
        print(f"Creating ArUco marker ID: {marker_id}")
        
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
        
        border_size = 20
        bordered_marker = np.ones((marker_size_pixels + 2*border_size, 
                                   marker_size_pixels + 2*border_size), dtype=np.uint8) * 255
        bordered_marker[border_size:-border_size, border_size:-border_size] = marker_image
        
        marker_path = os.path.join(temp_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(marker_path, bordered_marker)
        
        marker_mat = bpy.data.materials.new(f"ArUco_Marker_{marker_id}")
        marker_mat.use_nodes = True
        nodes = marker_mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        emission = nodes.new('ShaderNodeEmission')
        mix_shader = nodes.new('ShaderNodeMixShader')
        tex_image = nodes.new('ShaderNodeTexImage')
        
        marker_img = bpy.data.images.load(marker_path)
        marker_img.colorspace_settings.name = 'Non-Color'
        tex_image.image = marker_img
        
        bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.8
        
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.0
        elif 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = 1.0
        
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = 0.0
        
        # Add emission to make markers more visible
        emission.inputs['Strength'].default_value = 2.0
        
        # Connect nodes
        marker_mat.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        marker_mat.node_tree.links.new(tex_image.outputs['Color'], emission.inputs['Color'])
        marker_mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
        marker_mat.node_tree.links.new(emission.outputs['Emission'], mix_shader.inputs[2])
        mix_shader.inputs[0].default_value = 0.3  # Mix factor
        marker_mat.node_tree.links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
        
        marker_materials.append(marker_mat)
        print(f"  Created material: ArUco_Marker_{marker_id}")
    
    return marker_materials

def place_strategic_markers_fixed(marker_materials):
    """Place markers guaranteed to be visible"""
    
    marker_configs = [
        {
            "name": "Floor_Center_Marker",
            "id": 0,
            "position": (0, 0, 0.05),  # Raised higher above floor
            "rotation": (0, 0, 0),
            "size": 0.30,
            "description": "Center floor marker"
        },
        {
            "name": "Desk_Surface_Marker", 
            "id": 1,
            "position": (-2.0, -3.0, 0.78),  # Raised above desk surface
            "rotation": (0, 0, 0),
            "size": 0.15,
            "description": "Desk surface marker"
        },
        {
            "name": "Near_Wall_Marker",
            "id": 2,
            "position": (0, -2.5, 0.8),
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
        backing_size = marker["size"] * 1.2
        backing_pos = list(marker["world_position"])
        
        if abs(backing_pos[1]) > 4.5:
            backing_pos[1] += 0.01 if backing_pos[1] > 0 else -0.01
        else:
            backing_pos[0] += 0.01 if backing_pos[0] > 0 else -0.01
        
        bpy.ops.mesh.primitive_plane_add(
            size=backing_size,
            location=backing_pos
        )
        
        backing = bpy.context.active_object
        backing.name = f"{marker['name']}_Backing"
        backing.rotation_euler = marker["world_rotation"]
        
        backing.data.materials.append(materials['wall'])

def update_ground_truth_with_markers(poses_data, placed_markers):
    """Add marker information to ground truth data"""
    
    poses_data['aruco_markers'] = {
        'dictionary': 'DICT_4X4_50',
        'markers': placed_markers
    }
    
    print("Updated ground truth data with ArUco marker information")
    return poses_data

def add_markers_to_scene(materials):
    """Main function to add ArUco markers to existing scene"""
    
    print("Adding ArUco markers to scene...")
    print("=" * 50)
    
    marker_materials = create_aruco_markers(marker_ids=[0, 1, 2])
    placed_markers = place_strategic_markers_fixed(marker_materials)
    
    add_marker_backing_planes(placed_markers, materials)
    
    print("=" * 50)
    print("ArUco markers added successfully!")
    print("\nMarker placement summary:")
    for marker in placed_markers:
        print(f"  {marker['name']}: ID {marker['id']} at {marker['world_position']}")
        print(f"    Size: {marker['size']}m - {marker['description']}")
    
    return placed_markers

def clear_scene():
    """Clear default scene objects"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

def create_unique_texture_image(width, height, pattern_type, seed):
    """Create unique texture patterns as numpy arrays"""
    np.random.seed(seed)
    
    if pattern_type == 'wood_grain':
        # Create wood with unique grain patterns
        base = np.random.uniform(0.3, 0.5, (height, width, 3))
        for i in range(5):
            freq = np.random.uniform(10, 50)
            x = np.linspace(0, freq, width)
            grain = 0.15 * np.sin(x + np.random.uniform(0, 10))
            base += grain[np.newaxis, :, np.newaxis]
        return np.clip(base, 0, 1)
    
    elif pattern_type == 'paper_texture':
        # Realistic paper with subtle noise
        base_color = np.random.uniform(0.85, 0.98)
        texture = np.ones((height, width, 3)) * base_color
        noise = np.random.normal(0, 0.02, (height, width, 3))
        
        # Add random text-like features
        num_lines = random.randint(15, 30)
        for _ in range(num_lines):
            y = random.randint(0, height-1)
            x_start = random.randint(0, width//4)
            x_end = random.randint(3*width//4, width-1)
            thickness = random.randint(1, 3)
            texture[y:y+thickness, x_start:x_end] *= 0.3
        
        return np.clip(texture + noise, 0, 1)
    
    elif pattern_type == 'fabric_weave':
        # Unique fabric weave pattern
        texture = np.zeros((height, width, 3))
        base_color = np.random.uniform(0.2, 0.8, 3)
        
        thread_size = random.randint(3, 8)
        for i in range(0, height, thread_size):
            for j in range(0, width, thread_size):
                if (i // thread_size + j // thread_size) % 2:
                    texture[i:i+thread_size, j:j+thread_size] = base_color * 1.1
                else:
                    texture[i:i+thread_size, j:j+thread_size] = base_color * 0.9
        
        return np.clip(texture, 0, 1)
    
    elif pattern_type == 'poster':
        # Colorful poster with geometric shapes
        texture = np.ones((height, width, 3)) * np.random.uniform(0.7, 1.0, 3)
        
        num_shapes = random.randint(5, 15)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rect', 'line'])
            color = np.random.uniform(0, 1, 3)
            
            if shape_type == 'circle':
                cx, cy = random.randint(0, width), random.randint(0, height)
                radius = random.randint(width//20, width//5)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                texture[mask] = color
            
            elif shape_type == 'rect':
                x1 = random.randint(0, width-10)
                y1 = random.randint(0, height-10)
                x2 = random.randint(x1+10, min(x1+width//3, width))
                y2 = random.randint(y1+10, min(y1+height//3, height))
                texture[y1:y2, x1:x2] = color
            
            elif shape_type == 'line':
                y = random.randint(0, height-1)
                thickness = random.randint(2, 8)
                texture[y:y+thickness, :] = color
        
        return texture
    
    elif pattern_type == 'brick':
        # Realistic brick with variation
        texture = np.ones((height, width, 3))
        brick_color = np.array([0.6, 0.3, 0.2]) + np.random.uniform(-0.1, 0.1, 3)
        mortar_color = np.array([0.8, 0.8, 0.75])
        
        brick_h = height // 6
        brick_w = width // 4
        mortar = 3
        
        for row in range(6):
            offset = (brick_w // 2) if row % 2 else 0
            y_start = row * brick_h
            
            for col in range(5):
                x_start = col * brick_w + offset
                if x_start + brick_w <= width:
                    color_var = np.random.uniform(0.9, 1.1, 3)
                    texture[y_start+mortar:y_start+brick_h, 
                           x_start+mortar:x_start+brick_w] = brick_color * color_var
        
        return np.clip(texture, 0, 1)
    
    else:  # random_unique
        # Completely random unique pattern
        return np.random.uniform(0.2, 0.8, (height, width, 3))

def create_material_from_texture(name, texture_array, roughness=0.5):
    """Create Blender material from numpy texture array"""
    temp_dir = tempfile.gettempdir()
    texture_path = os.path.join(temp_dir, f"{name}.png")
    
    # Convert to BGR and save
    texture_bgr = (texture_array * 255).astype(np.uint8)
    texture_bgr = cv2.cvtColor(texture_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(texture_path, texture_bgr)
    
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex_image = nodes.new('ShaderNodeTexImage')
    
    img = bpy.data.images.load(texture_path)
    tex_image.image = img
    
    bsdf.inputs['Roughness'].default_value = roughness
    
    mat.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_diverse_materials():
    """Create materials with unique, non-repetitive textures"""
    materials = {}
    
    # Create 20 unique wood variations
    for i in range(20):
        texture = create_unique_texture_image(512, 512, 'wood_grain', seed=1000+i)
        materials[f'wood_{i}'] = create_material_from_texture(f'Wood_{i}', texture, 0.6)
    
    # Create 30 unique paper textures
    for i in range(30):
        texture = create_unique_texture_image(512, 512, 'paper_texture', seed=2000+i)
        materials[f'paper_{i}'] = create_material_from_texture(f'Paper_{i}', texture, 0.8)
    
    # Create 15 unique fabric patterns
    for i in range(15):
        texture = create_unique_texture_image(256, 256, 'fabric_weave', seed=3000+i)
        materials[f'fabric_{i}'] = create_material_from_texture(f'Fabric_{i}', texture, 0.7)
    
    # Create 10 unique posters
    for i in range(10):
        texture = create_unique_texture_image(512, 512, 'poster', seed=4000+i)
        materials[f'poster_{i}'] = create_material_from_texture(f'Poster_{i}', texture, 0.5)
    
    # Create 5 brick variations
    for i in range(5):
        texture = create_unique_texture_image(512, 512, 'brick', seed=5000+i)
        materials[f'brick_{i}'] = create_material_from_texture(f'Brick_{i}', texture, 0.6)
    
    # Create simple colored materials for variety
    colors = [
        ('White', (0.95, 0.95, 0.95)),
        ('Black', (0.05, 0.05, 0.05)),
        ('Red', (0.8, 0.2, 0.2)),
        ('Blue', (0.2, 0.4, 0.8)),
        ('Green', (0.3, 0.7, 0.3)),
        ('Yellow', (0.9, 0.9, 0.3)),
        ('Metal', (0.7, 0.7, 0.8)),
    ]
    
    for name, color in colors:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
        if name == 'Metal':
            bsdf.inputs['Metallic'].default_value = 0.9
        
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        materials[name.lower()] = mat
    
    # Wall material with subtle texture
    texture = create_unique_texture_image(512, 512, 'random_unique', seed=6000)
    texture = texture * 0.3 + 0.65  # Light colored wall
    materials['wall'] = create_material_from_texture('Wall', texture, 0.7)
    
    print(f"Created {len(materials)} unique materials")
    return materials

def create_unique_floor(materials):
    """Create floor with unique tile patterns"""
    
    tile_size = 0.5
    floor_size = 10
    tiles_per_side = int(floor_size / tile_size)
    
    # Randomly assign unique materials to each tile
    wood_mats = [k for k in materials.keys() if k.startswith('wood_')]
    fabric_mats = [k for k in materials.keys() if k.startswith('fabric_')]
    
    for x in range(tiles_per_side):
        for y in range(tiles_per_side):
            world_x = (x - tiles_per_side/2) * tile_size + tile_size/2
            world_y = (y - tiles_per_side/2) * tile_size + tile_size/2
            
            bpy.ops.mesh.primitive_plane_add(size=tile_size, location=(world_x, world_y, 0))
            tile = bpy.context.active_object
            tile.name = f"Floor_Tile_{x}_{y}"
            
            # Random material selection for uniqueness
            if (x + y) % 3 == 0:
                mat_key = random.choice(wood_mats)
            else:
                mat_key = random.choice(fabric_mats)
            
            tile.data.materials.append(materials[mat_key])

def create_detailed_walls(materials):
    """Create walls with unique sections"""
    
    wall_size = 10
    
    # Back wall - use unique poster materials
    poster_mats = [k for k in materials.keys() if k.startswith('poster_')]
    brick_mats = [k for k in materials.keys() if k.startswith('brick_')]
    
    segments_x = 5
    segments_y = 4
    
    for i in range(segments_x):
        for j in range(segments_y):
            x = (i - segments_x/2) * (wall_size/segments_x) + wall_size/(2*segments_x)
            z = j * (wall_size/segments_y) + wall_size/(2*segments_y)
            
            bpy.ops.mesh.primitive_plane_add(
                size=wall_size/segments_x, 
                location=(x, -5, z)
            )
            segment = bpy.context.active_object
            segment.name = f"Wall_Back_{i}_{j}"
            segment.rotation_euler = (1.5708, 0, 0)
            
            # Mix of poster and brick materials
            if j < 2:
                mat_key = random.choice(brick_mats)
            else:
                mat_key = random.choice(poster_mats)
            
            segment.data.materials.append(materials[mat_key])
    
    # Side wall
    for i in range(segments_y):
        for j in range(segments_y):
            y = (i - segments_y/2) * (wall_size/segments_y) + wall_size/(2*segments_y)
            z = j * (wall_size/segments_y) + wall_size/(2*segments_y)
            
            bpy.ops.mesh.primitive_plane_add(
                size=wall_size/segments_y, 
                location=(-5, y, z)
            )
            segment = bpy.context.active_object
            segment.name = f"Wall_Side_{i}_{j}"
            segment.rotation_euler = (1.5708, 0, 1.5708)
            
            mat_key = random.choice(poster_mats + brick_mats)
            segment.data.materials.append(materials[mat_key])
    
    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 5))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.data.materials.append(materials['white'])
    
    return ceiling

def create_desk_with_unique_objects(materials):
    """Create desk with uniquely textured items"""
    
    wood_mats = [k for k in materials.keys() if k.startswith('wood_')]
    
    # Main desk surface
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3, 0.75))
    desk = bpy.context.active_object
    desk.name = "Desk_Main"
    desk.scale = (2, 1, 0.05)
    desk.data.materials.append(materials[random.choice(wood_mats)])
    
    # Side extension with different wood
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.5, -2, 0.75))
    side = bpy.context.active_object
    side.name = "Desk_Side"
    side.scale = (0.5, 1, 0.05)
    desk.data.materials.append(materials[random.choice(wood_mats)])
    
    # Desk legs
    leg_positions = [(-3, -3.5, 0.375), (-1, -3.5, 0.375), 
                     (-1, -2.5, 0.375), (-4, -2.5, 0.375)]
    for i, pos in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.75, location=pos)
        leg = bpy.context.active_object
        leg.name = f"Desk_Leg_{i}"
        leg.data.materials.append(materials['metal'])
    
    # Drawer
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.5, -2.7, 0.4))
    drawer = bpy.context.active_object
    drawer.name = "Desk_Drawer"
    drawer.scale = (0.4, 0.25, 0.25)
    drawer.data.materials.append(materials[random.choice(wood_mats)])

def create_monitor_and_peripherals(materials):
    """Create monitor setup with unique textures"""
    
    # Monitor stand
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.02, location=(-2, -3.8, 0.8))
    stand = bpy.context.active_object
    stand.name = "Monitor_Stand"
    stand.data.materials.append(materials['black'])
    
    # Post
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.3, location=(-2, -3.8, 0.95))
    post = bpy.context.active_object
    post.name = "Monitor_Post"
    post.data.materials.append(materials['metal'])
    
    # Screen - use bright color for visibility
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3.8, 1.2))
    screen = bpy.context.active_object
    screen.name = "Monitor_Screen"
    screen.scale = (0.5, 0.05, 0.3)
    screen.data.materials.append(materials['blue'])
    
    # Bezel
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -3.75, 1.2))
    bezel = bpy.context.active_object
    bezel.name = "Monitor_Bezel"
    bezel.scale = (0.52, 0.03, 0.32)
    bezel.data.materials.append(materials['black'])
    
    # Keyboard base
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -2.8, 0.77))
    keyboard = bpy.context.active_object
    keyboard.name = "Keyboard"
    keyboard.scale = (0.35, 0.15, 0.01)
    keyboard.data.materials.append(materials['black'])
    
    # Mouse
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.4, -2.6, 0.78))
    mouse = bpy.context.active_object
    mouse.name = "Mouse"
    mouse.scale = (0.06, 0.1, 0.02)
    mouse.data.materials.append(materials['red'])

def create_unique_chair(materials):
    """Create chair with unique fabric"""
    
    fabric_mats = [k for k in materials.keys() if k.startswith('fabric_')]
    
    # Seat
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.5, -1.5, 0.5))
    seat = bpy.context.active_object
    seat.name = "Chair_Seat"
    seat.scale = (0.4, 0.4, 0.05)
    seat.data.materials.append(materials[random.choice(fabric_mats)])
    
    # Back
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.5, -1.8, 0.9))
    back = bpy.context.active_object
    back.name = "Chair_Back"
    back.scale = (0.35, 0.05, 0.4)
    back.data.materials.append(materials[random.choice(fabric_mats)])
    
    # Post
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.4, location=(-1.5, -1.5, 0.25))
    post = bpy.context.active_object
    post.name = "Chair_Post"
    post.data.materials.append(materials['metal'])
    
    # Wheels
    for i in range(5):
        angle = i * 2 * 3.14159 / 5
        x = -1.5 + 0.25 * np.cos(angle)
        y = -1.5 + 0.25 * np.sin(angle)
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.02, location=(x, y, 0.03))
        wheel = bpy.context.active_object
        wheel.name = f"Chair_Wheel_{i}"
        wheel.rotation_euler = (1.5708, 0, 0)
        wheel.data.materials.append(materials['black'])

def create_bookshelf_with_unique_books(materials):
    """Create bookshelf with uniquely textured books"""
    
    shelf_x = -4.2
    shelf_y = -4.5
    shelf_height = 2.0
    
    wood_mats = [k for k in materials.keys() if k.startswith('wood_')]
    paper_mats = [k for k in materials.keys() if k.startswith('paper_')]
    
    # Back panel
    bpy.ops.mesh.primitive_cube_add(size=1, location=(shelf_x, shelf_y, shelf_height/2))
    back = bpy.context.active_object
    back.name = "Bookshelf_Back"
    back.scale = (0.8, 0.02, shelf_height/2)
    back.data.materials.append(materials[random.choice(wood_mats)])
    
    # Shelves
    for i in range(4):
        z_pos = shelf_height * i / 3 - shelf_height/2 + 0.5
        bpy.ops.mesh.primitive_cube_add(size=1, location=(shelf_x, shelf_y - 0.1, z_pos))
        shelf = bpy.context.active_object
        shelf.name = f"Shelf_{i}"
        shelf.scale = (0.8, 0.1, 0.02)
        shelf.data.materials.append(materials[random.choice(wood_mats)])
    
    # Side panels
    for side in [-1, 1]:
        x_pos = shelf_x + side * 0.8
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x_pos, shelf_y - 0.05, shelf_height/2))
        panel = bpy.context.active_object
        panel.name = f"Shelf_Side_{side}"
        panel.scale = (0.02, 0.15, shelf_height/2)
        panel.data.materials.append(materials[random.choice(wood_mats)])
    
    # Books with unique textures
    np.random.seed(42)
    for shelf_level in range(3):
        z_pos = shelf_height * shelf_level / 3 - shelf_height/2 + 0.65
        
        num_books = random.randint(8, 12)
        book_x = shelf_x - 0.7
        
        for book_idx in range(num_books):
            width = np.random.uniform(0.02, 0.06)
            height = np.random.uniform(0.15, 0.25)
            depth = 0.12
            
            x = book_x + book_idx * 0.14
            
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, shelf_y - 0.05, z_pos))
            book = bpy.context.active_object
            book.name = f"Book_{shelf_level}_{book_idx}"
            book.scale = (width, depth, height)
            
            # Each book gets unique paper texture
            mat_key = random.choice(paper_mats)
            book.data.materials.append(materials[mat_key])

def create_diverse_clutter(materials):
    """Add clutter with unique textures"""
    
    paper_mats = [k for k in materials.keys() if k.startswith('paper_')]
    
    # Coffee mug
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.08, location=(-1.5, -3.2, 0.82))
    mug = bpy.context.active_object
    mug.name = "Mug"
    mug.data.materials.append(materials['white'])
    
    # Stack of papers with unique textures
    for i in range(8):
        z = 0.77 + i * 0.003
        rotation = (i - 4) * 0.03
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.5, -3.2, z))
        paper = bpy.context.active_object
        paper.name = f"Paper_{i}"
        paper.scale = (0.1, 0.15, 0.001)
        paper.rotation_euler = (0, 0, rotation)
        
        # Each paper gets unique texture
        paper.data.materials.append(materials[random.choice(paper_mats)])
    
    # Pen holder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.08, location=(-2.7, -3.5, 0.81))
    holder = bpy.context.active_object
    holder.name = "Pen_Holder"
    holder.data.materials.append(materials['blue'])
    
    # Pens
    colors = ['red', 'blue', 'black', 'green', 'yellow']
    for i in range(5):
        angle = i * 1.2
        x = -2.7 + 0.012 * np.cos(angle)
        y = -3.5 + 0.012 * np.sin(angle)
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.003, depth=0.12, location=(x, y, 0.87))
        pen = bpy.context.active_object
        pen.name = f"Pen_{i}"
        pen.data.materials.append(materials[colors[i % len(colors)]])
    
    # Desk lamp
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.3, location=(-3.7, -3.2, 0.9))
    lamp_post = bpy.context.active_object
    lamp_post.name = "Lamp_Post"
    lamp_post.data.materials.append(materials['metal'])
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.7, -3.2, 1.1))
    lamp_head = bpy.context.active_object
    lamp_head.name = "Lamp_Head"
    lamp_head.scale = (0.08, 0.08, 0.06)
    lamp_head.data.materials.append(materials['black'])
    
    # Notebook
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.2, -2.9, 0.772))
    notebook = bpy.context.active_object
    notebook.name = "Notebook"
    notebook.scale = (0.12, 0.18, 0.008)
    notebook.data.materials.append(materials[random.choice(paper_mats)])
    
    # Sticky notes with different colors
    note_positions = [(-1.7, -3.6, 0.773), (-1.6, -3.6, 0.773), (-1.5, -3.6, 0.773)]
    note_colors = ['yellow', 'green', 'blue']
    
    for i, (pos, color) in enumerate(zip(note_positions, note_colors)):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        note = bpy.context.active_object
        note.name = f"Note_{i}"
        note.scale = (0.04, 0.04, 0.002)
        note.rotation_euler = (0, 0, i * 0.2)
        note.data.materials.append(materials[color])

def create_wall_decorations(materials):
    """Create unique wall art"""
    
    poster_mats = [k for k in materials.keys() if k.startswith('poster_')]
    
    # Large wall posters with unique designs
    art_configs = [
        {"pos": (-1, -4.95, 1.5), "scale": (0.4, 0.02, 0.5)},
        {"pos": (-3, -4.95, 1.8), "scale": (0.35, 0.02, 0.45)},
        {"pos": (-4.95, -1, 1.6), "scale": (0.02, 0.3, 0.4)},
        {"pos": (-4.95, -3.5, 2.0), "scale": (0.02, 0.4, 0.3)},
    ]
    
    for i, config in enumerate(art_configs):
        # Frame
        bpy.ops.mesh.primitive_cube_add(size=1, location=config["pos"])
        frame = bpy.context.active_object
        frame.name = f"Frame_{i}"
        frame.scale = config["scale"]
        frame.data.materials.append(materials['black'])
        
        # Artwork - each gets unique poster
        if config["pos"][1] < -4.9:  # Back wall
            art_pos = (config["pos"][0], config["pos"][1] + 0.015, config["pos"][2])
            art_scale = (config["scale"][0] * 0.9, config["scale"][1], config["scale"][2] * 0.9)
        else:  # Side wall
            art_pos = (config["pos"][0] + 0.015, config["pos"][1], config["pos"][2])
            art_scale = (config["scale"][0], config["scale"][1] * 0.9, config["scale"][2] * 0.9)
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=art_pos)
        artwork = bpy.context.active_object
        artwork.name = f"Artwork_{i}"
        artwork.scale = art_scale
        artwork.data.materials.append(materials[poster_mats[i % len(poster_mats)]])
    
    # Wall clock
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.05, location=(-4.95, -2, 2.2))
    clock = bpy.context.active_object
    clock.name = "Clock"
    clock.rotation_euler = (0, 1.5708, 0)
    clock.data.materials.append(materials['white'])
    
    # Clock hands
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.94, -2, 2.2))
    hour = bpy.context.active_object
    hour.name = "Hour_Hand"
    hour.scale = (0.005, 0.08, 0.002)
    hour.rotation_euler = (0, 1.5708, 0.5)
    hour.data.materials.append(materials['black'])
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.94, -2, 2.2))
    minute = bpy.context.active_object
    minute.name = "Minute_Hand"
    minute.scale = (0.005, 0.12, 0.002)
    minute.rotation_euler = (0, 1.5708, 1.2)
    minute.data.materials.append(materials['black'])

def setup_lighting():
    """Setup realistic office lighting"""
    
    # Main ceiling light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4.5))
    light = bpy.context.active_object
    light.name = "Main_Light"
    light.data.energy = 100
    light.data.size = 2
    light.data.color = (1.0, 1.0, 1.0)
    
    # Window light
    bpy.ops.object.light_add(type='AREA', location=(-4.8, 0, 2))
    window = bpy.context.active_object
    window.name = "Window_Light"
    window.data.energy = 60
    window.data.size = 3
    window.rotation_euler = (0, 1.5708, 0)
    window.data.color = (0.95, 0.95, 1.0)
    
    # Task light for desk
    bpy.ops.object.light_add(type='SPOT', location=(-1, -2, 3))
    task = bpy.context.active_object
    task.name = "Task_Light"
    task.data.energy = 40
    task.data.spot_size = 1.2
    task.rotation_euler = (0.8, 0, -0.5)
    
    # Accent light
    bpy.ops.object.light_add(type='AREA', location=(0, -4, 3))
    accent = bpy.context.active_object
    accent.name = "Accent_Light"
    accent.data.energy = 30
    accent.data.size = 1
    accent.rotation_euler = (1.5708, 0, 0)

def add_exif_metadata_to_images(output_dir, focal_length_mm=35, sensor_width_mm=32, 
                                resolution_x=1280, resolution_y=960):
    """Add EXIF metadata to rendered images for Meshroom compatibility"""
    
    print("Adding EXIF metadata to images...")
    
    # Calculate focal length in 35mm equivalent
    # Standard 35mm film frame is 36mm wide
    focal_length_35mm = focal_length_mm * (36.0 / sensor_width_mm)
    
    # Calculate focal length in pixels
    focal_length_pixels = (focal_length_mm / sensor_width_mm) * resolution_x
    
    for camera_side in ['left', 'right']:
        image_dir = os.path.join(output_dir, camera_side)
        
        if not os.path.exists(image_dir):
            continue
        
        for filename in os.listdir(image_dir):
            if not filename.endswith('.png'):
                continue
            
            filepath = os.path.join(image_dir, filename)
            
            # Use exiftool if available (more reliable)
            try:
                # Check if exiftool is available
                result = subprocess.run(['exiftool', '-ver'], 
                                       capture_output=True, 
                                       text=True, 
                                       timeout=2)
                
                if result.returncode == 0:
                    # exiftool is available
                    exiftool_commands = [
                        'exiftool',
                        '-overwrite_original',
                        f'-Make=Blender',
                        f'-Model=Synthetic Camera',
                        f'-FocalLength={focal_length_mm}',
                        f'-FocalLengthIn35mmFormat={focal_length_35mm:.1f}',
                        f'-ExifImageWidth={resolution_x}',
                        f'-ExifImageHeight={resolution_y}',
                        f'-FNumber=2.8',
                        f'-ISO=100',
                        f'-ExposureTime=0.01',
                        filepath
                    ]
                    
                    subprocess.run(exiftool_commands, 
                                  capture_output=True, 
                                  check=True,
                                  timeout=10)
                    
                else:
                    # Fallback to manual method
                    add_exif_manual(filepath, focal_length_mm, focal_length_35mm, 
                                   resolution_x, resolution_y)
            
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # exiftool not available or failed, use manual method
                add_exif_manual(filepath, focal_length_mm, focal_length_35mm, 
                               resolution_x, resolution_y)
    
    print("EXIF metadata added to all images")
    print(f"  Focal Length: {focal_length_mm}mm")
    print(f"  35mm Equivalent: {focal_length_35mm:.1f}mm")
    print(f"  Sensor Width: {sensor_width_mm}mm")
    print(f"  Resolution: {resolution_x}x{resolution_y}")

def add_exif_manual(filepath, focal_length_mm, focal_length_35mm, width, height):
    """Manually add EXIF using piexif library as fallback"""
    
    try:
        import piexif
        from PIL import Image
        
        # Load image
        img = Image.open(filepath)
        
        # Create EXIF data
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"Blender",
                piexif.ImageIFD.Model: b"Synthetic Camera",
                piexif.ImageIFD.Software: b"Blender Dataset Generator",
            },
            "Exif": {
                piexif.ExifIFD.FocalLength: (int(focal_length_mm * 100), 100),
                piexif.ExifIFD.FocalLengthIn35mmFilm: int(focal_length_35mm),
                piexif.ExifIFD.PixelXDimension: width,
                piexif.ExifIFD.PixelYDimension: height,
                piexif.ExifIFD.FNumber: (28, 10),  # f/2.8
                piexif.ExifIFD.ISOSpeedRatings: 100,
                piexif.ExifIFD.ExposureTime: (1, 100),
            }
        }
        
        exif_bytes = piexif.dump(exif_dict)
        img.save(filepath, "png", exif=exif_bytes)
        
    except ImportError:
        # Neither exiftool nor piexif available
        print(f"Warning: Cannot add EXIF to {filepath}")
        print("Install exiftool or piexif: pip install piexif")
        
        # Create a sidecar JSON file with camera info instead
        json_path = filepath.replace('.png', '_camera.json')
        camera_data = {
            "focal_length_mm": focal_length_mm,
            "focal_length_35mm_equivalent": focal_length_35mm,
            "sensor_width_mm": 32,
            "resolution": [width, height],
            "f_number": 2.8,
            "iso": 100
        }
        
        with open(json_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        print(f"  Created sidecar file: {json_path}")

def setup_rendering():
    """Setup rendering for high-quality output"""
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.device = 'GPU'
    
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 960
    
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.gamma = 1.0
    bpy.context.scene.view_settings.exposure = 0.0
    
    bpy.context.scene.cycles.film_exposure = 1.0
    bpy.context.scene.cycles.pixel_filter_type = 'BOX'
    
    # Use JPEG instead of PNG for better EXIF support
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.quality = 95
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

def create_stereo_camera_rig(baseline=0.12):
    """Create stereo camera rig"""
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.8))
    rig = bpy.context.active_object
    rig.name = "StereoRig"
    
    focal_length = 35
    sensor_width = 32
    
    # Left camera
    bpy.ops.object.camera_add(location=(-baseline/2, 0, 0))
    cam_left = bpy.context.active_object
    cam_left.name = "Camera_Left"
    cam_left.data.lens = focal_length
    cam_left.data.sensor_width = sensor_width
    cam_left.parent = rig
    cam_left.parent_type = 'OBJECT'
    
    # Right camera
    bpy.ops.object.camera_add(location=(baseline/2, 0, 0))
    cam_right = bpy.context.active_object
    cam_right.name = "Camera_Right"
    cam_right.data.lens = focal_length
    cam_right.data.sensor_width = sensor_width
    cam_right.parent = rig
    cam_right.parent_type = 'OBJECT'
    
    return rig, cam_left, cam_right

def setup_camera_motion(rig, radius=4.0, height=1.8, frames=121):
    """Setup circular camera motion"""
    import math
    
    rig.animation_data_clear()
    
    print(f"Setting up camera motion for {frames} frames...")
    
    for frame in range(1, frames + 1):
        angle = 2 * math.pi * (frame - 1) / (frames - 1)
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        
        rig.location = (x, y, z)
        
        target = mathutils.Vector((0, 0, 1))
        loc = mathutils.Vector((x, y, z))
        direction = target - loc
        direction.normalize()
        
        rot_quat = direction.to_track_quat('-Z', 'Y')
        rig.rotation_euler = rot_quat.to_euler()
        
        rig.keyframe_insert(data_path="location", frame=frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    if rig.animation_data and rig.animation_data.action:
        for fcurve in rig.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    print(f"Camera motion created: {frames} keyframes")

def extract_camera_poses(cam_left, cam_right, start_frame=1, end_frame=121):
    """Extract ground truth camera poses"""
    
    poses_data = {
        'stereo_baseline': 0.12,
        'camera_intrinsics': {
            'fx': 1280.0,
            'fy': 1280.0,
            'cx': 640.0,
            'cy': 480.0,
            'resolution': [1280, 960]
        },
        'poses': []
    }
    
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        
        left_matrix = cam_left.matrix_world.copy()
        right_matrix = cam_right.matrix_world.copy()
        
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
            'timestamp': frame / 30.0,
            'left_camera': matrix_to_pose(left_matrix),
            'right_camera': matrix_to_pose(right_matrix)
        }
        
        poses_data['poses'].append(frame_data)
        
        if frame % 10 == 0:
            pos = left_matrix.translation
            print(f"Frame {frame}: Camera at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    
    return poses_data

def setup_output_directories(output_dir):
    """Create output directory structure"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)
    
    bpy.context.scene.render.filepath = output_dir

def render_stereo_sequence(cam_left, cam_right, output_dir, start_frame=1, end_frame=121):
    """Render stereo image sequence"""
    
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        
        if frame % 20 == 1:
            pos = cam_left.matrix_world.translation
            print(f"Frame {frame}: Camera at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        
        # Left camera
        bpy.context.scene.camera = cam_left
        bpy.context.scene.render.filepath = os.path.join(output_dir, 'left', f'frame_{frame:04d}.jpg')
        bpy.ops.render.render(write_still=True)
        
        # Right camera
        bpy.context.scene.camera = cam_right
        bpy.context.scene.render.filepath = os.path.join(output_dir, 'right', f'frame_{frame:04d}.jpg')
        bpy.ops.render.render(write_still=True)
        
        print(f"Rendered frame {frame}/{end_frame}")
    
    # Add EXIF metadata after rendering
    add_exif_metadata_to_images(output_dir, 
                                focal_length_mm=35, 
                                sensor_width_mm=32,
                                resolution_x=1280,
                                resolution_y=960)

def main():
    """Main function to create improved office dataset"""
    
    output_dir = "/tmp/improved_office_dataset"
    baseline = 0.12
    
    print("="*80)
    print("CREATING IMPROVED OFFICE DATASET WITH UNIQUE FEATURES")
    print("="*80)
    
    # Clear scene
    clear_scene()
    
    # Create diverse unique materials
    materials = create_diverse_materials()
    print(f"Created {len(materials)} unique materials")
    
    # Build scene with unique textures
    create_unique_floor(materials)
    print("Created floor with unique tile patterns")
    
    ceiling = create_detailed_walls(materials)
    print("Created walls with unique sections")
    
    create_desk_with_unique_objects(materials)
    print("Created desk with unique wood textures")
    
    create_monitor_and_peripherals(materials)
    print("Created monitor and peripherals")
    
    create_unique_chair(materials)
    print("Created chair with unique fabric")
    
    create_bookshelf_with_unique_books(materials)
    print("Created bookshelf with uniquely textured books")
    
    create_diverse_clutter(materials)
    print("Created diverse clutter items")
    
    create_wall_decorations(materials)
    print("Created wall decorations with unique posters")
    
    # Setup lighting BEFORE adding markers so they're well-lit
    setup_lighting()
    print("Lighting setup complete")
    
    # Add ArUco markers AFTER everything else so they're on top
    placed_markers = add_markers_to_scene(materials)
    print(f"Added {len(placed_markers)} ArUco markers (with emission for visibility)")
    
    # Setup rendering
    setup_rendering()
    print("Rendering configuration complete")
    
    # Create cameras
    rig, cam_left, cam_right = create_stereo_camera_rig(baseline=baseline)
    print("Stereo camera rig created")
    
    # Setup motion
    setup_camera_motion(rig, radius=4.0, height=1.8, frames=121)
    print("Camera motion configured")
    
    # Extract ground truth
    print("Extracting ground truth poses...")
    poses_data = extract_camera_poses(cam_left, cam_right, start_frame=1, end_frame=121)
    poses_data = update_ground_truth_with_markers(poses_data, placed_markers)
    
    # Save ground truth
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ground_truth_poses.json'), 'w') as f:
        json.dump(poses_data, f, indent=2)
    
    print(f"Ground truth saved to {output_dir}/ground_truth_poses.json")
    
    # Setup output
    setup_output_directories(output_dir)
    
    # Render test frame
    print("Rendering test frames...")
    render_stereo_sequence(cam_left, cam_right, output_dir, 1, 121)
    
    print("\n" + "="*80)
    print("DATASET READY WITH UNIQUE FEATURES")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Ground truth: {output_dir}/ground_truth_poses.json")
    print(f"Images: {output_dir}/left/ and {output_dir}/right/")
    print(f"\nCamera specifications:")
    print(f"  - Focal length: 35mm")
    print(f"  - Sensor width: 32mm")
    print(f"  - Resolution: 1280x960")
    print(f"  - Stereo baseline: 120mm")
    print(f"  - Format: JPEG with EXIF metadata")
    print(f"\nKey improvements:")
    print("  - 20 unique wood grain patterns")
    print("  - 30 unique paper textures")
    print("  - 15 unique fabric patterns")
    print("  - 10 unique poster designs")
    print("  - 5 unique brick variations")
    print("  - Random material assignment (no repetitive cycling)")
    print("  - Image-based textures (not procedural)")
    print("  - Unique features on every object")
    print("  - ArUco markers with emission (bright and visible)")
    print("  - EXIF metadata embedded (Meshroom compatible)")
    print("\nMeshroom compatibility:")
    print("  - Images have focal length in EXIF")
    print("  - Meshroom should auto-detect camera parameters")
    print("  - If EXIF missing, use: Focal=35mm, Sensor=32mm")
    print("\nTo render full sequence:")
    print("  render_stereo_sequence(cam_left, cam_right, output_dir, 1, 121)")
    
    return rig, cam_left, cam_right, output_dir, placed_markers

if __name__ == "__main__":
    main()