"""
Volume rendering utilities for 3D skeletal data visualization using PyVista.
"""

import numpy as np
import pyvista as pv
import torch
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt


class VolumeRenderer:
    """
    Volume renderer for 3D binary skeletal data with customizable camera positioning using PyVista.
    
    Attributes:
        volume (np.ndarray): 3D binary volume data
        volume_center (np.ndarray): Center coordinates of the volume
        volume_bounds (tuple): Min and max coordinates of non-zero voxels
    """
    
    def __init__(self, volume):
        """
        Initialize volume renderer with 3D binary data.
        
        Args:
            volume (np.ndarray or torch.Tensor): 3D binary volume data
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        self.volume = volume.astype(np.uint8)
        self._compute_volume_properties()
    
    def _compute_volume_properties(self):
        """
        Compute volume center and bounding box from non-zero voxels.
        """
        non_zero_coords = np.where(self.volume > 0)
        if len(non_zero_coords[0]) == 0:
            raise ValueError("Volume contains no non-zero voxels")
        
        min_coords = np.array([coords.min() for coords in non_zero_coords])
        max_coords = np.array([coords.max() for coords in non_zero_coords])
        
        self.volume_center = (min_coords + max_coords) / 2.0
        self.volume_bounds = (min_coords, max_coords)
    
    def _create_pyvista_mesh(self, render_method='scatter', color='red', 
                            coloring_method='none', colormap='viridis',
                            camera_position=None, camera_target=None):
        """
        Create PyVista mesh from volume data based on render method with optional coloring.
        
        Args:
            render_method (str): Rendering method - 'scatter', 'voxels', 'surface', 'wireframe'
            color (str): Base color for rendering (used when coloring_method='none')
            coloring_method (str): 'none', 'X', 'Y', 'Z', 'Z_camera'
            colormap (str): 'viridis', 'blue'
            camera_position (np.ndarray): Camera position for Z_camera coloring
            camera_target (np.ndarray): Camera target for Z_camera coloring
            
        Returns:
            pyvista mesh object with color data
        """
        if render_method == 'scatter':
            # A. Points rendering
            x, y, z = np.where(self.volume > 0)
            points = np.column_stack((x, y, z))
            point_cloud = pv.PolyData(points)
            
            # Apply coloring
            if coloring_method != 'none':
                colors = self._compute_colors(points, coloring_method, colormap, 
                                            camera_position, camera_target)
                point_cloud.point_data['colors'] = colors
            
            return point_cloud
            
        elif render_method in ['voxels', 'surface', 'wireframe']:
            # B, C, D. Volume-based rendering - create structured grid
            grid = pv.ImageData(dimensions=self.volume.shape)
            grid.point_data['values'] = self.volume.flatten(order='F')
            
            # Apply coloring to grid points before contouring
            if coloring_method != 'none':
                # Get all grid points
                all_points = grid.points
                # Filter only active voxel points
                active_mask = grid.point_data['values'] > 0
                active_points = all_points[active_mask]
                
                if len(active_points) > 0:
                    colors = self._compute_colors(active_points, coloring_method, colormap,
                                                camera_position, camera_target)
                    # Create full color array
                    full_colors = np.zeros(len(all_points))
                    full_colors[active_mask] = colors[:, 0] if colors.ndim > 1 else colors
                    grid.point_data['colors'] = full_colors
            
            # Extract surface where volume > 0
            surface = grid.contour([0.5])
            return surface
        else:
            raise ValueError(f"Unknown render_method: {render_method}")
    
    def _compute_colors(self, points, coloring_method, colormap, camera_position=None, camera_target=None):
        """
        Compute colors for points based on coloring method and colormap.
        
        Args:
            points (np.ndarray): Points coordinates (N, 3)
            coloring_method (str): 'X', 'Y', 'Z', 'Z_camera'
            colormap (str): 'viridis', 'blue'
            camera_position (np.ndarray): Camera position for Z_camera
            camera_target (np.ndarray): Camera target for Z_camera
            
        Returns:
            np.ndarray: Color values or RGB colors
        """
        if coloring_method == 'X':
            values = points[:, 0]  # X coordinates
        elif coloring_method == 'Y':
            values = points[:, 1]  # Y coordinates
        elif coloring_method == 'Z':
            values = points[:, 2]  # Z coordinates
        elif coloring_method == 'Z_camera':
            if camera_position is None or camera_target is None:
                raise ValueError("Camera position and target required for Z_camera coloring")
            
            # Calculate Z_camera values (depth in camera coordinate system)
            camera_position = np.array(camera_position, dtype=float)
            camera_target = np.array(camera_target, dtype=float)
            
            # Camera Z axis (viewing direction)
            z_cam_axis = camera_target - camera_position
            z_cam_axis = z_cam_axis / np.linalg.norm(z_cam_axis)
            
            # Project each point onto camera Z axis
            relative_points = points - camera_position
            values = np.dot(relative_points, z_cam_axis)
        else:
            raise ValueError(f"Unknown coloring_method: {coloring_method}")
        
        # Normalize values to [0, 1]
        if len(values) > 0:
            values_min, values_max = values.min(), values.max()
            if values_max > values_min:
                values_normalized = (values - values_min) / (values_max - values_min)
            else:
                values_normalized = np.zeros_like(values)
        else:
            values_normalized = np.array([])
        
        # Apply colormap
        if colormap == 'viridis':
            # Use matplotlib viridis colormap
            cmap = plt.cm.viridis
            colors = cmap(values_normalized)[:, :3]  # RGB only, no alpha
        elif colormap == 'blue':
            # Blue gradient: dark blue to light blue
            colors = np.zeros((len(values_normalized), 3))
            colors[:, 2] = 0.2 + 0.8 * values_normalized  # Blue channel varies
            colors[:, 0] = 0.1 * values_normalized  # Slight red for lighter blues
            colors[:, 1] = 0.1 * values_normalized  # Slight green for lighter blues
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
        
        return colors
    
    def render_projection(self, camera_position, camera_target=None, 
                         up_vector=None, image_size=(224, 224), 
                         render_method='scatter', alpha=0.6, color='red',
                         coloring_method='none', colormap='viridis', point_size=2):
        """
        Render 2D projection of the volume from specified camera position using PyVista.
        
        Camera coordinate system:
        - Z axis: direction from camera_position to camera_target (viewing direction)
        - X axis: tangent to latitude circle in XY plane of ROI volume
        - Y axis: perpendicular to Z and X (completes right-hand coordinate system)
        
        Args:
            camera_position (tuple or list): Camera position (x, y, z)
            camera_target (tuple or list, optional): Point camera looks at. 
                                                   Defaults to volume center.
            up_vector (tuple or list, optional): Camera up direction. 
                                                Defaults to (0, 0, 1).
            image_size (tuple): Output image dimensions (width, height)
            render_method (str): Rendering method - 'scatter', 'voxels', 'surface', 'wireframe'
            alpha (float): Transparency value (0.0 to 1.0)
            color (str): Base color for rendering (used when coloring_method='none')
            coloring_method (str): Coloring method - 'none', 'X', 'Y', 'Z', 'Z_camera'
            colormap (str): Colormap type - 'viridis', 'blue'
            point_size (int): Size of points for scatter rendering
        
        Returns:
            np.ndarray: 2D rendered image as RGB array with shape (H, W, 3)
        """
        if camera_target is None:
            camera_target = self.volume_center
        if up_vector is None:
            up_vector = (0, 0, 1)
        
        camera_position = np.array(camera_position, dtype=float)
        camera_target = np.array(camera_target, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        
        # Calculate camera coordinate system
        # Z axis: viewing direction (camera to target)
        z_cam = camera_target - camera_position
        z_cam = z_cam / np.linalg.norm(z_cam)
        
        # X axis: tangent to latitude circle in XY plane of ROI
        # Project camera position to XY plane and compute tangent
        cam_xy_projection = np.array([camera_position[0], camera_position[1], 0])
        cam_xy_norm = np.linalg.norm(cam_xy_projection)
        
        if cam_xy_norm > 1e-6:  # Avoid division by zero if camera is on Z axis
            # Tangent to circle: rotate projection by 90° in XY plane
            x_cam = np.array([-camera_position[1], camera_position[0], 0])
            x_cam = x_cam / np.linalg.norm(x_cam)
        else:
            # Fallback: camera on Z axis, use arbitrary X direction
            x_cam = np.array([1, 0, 0])
        
        # Y axis: perpendicular to Z and X (recomputed for orthogonality)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        
        # Create PyVista plotter
        plotter = pv.Plotter(off_screen=True, window_size=image_size)
        plotter.background_color = 'white'
        
        # Create mesh based on render method with coloring
        mesh = self._create_pyvista_mesh(render_method, color, coloring_method, colormap,
                                        camera_position, camera_target)
        
        # Add mesh to plotter with appropriate rendering style
        if coloring_method == 'none':
            # Use uniform color
            if render_method == 'scatter':
                plotter.add_mesh(mesh, color=color, opacity=alpha, point_size=point_size, render_points_as_spheres=True)
            elif render_method == 'voxels':
                plotter.add_mesh(mesh, color=color, opacity=alpha)
            elif render_method == 'surface':
                plotter.add_mesh(mesh, color=color, opacity=alpha, smooth_shading=True)
            elif render_method == 'wireframe':
                plotter.add_mesh(mesh, color=color, opacity=alpha, style='wireframe', line_width=1)
        else:
            # Use computed colors from colormap
            if render_method == 'scatter':
                plotter.add_mesh(mesh, opacity=alpha, point_size=point_size, render_points_as_spheres=True,
                               scalars='colors', rgb=True)
            elif render_method == 'voxels':
                plotter.add_mesh(mesh, opacity=alpha, scalars='colors', rgb=True)
            elif render_method == 'surface':
                plotter.add_mesh(mesh, opacity=alpha, smooth_shading=True, scalars='colors', rgb=True)
            elif render_method == 'wireframe':
                plotter.add_mesh(mesh, opacity=alpha, style='wireframe', line_width=1,
                               scalars='colors', rgb=True)
        
        # Set camera position and orientation
        plotter.camera.position = camera_position
        plotter.camera.focal_point = camera_target
        plotter.camera.up = y_cam
        
        # Set camera bounds to show full volume
        bounds = [
            self.volume_bounds[0][0], self.volume_bounds[1][0],  # x_min, x_max
            self.volume_bounds[0][1], self.volume_bounds[1][1],  # y_min, y_max
            self.volume_bounds[0][2], self.volume_bounds[1][2]   # z_min, z_max
        ]
        plotter.camera.reset_clipping_range()
        
        # Render to numpy array
        image = plotter.screenshot(return_img=True, transparent_background=False)
        plotter.close()
        
        return image
    
    def save_projection(self, camera_position, output_path, camera_target=None,
                       up_vector=None, image_size=(224, 224), 
                       render_method='scatter', alpha=0.6, color='red',
                       coloring_method='none', colormap='viridis', point_size=2):
        """
        Render and save 2D projection to file using PyVista.
        
        Args:
            camera_position (tuple or list): Camera position (x, y, z)
            output_path (str): Path to save the rendered image
            camera_target (tuple or list, optional): Point camera looks at
            up_vector (tuple or list, optional): Camera up direction
            image_size (tuple): Output image dimensions (width, height)
            render_method (str): Rendering method - 'scatter', 'voxels', 'surface', 'wireframe'
            alpha (float): Transparency value (0.0 to 1.0)
            color (str): Base color for rendering (used when coloring_method='none')
            coloring_method (str): Coloring method - 'none', 'X', 'Y', 'Z', 'Z_camera'
            colormap (str): Colormap type - 'viridis', 'blue'
            point_size (int): Size of points for scatter rendering
        """
        # Create PyVista plotter
        plotter = pv.Plotter(off_screen=True, window_size=image_size)
        plotter.background_color = 'white'
        
        if camera_target is None:
            camera_target = self.volume_center
        if up_vector is None:
            up_vector = (0, 0, 1)
        
        camera_position = np.array(camera_position, dtype=float)
        camera_target = np.array(camera_target, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        
        # Calculate Y camera axis for proper orientation
        z_cam = camera_target - camera_position
        z_cam = z_cam / np.linalg.norm(z_cam)
        
        cam_xy_projection = np.array([camera_position[0], camera_position[1], 0])
        cam_xy_norm = np.linalg.norm(cam_xy_projection)
        
        if cam_xy_norm > 1e-6:
            x_cam = np.array([-camera_position[1], camera_position[0], 0])
            x_cam = x_cam / np.linalg.norm(x_cam)
        else:
            x_cam = np.array([1, 0, 0])
        
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        
        # Create mesh based on render method with coloring
        mesh = self._create_pyvista_mesh(render_method, color, coloring_method, colormap,
                                        camera_position, camera_target)
        
        # Add mesh to plotter with appropriate rendering style
        if coloring_method == 'none':
            # Use uniform color
            if render_method == 'scatter':
                plotter.add_mesh(mesh, color=color, opacity=alpha, point_size=point_size, render_points_as_spheres=True)
            elif render_method == 'voxels':
                plotter.add_mesh(mesh, color=color, opacity=alpha)
            elif render_method == 'surface':
                plotter.add_mesh(mesh, color=color, opacity=alpha, smooth_shading=True)
            elif render_method == 'wireframe':
                plotter.add_mesh(mesh, color=color, opacity=alpha, style='wireframe', line_width=1)
        else:
            # Use computed colors from colormap
            if render_method == 'scatter':
                plotter.add_mesh(mesh, opacity=alpha, point_size=point_size, render_points_as_spheres=True,
                               scalars='colors', rgb=True)
            elif render_method == 'voxels':
                plotter.add_mesh(mesh, opacity=alpha, scalars='colors', rgb=True)
            elif render_method == 'surface':
                plotter.add_mesh(mesh, opacity=alpha, smooth_shading=True, scalars='colors', rgb=True)
            elif render_method == 'wireframe':
                plotter.add_mesh(mesh, opacity=alpha, style='wireframe', line_width=1,
                               scalars='colors', rgb=True)
        
        # Set camera position and orientation
        plotter.camera.position = camera_position
        plotter.camera.focal_point = camera_target
        plotter.camera.up = y_cam
        plotter.camera.reset_clipping_range()
        
        # Save screenshot directly to file
        plotter.screenshot(output_path, transparent_background=False)
        plotter.close()


if __name__ == "__main__":
    pass