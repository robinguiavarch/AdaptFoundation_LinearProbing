"""
Volume preprocessing script for testing optimal rendering configurations.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import HCPOFCDataLoader


class VolumePreprocessor:
    """
    Volume preprocessor for testing optimal rendering configurations.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        loader (HCPOFCDataLoader): Dataset loader
        volume (np.ndarray): Current volume being processed
        volume_center (np.ndarray): Center of the volume
    """
    
    def __init__(self, config_path):
        """
        Initialize preprocessor with configuration file.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = None
        self.volume = None
        self.volume_center = None
        
        # Setup output directory
        output_dir = Path(self.config['output']['save_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self):
        """
        Load dataset and extract specified subject.
        """
        # Setup data path
        data_path = self.config['dataset']['data_path']
        data_path_obj = Path(data_path)
        
        if not data_path_obj.is_absolute():
            project_root = Path(__file__).parent.parent
            data_path_obj = project_root / data_path
        
        print(f"Loading dataset from: {data_path_obj}")
        
        # Load dataset
        self.loader = HCPOFCDataLoader(str(data_path_obj))
        
        # Get test split and select subject
        volumes, labels, subject_ids = self.loader.get_test_split()
        subject_idx = self.config['dataset']['subject_idx']
        
        if subject_idx >= len(volumes):
            raise IndexError(f"Subject index {subject_idx} out of range (max: {len(volumes)-1})")
        
        self.volume = volumes[subject_idx]
        subject_id = subject_ids[subject_idx]
        label = labels[subject_idx]
        
        # Compute volume center
        non_zero_coords = np.where(self.volume > 0)
        if len(non_zero_coords[0]) == 0:
            raise ValueError("Volume contains no non-zero voxels")
        
        min_coords = np.array([coords.min() for coords in non_zero_coords])
        max_coords = np.array([coords.max() for coords in non_zero_coords])
        self.volume_center = (min_coords + max_coords) / 2.0
        
        print(f"Selected subject: {subject_id}, label: {label}")
        print(f"Volume shape: {self.volume.shape}")
        print(f"Non-zero voxels: {np.sum(self.volume > 0)}")
        print(f"Volume center: {self.volume_center}")
    
    def _compute_alpha_values(self, points, camera_position, camera_target):
        """
        Compute alpha values based on configuration.
        
        Args:
            points (np.ndarray): Points coordinates (N, 3)
            camera_position (np.ndarray): Camera position
            camera_target (np.ndarray): Camera target
            
        Returns:
            np.ndarray or float: Alpha values
        """
        alpha_config = self.config['alpha']
        alpha_type = alpha_config['type']
        
        if alpha_type == 'constant':
            return alpha_config['value']
        
        elif alpha_type == 'gradient':
            # Linear gradient alpha array
            alpha_range = alpha_config['range']
            steps = alpha_config['steps']
            return np.linspace(alpha_range[0], alpha_range[1], steps)
        
        elif alpha_type == 'adaptive':
            # Alpha based on Z_camera depth
            alpha_range = alpha_config['range']
            
            # Calculate Z_camera values (depth in camera coordinate system)
            z_cam_axis = camera_target - camera_position
            z_cam_axis = z_cam_axis / np.linalg.norm(z_cam_axis)
            
            # Project points onto camera Z axis
            relative_points = points - camera_position
            z_depths = np.dot(relative_points, z_cam_axis)
            
            # Normalize depths to [0, 1]
            if len(z_depths) > 0 and z_depths.max() > z_depths.min():
                z_normalized = (z_depths - z_depths.min()) / (z_depths.max() - z_depths.min())
            else:
                z_normalized = np.zeros_like(z_depths)
            
            # Map to alpha range (closer = more opaque)
            alphas = alpha_range[1] - z_normalized * (alpha_range[1] - alpha_range[0])
            return alphas
        
        else:
            raise ValueError(f"Unknown alpha type: {alpha_type}")
    
    def _compute_colormap_scalars(self, points, camera_position, camera_target):
        """
        Compute scalar values for colormap based on configuration.
        
        Args:
            points (np.ndarray): Points coordinates (N, 3)
            camera_position (np.ndarray): Camera position
            camera_target (np.ndarray): Camera target
            
        Returns:
            np.ndarray: Scalar values for coloring
        """
        colormap_config = self.config['colormap']
        colormap_type = colormap_config['type']
        
        if colormap_type == 'standard':
            # Use Z coordinate as scalar
            return points[:, 2]
        
        elif colormap_type == 'depth':
            # Use Z_camera depth as scalar
            z_cam_axis = camera_target - camera_position
            z_cam_axis = z_cam_axis / np.linalg.norm(z_cam_axis)
            
            relative_points = points - camera_position
            z_depths = np.dot(relative_points, z_cam_axis)
            return z_depths
        
        elif colormap_type == 'orientation':
            # Use X coordinate as proxy for orientation
            return points[:, 0]
        
        else:
            raise ValueError(f"Unknown colormap type: {colormap_type}")
    
    def _create_mesh(self, camera_position, camera_target):
        """
        Create PyVista mesh from volume with configured parameters.
        
        Args:
            camera_position (np.ndarray): Camera position
            camera_target (np.ndarray): Camera target
            
        Returns:
            pyvista mesh object
        """
        render_method = self.config['rendering']['method']
        
        if render_method == 'scatter':
            # Create point cloud
            x, y, z = np.where(self.volume > 0)
            points = np.column_stack((x, y, z))
            mesh = pv.PolyData(points)
            
            # Compute scalars and alpha
            scalars = self._compute_colormap_scalars(points, camera_position, camera_target)
            alphas = self._compute_alpha_values(points, camera_position, camera_target)
            
            mesh.point_data['colors'] = scalars
            if isinstance(alphas, np.ndarray):
                mesh.point_data['alpha'] = alphas
            
            return mesh, alphas
        
        elif render_method == 'voxels':
            # Create volume grid
            grid = pv.ImageData(dimensions=self.volume.shape)
            grid.point_data['values'] = self.volume.flatten(order='F')
            
            # For voxels, alpha is handled differently
            alpha_values = self._compute_alpha_values([], camera_position, camera_target)
            
            return grid, alpha_values
        
        else:
            raise ValueError(f"Unsupported render method: {render_method}")
    
    def render_volume(self, interactive=False):
        """
        Render volume with current configuration.
        
        Args:
            interactive (bool): If True, open interactive window instead of off-screen rendering
        
        Returns:
            np.ndarray or None: Rendered image as RGB array (None if interactive)
        """
        # Get configuration
        camera_config = self.config['camera']
        lighting_config = self.config['lighting']
        output_config = self.config['output']
        colormap_config = self.config['colormap']
        
        # Setup camera
        camera_position = np.array(camera_config['position'], dtype=float)
        camera_target = np.array(self.volume_center if camera_config['target'] is None 
                                else camera_config['target'], dtype=float)
        up_vector = np.array(camera_config['up_vector'], dtype=float)
        
        # Calculate camera coordinate system
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
        
        # Create plotter
        if interactive:
            print("Opening interactive visualization window...")
            print("Close the window when you're done exploring the volume.")
            window_size = [800, 600]  # Larger window for interaction
            plotter = pv.Plotter(window_size=window_size)
        else:
            window_size = output_config['resolution']
            plotter = pv.Plotter(off_screen=True, window_size=window_size)
        
        plotter.background_color = 'white'
        
        # Create mesh
        mesh, alpha_values = self._create_mesh(camera_position, camera_target)
        
        # Add mesh to plotter
        render_method = self.config['rendering']['method']
        colormap_name = colormap_config['name']
        
        if render_method == 'scatter':
            point_size = self.config['rendering']['point_size']
            
            if isinstance(alpha_values, np.ndarray):
                # Use computed alpha per point
                plotter.add_mesh(
                    mesh, 
                    scalars='colors',
                    cmap=colormap_name,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    opacity='alpha'  # Use alpha from point data
                )
            else:
                # Use constant alpha
                plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    cmap=colormap_name,
                    opacity=alpha_values,
                    point_size=point_size,
                    render_points_as_spheres=True
                )
        
        elif render_method == 'voxels':
            # Add volume with opacity
            if isinstance(alpha_values, np.ndarray):
                opacity = alpha_values
            else:
                # Create opacity array for constant alpha
                opacity = np.ones(256) * alpha_values
            
            plotter.add_volume(
                mesh,
                cmap=colormap_name,
                opacity=opacity,
                shade=lighting_config['shade']
            )
        
        # Configure lighting
        if lighting_config.get('eye_dome_lighting', False):
            plotter.enable_eye_dome_lighting()
        
        # Set camera
        plotter.camera.position = camera_position
        plotter.camera.focal_point = camera_target
        plotter.camera.up = y_cam
        plotter.camera.reset_clipping_range()
        
        if interactive:
            # Show interactive window
            plotter.show()
            return None
        else:
            # Render image
            image = plotter.screenshot(return_img=True, transparent_background=False)
            plotter.close()
            return image
    
    def save_result(self, image):
        """
        Save rendered image to file.
        
        Args:
            image (np.ndarray): Rendered image
        """
        output_config = self.config['output']
        save_path = Path(output_config['save_path'])
        filename = output_config['filename']
        
        output_file = save_path / filename
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        print(f"Saved rendered image to: {output_file}")
    
    def process(self, interactive=False):
        """
        Complete preprocessing pipeline.
        
        Args:
            interactive (bool): If True, open interactive visualization
        """
        print("=" * 60)
        print("Volume Preprocessing Test")
        print("=" * 60)
        
        # Load dataset
        self.load_dataset()
        
        # Render volume
        if interactive:
            print("\nStarting interactive visualization...")
            self.render_volume(interactive=True)
            print("Interactive session completed.")
            
            # Ask if user wants to save the configuration
            save_choice = input("\nSave an image with current configuration? (y/n): ").strip().lower()
            if save_choice == 'y':
                print("Rendering image for saving...")
                image = self.render_volume(interactive=False)
                self.save_result(image)
        else:
            print("\nRendering volume...")
            image = self.render_volume(interactive=False)
            self.save_result(image)
        
        print("\nPreprocessing completed successfully!")


def main():
    """
    Main function to run volume preprocessing.
    """
    parser = argparse.ArgumentParser(description='Volume preprocessing for optimal rendering')
    parser.add_argument('--config', type=str, default='preprocess_volume.yaml',
                       help='Path to configuration file')
    parser.add_argument('--visualize', action='store_true',
                       help='Open interactive visualization window')
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(__file__).parent / args.config
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Available configs:")
        configs_dir = Path(__file__).parent / "configs"
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yaml"):
                print(f"  configs/{config_file.name}")
        return
    
    try:
        # Run preprocessing
        preprocessor = VolumePreprocessor(config_path)
        preprocessor.process(interactive=args.visualize)
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user.")
    except Exception as e:
        print(f"Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()