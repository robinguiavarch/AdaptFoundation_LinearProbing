"""
Test script for volume rendering with HCP OFC dataset using YAML configuration.
"""

import sys
import os
import yaml
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import HCPOFCDataLoader
from volume_renderer import VolumeRenderer


class HCPRenderingTester:
    """
    Test volume rendering on HCP OFC dataset with configurable parameters.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        loader (HCPOFCDataLoader): Dataset loader
        output_dir (Path): Directory for saving renders
    """
    
    def __init__(self, config_path):
        """
        Initialize tester with configuration file.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Handle data path - prompt if empty or invalid
        self._setup_data_path()
        
        self.loader = None
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_data_path(self):
        """
        Setup data path - validate and convert to absolute path if needed.
        """
        data_path = self.config['dataset']['data_path']
        
        # Convert to Path object
        data_path_obj = Path(data_path)
        
        # If relative path, make it relative to the project root
        if not data_path_obj.is_absolute():
            # Get project root (parent of vol_rendering directory)
            project_root = Path(__file__).parent.parent
            data_path_obj = project_root / data_path
        
        # Validate path exists
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path_obj}")
        
        # Check required files
        required_files = [
            "Lskeleton.npy",
            "Lskeleton_subject.csv", 
            "hcp_OFC_labels.csv"
        ]
        
        missing_files = []
        for file in required_files:
            if not (data_path_obj / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {data_path_obj}: {', '.join(missing_files)}")
        
        # Check splits directory exists
        if not (data_path_obj / "splits").exists():
            print(f"Warning: 'splits' directory not found in {data_path_obj}")
        
        # Update config with absolute path
        self.config['dataset']['data_path'] = str(data_path_obj)
        print(f"Using dataset path: {data_path_obj}")
    
    def load_dataset(self):
        """
        Load HCP OFC dataset using configuration parameters.
        """
        data_path = self.config['dataset']['data_path']
        print(f"Loading HCP OFC dataset from: {data_path}")
        
        self.loader = HCPOFCDataLoader(data_path)
        print("Dataset loaded successfully")
    
    def analyze_volume_properties(self):
        """
        Analyze properties of HCP volumes if enabled in configuration.
        """
        if not self.config['analysis']['enable_volume_stats']:
            return
        
        print("\nAnalyzing volume properties...")
        
        volumes, labels, subject_ids = self.loader.get_test_split()
        max_subjects = self.config['dataset']['max_test_subjects']
        n_samples = min(max_subjects, len(volumes))
        
        volume_stats = {
            'shapes': [],
            'densities': [],
            'centers': [],
            'bounds': []
        }
        
        for i in range(n_samples):
            volume = volumes[i]
            
            # Basic stats
            volume_stats['shapes'].append(volume.shape)
            density = np.sum(volume > 0) / volume.size * 100
            volume_stats['densities'].append(density)
            
            # Spatial properties
            if np.sum(volume > 0) > 0:
                non_zero_coords = np.where(volume > 0)
                min_coords = np.array([coords.min() for coords in non_zero_coords])
                max_coords = np.array([coords.max() for coords in non_zero_coords])
                center = (min_coords + max_coords) / 2.0
                
                volume_stats['centers'].append(center)
                volume_stats['bounds'].append((min_coords, max_coords))
        
        # Print statistics
        print(f"Analyzed {n_samples} volumes:")
        print(f"Shapes: {set(volume_stats['shapes'])}")
        print(f"Density range: {min(volume_stats['densities']):.2f}% - {max(volume_stats['densities']):.2f}%")
        print(f"Average density: {np.mean(volume_stats['densities']):.2f}%")
        
        if volume_stats['centers']:
            avg_center = np.mean(volume_stats['centers'], axis=0)
            print(f"Average center: ({avg_center[0]:.1f}, {avg_center[1]:.1f}, {avg_center[2]:.1f})")
    
    def test_single_subject_all_configs(self, subject_idx=0):
        """
        Test all rendering configurations on a single subject.
        
        Args:
            subject_idx (int): Index of subject to test
        """
        volumes, labels, subject_ids = self.loader.get_test_split()
        
        if subject_idx >= len(volumes):
            print(f"Subject index {subject_idx} out of range")
            return
        
        volume = volumes[subject_idx]
        subject_id = subject_ids[subject_idx]
        label = labels[subject_idx]
        
        print(f"\nTesting subject {subject_id}, label: {label}")
        print(f"Volume shape: {volume.shape}")
        print(f"Non-zero voxels: {np.sum(volume > 0)}")
        
        # Initialize renderer
        renderer = VolumeRenderer(volume)
        
        # Get rendering parameters
        rendering_params = self.config['rendering']
        camera_config = self.config['camera']
        
        # Test each configuration
        for config in self.config['test_configs']:
            config_name = config['name']
            print(f"  Testing configuration: {config_name}")
            
            # Test each camera position
            for cam_idx, cam_pos in enumerate(camera_config['positions']):
                position_name = f"cam_{cam_idx}"
                
                output_path = self.output_dir / f"subject_{subject_id}_label_{label}_{config_name}_{position_name}.png"
                
                try:
                    renderer.save_projection(
                        camera_position=cam_pos,
                        output_path=str(output_path),
                        camera_target=camera_config['target'],
                        up_vector=camera_config['up_vector'],
                        image_size=rendering_params['image_size'],
                        render_method=config['render_method'],
                        alpha=rendering_params['alpha'],
                        color=rendering_params['base_color'],
                        coloring_method=config['coloring_method'],
                        colormap=config['colormap'],
                        point_size=rendering_params['point_size']
                    )
                    print(f"    Saved: {output_path.name}")
                    
                except Exception as e:
                    print(f"    Error with {config_name} - {position_name}: {e}")
    
    def test_multiple_subjects_single_config(self, config_name="scatter_uniform"):
        """
        Test single configuration on multiple subjects.
        
        Args:
            config_name (str): Name of configuration to test
        """
        volumes, labels, subject_ids = self.loader.get_test_split()
        max_subjects = self.config['dataset']['max_test_subjects']
        n_subjects = min(max_subjects, len(volumes))
        
        # Find configuration
        config = None
        for cfg in self.config['test_configs']:
            if cfg['name'] == config_name:
                config = cfg
                break
        
        if config is None:
            print(f"Configuration '{config_name}' not found")
            return
        
        print(f"\nTesting configuration '{config_name}' on {n_subjects} subjects")
        
        # Get rendering parameters
        rendering_params = self.config['rendering']
        camera_config = self.config['camera']
        
        for i in range(n_subjects):
            volume = volumes[i]
            subject_id = subject_ids[i]
            label = labels[i]
            
            print(f"  Subject {subject_id} (label: {label})")
            
            # Initialize renderer
            renderer = VolumeRenderer(volume)
            
            # Test first camera position only
            cam_pos = camera_config['positions'][0]
            output_path = self.output_dir / f"subject_{subject_id}_label_{label}_{config_name}.png"
            
            try:
                renderer.save_projection(
                    camera_position=cam_pos,
                    output_path=str(output_path),
                    camera_target=camera_config['target'],
                    up_vector=camera_config['up_vector'],
                    image_size=rendering_params['image_size'],
                    render_method=config['render_method'],
                    alpha=rendering_params['alpha'],
                    color=rendering_params['base_color'],
                    coloring_method=config['coloring_method'],
                    colormap=config['colormap'],
                    point_size=rendering_params['point_size']
                )
                print(f"    Saved: {output_path.name}")
                
            except Exception as e:
                print(f"    Error: {e}")
    
    def run_comprehensive_test(self):
        """
        Run comprehensive test suite based on configuration.
        """
        print("=" * 60)
        print("HCP OFC Volume Rendering Comprehensive Test")
        print("=" * 60)
        
        # Load dataset
        self.load_dataset()
        
        # Analyze volume properties
        self.analyze_volume_properties()
        
        # Test single subject with all configurations
        if self.config['analysis']['enable_rendering_comparison']:
            print("\n" + "=" * 40)
            print("Testing all configurations on single subject...")
            self.test_single_subject_all_configs(subject_idx=0)
        
        # Test multiple subjects with single configuration
        if self.config['analysis']['save_individual_renders']:
            print("\n" + "=" * 40)
            print("Testing single configuration on multiple subjects...")
            self.test_multiple_subjects_single_config("scatter_uniform")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print(f"Results saved in: {self.output_dir}")


def main():
    """
    Main function to run HCP rendering tests.
    """
    # Define paths
    config_path = Path(__file__).parent / "test_hcp_rendering.yaml"
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Please create the configuration file or adjust the path.")
        return
    
    try:
        # Run tests
        tester = HCPRenderingTester(config_path)
        tester.run_comprehensive_test()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()