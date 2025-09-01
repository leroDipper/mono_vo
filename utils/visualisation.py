import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque

class TrajectoryVisualizer:
    def __init__(self, max_points=100, window_size=(10, 8)):
        """
        Initialize the trajectory visualizer
        
        Args:
            max_points: Maximum number of trajectory points to keep in memory
            window_size: Size of the matplotlib window (width, height)
        """
        self.max_points = max_points
        self.trajectory_points = deque(maxlen=max_points)
        
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=window_size)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize empty line for trajectory
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Camera Trajectory')
        self.current_pos_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
        
        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Real-time Camera Trajectory')
        self.ax.legend()
        
        # Set initial view
        self.ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
    def add_pose(self, pose_matrix):
        """
        Add a new pose to the trajectory
        
        Args:
            pose_matrix: 4x4 transformation matrix
        """
        if pose_matrix is not None:
            # Extract position from pose matrix
            position = pose_matrix[:3, 3]
            self.trajectory_points.append(position)
            
    def update_plot(self):
        """
        Update the 3D plot with current trajectory
        """
        if len(self.trajectory_points) < 1:
            return
            
        # Convert trajectory points to arrays
        trajectory_array = np.array(self.trajectory_points)
        x_traj = trajectory_array[:, 0]
        y_traj = trajectory_array[:, 1]
        z_traj = trajectory_array[:, 2]
        
        # Update trajectory line
        self.trajectory_line.set_data_3d(x_traj, y_traj, z_traj)
        
        # Update current position point
        if len(self.trajectory_points) > 0:
            current_pos = self.trajectory_points[-1]
            self.current_pos_point.set_data_3d([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Auto-adjust the plot limits
        if len(trajectory_array) > 0:
            margin = 1.0  # Add some margin around the trajectory
            
            x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
            y_min, y_max = y_traj.min() - margin, y_traj.max() + margin
            z_min, z_max = z_traj.min() - margin, z_traj.max() + margin
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save_trajectory(self, filename='trajectory.png'):
        """
        Save the current trajectory plot
        
        Args:
            filename: Output filename for the saved plot
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Trajectory saved as {filename}")
        
    def close(self):
        """
        Close the visualization window
        """
        plt.ioff()
        plt.close(self.fig)
        
    def get_trajectory_array(self):
        """
        Get the trajectory as a numpy array
        
        Returns:
            numpy array of shape (N, 3) containing trajectory points
        """
        if len(self.trajectory_points) > 0:
            return np.array(self.trajectory_points)
        else:
            return np.array([]).reshape(0, 3)