import sys
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSlider, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QComboBox, QFileDialog, QCheckBox, 
                            QGroupBox, QFormLayout, QSplitter, QTreeWidget, QTreeWidgetItem, 
                            QDoubleSpinBox, QHeaderView, QScrollArea)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
from ..utils.cow_loader import load_cow_model, apply_cow_params_to_model, get_available_cow_models
import os

class SMALViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super(SMALViewer, self).__init__(parent)
        self.model = None
        self.vertices = None
        self.faces = None
        self.animation_frames = []
        self.current_frame = 0
        
        # Camera parameters
        self.distance = 3.0
        
        # Initialize camera with a more intuitive orientation
        # Forward is looking at the model from a slight angle
        self.camera_forward = np.array([0.0, -0.5, -1.0])
        # Up is pointing up (positive y)
        self.camera_up = np.array([0.0, 1.0, 0.0])
        # Right is pointing right
        self.camera_right = np.array([1.0, 0.0, 0.0])
        
        # Normalize vectors
        self.camera_forward = self.camera_forward / np.linalg.norm(self.camera_forward)
        self.camera_right = np.cross(self.camera_up, self.camera_forward)
        self.camera_right = self.camera_right / np.linalg.norm(self.camera_right)
        self.camera_up = np.cross(self.camera_forward, self.camera_right)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)
        
        # Calculate camera position based on distance and forward vector
        self.camera_position = -self.distance * self.camera_forward
        
        # Mouse tracking
        self.last_pos = None
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_playing = False
        
        # Display options
        self.show_skeleton = False
        self.show_joints = False
        
        # Transparency
        self.transparency_enabled = True
        self.transparency_alpha = 0.3  # Default alpha value
        
        # FPS calculation
        self.frame_times = []
        self.last_frame_time = time.time()
        self.fps = 0
        
        # Add these new attributes for joint manipulation
        self.selected_joint = None
        self.joint_manipulation_mode = False
        self.joint_rotation_axis = 0  # 0=X, 1=Y, 2=Z
        self.last_joint_rotation = None
        
    def set_model(self, model, animation_frames=None, model_path=None, animation_path=None):
        """Set the SMAL model to visualize"""
        self.model = model
        self.faces = model.f
        
        # Update vertices based on current model state
        # This is crucial - we need to get the vertices after shape parameters have been applied
        self.vertices = model.forward(pose=model.pose, betas=model.betas)
        
        if animation_frames is not None:
            self.animation_frames = animation_frames
        
        self.update()
    
    def update_animation(self):
        """Update the animation to the next frame"""
        if not self.animation_frames or not self.model:
            return
        
        self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
        pose = self.animation_frames[self.current_frame]
        
        # Use forward instead of set_params to ensure shape parameters are preserved
        self.vertices = self.model.forward(pose=pose, betas=self.model.betas)
        self.update()
    
    def toggle_animation(self):
        """Toggle animation playback"""
        if self.animation_playing:
            self.animation_timer.stop()
            self.animation_playing = False
        else:
            self.animation_timer.start(33)  # ~30 FPS
            self.animation_playing = True
    
    def initializeGL(self):
        """Initialize OpenGL"""
        # Change background to white
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        
        # Set up material
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1])
        glMaterialf(GL_FRONT, GL_SHININESS, 50)
    
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height)
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the scene using OpenGL"""
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Keep a rolling average of frame times
        self.frame_times.append(dt)
        if len(self.frame_times) > 30:  # Average over 30 frames
            self.frame_times.pop(0)
        
        # Calculate FPS from average frame time
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Clear the buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up the camera
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Position the camera using our trackball system
        eye = self.camera_position
        center = eye + self.camera_forward
        up = self.camera_up
        gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
        
        # Draw the grid first
        self.draw_grid()
        
        # Draw coordinate axes (will be drawn in corner)
        self.draw_axes()
        
        # Check if we have a model to render
        if self.model is None or self.vertices is None or self.faces is None:
            return
        
        # First draw the skeleton and joints if they're enabled
        # This way they'll be visible through the transparent mesh
        if (self.show_skeleton or self.show_joints) and self.transparency_enabled:
            # Disable lighting for skeleton and joints
            glDisable(GL_LIGHTING)
            
            # Draw the skeleton if enabled
            if self.show_skeleton:
                self.draw_skeleton()
            
            # Draw the joints if enabled
            if self.show_joints:
                self.draw_joints()
            
            # Re-enable lighting for the mesh
            glEnable(GL_LIGHTING)
        
        # Now draw the mesh
        
        # Enable lighting for better 3D appearance
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up light position
        light_position = [1.0, 1.0, 1.0, 0.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Set up transparency state
        glDisable(GL_BLEND)  # Start with blending disabled
        glDepthMask(GL_TRUE)  # Start with depth mask enabled
        
        if self.transparency_enabled:
            # Enable transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)  # Don't write to depth buffer for transparent objects
            glColor4f(0.8, 0.8, 0.8, self.transparency_alpha)  # Use the stored alpha value
        else:
            # Opaque rendering
            glColor4f(0.8, 0.8, 0.8, 1.0)  # Light gray with no transparency
        
        # Calculate vertex normals for smooth shading
        if not hasattr(self, 'vertex_normals') or self.vertex_normals is None:
            self.calculate_vertex_normals()
        
        # Draw the model with smooth shading and transparency
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex_idx in face:
                # Use vertex normal for smooth shading
                if hasattr(self, 'vertex_normals') and self.vertex_normals is not None:
                    normal = self.vertex_normals[vertex_idx]
                    glNormal3fv(normal)
                vertex = self.vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
        
        # Reset OpenGL state after drawing the mesh
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
    
    def calculate_vertex_normals(self):
        """Calculate vertex normals for smooth shading"""
        # Initialize vertex normals
        self.vertex_normals = np.zeros_like(self.vertices, dtype=np.float32)
        
        # For each face, calculate its normal and add to vertex normals
        for face in self.faces:
            # Get vertices of the face
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Calculate face normal using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Normalize the face normal
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal = face_normal / norm
            
            # Add face normal to each vertex of the face
            for vertex_idx in face:
                self.vertex_normals[vertex_idx] += face_normal
        
        # Normalize all vertex normals
        for i in range(len(self.vertex_normals)):
            norm = np.linalg.norm(self.vertex_normals[i])
            if norm > 0:
                self.vertex_normals[i] = self.vertex_normals[i] / norm
    
    def draw_axes(self):
        """Draw coordinate axes in the corner of the screen with labels"""
        # Save current matrices
        glPushMatrix()
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        
        # Disable lighting for the axes
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Move to bottom right corner and scale down
        glViewport(self.width() - 100, 0, 100, 100)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45, 1.0, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Use the same camera orientation as the main view
        # This makes the axes rotate with the camera
        eye = np.array([1.0, 1.0, 1.0])  # Fixed position for the corner view
        center = np.array([0.0, 0.0, 0.0])  # Looking at origin
        up = self.camera_up  # Use the same up vector as the main camera
        
        gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
        
        # Draw axes with thinner lines
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0.5, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.5, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 0.5)
        glEnd()
        
        # Draw axis labels (X, Y, Z)
        # X label (at the end of X axis)
        glColor3f(1, 0, 0)  # Red
        self.draw_x_label(0.55, 0, 0, 0.1)
        
        # Y label (at the end of Y axis)
        glColor3f(0, 1, 0)  # Green
        self.draw_y_label(0, 0.55, 0, 0.1)
        
        # Z label (at the end of Z axis)
        glColor3f(0, 0, 1)  # Blue
        self.draw_z_label(0, 0, 0.55, 0.1)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        # Restore viewport
        glViewport(0, 0, self.width(), self.height())
        
        # Restore attributes
        glPopAttrib()
        glPopMatrix()

    def draw_x_label(self, x, y, z, size):
        """Draw an X label using lines"""
        glBegin(GL_LINES)
        # Draw an X shape
        glVertex3f(x - size/2, y - size/2, z)
        glVertex3f(x + size/2, y + size/2, z)
        
        glVertex3f(x - size/2, y + size/2, z)
        glVertex3f(x + size/2, y - size/2, z)
        glEnd()

    def draw_y_label(self, x, y, z, size):
        """Draw a Y label using lines"""
        glBegin(GL_LINES)
        # Draw a Y shape
        glVertex3f(x, y, z)
        glVertex3f(x, y + size/2, z)
        
        glVertex3f(x - size/2, y + size, z)
        glVertex3f(x, y + size/2, z)
        
        glVertex3f(x + size/2, y + size, z)
        glVertex3f(x, y + size/2, z)
        glEnd()

    def draw_z_label(self, x, y, z, size):
        """Draw a Z label using lines"""
        glBegin(GL_LINES)
        # Draw a Z shape
        glVertex3f(x - size/2, y + size/2, z)
        glVertex3f(x + size/2, y + size/2, z)
        
        glVertex3f(x + size/2, y + size/2, z)
        glVertex3f(x - size/2, y - size/2, z)
        
        glVertex3f(x - size/2, y - size/2, z)
        glVertex3f(x + size/2, y - size/2, z)
        glEnd()
    
    def draw_grid(self):
        """Draw a grid on the y=0 plane (ground plane)"""
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        
        # Disable lighting for the grid
        glDisable(GL_LIGHTING)
        
        # Set grid color (darker gray for better visibility on white background)
        glColor3f(0.5, 0.5, 0.5)  # Darker gray
        
        # Use thinner lines for the grid
        glLineWidth(1.0)
        
        # Draw grid on y=0 plane (horizontal ground plane)
        glBegin(GL_LINES)
        grid_size = 5
        grid_step = 0.5
        
        # Draw lines along x-axis
        for i in range(-grid_size, grid_size + 1):
            z = i * grid_step
            glVertex3f(-grid_size * grid_step, 0, z)
            glVertex3f(grid_size * grid_step, 0, z)
        
        # Draw lines along z-axis
        for i in range(-grid_size, grid_size + 1):
            x = i * grid_step
            glVertex3f(x, 0, -grid_size * grid_step)
            glVertex3f(x, 0, grid_size * grid_step)
        glEnd()
        
        # Draw origin marker (0,0,0)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(0.0, 0.0, 0.0)  # Black color for origin
        glVertex3f(0, 0, 0)
        glEnd()
        
        # Draw small axes at origin
        glLineWidth(3.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0.3, 0, 0)
        # Y axis (green) - now pointing up
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.3, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 0.3)
        glEnd()
        
        # Restore state
        glPopAttrib()
    
    def draw_skeleton(self):
        """Draw the skeleton with improved visibility"""
        if hasattr(self.model, 'get_transformed_joints'):
            joints = self.model.get_transformed_joints()
            
            # Make lines thicker and brighter
            glLineWidth(5.0)  # Thicker lines
            
            # Use a more vibrant color - bright cyan with high saturation
            glColor3f(0.0, 0.8, 1.0)  # Bright cyan/light blue
            
            glBegin(GL_LINES)
            for i in range(1, len(joints)):
                parent = self.model.kintree_table[0, i]
                glVertex3fv(joints[parent])
                glVertex3fv(joints[i])
            glEnd()
    
    def draw_joints(self):
        """Draw the joints with improved visibility"""
        if hasattr(self.model, 'get_transformed_joints'):
            joints = self.model.get_transformed_joints()
            
            # First draw all regular joints
            glPointSize(10.0)  # Set point size for regular joints
            glBegin(GL_POINTS)
            for i, joint in enumerate(joints):
                # Skip the selected joint, we'll draw it separately
                if i == self.selected_joint:
                    continue
                    
                # Use different colors for different joint types with higher saturation
                if i == 0:  # Root joint
                    glColor3f(1.0, 0.8, 0.0)  # Bright gold
                elif i in [1, 2, 3, 4]:  # Leg joints
                    glColor3f(1.0, 0.3, 0.3)  # Bright red
                elif i in [5, 6, 7, 8]:  # Spine/neck/head
                    glColor3f(0.3, 1.0, 0.3)  # Bright green
                else:  # Other joints
                    glColor3f(0.3, 0.3, 1.0)  # Bright blue
                glVertex3fv(joint)
            glEnd()
            
            # Now draw the selected joint with a larger point size
            if self.selected_joint is not None and self.selected_joint < len(joints):
                glPointSize(15.0)  # Larger size for selected joint
                glBegin(GL_POINTS)
                glColor3f(1.0, 0.5, 0.0)  # Bright orange for selected joint
                glVertex3fv(joints[self.selected_joint])
                glEnd()
                
                # Draw rotation axis indicator for the selected joint
                if self.selected_joint is not None:
                    joint_pos = joints[self.selected_joint]
                    axis_length = 0.2  # Length of the axis indicator
                    
                    glLineWidth(4.0)  # Thicker line for the axis
                    glBegin(GL_LINES)
                    
                    # Draw the current rotation axis with a bright color
                    if self.joint_rotation_axis == 0:  # X-axis
                        glColor3f(1.0, 0.2, 0.2)  # Brighter red for X
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0] + axis_length, joint_pos[1], joint_pos[2])
                    elif self.joint_rotation_axis == 1:  # Y-axis
                        glColor3f(0.2, 1.0, 0.2)  # Brighter green for Y
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0], joint_pos[1] + axis_length, joint_pos[2])
                    else:  # Z-axis
                        glColor3f(0.2, 0.2, 1.0)  # Brighter blue for Z
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0], joint_pos[1], joint_pos[2] + axis_length)
                    glEnd()
                    
                    # Draw the other axes with dimmer colors
                    glLineWidth(2.0)  # Thinner lines for other axes
                    glBegin(GL_LINES)
                    
                    # X-axis (if not selected)
                    if self.joint_rotation_axis != 0:
                        glColor3f(0.7, 0.2, 0.2)  # Dimmer red
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0] + axis_length, joint_pos[1], joint_pos[2])
                    
                    # Y-axis (if not selected)
                    if self.joint_rotation_axis != 1:
                        glColor3f(0.2, 0.7, 0.2)  # Dimmer green
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0], joint_pos[1] + axis_length, joint_pos[2])
                    
                    # Z-axis (if not selected)
                    if self.joint_rotation_axis != 2:
                        glColor3f(0.2, 0.2, 0.7)  # Dimmer blue
                        glVertex3fv(joint_pos)
                        glVertex3f(joint_pos[0], joint_pos[1], joint_pos[2] + axis_length)
                    glEnd()
            
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        self.last_pos = event.position()
        
        # Check if we're in joint manipulation mode and left button is pressed
        if event.buttons() & Qt.MouseButton.LeftButton and self.show_joints:
            # Try to select a joint under the cursor
            self.select_joint_at_cursor(event.position())
            # If a joint was selected, return early to prevent camera rotation
            if self.selected_joint is not None:
                return
    
    def mouseMoveEvent(self, event):
        """Handle mouse move event"""
        if self.last_pos is None:
            return
        
        # Calculate mouse movement
        pos = event.position()
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        
        # Check if we're manipulating a joint
        if self.selected_joint is not None and event.buttons() & Qt.MouseButton.LeftButton:
            # We're in joint manipulation mode
            self.joint_manipulation_mode = True
            self.rotate_selected_joint(dx, dy)
            
            # Make sure the window has focus to receive key events during dragging
            self.setFocus()
        # Otherwise use normal camera controls
        elif event.buttons() & Qt.MouseButton.LeftButton:
            # Rotate camera with more intuitive controls
            self.rotate_camera(-dx * 0.01, -dy * 0.01)
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            # Calculate the camera target (look-at point) before changing distance
            camera_target = self.camera_position + self.distance * self.camera_forward
            
            # Zoom in/out
            zoom_factor = 1.0 - dy * 0.01
            self.distance /= zoom_factor
            self.distance = max(min(self.distance, 10.0), 0.5)
            
            # Update camera position based on target point and new distance
            # This preserves any panning that was done
            self.camera_position = camera_target - self.distance * self.camera_forward
            self.update()
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            # Pan the camera in the view plane
            # Calculate pan amount in camera space
            pan_speed = 0.003 * self.distance
            right_amount = dx * pan_speed
            up_amount = dy * pan_speed
            
            # Move camera position in the right and up directions
            self.camera_position += right_amount * self.camera_right + up_amount * self.camera_up
            self.update()
        
        self.last_pos = pos
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        self.last_pos = None
        
        # If we're releasing after joint manipulation, end the manipulation
        if self.joint_manipulation_mode and self.selected_joint is not None:
            self.joint_manipulation_mode = False
            self.last_joint_rotation = None
    
    def reset_view(self):
        """Reset the camera view to the default position"""
        self.distance = 3.0
        
        # Initialize camera with a more intuitive orientation
        self.camera_forward = np.array([0.0, -0.5, -1.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        
        # Normalize vectors
        self.camera_forward = self.camera_forward / np.linalg.norm(self.camera_forward)
        self.camera_right = np.cross(self.camera_up, self.camera_forward)
        self.camera_right = self.camera_right / np.linalg.norm(self.camera_right)
        self.camera_up = np.cross(self.camera_forward, self.camera_right)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)
        
        # Calculate camera position
        self.camera_position = -self.distance * self.camera_forward
        
        self.update()

    def update_cow_model(self, cow_params):
        """Update the model with new cow parameters"""
        if self.model is None:
            return
        
        # Extract shape parameters - check for both 'betas' and 'beta'
        betas = cow_params.get('betas', None)
        if betas is None:
            betas = cow_params.get('beta', None)  # Try singular form
        
        pose = cow_params.get('pose', None) or self.model.pose
        
        if betas is not None:
            print(f"Applying cow shape parameters to viewer: {betas[:5]}...")
            # Ensure betas is a numpy array
            if not isinstance(betas, np.ndarray):
                betas = np.array(betas, dtype=np.float64)
            
            # Update the model using forward
            self.vertices = self.model.forward(pose=pose, betas=betas)
            self.update()
        else:
            print("No shape parameters found in cow model!")

    def set_transparency(self, enabled, alpha=0.3):
        """Set whether the mesh should be transparent and its alpha value"""
        # Update the transparency state
        self.transparency_enabled = enabled
        self.transparency_alpha = alpha  # Store the alpha value
        
        # Print debug information
        print(f"Setting transparency to: {enabled}, alpha: {alpha}")
        
        # Force recalculation of vertex normals on next render
        self.vertex_normals = None
        
        # Force a complete redraw
        self.update()

    def rotate_camera(self, dx, dy):
        """Rotate the camera using a trackball-like approach"""
        # Rotation around the up vector (y-axis)
        rotation_y = np.array([
            [np.cos(dx), 0, np.sin(dx)],
            [0, 1, 0],
            [-np.sin(dx), 0, np.cos(dx)]
        ])
        
        # Rotation around the right vector (x-axis)
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(dy), -np.sin(dy)],
            [0, np.sin(dy), np.cos(dy)]
        ])
        
        # Calculate the camera target (look-at point)
        camera_target = self.camera_position + self.distance * self.camera_forward
        
        # Apply rotations in the correct order for intuitive control
        # First rotate around the right vector (x-axis), then around the world up vector (y-axis)
        self.camera_forward = rotation_y @ (rotation_x @ self.camera_forward)
        self.camera_right = rotation_y @ (rotation_x @ self.camera_right)
        
        # Recalculate the up vector to ensure orthogonality
        self.camera_up = np.cross(self.camera_forward, self.camera_right)
        
        # Normalize vectors
        self.camera_forward = self.camera_forward / np.linalg.norm(self.camera_forward)
        self.camera_right = self.camera_right / np.linalg.norm(self.camera_right)
        self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)
        
        # Update camera position based on the target point and new forward direction
        # This preserves any panning that was done
        self.camera_position = camera_target - self.distance * self.camera_forward

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        # Get the wheel delta (positive for zoom in, negative for zoom out)
        delta = event.angleDelta().y()
        
        # Calculate zoom factor based on wheel delta
        zoom_factor = 1.0 + delta * 0.001  # Adjust sensitivity as needed
        
        # Calculate the camera target (look-at point) before changing distance
        camera_target = self.camera_position + self.distance * self.camera_forward
        
        # Apply zoom
        self.distance /= zoom_factor
        self.distance = max(min(self.distance, 10.0), 0.5)  # Limit zoom range
        
        # Update camera position based on target point and new distance
        # This preserves any panning that was done
        self.camera_position = camera_target - self.distance * self.camera_forward
        
        self.update()

    def update_joint_angle(self, index, value):
        """Update a joint angle parameter"""
        if not hasattr(self, 'model') or self.model is None:
            return
        
        model = self.model
        
        # Create a copy of the current pose
        new_pose = np.array(model.pose).copy()
        
        # Update the specified parameter
        new_pose[index] = value
        
        # Apply the new pose
        self.vertices = model.forward(pose=new_pose, betas=model.betas)
        
        # Update the display
        self.update()

    def update_shape_param(self, index, value):
        """Update a shape parameter"""
        if not hasattr(self, 'model') or self.model is None:
            return
        
        model = self.model
        
        # Create a copy of the current shape parameters
        new_betas = np.array(model.betas).copy()
        
        # Update the specified parameter
        new_betas[index] = value
        
        # Apply the new shape parameters
        self.vertices = model.forward(pose=model.pose, betas=new_betas)
        
        # Update the display
        self.update()

    def update_joint_angle_from_slider(self, index, value, label):
        """Update a joint angle parameter from a slider value"""
        # Convert slider value to actual parameter value
        actual_value = value / 100.0
        
        # Update the label
        label.setText(f"{actual_value:.2f}")
        
        # Update the model
        self.update_joint_angle(index, actual_value)

    def update_shape_param_from_slider(self, index, value, label):
        """Update a shape parameter from a slider value"""
        # Convert slider value to actual parameter value
        actual_value = value / 100.0
        
        # Update the label
        label.setText(f"{actual_value:.2f}")
        
        # Update the model
        self.update_shape_param(index, actual_value)

    def select_joint_at_cursor(self, cursor_pos):
        """Select a joint at the cursor position"""
        if not hasattr(self, 'model') or self.model is None:
            return
        
        if not hasattr(self.model, 'get_transformed_joints'):
            return
        
        # Get the current joint positions
        joints = self.model.get_transformed_joints()
        
        # Make sure joints are visible
        if not self.show_joints:
            print("Joints are not visible. Enable 'Show Joints' to select joints.")
            return
        
        # Force a render to ensure matrices are up to date
        self.makeCurrent()
        self.paintGL()
        
        # Get the current viewport, modelview and projection matrices
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Convert cursor position to OpenGL coordinates
        x = cursor_pos.x()
        y = self.height() - cursor_pos.y()  # Flip y coordinate
        
        # Find the closest joint to the cursor
        closest_joint = None
        closest_distance = float('inf')
        
        for i, joint in enumerate(joints):
            try:
                # Project 3D joint position to 2D screen coordinates
                win_coords = gluProject(
                    joint[0], joint[1], joint[2],
                    modelview_matrix, projection_matrix, viewport
                )
                
                win_x, win_y, win_z = win_coords
                
                # Calculate distance to cursor
                distance = ((win_x - x) ** 2 + (win_y - y) ** 2) ** 0.5
                
                # Check if this joint is closer than the current closest
                # Use a smaller threshold for more precise selection
                if distance < closest_distance and distance < 30:  # 30 pixel threshold
                    closest_joint = i
                    closest_distance = distance
            except Exception as e:
                # Silently handle projection errors
                pass
        
        # Set the selected joint
        self.selected_joint = closest_joint
        
        if self.selected_joint is not None:
            print(f"Selected joint: {self.selected_joint}")
        else:
            print("No joint selected")
        
        # Force a redraw to highlight the selected joint
        self.update()
    
    def rotate_selected_joint(self, dx, dy):
        """Rotate the selected joint based on mouse movement"""
        if self.selected_joint is None or not hasattr(self, 'model') or self.model is None:
            return
        
        # Determine which joint parameter indices to modify
        # For joint 0 (root), use the first 3 parameters
        # For other joints, use parameters starting from index 3
        if self.selected_joint == 0:
            param_start_idx = 0
        else:
            param_start_idx = 3 + (self.selected_joint - 1) * 3
        
        # Create a copy of the current pose
        new_pose = np.array(self.model.pose).copy()
        
        # Apply rotation based on mouse movement
        # Scale the movement to get reasonable rotation amounts
        rotation_scale = 0.01
        
        # Rotate around the current axis
        if self.joint_rotation_axis == 0:  # X-axis
            new_pose[param_start_idx] += dy * rotation_scale
        elif self.joint_rotation_axis == 1:  # Y-axis
            new_pose[param_start_idx + 1] += dx * rotation_scale
        else:  # Z-axis
            new_pose[param_start_idx + 2] += (dx + dy) * rotation_scale
        
        # Apply the new pose
        self.vertices = self.model.forward(pose=new_pose, betas=self.model.betas)
        
        # Update the display
        self.update()
        
        # Update the sliders in the main window
        parent = self.parent()
        if parent and hasattr(parent, 'update_sliders_from_pose'):
            parent.update_sliders_from_pose(new_pose)
    
    def keyPressEvent(self, event):
        """Handle key press events for joint manipulation"""
        # Toggle between rotation axes with X, Y, Z keys
        if event.key() == Qt.Key.Key_X:
            self.joint_rotation_axis = 0
            print("Joint rotation axis: X")
        elif event.key() == Qt.Key.Key_Y:
            self.joint_rotation_axis = 1
            print("Joint rotation axis: Y")
        elif event.key() == Qt.Key.Key_Z:
            self.joint_rotation_axis = 2
            print("Joint rotation axis: Z")
        
        # Update the display
        self.update()

    def update_sliders_from_pose(self, new_pose):
        """Update the joint sliders to reflect the new pose"""
        for i, value in enumerate(new_pose):
            if i in self.joint_spinboxes:
                # Temporarily block signals to avoid recursive updates
                self.joint_spinboxes[i].blockSignals(True)
                self.joint_spinboxes[i].setValue(int(float(value) * 100))
                self.joint_spinboxes[i].blockSignals(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.setWindowTitle("SMAL Cow Model Viewer")
        self.resize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create top section with viewer and status panel
        top_section = QHBoxLayout()
        
        # Create OpenGL viewer (left side)
        self.viewer = SMALViewer()
        top_section.addWidget(self.viewer, 3)  # Reduce from 4 to 3
        
        # Create status panel (right side)
        status_panel = QWidget()
        status_panel.setMinimumWidth(250)  # Set minimum width for the right panel
        status_layout = QVBoxLayout(status_panel)
        
        # FPS display
        fps_group = QGroupBox("Performance")
        fps_layout = QFormLayout()
        self.fps_label = QLabel("0.0 FPS")
        fps_layout.addRow("Frame Rate:", self.fps_label)
        fps_group.setLayout(fps_layout)
        status_layout.addWidget(fps_group)
        
        # Camera parameters
        camera_group = QGroupBox("Camera Parameters")
        camera_layout = QFormLayout()
        self.camera_pos_label = QLabel("(0.0, 0.0, 0.0)")
        self.camera_forward_label = QLabel("(0.0, 0.0, -1.0)")
        self.camera_up_label = QLabel("(0.0, 1.0, 0.0)")
        self.camera_distance_label = QLabel("3.0")
        
        camera_layout.addRow("Position:", self.camera_pos_label)
        camera_layout.addRow("Forward:", self.camera_forward_label)
        camera_layout.addRow("Up:", self.camera_up_label)
        camera_layout.addRow("Distance:", self.camera_distance_label)
        
        camera_group.setLayout(camera_layout)
        status_layout.addWidget(camera_group)
        
        # Add instructions for joint manipulation
        instructions_group = QGroupBox("Joint Manipulation")
        instructions_layout = QVBoxLayout()
        instructions_label = QLabel(
            "1. Enable 'Show Joints'\n"
            "2. Click on a joint to select it\n"
            "3. Press X, Y, Z to choose axis\n"
            "4. Drag with left mouse to rotate"
        )
        instructions_layout.addWidget(instructions_label)
        instructions_group.setLayout(instructions_layout)
        status_layout.addWidget(instructions_group)
        
        # Add transparency controls
        transparency_group = QGroupBox("Transparency")
        transparency_layout = QVBoxLayout()
        
        # Add a slider for transparency
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)  # 0-100 range (will be converted to 0.0-1.0)
        self.transparency_slider.setValue(30)  # Default 30% (0.3)
        self.transparency_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.transparency_slider.setTickInterval(10)
        self.transparency_slider.valueChanged.connect(self.update_transparency)
        
        # Add a label to show the current value
        transparency_value_layout = QHBoxLayout()
        transparency_value_layout.addWidget(QLabel("Opacity:"))
        self.transparency_value_label = QLabel("30%")
        transparency_value_layout.addWidget(self.transparency_value_label)
        
        transparency_layout.addLayout(transparency_value_layout)
        transparency_layout.addWidget(self.transparency_slider)
        
        transparency_group.setLayout(transparency_layout)
        status_layout.addWidget(transparency_group)
        
        # Add shape parameter controls
        shape_group = QGroupBox("Shape Parameters")
        shape_layout = QVBoxLayout()
        
        # Add checkbox to enable/disable shape parameter adjustment
        self.shape_checkbox = QCheckBox("Enable Shape Adjustment")
        self.shape_checkbox.setChecked(False)
        self.shape_checkbox.toggled.connect(self.toggle_shape_adjustment)
        shape_layout.addWidget(self.shape_checkbox)
        
        # Create a scroll area for shape parameters (since there might be many)
        shape_scroll = QScrollArea()
        shape_scroll.setWidgetResizable(True)
        shape_scroll.setMinimumHeight(300)  # Set minimum height to make it taller
        shape_scroll_content = QWidget()
        shape_scroll_layout = QVBoxLayout(shape_scroll_content)
        
        # Create a container for all shape parameter sliders
        self.shape_sliders_container = QWidget()
        self.shape_sliders_layout = QVBoxLayout(self.shape_sliders_container)
        self.shape_sliders = []  # List to store shape parameter sliders
        
        # Initially hide the shape sliders container
        self.shape_sliders_container.setVisible(False)
        
        # Add the container to the scroll area
        shape_scroll_layout.addWidget(self.shape_sliders_container)
        shape_scroll.setWidget(shape_scroll_content)
        
        shape_layout.addWidget(shape_scroll)
        
        shape_group.setLayout(shape_layout)
        status_layout.addWidget(shape_group)
        
        # Add a spacer to push everything to the top
        status_layout.addStretch()
        
        # Add the status panel to the top section
        top_section.addWidget(status_panel, 1)  # Keep at 1, but the panel will be wider
        
        # Add the top section to the main layout
        main_layout.addLayout(top_section)
        
        # Create controls at the bottom
        controls_layout = QHBoxLayout()
        
        # Add a label indicating this is a cow viewer
        cow_label = QLabel("Cow Model Viewer")
        controls_layout.addWidget(cow_label)
        
        # Cow model selector
        cow_layout = QHBoxLayout()
        cow_layout.addWidget(QLabel("Cow Model:"))
        self.cow_combo = QComboBox()
        self.cow_combo.addItem("Default")
        self.cow_combo.currentIndexChanged.connect(self.change_cow_model)
        cow_layout.addWidget(self.cow_combo)
        
        # Add button to load cow model from file
        self.load_cow_button = QPushButton("Load Cow...")
        self.load_cow_button.clicked.connect(self.load_cow_from_file)
        cow_layout.addWidget(self.load_cow_button)
        
        controls_layout.addLayout(cow_layout)
        
        # Visualization options
        viz_layout = QHBoxLayout()
        
        # Skeleton toggle
        self.skeleton_button = QPushButton("Show Skeleton")
        self.skeleton_button.setCheckable(True)
        self.skeleton_button.clicked.connect(self.toggle_skeleton)
        viz_layout.addWidget(self.skeleton_button)
        
        # Joints toggle
        self.joints_button = QPushButton("Show Joints")
        self.joints_button.setCheckable(True)
        self.joints_button.clicked.connect(self.toggle_joints)
        viz_layout.addWidget(self.joints_button)
        
        # Debug button
        self.debug_button = QPushButton("Debug Joints")
        self.debug_button.clicked.connect(self.print_joint_info)
        viz_layout.addWidget(self.debug_button)
        
        # Add reset view button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        viz_layout.addWidget(self.reset_view_button)
        
        # Transparency checkbox
        self.transparency_checkbox = QCheckBox("Transparent Mesh")
        self.transparency_checkbox.setChecked(True)
        self.transparency_checkbox.toggled.connect(self.toggle_transparency)
        viz_layout.addWidget(self.transparency_checkbox)
        
        controls_layout.addLayout(viz_layout)
        
        # Animation controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        controls_layout.addWidget(self.play_button)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Will be updated when animation is loaded
        self.frame_slider.valueChanged.connect(self.change_frame)
        controls_layout.addWidget(self.frame_slider)
        
        # Add the controls to the main layout
        main_layout.addLayout(controls_layout)
        
        self.setCentralWidget(central_widget)
        
        # Store the model path for reloading
        self.model_path = None
        self.animation_path = None
        self.cow_models = []
        
        # Load available cow models
        self.load_available_cow_models()
        
        # Set up a timer to update the parameter display
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_parameter_display)
        self.update_timer.start(100)  # Update 10 times per second
        
        # Store the original pose and shape for reset functionality
        self.original_pose = None
        self.original_shape = None
    
    def load_available_cow_models(self):
        """Load available cow models from the data directory"""
        try:
            self.cow_models = get_available_cow_models()
        except Exception as e:
            print(f"Error loading cow models: {e}")
            self.cow_models = []
        
        # Add cow models to the dropdown
        self.cow_combo.clear()
        self.cow_combo.addItem("Default")
        
        for cow_file in self.cow_models:
            self.cow_combo.addItem(os.path.basename(cow_file))
    
    def change_cow_model(self, index):
        """Change the cow model"""
        if index == 0:
            # Default model - do nothing
            return
        
        # Ensure we have cow models and the index is valid
        if not self.cow_models:
            print("No cow models available")
            return
        
        # Adjust index to account for "Default" option at index 0
        adjusted_index = index - 1
        
        # Check if the adjusted index is valid
        if adjusted_index < 0 or adjusted_index >= len(self.cow_models):
            print(f"Invalid cow model index: {adjusted_index} (available models: {len(self.cow_models)})")
            return
        
        # Load the selected cow model
        cow_file = self.cow_models[adjusted_index]
        print(f"Loading cow model from: {cow_file}")
        
        try:
            cow_params = load_cow_model(cow_file)
            print(f"Loaded cow parameters with keys: {list(cow_params.keys())}")
            
            # Check for shape parameters
            if 'beta' in cow_params:
                print(f"Found beta parameter with shape: {cow_params['beta'].shape}")
            elif 'betas' in cow_params:
                print(f"Found betas parameter with shape: {cow_params['betas'].shape}")
            else:
                print("No shape parameters found in cow model!")
            
            # Apply the cow parameters to the model
            self.viewer.update_cow_model(cow_params)
            
        except Exception as e:
            print(f"Error loading cow model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_cow_from_file(self):
        """Load a cow model from a file selected by the user"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Cow Model", "", "Pickle Files (*.pkl)"
        )
        
        if file_path and file_path.lower().endswith('.pkl') and not os.path.basename(file_path).startswith('.'):
            try:
                # Load the cow model
                cow_params = load_cow_model(file_path)
                
                # Apply the cow parameters to the model
                self.viewer.update_cow_model(cow_params)
                
            except Exception as e:
                print(f"Error loading cow model: {e}")
                # Show error message to user
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to load cow model: {e}")
        elif file_path:
            # Show error for invalid files
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid File", "Please select a valid cow model file (*.pkl)")
    
    def set_model(self, model, animation_frames=None, model_path=None, animation_path=None):
        """Set the SMAL model to visualize"""
        # Store the model path for reloading
        self.model_path = model_path
        self.animation_path = animation_path
        
        # Pass the model to the viewer
        self.viewer.set_model(model, animation_frames)
        
        # Store original pose and shape for reset functionality
        if hasattr(model, 'pose'):
            self.original_pose = np.array(model.pose).copy()
        
        if hasattr(model, 'betas'):
            self.original_shape = np.array(model.betas).copy()
        
        # Update the frame slider if animation frames are provided
        if animation_frames:
            self.frame_slider.setMaximum(len(animation_frames) - 1)
            self.frame_slider.setValue(0)
    
    def toggle_animation(self):
        """Toggle animation playback"""
        self.viewer.toggle_animation()
        # Update button text
        if self.viewer.animation_playing:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")
    
    def change_frame(self, frame_index):
        """Change the current animation frame"""
        if not self.viewer.animation_frames:
            return
        
        self.viewer.current_frame = frame_index
        pose = self.viewer.animation_frames[frame_index]
        
        # Use forward instead of set_params
        self.viewer.vertices = self.viewer.model.forward(pose=pose, betas=self.viewer.model.betas)
        self.viewer.update()
    
    def toggle_skeleton(self):
        """Toggle skeleton visibility"""
        self.viewer.show_skeleton = self.skeleton_button.isChecked()
        self.viewer.update()
    
    def toggle_joints(self):
        """Toggle joint visibility"""
        self.viewer.show_joints = self.joints_button.isChecked()
        self.viewer.update()
    
    def reset_view(self):
        """Reset the camera view to the default position"""
        self.viewer.reset_view()
    
    def print_joint_info(self):
        """Print information about the model's joints for debugging"""
        if not hasattr(self.viewer, 'model') or self.viewer.model is None:
            print("No model loaded")
            return
        
        model = self.viewer.model
        print("\nJoint Information:")
        print(f"Number of joints: {model.num_joints}")
        
        # Print original joint positions
        print("\nOriginal Joint Positions:")
        for i in range(model.num_joints):
            print(f"Joint {i}: {model.J[i]}")
        
        # Print transformed joint positions
        if hasattr(model, 'get_transformed_joints'):
            transformed_joints = model.get_transformed_joints()
            print("\nTransformed Joint Positions:")
            for i in range(model.num_joints):
                print(f"Joint {i}: {transformed_joints[i]}")
        
        # Print kinematic tree
        print("\nKinematic Tree:")
        for i in range(1, model.num_joints):
            parent = model.kintree_table[0, i]
            print(f"Joint {i} -> Parent {parent}")
    
    def toggle_transparency(self, checked):
        """Toggle mesh transparency based on checkbox state"""
        # The 'checked' parameter is a boolean that indicates the new state
        print(f"Checkbox toggled to: {checked}")
        
        # Update the viewer's transparency state
        self.viewer.set_transparency(checked)
    
    def update_parameter_display(self):
        """Update the parameter display with current values"""
        # Update FPS
        self.fps_label.setText(f"{self.viewer.fps:.1f} FPS")
        
        # Update camera parameters
        pos = self.viewer.camera_position
        self.camera_pos_label.setText(f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        forward = self.viewer.camera_forward
        self.camera_forward_label.setText(f"({forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f})")
        
        up = self.viewer.camera_up
        self.camera_up_label.setText(f"({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f})")
        
        self.camera_distance_label.setText(f"{self.viewer.distance:.2f}")
        
        # Update animation button text if needed
        if hasattr(self.viewer, 'animation_playing'):
            if self.viewer.animation_playing:
                self.play_button.setText("Pause")
            else:
                self.play_button.setText("Play")

    def update_joint_controls(self):
        """Update the joint angle controls"""
        if hasattr(self.viewer, 'model') and self.viewer.model is not None:
            model = self.viewer.model
            self.joint_tree.clear()
            self.shape_tree.clear()
            
            # Populate joint angle controls
            for i in range(model.num_joints):
                joint_item = QTreeWidgetItem(self.joint_tree)
                joint_item.setText(0, f"Joint {i}")
                joint_item.setText(1, f"{model.J[i]}")
            
            # Populate shape parameter controls
            for i, param in enumerate(model.betas):
                shape_item = QTreeWidgetItem(self.shape_tree)
                shape_item.setText(0, f"Parameter {i}")
                shape_item.setText(1, f"{param:.4f}")
            
            # Store original pose and shape
            self.original_pose = model.pose
            self.original_shape = model.betas
    
    def update_joint_angle(self, index, value):
        """Update a joint angle parameter"""
        if not hasattr(self.viewer, 'model') or self.viewer.model is None:
            return
        
        model = self.viewer.model
        
        # Create a copy of the current pose
        new_pose = np.array(model.pose).copy()
        
        # Update the specified parameter
        new_pose[index] = value
        
        # Apply the new pose
        self.viewer.vertices = model.forward(pose=new_pose, betas=model.betas)
        
        # Update the display
        self.viewer.update()

    def update_shape_param(self, index, value):
        """Update a shape parameter"""
        if not hasattr(self.viewer, 'model') or self.viewer.model is None:
            return
        
        model = self.viewer.model
        
        # Create a copy of the current shape parameters
        new_betas = np.array(model.betas).copy()
        
        # Update the specified parameter
        new_betas[index] = value
        
        # Apply the new shape parameters
        self.viewer.vertices = model.forward(pose=model.pose, betas=new_betas)
        
        # Update the display
        self.viewer.update()

    def update_transparency(self, value):
        """Update the mesh transparency based on slider value"""
        # Convert slider value (0-100) to alpha value (0.0-1.0)
        alpha = value / 100.0
        
        # Update the label
        self.transparency_value_label.setText(f"{value}%")
        
        # Update the viewer's transparency
        self.viewer.set_transparency(self.transparency_checkbox.isChecked(), alpha)

    def toggle_shape_adjustment(self, enabled):
        """Toggle the visibility of shape parameter controls"""
        if enabled:
            # Create shape parameter sliders if they don't exist
            self.create_shape_parameter_sliders()
            self.shape_sliders_container.setVisible(True)
        else:
            self.shape_sliders_container.setVisible(False)

    def create_shape_parameter_sliders(self):
        """Create sliders for each shape parameter"""
        # Clear existing sliders
        for i in reversed(range(self.shape_sliders_layout.count())):
            self.shape_sliders_layout.itemAt(i).widget().setParent(None)
        self.shape_sliders = []
        
        # Check if we have a model with shape parameters
        if not hasattr(self.viewer, 'model') or self.viewer.model is None:
            label = QLabel("No model loaded")
            self.shape_sliders_layout.addWidget(label)
            return
        
        model = self.viewer.model
        if not hasattr(model, 'betas') or model.betas is None:
            label = QLabel("Model has no shape parameters")
            self.shape_sliders_layout.addWidget(label)
            return
        
        # Create a slider for each shape parameter
        for i, beta_value in enumerate(model.betas):
            # Create a group for this parameter
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add label with current value
            label_layout = QHBoxLayout()
            label = QLabel(f"Shape {i+1}:")
            value_label = QLabel(f"{float(beta_value):.2f}")
            label_layout.addWidget(label)
            label_layout.addWidget(value_label)
            param_layout.addLayout(label_layout)
            
            # Add slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-500, 500)  # -5.0 to 5.0 (scaled by 100)
            slider.setValue(int(float(beta_value) * 100))
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            
            # Connect the slider to update function
            slider.valueChanged.connect(
                lambda value, idx=i, lbl=value_label: self.update_shape_parameter(idx, value/100.0, lbl))
            
            param_layout.addWidget(slider)
            self.shape_sliders_layout.addWidget(param_widget)
            self.shape_sliders.append((slider, value_label))
        
        # Add a reset button at the bottom
        reset_button = QPushButton("Reset All Shape Parameters")
        reset_button.clicked.connect(self.reset_shape_parameters)
        self.shape_sliders_layout.addWidget(reset_button)

    def update_shape_parameter(self, index, value, label=None):
        """Update a shape parameter based on slider value"""
        if not hasattr(self.viewer, 'model') or self.viewer.model is None:
            return
        
        # Update the label if provided
        if label:
            label.setText(f"{value:.2f}")
        
        # Update the model's shape parameter
        self.viewer.update_shape_param(index, value)

    def reset_shape_parameters(self):
        """Reset all shape parameters to their original values"""
        if not hasattr(self.viewer, 'model') or self.viewer.model is None or self.original_shape is None:
            return
        
        # Apply the original shape parameters
        model = self.viewer.model
        self.viewer.vertices = model.forward(pose=model.pose, betas=self.original_shape)
        
        # Update the sliders to reflect the original values
        for i, (slider, label) in enumerate(self.shape_sliders):
            if i < len(self.original_shape):
                value = float(self.original_shape[i])
                slider.setValue(int(value * 100))
                label.setText(f"{value:.2f}")
        
        # Update the display
        self.viewer.update() 