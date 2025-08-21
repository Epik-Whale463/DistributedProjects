# live_glass_raytracer.py
import socket
import numpy as np
import json
from PIL import Image
import time
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class LiveGlassRayTracer:
    def __init__(self):
        self.width = 600
        self.height = 400
        self.max_depth = 6  # Ray bounces for reflections/refractions
        self.image_queue = queue.Queue()
        self.worker_conn = None
        self.server = None
        
    def setup_live_display(self):
        """Setup matplotlib for live display"""
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title("🔮 Live Distributed Glass Ray Tracing", fontsize=16)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        
        # Initial empty image
        self.im = self.ax.imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8), 
                                origin='upper', extent=[0, self.width, self.height, 0])
        
        # Progress indicators
        self.left_progress = patches.Rectangle((10, self.height-30), self.width//2-20, 20, 
                                              facecolor='red', alpha=0.3, label='Your GPU')
        self.right_progress = patches.Rectangle((self.width//2+10, self.height-30), self.width//2-20, 20,
                                               facecolor='green', alpha=0.3, label="Friend's GPU")
        self.ax.add_patch(self.left_progress)
        self.ax.add_patch(self.right_progress)
        
        # Status text
        self.status_text = self.ax.text(self.width//2, 40, "Initializing...", 
                                       ha='center', fontsize=12, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Add legend
        self.ax.text(20, self.height-50, "Your GPU", color='red', fontweight='bold')
        self.ax.text(self.width//2+20, self.height-50, "Friend's GPU", color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.show(block=False)
        
    def get_local_ip(self):
        """Get local IP address"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
        
    def start_server(self):
        """Start server and wait for worker"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('0.0.0.0', 7777))
            self.server.listen(1)
            
            local_ip = self.get_local_ip()
            print(f"🌐 Waiting for friend's connection on port 7777...")
            print(f"📍 Tell your friend to connect to: {local_ip}")
            
            self.worker_conn, addr = self.server.accept()
            print(f"✅ Friend connected from {addr[0]}")
            return True
        except Exception as e:
            print(f"❌ Server error: {e}")
            return False
            
    def fresnel_reflectance(self, cos_i, n1, n2):
        """Calculate Fresnel reflectance"""
        sin_t = (n1 / n2) * np.sqrt(max(0, 1.0 - cos_i * cos_i))
        
        if sin_t >= 1.0:  # Total internal reflection
            return 1.0
            
        cos_t = np.sqrt(max(0, 1.0 - sin_t * sin_t))
        
        # Fresnel equations
        rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
        rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
        
        return (rs + rp) / 2.0
    
    def refract(self, incident, normal, n1, n2):
        """Calculate refracted ray direction"""
        cos_i = -np.dot(normal, incident)
        sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i ** 2)
        
        if sin_t2 > 1.0:  # Total internal reflection
            return None
            
        cos_t = np.sqrt(1.0 - sin_t2)
        return (n1 / n2) * incident + ((n1 / n2) * cos_i - cos_t) * normal
    
    def reflect(self, incident, normal):
        """Calculate reflected ray direction"""
        return incident - 2 * np.dot(incident, normal) * normal
        
    def environment_color(self, direction):
        """Environment/background color"""
        # Sky gradient
        t = 0.5 * (direction[1] + 1.0)
        sky_color = (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
        
        # Add sun
        sun_dir = np.array([0.3, 0.6, -0.8])
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        sun_intensity = max(0, np.dot(direction, sun_dir)) ** 16
        
        return sky_color + sun_intensity * np.array([1.0, 0.9, 0.6]) * 0.5
        
    def trace_ray(self, origin, direction, depth=0):
        """Advanced ray tracing with glass materials"""
        if depth > self.max_depth:
            return np.array([0.0, 0.0, 0.0])
            
        # Glass sphere parameters
        sphere_center = np.array([0.0, 0.0, -4.0])
        sphere_radius = 1.2
        
        # Ray-sphere intersection
        oc = origin - sphere_center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return self.environment_color(direction)
            
        # Find intersection points
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        
        t = t1 if t1 > 0.001 else t2
        if t < 0.001:
            return self.environment_color(direction)
            
        # Hit point and normal
        hit_point = origin + t * direction
        normal = (hit_point - sphere_center) / sphere_radius
        
        # Glass properties
        glass_ior = 1.5
        air_ior = 1.0
        
        # Determine if entering or exiting glass
        entering = np.dot(direction, normal) < 0
        if entering:
            n1, n2 = air_ior, glass_ior
            normal_adj = normal
        else:
            n1, n2 = glass_ior, air_ior
            normal_adj = -normal
            
        cos_i = abs(np.dot(direction, normal_adj))
        fresnel = self.fresnel_reflectance(cos_i, n1, n2)
        
        color = np.array([0.0, 0.0, 0.0])
        
        # Reflection component
        if fresnel > 0.01:
            reflect_dir = self.reflect(direction, normal_adj)
            reflect_color = self.trace_ray(hit_point + 0.001 * reflect_dir, 
                                         reflect_dir, depth + 1)
            color += fresnel * reflect_color
            
        # Refraction component
        if fresnel < 0.99:
            refract_dir = self.refract(direction, normal_adj, n1, n2)
            if refract_dir is not None:
                refract_color = self.trace_ray(hit_point + 0.001 * refract_dir,
                                             refract_dir, depth + 1)
                # Glass tint
                glass_tint = np.array([0.98, 1.0, 0.98])
                color += (1.0 - fresnel) * refract_color * glass_tint
                
        return np.clip(color, 0.0, 1.0)
        
    def render_region_advanced(self, region):
        """Advanced rendering with glass materials"""
        x_start, x_end, y_start, y_end = region
        width = x_end - x_start
        height = y_end - y_start
        
        img = np.zeros((height, width, 3), dtype=np.float32)
        
        print(f"🔮 Rendering realistic glass sphere ({width}x{height})...")
        start_time = time.time()
        
        # Camera parameters
        camera_pos = np.array([0.0, 0.0, 0.0])
        fov = 45.0
        aspect = self.width / self.height
        
        for y in range(height):
            # Update progress every few rows
            if y % 20 == 0:
                progress = y / height
                self.update_progress_live(f"🔮 Rendering glass... {int(progress*100)}%", 
                                        left_progress=progress)
            
            for x in range(width):
                # Convert pixel to world coordinates
                px = (x + x_start) / self.width
                py = (y + y_start) / self.height
                
                # Map to [-1, 1] range
                screen_x = (px - 0.5) * 2.0 * aspect
                screen_y = (0.5 - py) * 2.0
                
                # Calculate ray direction
                ray_dir = np.array([
                    screen_x * np.tan(np.radians(fov/2)),
                    screen_y * np.tan(np.radians(fov/2)),
                    -1.0
                ])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Trace the ray
                color = self.trace_ray(camera_pos, ray_dir)
                
                # Tone mapping and gamma correction
                color = color / (color + 1.0)
                color = np.power(color, 1.0/2.2)
                
                img[y, x] = color
                
        elapsed = time.time() - start_time
        print(f"⏱️  Glass rendering completed in {elapsed:.2f}s")
        
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        
    def update_progress_live(self, status, left_progress=0, right_progress=0):
        """Update live display with progress"""
        try:
            # Update progress bars
            self.left_progress.set_alpha(0.3 + 0.5 * left_progress)
            self.right_progress.set_alpha(0.3 + 0.5 * right_progress)
            
            # Update status text
            self.status_text.set_text(status)
            
            # Refresh display
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except:
            pass
            
    def send_task(self, region):
        """Send rendering task to worker"""
        task = {
            'region': region,
            'width': self.width,
            'height': self.height,
            'max_depth': self.max_depth
        }
        data = json.dumps(task).encode()
        self.worker_conn.send(len(data).to_bytes(4, 'big'))
        self.worker_conn.send(data)
        
    def receive_result(self):
        """Receive result from worker"""
        try:
            length_bytes = self.worker_conn.recv(4)
            if not length_bytes:
                return None
            length = int.from_bytes(length_bytes, 'big')
            
            data = b''
            while len(data) < length:
                chunk = self.worker_conn.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk
                
                # Update progress while receiving
                progress = len(data) / length
                self.update_progress_live(f"📥 Receiving friend's result... {int(progress*100)}%",
                                        left_progress=1.0, right_progress=progress)
                
            result = np.frombuffer(data, dtype=np.uint8)
            region_height = self.height
            region_width = self.width // 2
            return result.reshape((region_height, region_width, 3))
        except Exception as e:
            print(f"❌ Error receiving data: {e}")
            return None
            
    def render_distributed_live(self):
        """Main rendering with live updates"""
        # Setup live display
        self.setup_live_display()
        self.update_progress_live("🌐 Starting server...")
        
        if not self.start_server():
            return None
            
        # Split work
        left_region = (0, self.width//2, 0, self.height)
        right_region = (self.width//2, self.width, 0, self.height)
        
        # Create initial image
        current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.im.set_array(current_image)
        self.fig.canvas.draw_idle()
        
        self.update_progress_live("✅ Connected! Starting render...")
        
        # Send task to friend
        print("📤 Sending glass rendering task to friend...")
        self.send_task(right_region)
        self.update_progress_live("📤 Task sent to friend...")
        
        # Render our part
        print("🖥️  Rendering our glass part...")
        left_img = self.render_region_advanced(left_region)
        current_image[:, :self.width//2] = left_img
        
        # Update display with our result
        self.im.set_array(current_image)
        self.fig.canvas.draw_idle()
        self.update_progress_live("📥 Waiting for friend's result...", left_progress=1.0)
        
        # Wait for friend's result
        right_img = self.receive_result()
        if right_img is not None:
            current_image[:, self.width//2:] = right_img
            self.im.set_array(current_image)
            self.fig.canvas.draw_idle()
            self.update_progress_live("✅ Distributed glass rendering complete!", 
                                    left_progress=1.0, right_progress=1.0)
            
            # Save final result
            img = Image.fromarray(current_image)
            img.save('glass_sphere_distributed.png')
            print("💾 Saved as 'glass_sphere_distributed.png'")
            
        return current_image

if __name__ == "__main__":
    print("🚀 Starting Live Glass Ray Tracer - Master Node")
    raytracer = LiveGlassRayTracer()
    
    try:
        result = raytracer.render_distributed_live()
        if result is not None:
            print("✅ Rendering complete! Press any key to exit...")
            input()
    except KeyboardInterrupt:
        print("\n🛑 Rendering interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if raytracer.server:
            raytracer.server.close()
        plt.close('all')
