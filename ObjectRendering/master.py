# photorealistic_glass_raytracer.py
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

class PhotorealisticGlassRayTracer:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.max_depth = 8  # Higher for more realistic reflections
        self.image_queue = queue.Queue()
        self.worker_conn = None
        self.server = None
        
    def setup_live_display(self):
        """Setup matplotlib for live display"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.set_title("🔮 Photorealistic Distributed Glass Ray Tracing", fontsize=18)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        
        # Initial black image
        self.im = self.ax.imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8), 
                                origin='upper', extent=[0, self.width, self.height, 0])
        
        # Enhanced progress indicators
        self.left_progress = patches.Rectangle((20, self.height-40), self.width//2-40, 30, 
                                              facecolor='#FF4444', alpha=0.3, 
                                              linewidth=2, edgecolor='#CC0000')
        self.right_progress = patches.Rectangle((self.width//2+20, self.height-40), 
                                               self.width//2-40, 30,
                                               facecolor='#44FF44', alpha=0.3,
                                               linewidth=2, edgecolor='#00CC00')
        self.ax.add_patch(self.left_progress)
        self.ax.add_patch(self.right_progress)
        
        # Status text with better styling
        self.status_text = self.ax.text(self.width//2, 60, "Initializing...", 
                                       ha='center', fontsize=14, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.5", 
                                               facecolor="white", alpha=0.9,
                                               edgecolor="gray", linewidth=1))
        
        # Labels
        self.ax.text(30, self.height-60, "Your NVIDIA RTX 3050", 
                    color='#CC0000', fontweight='bold', fontsize=12)
        self.ax.text(self.width//2+30, self.height-60, "Friend's AMD RX 6600", 
                    color='#00CC00', fontweight='bold', fontsize=12)
        
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
        """Enhanced Fresnel reflectance calculation"""
        sin_t = (n1 / n2) * np.sqrt(max(0, 1.0 - cos_i * cos_i))
        
        if sin_t >= 1.0:
            return 1.0
            
        cos_t = np.sqrt(max(0, 1.0 - sin_t * sin_t))
        
        rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
        rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
        
        return (rs + rp) / 2.0
    
    def refract(self, incident, normal, n1, n2):
        """Calculate refracted ray direction"""
        cos_i = -np.dot(normal, incident)
        sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i ** 2)
        
        if sin_t2 > 1.0:
            return None
            
        cos_t = np.sqrt(1.0 - sin_t2)
        return (n1 / n2) * incident + ((n1 / n2) * cos_i - cos_t) * normal
    
    def reflect(self, incident, normal):
        """Calculate reflected ray direction"""
        return incident - 2 * np.dot(incident, normal) * normal
        
    def environment_color(self, direction):
        """Enhanced environment with brighter lighting"""
        # Bright sky gradient (like in the reference image)
        t = 0.5 * (direction[1] + 1.0)
        
        # Brighter sky colors
        sky_top = np.array([0.8, 0.9, 1.0])      # Light blue
        sky_horizon = np.array([1.0, 1.0, 0.95])  # Warm white
        sky_color = (1.0 - t) * sky_horizon + t * sky_top
        
        # Multiple light sources for brighter environment
        lights = [
            {'dir': np.array([0.5, 0.8, -0.3]), 'color': np.array([1.0, 0.95, 0.9]), 'power': 64},
            {'dir': np.array([-0.3, 0.6, -0.7]), 'color': np.array([0.9, 0.95, 1.0]), 'power': 32},
            {'dir': np.array([0.0, 1.0, 0.0]), 'color': np.array([1.0, 1.0, 1.0]), 'power': 16}
        ]
        
        total_light = np.array([0.0, 0.0, 0.0])
        for light in lights:
            light_dir = light['dir'] / np.linalg.norm(light['dir'])
            intensity = max(0, np.dot(direction, light_dir)) ** light['power']
            total_light += intensity * light['color'] * 0.3
            
        return np.clip(sky_color + total_light, 0.0, 2.0)  # Allow overbright for HDR
        
    def trace_ray(self, origin, direction, depth=0):
        """Enhanced ray tracing for photorealistic glass"""
        if depth > self.max_depth:
            return np.array([0.0, 0.0, 0.0])
            
        # Glass sphere (positioned like in reference image)
        sphere_center = np.array([0.0, -0.8, -4.5])  # Slightly lower
        sphere_radius = 1.5  # Larger sphere
        
        # Ground plane for shadow
        ground_y = -2.3
        
        closest_t = float('inf')
        hit_object = None
        hit_normal = None
        hit_point = None
        
        # Ray-sphere intersection
        oc = origin - sphere_center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            
            t = t1 if t1 > 0.001 else t2
            if 0.001 < t < closest_t:
                closest_t = t
                hit_object = 'glass_sphere'
                hit_point = origin + t * direction
                hit_normal = (hit_point - sphere_center) / sphere_radius
                
        # Ray-ground intersection
        if direction[1] < -0.001:  # Ray going downward
            t = (ground_y - origin[1]) / direction[1]
            if 0.001 < t < closest_t:
                ground_hit = origin + t * direction
                # Only render ground near sphere
                if np.linalg.norm(ground_hit - np.array([0, ground_y, -4.5])) < 6:
                    closest_t = t
                    hit_object = 'ground'
                    hit_point = ground_hit
                    hit_normal = np.array([0.0, 1.0, 0.0])
        
        if hit_object is None:
            return self.environment_color(direction)
            
        if hit_object == 'ground':
            # Ground material (like in reference - light gray/white)
            base_color = np.array([0.9, 0.9, 0.95])
            
            # Simple lighting
            light_dir = np.array([0.5, 0.8, -0.3])
            light_dir = light_dir / np.linalg.norm(light_dir)
            diffuse = max(0.1, np.dot(hit_normal, light_dir))
            
            # Shadow from sphere
            shadow_ray_origin = hit_point + 0.001 * hit_normal
            shadow_ray_dir = light_dir
            
            # Check if shadow ray hits sphere
            oc_shadow = shadow_ray_origin - sphere_center
            a_shadow = np.dot(shadow_ray_dir, shadow_ray_dir)
            b_shadow = 2.0 * np.dot(oc_shadow, shadow_ray_dir)
            c_shadow = np.dot(oc_shadow, oc_shadow) - sphere_radius * sphere_radius
            
            shadow_discriminant = b_shadow * b_shadow - 4 * a_shadow * c_shadow
            if shadow_discriminant >= 0:
                shadow_t = (-b_shadow - np.sqrt(shadow_discriminant)) / (2.0 * a_shadow)
                if shadow_t > 0.001:
                    diffuse *= 0.3  # In shadow
                    
            return base_color * diffuse
            
        elif hit_object == 'glass_sphere':
            # Enhanced glass properties
            glass_ior = 1.52  # More realistic glass IOR
            air_ior = 1.0
            
            entering = np.dot(direction, hit_normal) < 0
            if entering:
                n1, n2 = air_ior, glass_ior
                normal_adj = hit_normal
            else:
                n1, n2 = glass_ior, air_ior
                normal_adj = -hit_normal
                
            cos_i = abs(np.dot(direction, normal_adj))
            fresnel = self.fresnel_reflectance(cos_i, n1, n2)
            
            color = np.array([0.0, 0.0, 0.0])
            
            # Reflection (enhanced for brighter reflections)
            if fresnel > 0.01:
                reflect_dir = self.reflect(direction, normal_adj)
                reflect_color = self.trace_ray(hit_point + 0.001 * reflect_dir, 
                                             reflect_dir, depth + 1)
                color += fresnel * reflect_color * 1.1  # Brighter reflections
                
            # Refraction (with slight tint)
            if fresnel < 0.99:
                refract_dir = self.refract(direction, normal_adj, n1, n2)
                if refract_dir is not None:
                    refract_color = self.trace_ray(hit_point + 0.001 * refract_dir,
                                                 refract_dir, depth + 1)
                    # Very subtle glass tint
                    glass_tint = np.array([0.995, 1.0, 0.998])
                    color += (1.0 - fresnel) * refract_color * glass_tint
                    
            return np.clip(color, 0.0, 3.0)  # Allow bright highlights
        
    def render_region_photorealistic(self, region):
        """Photorealistic rendering matching the reference image"""
        x_start, x_end, y_start, y_end = region
        width = x_end - x_start
        height = y_end - y_start
        
        img = np.zeros((height, width, 3), dtype=np.float32)
        
        print(f"🔮 Rendering photorealistic glass sphere ({width}x{height})...")
        start_time = time.time()
        
        # Camera setup
        camera_pos = np.array([0.0, 0.0, 0.0])
        fov = 35.0  # Narrower FOV like the reference
        aspect = self.width / self.height
        
        for y in range(height):
            if y % 15 == 0:
                progress = y / height
                self.update_progress_live(f"🔮 Photorealistic rendering... {int(progress*100)}%", 
                                        left_progress=progress if x_start == 0 else 0,
                                        right_progress=progress if x_start > 0 else 0)
            
            for x in range(width):
                # Anti-aliasing with multiple samples
                color_samples = []
                samples = 2  # 2x2 supersampling
                
                for sx in range(samples):
                    for sy in range(samples):
                        # Sub-pixel sampling
                        px = (x + (sx + 0.5)/samples + x_start) / self.width
                        py = (y + (sy + 0.5)/samples + y_start) / self.height
                        
                        screen_x = (px - 0.5) * 2.0 * aspect
                        screen_y = (0.5 - py) * 2.0
                        
                        ray_dir = np.array([
                            screen_x * np.tan(np.radians(fov/2)),
                            screen_y * np.tan(np.radians(fov/2)),
                            -1.0
                        ])
                        ray_dir = ray_dir / np.linalg.norm(ray_dir)
                        
                        sample_color = self.trace_ray(camera_pos, ray_dir)
                        color_samples.append(sample_color)
                
                # Average the samples
                color = np.mean(color_samples, axis=0)
                
                # Enhanced tone mapping for brighter image
                color = color / (color + 0.5)  # Less aggressive tone mapping
                color = np.power(color, 1.0/2.4)  # sRGB gamma
                
                img[y, x] = color
                
        elapsed = time.time() - start_time
        print(f"⏱️  Photorealistic rendering completed in {elapsed:.2f}s")
        
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        
    def update_progress_live(self, status, left_progress=0, right_progress=0):
        """Update live display with progress"""
        try:
            self.left_progress.set_alpha(0.3 + 0.6 * left_progress)
            self.right_progress.set_alpha(0.3 + 0.6 * right_progress)
            self.status_text.set_text(status)
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
            'max_depth': self.max_depth,
            'render_type': 'photorealistic'
        }
        data = json.dumps(task).encode()
        self.worker_conn.send(len(data).to_bytes(4, 'big'))
        self.worker_conn.send(data)
        
    def receive_result(self):
        """Receive result from worker with progress updates"""
        try:
            length_bytes = self.worker_conn.recv(4)
            if not length_bytes:
                return None
            length = int.from_bytes(length_bytes, 'big')
            
            data = b''
            while len(data) < length:
                chunk = self.worker_conn.recv(min(8192, length - len(data)))
                if not chunk:
                    return None
                data += chunk
                
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
            
    def render_distributed_photorealistic(self):
        """Main photorealistic rendering function"""
        self.setup_live_display()
        self.update_progress_live("🌐 Starting server...")
        
        if not self.start_server():
            return None
            
        left_region = (0, self.width//2, 0, self.height)
        right_region = (self.width//2, self.width, 0, self.height)
        
        current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.im.set_array(current_image)
        self.fig.canvas.draw_idle()
        
        self.update_progress_live("✅ Connected! Starting photorealistic render...")
        
        print("📤 Sending photorealistic glass task to friend...")
        self.send_task(right_region)
        self.update_progress_live("📤 Task sent to friend...")
        
        print("🖥️  Rendering photorealistic glass (your part)...")
        left_img = self.render_region_photorealistic(left_region)
        current_image[:, :self.width//2] = left_img
        
        self.im.set_array(current_image)
        self.fig.canvas.draw_idle()
        self.update_progress_live("📥 Waiting for friend's result...", left_progress=1.0)
        
        right_img = self.receive_result()
        if right_img is not None:
            current_image[:, self.width//2:] = right_img
            self.im.set_array(current_image)
            self.fig.canvas.draw_idle()
            self.update_progress_live("✅ Photorealistic distributed rendering complete!", 
                                    left_progress=1.0, right_progress=1.0)
            
            img = Image.fromarray(current_image)
            img.save('photorealistic_glass_distributed.png')
            print("💾 Saved as 'photorealistic_glass_distributed.png'")
            
        return current_image

if __name__ == "__main__":
    print("🚀 Starting Photorealistic Glass Ray Tracer - Master Node")
    raytracer = PhotorealisticGlassRayTracer()
    
    try:
        result = raytracer.render_distributed_photorealistic()
        if result is not None:
            print("✅ Photorealistic rendering complete! Press Enter to exit...")
            input()
    except KeyboardInterrupt:
        print("\n🛑 Rendering interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if raytracer.server:
            raytracer.server.close()
        plt.close('all')
