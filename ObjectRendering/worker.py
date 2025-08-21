# photorealistic_glass_worker.py
import socket
import numpy as np
import json
import time

class PhotorealisticGlassWorker:
    def __init__(self, master_ip):
        self.master_ip = master_ip
        self.max_depth = 8
        
    def connect_to_master(self):
        """Connect to master"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.master_ip, 7777))
            print(f"✅ Connected to master at {self.master_ip}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
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
        """Enhanced environment matching master"""
        t = 0.5 * (direction[1] + 1.0)
        
        sky_top = np.array([0.8, 0.9, 1.0])
        sky_horizon = np.array([1.0, 1.0, 0.95])
        sky_color = (1.0 - t) * sky_horizon + t * sky_top
        
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
            
        return np.clip(sky_color + total_light, 0.0, 2.0)
        
    def trace_ray(self, origin, direction, depth=0):
        """Same photorealistic ray tracing as master"""
        if depth > self.max_depth:
            return np.array([0.0, 0.0, 0.0])
            
        sphere_center = np.array([0.0, -0.8, -4.5])
        sphere_radius = 1.5
        ground_y = -2.3
        
        closest_t = float('inf')
        hit_object = None
        hit_normal = None
        hit_point = None
        
        # Same intersection logic as master...
        # [Copy the exact same ray-sphere and ray-ground intersection code from master]
        
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
                
        if direction[1] < -0.001:
            t = (ground_y - origin[1]) / direction[1]
            if 0.001 < t < closest_t:
                ground_hit = origin + t * direction
                if np.linalg.norm(ground_hit - np.array([0, ground_y, -4.5])) < 6:
                    closest_t = t
                    hit_object = 'ground'
                    hit_point = ground_hit
                    hit_normal = np.array([0.0, 1.0, 0.0])
        
        if hit_object is None:
            return self.environment_color(direction)
            
        if hit_object == 'ground':
            base_color = np.array([0.9, 0.9, 0.95])
            light_dir = np.array([0.5, 0.8, -0.3])
            light_dir = light_dir / np.linalg.norm(light_dir)
            diffuse = max(0.1, np.dot(hit_normal, light_dir))
            
            # Same shadow calculation as master
            shadow_ray_origin = hit_point + 0.001 * hit_normal
            shadow_ray_dir = light_dir
            
            oc_shadow = shadow_ray_origin - sphere_center
            a_shadow = np.dot(shadow_ray_dir, shadow_ray_dir)
            b_shadow = 2.0 * np.dot(oc_shadow, shadow_ray_dir)
            c_shadow = np.dot(oc_shadow, oc_shadow) - sphere_radius * sphere_radius
            
            shadow_discriminant = b_shadow * b_shadow - 4 * a_shadow * c_shadow
            if shadow_discriminant >= 0:
                shadow_t = (-b_shadow - np.sqrt(shadow_discriminant)) / (2.0 * a_shadow)
                if shadow_t > 0.001:
                    diffuse *= 0.3
                    
            return base_color * diffuse
            
        elif hit_object == 'glass_sphere':
            glass_ior = 1.52
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
            
            if fresnel > 0.01:
                reflect_dir = self.reflect(direction, normal_adj)
                reflect_color = self.trace_ray(hit_point + 0.001 * reflect_dir, 
                                             reflect_dir, depth + 1)
                color += fresnel * reflect_color * 1.1
                
            if fresnel < 0.99:
                refract_dir = self.refract(direction, normal_adj, n1, n2)
                if refract_dir is not None:
                    refract_color = self.trace_ray(hit_point + 0.001 * refract_dir,
                                                 refract_dir, depth + 1)
                    glass_tint = np.array([0.995, 1.0, 0.998])
                    color += (1.0 - fresnel) * refract_color * glass_tint
                    
            return np.clip(color, 0.0, 3.0)
        
    def render_region_photorealistic(self, task):
        """Photorealistic rendering for worker"""
        region = task['region']
        width = task['width']
        height = task['height']
        self.max_depth = task['max_depth']
        
        x_start, x_end, y_start, y_end = region
        region_width = x_end - x_start
        region_height = y_end - y_start
        
        print(f"🔮 Friend rendering photorealistic glass ({region_width}x{region_height})...")
        start_time = time.time()
        
        img = np.zeros((region_height, region_width, 3), dtype=np.float32)
        
        camera_pos = np.array([0.0, 0.0, 0.0])
        fov = 35.0
        aspect = width / height
        
        for y in range(region_height):
            if y % 15 == 0:
                progress = (y / region_height) * 100
                print(f"🟢 Friend progress: {progress:.1f}%")
                
            for x in range(region_width):
                # Same anti-aliasing as master
                color_samples = []
                samples = 2
                
                for sx in range(samples):
                    for sy in range(samples):
                        px = (x + (sx + 0.5)/samples + x_start) / width
                        py = (y + (sy + 0.5)/samples + y_start) / height
                        
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
                
                color = np.mean(color_samples, axis=0)
                
                # Same tone mapping as master
                color = color / (color + 0.5)
                color = np.power(color, 1.0/2.4)
                
                img[y, x] = color
        
        elapsed = time.time() - start_time
        print(f"⏱️  Friend completed photorealistic rendering in {elapsed:.2f}s")
        
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        
    def send_result(self, img):
        """Send result back to master"""
        data = img.tobytes()
        self.socket.send(len(data).to_bytes(4, 'big'))
        
        # Send in chunks to show progress
        chunk_size = 8192
        sent = 0
        while sent < len(data):
            chunk = data[sent:sent + chunk_size]
            self.socket.send(chunk)
            sent += len(chunk)
            progress = (sent / len(data)) * 100
            if progress % 10 < 1:
                print(f"📤 Friend sending: {progress:.0f}%")
        
        print("📤 Friend sent photorealistic result back")
        
    def work_loop(self):
        """Main worker loop"""
        if not self.connect_to_master():
            return
            
        try:
            while True:
                print("👂 Friend waiting for photorealistic rendering task...")
                
                length_bytes = self.socket.recv(4)
                if not length_bytes:
                    break
                    
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = self.socket.recv(length - len(data))
                    if not chunk:
                        break
                    data += chunk
                
                task = json.loads(data.decode())
                
                result = self.render_region_photorealistic(task)
                self.send_result(result)
                
        except Exception as e:
            print(f"❌ Friend error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.socket.close()

if __name__ == "__main__":
    print("🚀 Starting Photorealistic Glass Worker - Friend's Machine")
    master_ip = input("Enter master's IP address: ")
    
    worker = PhotorealisticGlassWorker(master_ip)
    worker.work_loop()
