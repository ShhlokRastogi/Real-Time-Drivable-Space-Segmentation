import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

class NuScenesDrivableDataset(Dataset):
    def __init__(self, nusc_version='v1.0-mini', dataroot='c:/drivableseg/v1.0-mini', split='train', target_size=(640, 360)):
        self.dataroot = dataroot
        self.target_size = target_size
        
        print("Initializing NuScenes Dataset...")
        self.nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=False)
        
        self.maps = {}
        try:
            self.maps['singapore-onenorth'] = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
            self.maps['singapore-hollandvillage'] = NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage')
            self.maps['singapore-queenstown'] = NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown')
            self.maps['boston-seaport'] = NuScenesMap(dataroot=dataroot, map_name='boston-seaport')
        except FileNotFoundError:
            print("Note: Map Expansion pack missing.")
            
        self.samples = []
        for scene in self.nusc.scene:
            current_sample_token = scene['first_sample_token']
            while current_sample_token != '':
                sample = self.nusc.get('sample', current_sample_token)
                self.samples.append(sample)
                current_sample_token = sample['next']
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_path = os.path.join(self.dataroot, cam_front_data['filename'])
        
        image = Image.open(cam_path).convert('RGB')
        mask = self._generate_drivable_mask_projection(cam_front_data)
        
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        image_np = np.array(image, dtype=np.float32) / 255.0  
        mask_np = np.array(mask, dtype=np.float32)            
        
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1) 
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  
            
        return image_tensor, mask_tensor

    def _project_and_draw_polygon(self, mask_image, coords, ego_translation, ego_yaw, cs_record, cs_yaw, camera_intrinsic, axle_height, color):
        """Helper matrix function to mathematically slice and draw 3D map polygons to 2D image pixels via pure geometry."""
        # Elevate flat 2D map to specific 3D physical axle drop
        coords_3d = np.hstack((coords[:, :2], np.full((coords.shape[0], 1), ego_translation[2] - axle_height)))
        
        # Transform to Ego Frame
        coords_3d -= ego_translation
        coords_3d = np.dot(ego_yaw.inverse.rotation_matrix, coords_3d.T).T
        
        # Transform to Camera Lens Frame
        coords_3d -= cs_record['translation']
        coords_3d = np.dot(cs_yaw.inverse.rotation_matrix, coords_3d.T).T
        
        # Frustum Near-Plane Math Clipping (z > 0.1)
        clipped_coords = []
        for i in range(len(coords_3d)):
            p1, p2 = coords_3d[i], coords_3d[(i + 1) % len(coords_3d)]
            if p1[2] >= 0.1:
                clipped_coords.append(p1)
                if p2[2] < 0.1:
                    clipped_coords.append(p1 + ((0.1 - p1[2]) / (p2[2] - p1[2])) * (p2 - p1))
            elif p2[2] >= 0.1:
                clipped_coords.append(p1 + ((0.1 - p1[2]) / (p2[2] - p1[2])) * (p2 - p1))
                    
        clipped_coords = np.array(clipped_coords)
        if len(clipped_coords) < 3: 
            return mask_image # Discard if non-geometric

        # Matrix projection directly into 2D camera resolution
        points_2d = view_points(clipped_coords.T, camera_intrinsic, normalize=True)
        uv = points_2d[:2, :].T
        pts = np.array([uv], dtype=np.int32)
        
        # Apply rasterization
        cv2.fillPoly(mask_image, pts, color)
        return mask_image

    def _generate_drivable_mask_projection(self, cam_data):
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        sample_record = self.nusc.get('sample', cam_data['sample_token'])
        scene_record = self.nusc.get('scene', sample_record['scene_token'])
        log_record = self.nusc.get('log', scene_record['log_token'])
        
        nusc_map = self.maps.get(log_record['location'], None)
        im_size = (cam_data['width'], cam_data['height'])
        mask_image = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
        
        if nusc_map is None:
            return mask_image

        ego_translation = pose_record['translation']
        radius = 50.0 
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        ego_yaw = Quaternion(pose_record['rotation'])
        cs_yaw = Quaternion(cs_record['rotation'])
        axle_height = 0.35 
        
        # ---------------------------------------------------------------------------------- #
        # LAYER 1: ADD EXTERIOR DRIVABLE POLYGONS
        # ---------------------------------------------------------------------------------- #
        records = nusc_map.get_records_in_radius(ego_translation[0], ego_translation[1], radius, ['drivable_area'])
        for token in records['drivable_area']:
            record = nusc_map.get('drivable_area', token)
            for polygon_token in record['polygon_tokens']:
                polygon = nusc_map.extract_polygon(polygon_token) 
                
                # Render Drivable Zone Base (White/1)
                coords = np.array(polygon.exterior.coords)
                mask_image = self._project_and_draw_polygon(
                    mask_image, coords, ego_translation, ego_yaw, cs_record, cs_yaw, camera_intrinsic, axle_height, color=1
                )
                
                # CRITICAL FIX 3: SUBTRACT MAP HOLES (Grassy Medians, Islands, Physical Dividers)
                # Previously, Shapely .exterior ignored internal geographic islands!
                for interior in polygon.interiors:
                    coords_inner = np.array(interior.coords)
                    # Erase these inner islands back to Black/0
                    mask_image = self._project_and_draw_polygon(
                        mask_image, coords_inner, ego_translation, ego_yaw, cs_record, cs_yaw, camera_intrinsic, axle_height, color=0
                    )

        # ---------------------------------------------------------------------------------- #
        # LAYER 2: SUBTRACT SIDEWALKS / FOOTPATHS EXPLICITLY
        # ---------------------------------------------------------------------------------- #
        records_walkway = nusc_map.get_records_in_radius(ego_translation[0], ego_translation[1], radius, ['walkway'])
        for token in records_walkway['walkway']:
            record = nusc_map.get('walkway', token)
            polygon_token = record.get('polygon_token', None)
            if not polygon_token: continue
            
            polygon = nusc_map.extract_polygon(polygon_token) 
            coords_walkway = np.array(polygon.exterior.coords)
            
            # Physically punch the sidewalk out of the Drivable mask
            mask_image = self._project_and_draw_polygon(
                mask_image, coords_walkway, ego_translation, ego_yaw, cs_record, cs_yaw, camera_intrinsic, axle_height, color=0
            )

        # ---------------------------------------------------------------------------------- #
        # LAYER 3: DYNAMIC OBSTACLE OCCLUSION (Vehicles, Pedestrians, Barricades)
        # ---------------------------------------------------------------------------------- #
        _, boxes, _ = self.nusc.get_sample_data(cam_data['token'])
        for box in boxes:
            corners_3d = box.corners()
            if np.any(corners_3d[2, :] < 0.1): continue
            corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
            
            pts = corners_2d.T.astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask_image, hull, 0)
        
        return mask_image
