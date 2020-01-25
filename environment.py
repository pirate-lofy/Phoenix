from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.settings import Camera
from carla.image_converter import to_rgb_array,labels_to_cityscapes_palette

import random
import cv2 as cv
import numpy as np

class CarlaEnv:
    n_vehicles=0
    n_peds=0
    n_actions=3
    
    observation_shape=(600,800,3)
    callibaration_shape=(84,84,3)
    
    cameras={'rgb':'SceneFinal',
             'seg':'SemanticSegmentation',
             'depth':'Depth'}
    
    def __init__(self,host='localhost',port=2000):
        
        self.client=CarlaClient(host,port)
        self.settings=self._get_settings(
                self.n_vehicles,self.n_peds)
        self._add_cameras()
    
    def start(self):
        self.client.connect()
        print('CarlaEnv log: client connected successfully.')
        
        self.scene=self.client.load_settings(self.settings)
        
    def reset(self):
        self._start_new_episod()
        measures,data=self.client.read_data()
        data=self._data_preprocessing(data)
        measures=self._measures_preprocessing(measures) 
        
        
    def _measures_preprocessing(self,measures):
        PM = measures.player_measurements

        pos_x = PM.transform.location.x / 100 # cm -> m
        pos_y = PM.transform.location.y / 100
        speed = PM.forward_speed/10.0 if PM.forward_speed>0 else 0.0

        col_cars = PM.collision_vehicles
        col_ped = PM.collision_pedestrians
        col_other = PM.collision_other

        other_lane = PM.intersection_otherlane
        offroad = PM.intersection_offroad
    
        return np.array([pos_x, pos_y, speed]), np.array([col_cars, col_ped, col_other]), np.array([other_lane, offroad])
    
    def _data_preprocessing(self,data):
        ready_data=[]
        for key in self.cameras.keys():
            img=data[key].data
            
            if not key=='depth':
                img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            
            img=np.resize(img,self.callibaration_shape)/255.0
            ready_data.append(img)
        
        return np.stack(ready_data,axis=2)
        
    def _start_new_episod(self):
        n_spots=len(self.scene.player_start_spots)
        start_point=random.randint(0,max(0,n_spots-1))
        goal_point=random.randint(0,max(0,n_spots-1))
        
        while start_point==goal_point:
            goal_point=random.randint(0,max(0,n_spots-1))
        
        start=self.scene.player_start_spots[start_point]
        goal=self.scene.player_start_spots[goal_point]
        
        self.start=[start.location.x/100.0, start.location.y/100.0]
        self.goal=[goal.location.x/100.0, goal.location.y/100.0]
        
        self.client.start_episode(start_point)
        
    
    def _get_settings(self,n_vehicles,n_peds):
        settings=CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=n_vehicles,
            NumberOfPedestrians=n_peds,
            WeatherId=1,
        )
        settings.randomize_seeds()
        return settings
    
    def _add_cameras(self):            
        for name,postproc in self.cameras.items():
            cam=Camera(name,PostProcessing=postproc)
            cam.set(FOV=90.0)
            cam.set_image_size(self.observation_shape[1],
                               self.observation_shape[0])
            cam.set_position(x=0.30, y=0, z=1.30)
            cam.set_rotation(pitch=0, yaw=0, roll=0)
            self.settings.add_sensor(cam)