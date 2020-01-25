from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla.tcp import TCPConnectionError
from carla.image_converter import to_rgb_array,labels_to_cityscapes_palette

import random
import cv2 as cv
import numpy as np

class CarlaEnv:
    n_vehicles=60
    n_peds=10
    n_actions=3
    
    observation_shape=(600,800,3)
    callibaration_shape=(84,84)
    
    speed_list=[]
    speed_list_limit=10
    wait=15
    
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
        self._empty_cycle()
        data,measures=self._get_data(reset=True)
        return data,measures
    
    def step(self,actions):
        steer=np.clip(actions[0],-1,1)
        throttle=np.clip(actions[1],0,1)
        brake=round(actions[2])
        
        self.client.send_control(
                steer=steer,
                throttle=throttle,
                brake=brake,
                hand_brake=False,
                reverse=False                
                )
        # speed,dist_to_goal,colls,inters
        data,measures=self._get_data()
        
        reward=self._compute_reward(measures)
        done=self._is_done(measures)
        return (data,measures),reward,done,{}
    
    
    def _empty_cycle(self):
        print('CarlaEnv log: empty cycle started...')
        for _ in range(self.wait):
            self.client.read_data()
            self.client.send_control(
                steer=0,
                throttle=0,
                brake=0,
                hand_brake=False,
                reverse=False                
                )
        print('CarlaEnv log: empty cycle ended.')
    
    def _is_done(self,measures):
        dist=measures[1]
        return self._is_goal(dist) or self._is_bad_pos(measures)
                
    
    def _is_bad_pos(self,measures):
        colls=measures[2]
        inters=measures[3]
        return self._did_collide(colls) or self._sidewalk(inters)\
            or self._is_quiet()

                
    def _is_quiet(self):
        if len(self.speed_list)>self.speed_list_limit:
            self.speed_list=self.speed_list[1:self.speed_list_limit+1]
        return sum(self.speed_list)==0
    
    #TODO: should be revised
    def _compute_reward(self,measures):
        speed=measures[0]
        dist=measures[1]
        colls=measures[2]
        inters=measures[3]
        
        alpha=0.1 # should be revised
        
        if self._is_goal(dist):
            return 1000
        if self._is_bad_pos(measures):
            return -1000
        r=alpha*(speed+dist-inters)
        return r
        
    def _sidewalk(self,inters):
        return inters>0.3
    
    def _is_goal(self,dist):
        return dist<2.0
    
    def _did_collide(self,colls):
        return colls>0
    
    def _get_data(self,reset=False):
        measures,data=self.client.read_data()
        
        if reset:
            self.start_time=measures.game_timestamp
        
        data=self._data_preprocessing(data)
        state,colls,inters=self._measures_preprocessing(measures)
        measures=self._make_measures_blob(state,colls,inters)
    
        return data,measures
    
    def _make_measures_blob(self,state,colls,inters):
        pos=np.array(state[0:2])
        dist_to_goal=np.linalg.norm(pos-self.goal)
        
        speed=state[2]
        colls=sum(colls)
        inters=sum(inters)
        
        return np.array([speed,dist_to_goal,colls,inters])
    
    def _measures_preprocessing(self,measures):
        PM = measures.player_measurements

        pos_x = PM.transform.location.x / 100 # cm -> m
        pos_y = PM.transform.location.y / 100
        
        speed = PM.forward_speed/10.0 if PM.forward_speed>=0 else 0.0001
        self.speed_list.append(speed)
        
        col_cars = PM.collision_vehicles
        col_ped = PM.collision_pedestrians
        col_other = PM.collision_other

        other_lane = PM.intersection_otherlane
        offroad = PM.intersection_offroad
    
        return [pos_x, pos_y, speed], [col_cars, col_ped, col_other], [other_lane, offroad]
    
    def _data_preprocessing(self,data):
        ready_data=[]
        for key in self.cameras.keys():
            img,img_data=data[key],data[key].data
            
            if key=='seg':
                img_data=labels_to_cityscapes_palette(img)
            if not key=='depth':
                img_data=cv.cvtColor(img_data.astype('float32'),cv.COLOR_BGR2GRAY)
            
            img_data=np.resize(img_data,self.callibaration_shape)/255.0
            ready_data.append(img_data)
        
        # TODO: check the stack technique to get 
        # RGB-like tensor
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