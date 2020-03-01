from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla.tcp import TCPConnectionError
from carla.image_converter import labels_to_cityscapes_palette

from gym import spaces

import random
import cv2 as cv
import numpy as np


class CarlaEnv:
    n_vehicles=40
    n_peds=350
    n_actions=3
    #hi nour
    
    # we get all kinds of cameras and choose between them later
    #key: the name of the camera
    #value: the post processing paramerter to function Camera
    cameras={'rgb':'SceneFinal',
             'depth':'Depth'}
#             'seg':'SemanticSegmentation',}   
    
    callibaration_shape=(84,84)
    callibarationRGB_shape=(84,84,len(cameras))
    
    
    wait=15
    num_envs=1
    
    stand=300
    stand_limit=300

    
    def __init__(self,host='localhost',port=2000, repeat_frames=3):
        self.repeat_frames=repeat_frames
        self.host=host
        self.port=port
        self.client=None
        
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 255, self.callibarationRGB_shape), 
            spaces.Box(-np.inf, np.inf, (5,))
            )
        )
        self.action_space = spaces.Box(-1, 1, shape=(
                self.n_actions,))
        
        self.settings=self._get_settings(
                self.n_vehicles,self.n_peds)
        self._add_cameras()

 
    def _initialize_connection(self,host,port):
        if self.client.connected():
            self.client.disconnect()
            self.client=None
        self._connect(host,port)
        self.scene=self.client.load_settings(self.settings)
 
    
    def _connect(self,host,port):
        while True:
            try:
                self.client = CarlaClient(host, port,timeout=200000)
                self.client.connect()
                print('CarlaEnv log: client connected successfully.')
                break
            except TCPConnectionError:
                print('''CarlaEnv log: Client can not connect to the server..
                      Server may not launched yet...''')
        
    def reset(self):
        print('CarlaEnv log: reseting the world, starting new session..')
        self._initialize_connection(self.host,self.port)
        self._start_new_episod()
        
        # just to wait until the car falls from the sky and
        # becomes ready 
        # may prevent the collision sensor from recording 
        # falling as collision
        self._empty_cycle()
        
        data,measures=self._get_data(reset=True)
        return data,measures
    
    def step(self,actions):
        actions=actions[0]
        steer=np.clip(actions[0],-1,1)
        throttle=np.clip(actions[1],0,1)
        brake=np.clip(actions[2],0,1)

        # remember: Carla needs to get_data once and followed
        # by send_control
        # getting data twice in row causes craching
        for _ in range(self.repeat_frames):
            self.client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    hand_brake=False,
                    reverse=False                
                    )
            data,measures=self._get_data()
        self.client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    hand_brake=False,
                    reverse=False                
                    )
        
        # speed,dist_to_goal,dist_from_start,colls,inters
        data,measures=self._get_data()
        
        reward=self._compute_reward(measures)
        done=self._is_done(measures)
        return data,measures,reward,done,{}
    
    
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
        print('CarlaEnv log: empty cycle ended.\n')
    
    def _is_done(self,measures):
        dist=measures[1]
        goal=self._is_goal(dist)
        bad=self._is_bad_pos(measures)
        return goal or bad
                
    #speed,dist_to_goal,dist_from_start,colls,inters
    def _is_bad_pos(self,measures):
        '''
        checks if the car has collided with any thing or
        crossed the sidewalk or it has been stand still for too long
        
        :parm measures: all measurments came from _make_measures_blob function
        :return: boolean value
        
        '''
        colls=measures[3]
        inters=measures[4]
        return self._did_collide(colls) or self._sidewalk(inters)\
            or self._is_quiet()

               
    def compute_dif_between_positions(self,cur):
        dif=np.linalg.norm(np.array(cur)-np.array(self.prev_pos))
        self.prev_pos[:]=cur[:]
        return 1 if dif>0.001 else 0
    
    def _init_stand(self):
        self.stand=self.stand_limit
    
    def _is_quiet(self):
        return self.stand==0
    
    def _update_stand(self,value):
        if value and self.stand<self.stand_limit:
            self.stand=self.stand_limit
        elif self.stand>0:
            self.stand-=1
    
    
    #TODO: should be revised
    #speed,dist_to_goal,dist_from_start,colls,inters
    def _compute_reward(self,measures):
        speed=measures[0]
        dist_to_goal=measures[1]
        dist_from_start=measures[2]
        inters=measures[4]
        
        alpha=0.1 # should be revised
        
        if self._is_goal(dist_to_goal):
            return 100
        if self._is_bad_pos(measures):
            return -100
        r=alpha*(speed+dist_from_start-inters) #------------------
        return r
        
    def _sidewalk(self,inters):
        return inters>0.2
    
    def _is_goal(self,dist):
        return dist<0.05
#        return False
    
    def _did_collide(self,colls):
        return colls>0
    
    def _get_data(self,reset=False):
        measures,data=self.client.read_data()
        
        # dave it just in case!
        if reset:
            self.start_time=measures.game_timestamp
        
        data=self._data_preprocessing(data)
        state,colls,inters=self._measures_preprocessing(measures)
        measures=self._make_measures_blob(state,colls,inters)
    
        return data,measures
    
    def _make_measures_blob(self,state,colls,inters):
        pos=np.array(state[0:2])
        dist_to_goal=np.linalg.norm(pos-self.goal)
        dist_from_start=np.linalg.norm(pos-self.start)
        
        speed=state[2]
        colls=sum(colls)
        
        return np.array([speed,dist_to_goal,dist_from_start,colls,inters])
    
    # TODO: need to reconsider the measures we need
    def _measures_preprocessing(self,measures):
        PM = measures.player_measurements

        pos_x = PM.transform.location.x / 100 # cm -> m
        pos_y = PM.transform.location.y / 100
        
        # to prevent negative values
        speed = PM.forward_speed/10.0 if PM.forward_speed>=0 else 0
        dif=self.compute_dif_between_positions([pos_x,pos_y])
#        self.movement_list.append(dif)
        self._update_stand(dif)
        
        col_cars = PM.collision_vehicles
        col_ped = PM.collision_pedestrians
        col_other = PM.collision_other

        offroad = PM.intersection_offroad
    
        return [pos_x, pos_y, speed], [col_cars, col_ped, col_other],  offroad
    
    def _data_preprocessing(self,data):
        ready_data=[]
        for key in self.cameras.keys():
            img,img_data=data[key],data[key].data
            
            if key=='seg':
                img_data=labels_to_cityscapes_palette(img)
            # depth camera produces gray scale image
            if not key=='depth':
                # astype('float32') is needed as attribute data
                # returns data in float64 format which generates
                # error with opencv
                img_data=cv.cvtColor(img_data.astype('float32'),cv.COLOR_BGR2GRAY)
            
            img_data=np.resize(img_data,self.callibaration_shape)/255.0
            ready_data.append(img_data)

        return np.stack(ready_data,axis=2)
        
    def _start_new_episod(self):
        '''
        chooses a random position for the car to start from 
        and chooses another random position to be considered 
        as the destination of the navigation... this method 
        is needed to calculate the dist_to_goal value which
        is used as parameter feeded to the model later
        
        '''
        n_spots=len(self.scene.player_start_spots)
        start_point=random.randint(0,max(0,n_spots-1))
        goal_point=random.randint(0,max(0,n_spots-1))
        
        # try to find another position which is not the same 
        # as the starting position
        while start_point==goal_point:
            goal_point=random.randint(0,max(0,n_spots-1))
        
        start=self.scene.player_start_spots[start_point]
        goal=self.scene.player_start_spots[goal_point]
        
        self.start=[start.location.x/100.0, start.location.y/100.0]
        self.prev_pos=self.start[:]
        self.goal=[goal.location.x/100.0, goal.location.y/100.0]
        
        self.client.start_episode(start_point)
        self._init_stand()
        
    
    def _get_settings(self,n_vehicles,n_peds):
        settings=CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=n_vehicles,
            NumberOfPedestrians=n_peds,
            WeatherId=1,
            QualityLevel='Low'
        )
        settings.randomize_seeds()
        return settings
    
    def _add_cameras(self):            
        for name,postproc in self.cameras.items():
            cam=Camera(name,PostProcessing=postproc)
            cam.set(FOV=90.0)
            cam.set_image_size(self.observation_space[1],
                               self.observation_space[0])
            cam.set_position(x=0.30, y=0, z=1.30)
            cam.set_rotation(pitch=0, yaw=0, roll=0)
            self.settings.add_sensor(cam)
