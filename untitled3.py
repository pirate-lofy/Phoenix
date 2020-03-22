#import glob
#import os
#import sys
from gym import spaces
import numpy as np
import random
import time
from math import exp,sqrt

# ==============================================================================
#try:
#    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
#    sys.version_info.major,
#    sys.version_info.minor,
#    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#except IndexError:
#    print('cant append carla egg')

print('here')
import carla

from carla import ColorConverter as cc

import cv2 as cv

class CarlaEnv:
    # initial variables
    callibaration_shape=(182,192)
    callibarationRGB_shape=(182,192,2)
    
    n_actions=3
    stand=300
    stand_limit=300
    actors=[]
    collision_data=[]
    SHOW_VIEW=True
    is_goal=None
    bad_pos=None
    reset_timer=0
    
    rgb_data=None
    seg_data=None

    
    def __init__(self,host='127.0.0.1', port=2000,timeout=100):
        self.host=host
        self.port=port
        self.timeout=timeout
        
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 255, self.callibarationRGB_shape), 
            spaces.Box(-np.inf, np.inf, (5,))
            )
        )
        self.action_space = spaces.Box(-1, 1, shape=(
                self.n_actions,))
        
        self.blueprint=self._connect()
        self.vehicle=self._add_vehicle(self.blueprint)
        self._add_actors()
    
    def _connect(self):
        self.client = carla.Client(self.host,self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map=self.world.get_map()
        return self.world.get_blueprint_library()
    
    def _add_vehicle(self,blueprint):
        car = blueprint.filter('isetta')[0]
        transform = random.choice(self.map.get_spawn_points())
        vehicle = self.world.spawn_actor(car, transform)        
        self.actors.append(vehicle)
        return vehicle


    def _add_actors(self):
        transform = carla.Transform(carla.Location(x=0.30, y=0, z=1.30),
                                    carla.Rotation(pitch=0, yaw=0, roll=0))
        # rgb cam
        rgb = self.blueprint.find('sensor.camera.rgb')
        rgb.set_attribute('image_size_x', '896')
        rgb.set_attribute('image_size_y', '512')
        rgb.set_attribute('fov', '110')
        rgb = self.world.spawn_actor(rgb, transform, attach_to=self.vehicle)
        rgb.listen(lambda data: self._process_rgb(data))
        self.actors.append(rgb)
        
        # segmentation cam
        seg = self.blueprint.find('sensor.camera.semantic_segmentation')
        seg.set_attribute('image_size_x', '896')
        seg.set_attribute('image_size_y', '512')
        seg.set_attribute('fov', '110')
        seg = self.world.spawn_actor(seg, transform, attach_to=self.vehicle)
        seg.listen(lambda data: self._process_seg(data))
        self.actors.append(seg)
        
        # collision
        col= self.blueprint.find('sensor.other.collision')
        col = self.world.spawn_actor(col, transform, attach_to=self.vehicle)
        col.listen(lambda data,: self._detect_col(data))
        self.actors.append(col)
    
    def _to_gray(self,img):
        return img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
    
    def _prepare_raw_image(self,image,to_gray=True):
        img=np.array(image.raw_data)
        img=img.reshape((512,896,4))[:,:,:3]
        
        if to_gray:
            img_gray=self._to_gray(img)
            return img,img_gray
        else:
            return img
    
    def _process_rgb(self,image):
        img,img_gray=self._prepare_raw_image(image)
        img_gray=img_gray[:350,:]
        img_gray=cv.resize(img_gray,(192,182),cv.INTER_AREA)/255.
        self.rgb_data=img_gray[:]
        
        if self.SHOW_VIEW:
            cv.imshow('front view',img)
            cv.waitKey(1)
        
    def _prepare_seg(self,img):
        res=np.zeros(img.shape)
        
        mask1=img==[157, 234, 50]
        mask2=img==[128, 64, 128]
        mask=np.logical_or(mask1,mask2)
        res[mask]=1
        res[np.logical_not(mask)]=0
        r=res>0
        r2=res<255
        res[np.logical_and(r,r2)]=255
        res=self._to_gray(res)[:350,:]
        return res
    
    def _process_seg(self, image):
        image.convert(cc.CityScapesPalette)
        i3=np.array(image.raw_data)
        i3 = i3.reshape((512,896,4))[:,:,:3]
        i3=self._prepare_seg(i3)
#        if self.SHOW_VIEW:
#            cv.imshow("",i3)
#            cv.waitKey(1)
        res=cv.resize(i3,(192,182),cv.INTER_AREA)/255.
        self.seg_data=res[:]

        
    def _detect_col(self,data):
        self.collision_data.append(data)
        
    
    def _clear_history(self):
        self.collision_data=[]
        self.is_goal=False
        self.bad_pos=False
        self.reset_timer+=1

    
    def _empty_cycle(self):
        time.sleep(2)
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        transform = random.choice(self.map.get_spawn_points())
        self.vehicle.set_transform(transform)
        self.start_loc=self.vehicle.get_location()
        self.prev_pos=self.start_loc
        
        wp = random.choice(self.map.get_spawn_points())
        self.goal_loc=wp.location
        sp=np.array([self.start_loc.x,self.start_loc.y])
        gp=np.array([self.goal_loc.x,self.goal_loc.y])
        self.dist_from_start_to_end=np.linalg.norm(sp-gp)
    
    def _get_vector_value(self,vec):
        return sqrt(vec.x**2+vec.y**2)
    
    def _get_measures(self):
        speed=self.vehicle.get_velocity()
        speed=self._get_vector_value(speed)
        speed=speed/10. if speed>0 else 0.
        
        acc=self.vehicle.get_acceleration()
        acc=self._get_vector_value(acc)
        acc=acc/10. if acc>0 else 0.
        
        transform=self.vehicle.get_transform()
        loc=transform.location
        dist_from_start=loc.distance(self.start_loc)
        dist_to_goal=loc.distance(self.goal_loc)
        
        dif=self._compute_dif_between_positions(loc)
        self._update_stand(dif)
        
        colls=0
        for col in self.collision_data:
            colls+=self._get_vector_value(col.normal_impulse)
        
        return np.array([speed,acc,dist_from_start,dist_to_goal,colls])
    
    
    def _get_images_data(self):
        data=[]
        data.append(self.rgb_data)
        data.append(self.seg_data)
        data=np.stack(data,2)
        return data

    
    def _get_data(self):
        data=self._get_images_data()
        measures=self._get_measures()
        return data,measures


    def _is_goal(self,dist):
        return dist<0.05

    def _is_collid(self,colls):
        return colls>0

    def _is_so_far(self,dist):
        return dist>=self.dist_from_start_to_end*2.
    
    def _bad_pos(self,colls,dist_to_goal):
        if self._is_collid(colls) or self._is_so_far(dist_to_goal) or \
        self._is_quiet():
            return True
        return False


    ''' is quiet sensor'''
    def _compute_dif_between_positions(self,cur):
        
        dif=np.linalg.norm(np.array([cur.x,cur.y])-
                           np.array([self.prev_pos.x,self.prev_pos.y]))
        self.prev_pos=cur
        return 1 if dif>0.01 else 0
    
    def _init_stand(self):
        self.stand=self.stand_limit
    
    def _is_quiet(self):
        return self.stand==0
    
    def _update_stand(self,value):
        if value and self.stand<self.stand_limit:
            self.stand=self.stand_limit
        elif self.stand>0:
            self.stand-=1
    '''----------------'''

    
    def _compute_reward(self,measures):
        speed,acc,dist_from_start,dist_to_goal,colls=\
        measures[0],measures[1],measures[2],measures[3],measures[4]
        if self._is_goal(dist_to_goal):
            self.is_goal=True
            return 100
        if self._bad_pos(colls,dist_to_goal):
            self.bad_pos=True
            return -100
        alpha=0.1
        reward= alpha*(exp(speed)-exp(acc)+dist_from_start-dist_to_goal)
        return reward


    def _is_done(self,colls,dist_to_goal):
#        print('goal= ',self.is_goal,' bad= ',self.bad_pos)
        return self.is_goal or self.bad_pos

    
    # ---------------------------------
    def reset(self):
        print('reset for the {0} time'.format(self.reset_timer))
        self._clear_history()
        self._empty_cycle()
        self._init_stand()
        data,measures=self._get_data()
        return data,measures
    
    def step(self,actions):
        actions=actions[0]
        steer=np.clip(actions[0],-1,1).astype(np.float32)
        throttle=np.clip(actions[1],0,1).astype(np.float32)
        brake=np.clip(actions[2],0,1).astype(np.float32)
        
        steer=steer.item()
        throttle=throttle.item()
        brake=brake.item()
        
        control=carla.VehicleControl(throttle,steer,brake)
        self.vehicle.apply_control(control)
        data,measures=self._get_data()
        reward=self._compute_reward(measures)
        done=self._is_done(measures[4],measures[3])
        return data,measures,reward,done,{}
    
    def destroy(self):
        for actor in self.actors:
            actor.destroy()
            
    def log(self,measures):
        speed,acc,dist_from_start,dist_to_goal,colls=\
        measures[0],measures[1],measures[2],measures[3],measures[4]
        print('speed=',speed)
        print('acc=',acc)
        print('dis to goal=',dist_to_goal)
        print('dis from start=',dist_from_start)
        print('dist from start to end=',self.dist_from_start_to_end)
        print('colls=',colls)
        
        print('stand=',self.stand)