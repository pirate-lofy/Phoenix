import sys
import gym
from gym import spaces
import numpy as np
import random
import time
from math import sqrt
from colorama import Fore

#linux
try:
    sys.path.append("carla-0.9.5-py3.5-linux-x86_64.egg")
except IndexError:
    print(Fore.YELLOW+'CarlaEnv log: cant append carla #egg'+Fore.WHITE)

#
## windows
#try:
#    sys.path.append("carla-0.9.5-py3.7-win-amd64.egg")
#except IndexError:
#    print(Fore.YELLOW+'CarlaEnv log: cant append carla egg'+Fore.WHITE)


import carla
from carla import ColorConverter as cc
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from navigation.misc import draw_waypoints
import cv2 as cv

class CarlaEnv(gym.Env):
    # initial variables
    callibaration_shape=(182,192)
    callibarationRGB_shape=(182,192,2)
    measures_shape=(4,)
    hl_command_shape=(7,)
    
    commands=['VOID','LEFT','RIGHT' ,'STRAIGHT','LANEFOLLOW','CHANGELANELEFT', 'CHANGELANERIGHT']
    num_envs=1
    n_actions=1
    stand=150
    stand_limit=150
    actors=[]
    collision_data=[]
    invasion_data=[]
    SHOW_VIEW=True
    is_goal=None
    bad_pos=None
    off_road=None
    reset_timer=0
    wait=300
    c=0
    rgb_data=None
    seg_data=None
    checkpoint=None
    dist_to_goal=None
    visited=None
    episode_limit=50

    metadata = {'render.modes': ['human']}
    
    def __init__(self,env_id,host='127.0.0.1', port=2000,timeout=20):
        super(CarlaEnv,self).__init__()
    
        self.env_id=env_id
        self.host=host
        self.port=port
        self.timeout=timeout
        
#        self.observation_space = spaces.Tuple(
#            (spaces.Box(0, 255, self.callibarationRGB_shape), 
#            spaces.Box(-np.inf, np.inf, (5,)),
#            spaces.Box(0,1, (4,))
#            )
#        )
        self.observation_space=spaces.Box(0, 255, self.callibarationRGB_shape)
        self.measures_space=spaces.Box(-np.inf,np.inf, self.measures_shape)
        self.hl_command_space=spaces.Box(0, 1, self.hl_command_shape)
        self.action_space = spaces.Box(-1, 1, shape=(self.n_actions,))
        
        self.blueprint=self._connect()
        self.vehicle=self._add_vehicle(self.blueprint)
        self._add_actors()
        self.gpDAO=GlobalRoutePlannerDAO(self.map)
        self.gp=GlobalRoutePlanner(self.gpDAO)
        self.gp.setup()
    
    def _connect(self):
        self.client = carla.Client(self.host,self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.map=self.world.get_map()
        print(Fore.YELLOW+'CarlaEnv log: env no. {0} client connected'.format(self.env_id)+Fore.WHITE)
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
        
        # lane invasion
        inv= self.blueprint.find('sensor.other.lane_invasion')
        inv = self.world.spawn_actor(inv, transform, attach_to=self.vehicle)
        inv.listen(lambda data,: self._detect_lane_invasion(data))
        self.actors.append(inv)
    
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
        self.rgb_data=img_gray.copy()
        
#        if self.SHOW_VIEW:
#            cv.imshow('front view',img)
#            cv.waitKey(1)
        
    def _prepare_seg(self,img):
#        res=np.zeros(img.shape)
#        
#        mask1=img==[157, 234, 50]
#        mask2=img==[128, 64, 128]
#        mask=np.logical_or(mask1,mask2)
#        res[mask]=1
#        res[np.logical_not(mask)]=0
#        r=res>0
#        r2=res<255
#        res[np.logical_and(r,r2)]=255
#        res=self._to_gray(res)[:350,:]
        res=self._to_gray(img)[:350,:]
        return res
    
    def _process_seg(self, image):
        image.convert(cc.CityScapesPalette)
        i3=np.array(image.raw_data)
        i3 = i3.reshape((512,896,4))[:,:,:3]
        i3=self._prepare_seg(i3)
        res=cv.resize(i3,(192,182),cv.INTER_AREA)/255.
        self.seg_data=res.copy()
        
#        if self.SHOW_VIEW:
#            cv.imshow("",res)
#            cv.waitKey(1)

        
    def _detect_col(self,data):
        self.collision_data.append(data)
        
    
    def _detect_lane_invasion(self,data):
        self.invasion_data.append(data)
        
    
    def _clear_history(self):
        self.collision_data=[]
        self.invasion_data=[]
        self.is_goal=False
        self.bad_pos=False
        self.off_road=False
        self.reset_timer+=1
        self.timer=time.time()


    def get_waypoints(self,route):
        waypoints=[]
        for i in route:
            point=self.gp._graph.nodes[i]['vertex']
            point=carla.Location(*point)
            point=self.gpDAO.get_waypoint(point)
            waypoints.append(point)
        return waypoints
    
    def _get_between(self,p1,p2):
        x1,y1=p1[0],p1[1]
        x2,y2=p2[0],p2[1]
        z=p1[2]
        ydiff=y2-y1
        xdiff=x2-x1
        slope=(y2-y1)/(x2-x1)
        n=int((abs(ydiff)+abs(xdiff))*0.8)-1
        
        points=[]
        for i in range(n):
            y=0 if slope==0 else ydiff*(i/n)
            x=xdiff*(i/n) if slope==0 else y/slope
            point=(int(round(x)+x1),int(round(y)+y1))
            points.append(point)
        points.append((p2[0],p2[1]))
        
        points=[p+(z,) for p in points]
        return points
        
    def _get_new_waypoints(self,route):
        new_points=[]
        for i in range(len(route)-1):
            i1=route[i]
            i2=route[i+1]
            point1=self.gp._graph.nodes[i1]['vertex']
            point2=self.gp._graph.nodes[i2]['vertex']
            points=self._get_between(point1,point2)
            
            for point in points:
                point=carla.Location(*point)
                point=self.gpDAO.get_waypoint(point)
                new_points.append(point)
                
        return new_points
    
    def _initialize_position(self):
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        transform = random.choice(self.map.get_spawn_points())
        self.vehicle.set_transform(transform)
        time.sleep(2)
        self.start_loc=self.vehicle.get_location()
        self.prev_pos=self.start_loc
        
        wp = random.choice(self.map.get_spawn_points())
        self.goal_loc=wp.location
        sp=np.array([self.start_loc.x,self.start_loc.y])
        gp=np.array([self.goal_loc.x,self.goal_loc.y])
        self.dist_from_start_to_end=np.linalg.norm(sp-gp)
        
        self.route=self.gp._path_search(self.start_loc,self.goal_loc)
        self.waypoints=self._get_new_waypoints(self.route)
        self.visited=[False]*len(self.waypoints)
        self.c=0
        
    
    def _empty_cycle(self):
        print(Fore.YELLOW+'CarlaEnv log: empty cycle'+Fore.WHITE)
        for _ in  range(self.wait):
            self.step([[0.,0.,0.]])
        print(Fore.YELLOW+'CarlaEnv log: empty cycle ended'+Fore.WHITE)
    
    def _get_vector_value(self,vec):
        return sqrt(vec.x**2+vec.y**2)
    
    def draw_path(self,points):
        draw_waypoints(self.world,points)
    
    def check_path(self,loc):
        loc=self.gp._localize(loc)
        for i in range(len(self.waypoints)):
            if self.visited[i]:
                continue
            point=self.waypoints[i].transform.location
            point=self.gp._localize(point)
            if point==loc:
                self.visited[i]=True
                return True
        return False
    
    def _get_measures(self):
        speed=self.vehicle.get_velocity()
        speed=self._get_vector_value(speed)
        speed=speed/10. if speed>0 else 0.
        
        acc=self.vehicle.get_acceleration()
        acc=self._get_vector_value(acc)
        acc=acc/10. if acc>0 else 0.
        
        transform=self.vehicle.get_transform()
        loc=transform.location
        self.dist_to_goal=loc.distance(self.goal_loc)
        self.checkpoint=self.check_path(loc)
        
        dif=self._compute_dif_between_positions(loc)
        self._update_stand(dif)
        
        ret=self.map.get_waypoint(loc, project_to_road=False)
        if ret is None:
            self.off_road=True
        
        invasion=len(self.invasion_data)
        colls=0
        for col in self.collision_data:
            colls+=self._get_vector_value(col.normal_impulse)
        
#        self.draw_path(self.waypoints)
        return np.array([speed,acc,colls,invasion])
    
    
    def _get_images_data(self):
        data=[]
        data.append(self.rgb_data)
        data.append(self.seg_data)
        data=np.stack(data,2)
        return data

    
    def _get_data(self):
        data=self._get_images_data()
        measures=self._get_measures()
        hl_command=self.get_hl_command()
        return data,measures,hl_command


    def _is_goal(self):
        return self.dist_to_goal<0.05

    def _is_positive(self,val):
        return val>0

    
    def _bad_pos(self,colls,invasion):
        if self._is_positive(colls) or self._is_positive(invasion) or self._is_quiet() or self.off_road:
            return True
        return False


    ''' is quiet sensor'''
    def _compute_dif_between_positions(self,cur):
        
        dif=np.linalg.norm(np.array([cur.x,cur.y])-
                           np.array([self.prev_pos.x,self.prev_pos.y]))
        self.prev_pos=cur
        return 1 if dif>0.05 else 0
    
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

    '''
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
    '''

    def get_hl_command(self):
        cl=self.vehicle.get_location()
        direction=self.gp.abstract_route_plan(cl,self.goal_loc)[0].value        
        if direction==-1:
            direction=0
#        if direction==5 or direction==6 or direction==4:
#            direction=3
        directions_vector=np.zeros(self.hl_command_shape)
        directions_vector[direction]=1
        return directions_vector
    
    def _compute_reward(self,measures):
        colls,invasion=measures[2],measures[3]
        reward=0
        if self._is_goal():
            self.is_goal=True
            reward+= 100
        elif self._bad_pos(colls,invasion):
            self.bad_pos=True
            reward+= -100
        reward+=10 if self.checkpoint else 0.1
#        if reward>0.1:
#            print('env no. {0} reward'.format(self.env_id),reward)
        return reward


    def _is_done(self):
#        print('goal= ',self.is_goal,' bad= ',self.bad_pos)
#        print(self.is_goal,self.bad_pos)
        return self.is_goal or self.bad_pos or self.off_road

    
    def _time_out(self):
        if time.time()>=self.episode_limit+self.timer:
            print(Fore.YELLOW+'CarlaEnv log: env no. {0} exeeded episode limit'.format(self.env_id)+Fore.WHITE)
            return True
        return False
    
    
    # ---------------------------------
    def reset(self):
        print(Fore.YELLOW+'CarlaEnv log: env no. {0} reset for the {1} time'.format(self.env_id,
                      self.reset_timer)+Fore.WHITE)
        self._clear_history()
        self._initialize_position()
        self._init_stand()
#        self._empty_cycle()
        data,measures,hl_command=self._get_data()
        return data,measures,hl_command
    
    def step(self,actions,dead=False):
#        print(actions)
        steer=np.clip(actions[0],-1,1).astype(np.float32)
#        throttle=np.clip(actions[1],0,1).astype(np.float32)
        throttle=0.3
        
        steer=steer.item()
#        throttle=throttle.item()
        if not dead:
            control=carla.VehicleControl(throttle,steer,0)
        else:
            control=carla.VehicleControl(0,0,0)
        self.vehicle.apply_control(control)
        data,measures,hl_command=self._get_data()
        reward=self._compute_reward(measures)
        done=self._is_done()|self._time_out()
        return data,measures,hl_command,reward,done,{}
    
    def dead_command(self):
        self.step([0.,0.,0.],True)
    
    def close(self):
        for actor in self.actors:
            actor.destroy()
            
    def render(self):
        pass
            
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
