#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import numpy as np
import time
import zmq
from threading import Thread

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

class SimComms(object):
    def __init__(self, pub_topic='control_traj', sub_topic='robot_state', host='127.0.0.1', pub_port='5001',
                 sub_port='5002'):
        self.dt = 1 / 1000.0
        self.done = False
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://{}:{}".format(host, pub_port))
        
        
        self.pub_topic = pub_topic
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.sub_socket.connect("tcp://{}:{}".format(host, sub_port))
        self.sub_topic = sub_topic
        self.sub_socket.subscribe(self.sub_topic)
        #self.state = None
        self.state_message = None
        self.state_topic = None
        self.pub_array = None
        
        self.t1 = Thread(target=self.thread_fn_sub)
        self.t1.start()
        self.t2 = Thread(target=self.thread_fn_pub)
        self.t2.start()


    def thread_fn_sub(self):
        while not self.done:
            try:
                topic = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again as e:
                time.sleep(self.dt)
                continue
            state = recv_array(self.sub_socket, flags=zmq.NOBLOCK, copy=True, track=False)

            if(topic == self.sub_topic):
                self.state_message = state
                self.state_topic = topic
    def thread_fn_pub(self):
        #print(self.done)
        while(not self.done):
            if(self.pub_array is not None):
                #print(self.robot_state_pub)
                #print(self.pub_topic, self.pub_array)
                self.pub_socket.send_string(self.pub_topic, flags=zmq.SNDMORE)
                send_array(self.pub_socket,self.pub_array, flags=0, copy=True, track=False)
                self.pub_array = None
            time.sleep(self.dt)
        return True

        
    def close(self):
        self.done = True
        self.sub_socket.close()
        self.pub_socket.close()
        self.context.term()
    def send_command(self, cmd):
        #print("setting command...")
        self.pub_array = cmd
        #print(self.done)
        #print(self.pub_array)
        
    def get_state(self):
        state_message = self.state_message
        self.state_message = None
        return state_message




class RobotInterface(object):
    def __init__(self, pub_topic='control_traj', sub_topic='robot_state', host='127.0.0.1', pub_port='5001', sub_port='5002',pair_port='5003'):

        self.sub_hz = 500.0
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.host = host
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.state_topic = sub_topic
        self.zmq_comms = SimComms(sub_topic=sub_topic,
                                  pub_topic=pub_topic,
                                  host=host,
                                  sub_port=sub_port,
                                  pub_port=pub_port)
        
        
    def get_state(self):
        #print("waiting on state")
        self.state = self.zmq_comms.get_state()
        while(self.state is None):
            try:
                self.state = self.zmq_comms.get_state()
            except KeyboardInterrupt:
                exit()
        
        self.state = np.ravel(self.state)
        
        if(len(self.state) > 6*3):
            state = self.state[:-9]
            goal_pose = self.state[-9:-2]
            #print(self.state)
            t_idx = self.state[-2:-1]
            open_loop = self.state[-1:0]
        else:
            state = self.state
            goal_pose, t_idx, open_loop = None, None, 0
            
        state_dict = {'robot_state': state,'goal_pose': goal_pose, 't_idx': t_idx, 'open_loop':
                      bool(open_loop)}    
        self.state = None
        return state_dict

    def publish_state(self, state):
        pass
 
    def publish_action(self, action_traj, append_time=True,dt=0.1):
        num_commands = action_traj.shape[0]
        
        if append_time:
            command_times = np.arange(start=0.0,stop=dt*len(action_traj), step=dt).reshape(len(action_traj),1)
            #print(command_times)
        command = action_traj
        if append_time:        
            command = np.concatenate((command, command_times),axis=-1)
        #print(command)
        self.zmq_comms.send_command(command)
    
   
    def publish_command(self, command_state_seq, mode='acc', append_time=True):
        num_commands = command_state_seq.shape[0]
        q = np.array(command_state_seq[:,:6]) #TODO: Remove hardode!
        qd = np.array(command_state_seq[:,6:12])
        qdd = np.array(command_state_seq[:, 12:18])
        if append_time:
            command_times = np.array(command_state_seq[:, -1]).reshape(num_commands,1)

        if mode == 'pos':
            command = q
        elif mode == 'vel':
            command = qd
        elif mode == 'acc':
            command = qdd
        elif mode == 'full':
            command = command_state_seq
        if append_time:        
            command = np.concatenate((command, command_times),axis=-1)
        print('Publishing plan', command.shape)
        self.zmq_comms.send_command(command)
    
        
    def close(self):
        #close thread
        self.zmq_comms.close()
