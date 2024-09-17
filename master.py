#!/usr/bin/env pybricks-micropython
if __name__ == "__main__":
    from pybricks.iodevices import I2CDevice
    from pybricks.ev3devices import Motor
    from pybricks.parameters import Port, Stop, Direction
    from pybricks.robotics import DriveBase
    import math
    from pybricks.messaging import BluetoothMailboxServer, TextMailbox
    import ujson

    class Random:

        # values except for m can be changed freely (m must be prime). for best results, numbers should be around the same order of magnitude
        # algo is LGA

        def __init__(self, a=93024524, c=4523624, m=2147483647, seed=7192650892):

            self.a = a
            self.c = c
            self.m = m
            self.prev_val = seed

        # intended for class-private use only
        # will generate numbers in the range [0, m)

        def _int(self):
        
            self.prev_val = (self.prev_val*self.a + self.c) % self.m

            return self.prev_val
        
        # will generate numbers in the range [min, max)
        
        def int(self, min, max):
        
            return self.choice(list(range(min, max)))
        
        # will generate numbers in the range [0, 1)

        def float(self):

            return self._int() / self.m
        
        # chooses random element of array
        
        def choice(self, arr):

            i = int(self.float()*len(arr))

            return arr[i]
        
    # performs shallow copy of list
        
    def copy_list(l):

        new_list = list()

        for elem in l:

            new_list.append(elem)
        
        return new_list

    # performs shallow copy of dict
        
    def copy_dict(d):

        new_dict = dict()

        for key in d.keys():

            new_dict[key] = d[key]

        return new_dict

    # get max value in arr

    def max(arr):

        max_val = float("-inf")

        for elem in arr:

            if elem > max_val:

                max_val = elem

        return max_val

    random = Random()

    class QLearning:

        def __init__(self, possible_actions, connected, open_from_file=False, a=0.1, g=0.9, l=0.99, e=0.05): # a is learning rate, g is future discount rate, l is elegibility trace discount rate, e is epsilon for epsilon-greedy

            self.connected = connected

            self.possible_actions = possible_actions
            self.q_table = dict() if not open_from_file else ujson.loads(open("q_table_master_connected.pkl" if self.connected else "q_table_master_connected.pkl").read())
            self.a = a
            self.g = g
            self.e = e
            self.l = l

        # intended for class-private use only 
        def _generate_starting_state_dict(self, state):

            if self.connected:

                return {str(action): sum([0.1 if (False if (state[i*4 + 2] == 0) else action[i] == ((round((state[i*4 + 2]/12)*8)) % 8)) else -0.02 for i in range(2)]) for action in self.possible_actions}
            
            else:

                return {str(action): \
                0.1 if (False if (state[2] == 0) \
                else action[0] == \
                ((round((state[2]/12)*8)) % 8)) \
                else -0.1 for action in self.possible_actions}

        # intended for class-private use only 
        # will select action based using epsilon greedy on dict representing the possible actions in the state and the Q-values of each possible action 
        # will select random action if Q-values are equal

        def _epsilon_greedy(self, action_dict, training):

            actions = action_dict.keys()

            if training or random.float() < self.e:

                return eval(random.choice(list(actions)))

            else:

                max_action = None
                max_q_value = float("-inf")

                for action in actions:

                    if (action_dict[action] > max_q_value or (action_dict[action] == max_q_value and random.float() < 0.5)):
                        
                        max_action = action
                        max_q_value = action_dict[action]

                return eval(max_action)
            
        # intended for class-private use only 
        # transitions state to the new state after an action is applied
        # state_action_pair should be an array or tuple, where state_action_pair[0] is the old state, and state_action_pair[1] is action to be applied
        # change does not happen in place, a new state is returned
            
        def _state_transition(self, state_action_pair):

            new_state = copy_list(state_action_pair[0])

            for i in range(len(new_state)//4):

                if state_action_pair[1][i] == -1:

                    return new_state

                dx = 0
                dy = 0

                if state_action_pair[1][4*i] in [1, 2, 3]:

                    dx = 1

                if state_action_pair[1][4*i] in [5, 6, 7]:

                    dx = -1

                if state_action_pair[1][4*i] in [7, 0, 1]:

                    dy = 1

                if state_action_pair[1][4*i] in [5, 4, 3]:

                    dy = -1

                new_state[4*i] += dx
                new_state[4*i + 1] += dy

            return new_state
        
        # using epsilon greedy, getAction computes best action for state

        def get_action(self, state, training):

            action_dict = self.q_table.get(str(state), None)

            if not action_dict:

                self.q_table[str(state)] = self._generate_starting_state_dict(state)

            return self._epsilon_greedy(self.q_table[str(state)], training)
        
        # stateActionsPairs is an array of state_action_pairs, from least recent to most recent, and each element state_action_pair follows the format state_action_pair[0] = state, state_action_pair[1] = action
        
        def update_q_values(self, state_action_pairs, reward, save=False):

            state_action_pairs = list(reversed(state_action_pairs))

            for i in range(len(state_action_pairs)):

                state_action_pair = state_action_pairs[i]

                action_dict = self.q_table.get(str(state_action_pair[0]), None)

                if not action_dict:

                    self.q_table[str(state_action_pair[0])] = self._generate_starting_state_dict(state_action_pair[0])
                    action_dict = self.q_table[str(state_action_pair[0])]

                prev_q_value = action_dict[str(state_action_pair[1])] or 0.0

                max_future_reward = max(list(self.q_table[str(self._state_transition(state_action_pair))].values()))

                new_q_value = prev_q_value + (self.l**i)*self.a*(reward + self.g*max_future_reward - prev_q_value)

                self.q_table[str(state_action_pair[0])][str(state_action_pair[1])] = new_q_value

            if save:

                serialized_q_table = ujson.dump(self.q_table)

                if self.connected:

                    with open('q_table_master_connected.pkl', 'w') as file:
                
                        file.write(serialized_q_table)

                else:

                    with open('q_table_master_disconnected.pkl', 'w') as file:
                
                        file.write(serialized_q_table)

        def save(self):

            serialized_q_table = ujson.dump(self.q_table)

            if self.connected:

                with open('q_table_master_connected.pkl', 'w') as file:
                
                    file.write(serialized_q_table)

            else:

                with open('q_table_master_disconnected.pkl', 'w') as file:
                
                    file.write(serialized_q_table)

    # ALL MEASUREMENTS ARE IN MM

    # self.forward
    # self.backward
    # self.left
    # self.right

    class Vector2:
        def __init__(self, x:float,y:float):
            self.x = x
            self.y = y
        def __add__(self, other):
            return Vector2(self.x + other.x, self.y + other.y)
        def __sub__(self, other):
            return Vector2(self.x - other.x, self.y - other.y)
        def __mul__(self, other):
            return Vector2(self.x * other, self.y * other)
        def __truediv__(self, other):
            return Vector2(self.x / other, self.y / other)
        def __str__(self):
            return str(self.x)+", "+str(self.y)
        def __tuple__(self):
            return (self.x, self.y)
        def to_tuple(self):
            return (self.x, self.y)
        def to_list(self):
            return [self.x, self.y]
        def zero():
            return Vector2(0,0)
        def up():
            return Vector2(0,1)
        def right():
            return Vector2(1,0)
        def direction(angle:float):
            return Vector2(math.cos(angle), math.sin(angle))
        def normalize(self):
            return 1 / self.magnitude()
        def magnitude(self):
            return (self.x**2 + self.y**2)**0.5

    class IRSeeker:
        def __init__(self, port: Port):
            self.sensor = I2CDevice(port, 0x08)
        def read(self):
            data = self.sensor.read(2,2)
            return (((data[0] + 3)%12) or 12, data[1])
        def get_direction(self):
            data = self.read()
            return data[0]
        def get_strength(self):
            data = self.read()
            return data[1]

    class Wheels:
        def __init__(self, forward_port: Port, backward_port: Port, left_port: Port, right_port: Port, init_pos: tuple[float, float], pos_scale_coefs: tuple[int, int] = (9, 12)) -> None:
            self.forward = Motor(forward_port)
            self.backward = Motor(backward_port)
            self.left = Motor(left_port)
            self.right = Motor(right_port)
            self.diameter = 50 #mm
            self.circumference = math.pi * self.diameter
            self.init_pos = init_pos
            self.pos_scale_coefs = pos_scale_coefs
            self.forward.reset_angle(0)
            self.backward.reset_angle(0)
            self.left.reset_angle(0)
            self.right.reset_angle(0)
        def drive(self, direction:int, distance:int) -> None:
            vertical_mod = math.cos(direction)
            horizontal_mod = math.sin(direction)
            target_angle = distance/self.circumference*360
            print(vertical_mod, target_angle)
            print(abs(vertical_mod),target_angle*(vertical_mod**0))
            self.forward.run_angle(abs(vertical_mod),math.copysign(target_angle,vertical_mod),Stop.HOLD,False)
            self.backward.run_angle(abs(vertical_mod),-math.copysign(target_angle,vertical_mod),Stop.HOLD,False)
            self.left.run_angle(abs(horizontal_mod),-math.copysign(target_angle,vertical_mod),Stop.HOLD,False)
            self.right.run_angle(abs(horizontal_mod),math.copysign(target_angle,vertical_mod),Stop.HOLD,False)
            # self.forward.run_angle(1,180,Stop.HOLD,False)
            # self.backward.run_angle(1,-180,Stop.HOLD,False)
            # self.left.run_angle(1,180,Stop.HOLD,False)
            # self.right.run_angle(1,180,Stop.HOLD,False)
        def stop(self) -> None:
            self.forward.stop()
            self.backward.stop()
            self.left.stop()
            self.right.stop()
        def get_x(self) -> int:
            return math.floor((((self.right.angle() - self.left.angle()) / 720 * self.circumference) + self.init_pos[0])/self.pos_scale_coefs[0]) # get both for better accuracy
        def get_y(self) -> int:
            return math.floor((((self.backward.angle() - self.forward.angle()) / 720 * self.circumference) + self.init_pos[1])/self.pos_scale_coefs[1])
        def get_pos(self) -> Vector2:
            return Vector2(self.get_x(),self.get_y())
    
    class DriveBaseWheels:
        def __init__(self, forward_port: Port, backward_port: Port, left_port: Port, right_port: Port, init_pos: tuple[float, float], pos_scale_coefs: tuple[int, int] = (9, 12), use_gyro:Bool = False) -> None:
            self.forward = Motor(forward_port, Direction.COUNTERCLOCKWISE)
            self.backward = Motor(backward_port)
            self.left = Motor(left_port, Direction.COUNTERCLOCKWISE)
            self.right = Motor(right_port)
            self.diameter = 50 #mm
            self.axle_track = 150 # the distance between the wheels
            self.circumference = math.pi * self.diameter
            self.init_pos = init_pos
            self.pos_scale_coefs = pos_scale_coefs
            self.forwards_drive = DriveBase(self.left, self.right, wheeldiameter=self.diameter, axle_track = self.axle_track)
            self.sideways_drive = DriveBase(self.forward, self.backward, wheeldiameter=self.diameter, axle_track = self.axle_track)
            self.forwards_drive.use_gyro(use_gyro)
            self.sideways_drive.use_gyro(use_gyro)
        def drive(self, direction:int, distance:int) -> None:
            vertical_mod = math.cos(direction)
            horizontal_mod = math.sin(direction)
            forwards_drive.straight(distance*vertical_mod,wait=False)
            sideways_drive.straight(distance*horizontal_mod,wait=False)
        def stop(self) -> None:
            self.forward.stop()
            self.backward.stop()
            self.left.stop()
            self.right.stop()
        def get_x(self) -> int:
            return math.floor((sideways_drive.distance() + self.init_pos[0])/self.pos_scale_coefs[0]) # get both for better accuracy
        def get_y(self) -> int:
            return math.floor((forwards_drive.distance() + self.init_pos[1])/self.pos_scale_coefs[1])
        def get_pos(self) -> Vector2:
            return Vector2(self.get_x(),self.get_y())


    class Bot:

        def __init__(self, is_training: bool, connected=True):

            self.MM_SCALE_COEF = 100
            self.IR_SCALE_COEF = 10

            self.is_training = is_training

            self.connected = connected

            if connected:

                try:

                    self.server = BluetoothMailboxServer()

                    self.server.wait_for_connection()

                except OSError:

                    self.disconnect()

                self.mbox = TextMailbox("communication", self.server)

            self.qlearning_connected = QLearning([[i, j] for i in range(-1, 8) for j in range(-1, 8)], True)
            self.qlearning_disconnected = QLearning([[i] for i in range(-1, 8)], False)
            self.ir = IRSeeker(Port.S1)
            self.wheels = Wheels(Port.A, Port.D, Port.C, Port.B,(71.5, 60))

            if is_training:

                self.state_action_pairs = list()

        def disconnect(self):

            self.connected = False

            self.server.close()

        def main_loop(self):

            print("in main loop")

            while True:

                print("loop iter")

                x_pos, y_pos = self.wheels.get_pos().to_tuple()

                x_coord = int(x_pos/self.MM_SCALE_COEF)
                y_coord = int(y_pos/self.MM_SCALE_COEF)

                ir_bearing = self.ir.get_direction()
                ir_distance = int(self.ir.get_strength()/self.IR_SCALE_COEF)

                action_self = -1

                state_self = [x_coord, y_coord, ir_bearing, ir_distance]

                if self.connected:

                    self.mbox.wait()

                    state_client = ""

                    while state_client == "": # TODO: implement a max_iter on this loop
                        
                            self.mbox.wait()
                            state_client = int(self.mbox.read())

                    state = state_self + state_client

                    action_self, action_client = self.qlearning_connected.get_action(state, False)

                    try:

                        self.mbox.send(str(action_client))

                    except OSError:

                        self.disconnect()

                        action_self = self.qlearning_disconnected.get_action(state, False)[0]

                else:

                    action_self = self.qlearning_disconnected.get_action(state_self, False)[0]

                if action_self == -1:

                    self.wheels.stop()

                else:

                    move_dir = action_self*45

                    self.wheels.drive(move_dir, self.MM_SCALE_COEF)

print("starting")
bot = Bot(False, connected=False)
bot.main_loop()
