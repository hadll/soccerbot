#!/usr/bin/env pybricks-micropython
if __name__ == "__main__":
    import math
    import master
    import slave
    from tqdm import tqdm
    import turtle
    import numpy as np

    goal_scored_state = 0

    class Body:

        def __init__(self, init_pos_cm: list[float, float], dimensions_cm: list[float, float], mass_grams: float):

            self.pos_cm = init_pos_cm
            self.vel_mps = [0, 0]

            self.dimensions_cm = dimensions_cm
            self.mass_grams = mass_grams

        def calculate_distance(self, other_body):

            return math.sqrt((self.pos_cm[0] - other_body.pos_cm[0])**2 + (self.pos_cm[1] - other_body.pos_cm[1])**2)

        def tick(self, delta_time_s):

            self.pos_cm[0] += (self.vel_mps[0]*delta_time_s)
            self.pos_cm[1] += (self.vel_mps[1]*delta_time_s)

            if math.isnan(self.pos_cm[0]) or math.isnan(self.pos_cm[1]):

                raise ValueError("Position values cannot be NaN")

        def is_collided(self, other_body) -> bool:

            left1 = self.pos_cm[0] - self.dimensions_cm[0] / 2
            right1 = self.pos_cm[0] + self.dimensions_cm[0] / 2
            top1 = self.pos_cm[1] + self.dimensions_cm[1] / 2
            bottom1 = self.pos_cm[1] - self.dimensions_cm[1] / 2

            left2 = other_body.pos_cm[0] - other_body.dimensions_cm[0] / 2
            right2 = other_body.pos_cm[0] + other_body.dimensions_cm[0] / 2
            top2 = other_body.pos_cm[1] + other_body.dimensions_cm[1] / 2
            bottom2 = other_body.pos_cm[1] - other_body.dimensions_cm[1] / 2

            if right1 >= left2 and left1 <= right2 and top1 >= bottom2 and bottom1 <= top2:

                return True
            
            else:

                return False

        def collides(self, other_body):

            distance = self.calculate_distance(other_body) or 0.0001

            n = [(self.pos_cm[0] - other_body.pos_cm[0])/distance, (self.pos_cm[1] - other_body.pos_cm[1])/distance]
            t = [-n[1], n[0]]

            v1_n = self.vel_mps[0]*n[0] + self.vel_mps[1]*n[1]
            v1_t = self.vel_mps[0]*t[0] + self.vel_mps[1]*t[1]
            v2_n = other_body.vel_mps[0]*n[0] + other_body.vel_mps[1]*n[1]
            v2_t = other_body.vel_mps[0]*t[0] + other_body.vel_mps[1]*t[1]

            v1_n_new = (v1_n * (self.mass_grams - other_body.mass_grams) + 2 * other_body.mass_grams * v2_n) / (self.mass_grams + other_body.mass_grams)
            v2_n_new = (v2_n * (other_body.mass_grams - self.mass_grams) + 2 * self.mass_grams * v1_n) / (self.mass_grams + other_body.mass_grams)
        
            self.vel_mps = np.array(v1_n_new) * n + np.array(v1_t) * t
            other_body.vel_mps = np.array(v2_n_new) * n + np.array(v2_t) * t

    class Robot(Body):

        def __init__(self, init_pos_cm: list[float, float], dimensions_cm: float, mass_grams: float, average_vel_mps: float):

            super().__init__(init_pos_cm, dimensions_cm, mass_grams)

            self.average_vel_mps = average_vel_mps

        def move(self, direction):

            if direction == -1:

                self.vel_mps = [0, 0]

            else:

                movement_vector = [math.sin(math.radians(direction*45))*self.average_vel_mps, math.cos(math.radians(direction*45))*self.average_vel_mps]

                norm_coef = math.sqrt(movement_vector[0]**2 + movement_vector[1]**2)

                movement_vector[0] /= norm_coef
                movement_vector[1] /= norm_coef

                movement_vector[0] *= self.average_vel_mps
                movement_vector[1] *= self.average_vel_mps

                self.vel_mps = movement_vector

    class Ball(Body):

        def __init__(self, init_pos_cm: list[float, float], dimensions_cm: list[float, float], mass_grams: float):

            super().__init__(init_pos_cm, dimensions_cm, mass_grams)

    class Wall(Body):

        def __init__(self, init_pos_cm: list[float, float], dimensions_cm: list[float, float], mass_grams: float):

            super().__init__(init_pos_cm, dimensions_cm, mass_grams)

    class Goal(Body):

        def __init__(self, init_pos_cm: list[float, float], dimensions_cm: list[float, float], mass_grams: float, side: int):

            super().__init__(init_pos_cm, dimensions_cm, mass_grams)

        def colides(self, other_body):

            super().colides(other_body)

            if type(other_body) == Ball:

                global goal_scored_state

                goal_scored_state = self.side

    bot_ally_ais_connected = [master.QLearning([[i, j] for i in range(-1, 8) for j in range(-1, 8)], True)]
    bot_enemy_ais_connected = [master.QLearning([[i, j] for i in range(-1, 8) for j in range(-1, 8)], True)]
    bot_ally_ais_disconnected = [master.QLearning([[i] for i in range(-1, 8)], False), slave.QLearning([[i] for i in range(-1, 8)], False)]
    bot_enemy_ais_disconnected = [master.QLearning([[i] for i in range(-1, 8)], False), slave.QLearning([[i] for i in range(-1, 8)], False)]

    bodies = list()
    bot_ally_bodies = list()
    bot_enemy_bodies = list()
    ball = None

    delta_time = 0.01
    pos_scale_coefs = (20, 20)

    def configure_bodies():

        bodies = list()
        bot_ally_bodies = list()
        bot_enemy_bodies = list()

        bot_ally_1_body = Robot([71.5, 60], [18, 18], 870, 0.5)
        bot_ally_2_body = Robot([111.5, 60], [18, 18], 870, 0.5)

        bot_enemy_1_body = Robot([71.5, 183], [18, 18], 870, 0.5)
        bot_enemy_2_body = Robot([111.5, 183], [18, 18], 870, 0.5)

        wall_left = Wall([0, 121.5], [1, 243], 1e12)
        wall_right = Wall([182, 121.5], [1, 243], 1e12)
        wall_top = Wall([91.5, 0], [182, 1], 1e12)
        wall_bottom = Wall([91.5, 243], [182, 1], 1e12)

        goal_home = Goal([91.5, 26.3], [45, 7.4], 1e12, 1)
        goal_opposition = Goal([91.5, 216.7], [45, 7.4], 1e12, -1)

        ball = Ball([91.5, 121.5], [7.4, 7.4], 140.0)

        bodies.append(ball)

        bot_ally_bodies.append(bot_ally_1_body)
        bot_ally_bodies.append(bot_ally_2_body)

        bot_enemy_bodies.append(bot_enemy_1_body)
        bot_enemy_bodies.append(bot_enemy_2_body)

        bodies.append(bot_ally_1_body)
        bodies.append(bot_ally_2_body)
        bodies.append(bot_enemy_1_body)
        bodies.append(bot_enemy_2_body)

        bodies.append(wall_left)
        bodies.append(wall_right)
        bodies.append(wall_top)
        bodies.append(wall_bottom)

        bodies.append(goal_home)
        bodies.append(goal_opposition)

        return bodies, bot_ally_bodies, bot_enemy_bodies, ball

    def calculate_bearing(pos_1, pos_2):
        # Calculate differences in coordinates
        delta_x = pos_2[0] - pos_1[0]
        delta_y = pos_2[1] - pos_1[1]
        
        # Calculate the angle in radians
        angle_radians = math.atan2(-delta_y, delta_x)
        
        # Convert to degrees and normalize to 0-360 range
        angle_degrees = math.degrees(angle_radians)
        bearing = (angle_degrees + 360) % 360
        
        return bearing

    def get_bot_state(bot_bodies, ball_body):

        state = list()

        for bot_body in bot_bodies:

            state += [math.floor(bot_body.pos_cm[0]/pos_scale_coefs[0]), \
                    math.floor(bot_body.pos_cm[1]/pos_scale_coefs[1]), \
                    int(((round(calculate_bearing(bot_body.pos_cm, ball_body.pos_cm) / 30)+3) % 12)) or 12, \
                    int(round(max(1, -0.5 * bot_body.calculate_distance(ball_body) + 100)/10))]
            
        return state

    num_iter = int(input("Enter the number of training iterations: "))

    bodies, bot_ally_bodies, bot_enemy_bodies, ball = configure_bodies()

    print([type(body) for body in bodies])

    bot_ally_saps_connected = list()
    bot_enemy_saps_connected = list()
    bot_ally_saps_disconnected = list()
    bot_enemy_saps_disconnected = list()

    for iter_count in tqdm(range(num_iter)):

        simulation_steps = 0

        while True:

            if simulation_steps % 100 == 0:
            
                print(f"Iteration: {iter_count}, Simulation Steps: {simulation_steps}, Simulation Time: {simulation_steps*delta_time} seconds")
                print(f"Ally bot positions: {[bot_ally_body.pos_cm for bot_ally_body in bot_ally_bodies]}")
                print(f"Enemy bot positions: {[bot_enemy_body.pos_cm for bot_enemy_body in bot_enemy_bodies]}")
                print(f"Ball position: {ball.pos_cm}")

            if simulation_steps % 100 == 0 and simulation_steps > 94000:

                print(f"Wall positions: {[wall.pos_cm for wall in bodies if type(wall) == Wall]}")
                input("Press enter to continue...")

            if simulation_steps % int(0.4/delta_time) == 0:

                bot_ally_actions = list()
                bot_enemy_actions = list()

                bot_ally_prewards = [0]*len(bot_ally_bodies)
                bot_enemy_prewards = [0]*len(bot_enemy_bodies)

                if iter_count % 2 == 0:

                    ally_ai_state = get_bot_state(bot_ally_bodies, ball)
                    enemy_ai_state = get_bot_state(bot_enemy_bodies, ball)

                    bot_ally_actions = list(bot_ally_ais_connected[0].get_action(ally_ai_state, False))
                    bot_enemy_actions = list(bot_enemy_ais_connected[0].get_action(enemy_ai_state, False))

                    bot_ally_saps_connected.append((ally_ai_state, bot_ally_actions))
                    bot_enemy_saps_connected.append((enemy_ai_state, bot_enemy_actions))

                else:

                    ally_ai_states = [get_bot_state(bot_ally_bodies[i], ball) for i in range(len(bot_ally_bodies))]
                    enemy_ai_states = [get_bot_state(bot_enemy_bodies[i], ball) for i in range(len(bot_enemy_bodies))]

                    bot_ally_actions = [bot_ally_ais_disconnected[i].get_action(ally_ai_states[i], False) for i in range(len(bot_ally_bodies))]
                    enemy_ally_actions = [bot_enemy_ais_disconnected[i].get_action(ally_ai_states[i], False) for i in range(len(bot_enemy_bodies))]

                    bot_ally_saps_disconnected.append((ally_ai_states, bot_ally_actions))
                    bot_enemy_saps_disconnected.append((enemy_ai_states, bot_enemy_actions))

                #print(bot_ally_ais_connected[0].q_table)
                #input("Press Enter to continue...")

                for i in range(len(bot_ally_bodies)):

                    bot_ally_bodies[i].move(bot_ally_actions[i])
                    bot_enemy_bodies[i].move(bot_enemy_actions[i])

            [body.tick(delta_time) for body in bodies]

            for i in range(len(bodies)):

                if type(bodies[i]) == Wall:

                    continue

                for j in range(i, len(bodies)):

                    if i == j:

                        continue

                    if bodies[i].is_collided(bodies[j]):

                        print(f"{type(bodies[i])} collided with {type(bodies[j])}")

                        bodies[i].collides(bodies[j])

                    if goal_scored_state != 0:

                        break

            if goal_scored_state != 0:

                if iter_count % 2 == 0:

                    bot_ally_ais_connected[0].update_q_values(bot_ally_saps_connected, goal_scored_state, save=True)
                    bot_enemy_ais_connected[0].update_q_values(bot_enemy_saps_connected, -1*goal_scored_state)

                else:

                    for i in range(len(bot_ally_ais_disconnected)):

                        bot_ally_ais_disconnected[i].update_q_values([(x[i], y[i]) for x, y in bot_ally_saps_disconnected], goal_scored_state, save=True)
                        bot_enemy_ais_disconnected[i].update_q_values([(x[i], y[i]) for x, y in bot_enemy_saps_disconnected], -1*goal_scored_state)

                bot_ally_saps_connected = list()
                bot_enemy_saps_connected = list()
                bot_ally_saps_disconnected = list()
                bot_enemy_saps_disconnected = list()

                bodies, bot_ally_bodies, bot_enemy_bodies, ball = configure_bodies()

                break

            simulation_steps += 1