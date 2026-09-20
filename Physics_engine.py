import math
import pymunk 

class PhysicsSimulator:
    def __init__(self, pool_w, pool_h, pockets, pocket_radius):
        self.pool_w = pool_w
        self.pool_h = pool_h
        self.pockets = pockets
        self.pocket_radius = pocket_radius

    def run_shot(self, white_ball, black_ball, my_balls, opponent_balls, angle_deg, force_percent):
        """
        Input:  Balls positions and states, angle and force of the strike
        Output:  Balls positions and states updated
        """
        # Pymunk space generation
        space = pymunk.Space()
        space.damping = 0.85 # Friction simulation (1.0 = Ice, 0.0 = Glue)

        # Pool generation
        walls = [
            pymunk.Segment(space.static_body, (0, 0), (self.pool_w, 0), 5.0),
            pymunk.Segment(space.static_body, (self.pool_w, 0), (self.pool_w, self.pool_h), 5.0),
            pymunk.Segment(space.static_body, (self.pool_w, self.pool_h), (0, self.pool_h), 5.0),
            pymunk.Segment(space.static_body, (0, self.pool_h), (0, 0), 5.0)
        ]
        for wall in walls:
            wall.elasticity = 0.9 # Les bandes rebondissent bien
        space.add(*walls)

        # Generating balls physics
        def add_physics_ball(ball_obj):
            if ball_obj.is_potted:
                return None 
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, 10.0))
            body.position = (ball_obj.x, ball_obj.y)
            shape = pymunk.Circle(body, 10.0)
            shape.elasticity = 0.95
            space.add(body, shape)
            return body

        white_body = add_physics_ball(white_ball)
        black_body = add_physics_ball(black_ball)
        my_bodies = [(b, add_physics_ball(b)) for b in my_balls]
        opponent_bodies = [(b, add_physics_ball(b)) for b in opponent_balls]

        # Generating a strike
        if white_body:
            angle_rad = math.radians(angle_deg)
            # Converting the force into a strike
            fx = math.cos(angle_rad) * (force_percent * 50.0) 
            fy = math.sin(angle_rad) * (force_percent * 50.0)
            white_body.apply_impulse_at_local_point((fx, fy))

        # Simulation
        moving = True
        while moving:
            space.step(1/60.0)
            moving = False
            # Continue the hit simulation if a ball is still going fast enough
            for body in space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC and body.velocity.length > 2.0:
                    moving = True
                    break

        # Updating the balls state and positions
        def update_ball(ball_obj, phys_body):
            if phys_body is None: return
            for px, py in self.pockets:
                dist = math.hypot(phys_body.position.x - px, phys_body.position.y - py)
                if dist < self.pocket_radius:
                    ball_obj.is_potted = True
                    return
            ball_obj.x = phys_body.position.x
            ball_obj.y = phys_body.position.y

        update_ball(white_ball, white_body)
        update_ball(black_ball, black_body)
        for ball_obj, phys_body in my_bodies: update_ball(ball_obj, phys_body)
        for ball_obj, phys_body in opponent_bodies: update_ball(ball_obj, phys_body)
