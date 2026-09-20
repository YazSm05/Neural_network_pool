import math
import pymunk 

class PhysicsSimulator:
    def __init__(self, pool_w, pool_h, pockets, pocket_radius):
        self.pool_w = pool_w
        self.pool_h = pool_h
        self.pockets = pockets
        self.pocket_radius = pocket_radius

    def run_shot(self, white_ball, black_ball, my_balls, opponent_balls, angle_deg, force_percent, display=False, screen=None, clock=None):
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

        # Generating balls
        body_colors = {}
        def add_physics_ball(ball_obj, color):
            if ball_obj.is_potted:
                return None 
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, 10.0))
            body.position = (ball_obj.x, ball_obj.y)
            shape = pymunk.Circle(body, 10.0)
            shape.elasticity = 0.95
            space.add(body, shape)
            body_colors[body] = color
            return body

        white_body = add_physics_ball(white_ball, (255, 255, 255))
        black_body = add_physics_ball(black_ball, (30, 30, 30))
        my_bodies = [(b, add_physics_ball(b, (50, 150, 255))) for b in my_balls] # Blue
        opponent_bodies = [(b, add_physics_ball(b, (255, 50, 50))) for b in opponent_balls] # Red

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
            
            # Pygame simulation
            if display and screen and clock:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                        
                screen.fill((34, 100, 34))
                for px, py in self.pockets:
                    pygame.draw.circle(screen, (10, 10, 10), (int(px), int(py)), int(self.pocket_radius))
                for body, color in body_colors.items():
                    pygame.draw.circle(screen, color, (int(body.position.x), int(body.position.y)), 10)
                    
                pygame.display.flip()
                clock.tick(60) # Lock to 60 FPS to be watchable

            # Continue simulation if any ball still has enough motion
            moving = False
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
