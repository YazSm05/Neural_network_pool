import math
import pymunk 
import pygame
import sys

class PhysicsSimulator:
    def __init__(self, pool_w, pool_h, pockets, pocket_radius):
        self.pool_w = pool_w
        self.pool_h = pool_h
        self.pockets = pockets
        self.pocket_radius = pocket_radius

    def run_shot(self, white_ball, black_ball, my_balls, opponent_balls, angle_deg, force_percent, display=False, screen=None, clock=None):
        # Pymunk space generation
        space = pymunk.Space()
        space.damping = 0.85 # Friction simulation

        # Pool table generation
        pr = self.pocket_radius
        walls = [
            # Top cushion (2 segments, separated by the middle pocket)
            pymunk.Segment(space.static_body, (pr, 0), (self.pool_w / 2 - pr, 0), 5.0),
            pymunk.Segment(space.static_body, (self.pool_w / 2 + pr, 0), (self.pool_w - pr, 0), 5.0),
            # Bottom cushion (2 segments, separated by the middle pocket)
            pymunk.Segment(space.static_body, (pr, self.pool_h), (self.pool_w / 2 - pr, self.pool_h), 5.0),
            pymunk.Segment(space.static_body, (self.pool_w / 2 + pr, self.pool_h), (self.pool_w - pr, self.pool_h), 5.0),
            # Left cushion (leaves space at the top and bottom corners)
            pymunk.Segment(space.static_body, (0, pr), (0, self.pool_h - pr), 5.0),
            # Right cushion (leaves space at the top and bottom corners)
            pymunk.Segment(space.static_body, (self.pool_w, pr), (self.pool_w, self.pool_h - pr), 5.0)
        ]
        
        for wall in walls:
            wall.elasticity = 0.9 # High bounciness for the cushions
        space.add(*walls)

        # Dictionaries to manage rendering and potting logic
        body_colors = {}
        body_to_ball = {} # Links the physics engine body to the Python Ball class

        def add_physics_ball(ball_obj, color):
            if ball_obj.is_potted:
                return None 
                
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, 10.0))
            body.position = (ball_obj.x, ball_obj.y)
            shape = pymunk.Circle(body, 10.0)
            shape.elasticity = 0.95
            space.add(body, shape)
            
            body_colors[body] = color
            body_to_ball[body] = ball_obj # Save the link
            return body

        white_body = add_physics_ball(white_ball, (255, 255, 255))
        black_body = add_physics_ball(black_ball, (30, 30, 30))
        my_bodies = [(b, add_physics_ball(b, (50, 150, 255))) for b in my_balls]  # Blue
        opponent_bodies = [(b, add_physics_ball(b, (255, 50, 50))) for b in opponent_balls] # Red

        if white_body:
            angle_rad = math.radians(angle_deg)
            fx = math.cos(angle_rad) * (force_percent * 15.0) 
            fy = math.sin(angle_rad) * (force_percent * 15.0)
            white_body.apply_impulse_at_local_point((fx, fy))

        # Main simulation loop
        moving = True
        while moving:
            space.step(1/60.0)
            
            # --- REAL-TIME POCKET VERIFICATION ---
            bodies_to_remove = []
            for body in list(body_colors.keys()):
                for px, py in self.pockets:
                    dist = math.hypot(body.position.x - px, body.position.y - py)
                    # We increase the detection zone (+10.0) to counter the cushion's hitbox
                    if dist < (self.pocket_radius + 10.0):
                        bodies_to_remove.append(body)
                        break 
                        
            for body in bodies_to_remove:
                # 1. Officially declare the ball potted for the neural network referee
                body_to_ball[body].is_potted = True 
                # 2. Remove it from the physical world and the Pygame screen
                space.remove(body, list(body.shapes)[0])
                del body_colors[body]
                
            # --- PYGAME VISUAL RENDERING ---
            if display and screen and clock:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                        
                screen.fill((34, 100, 34))
                for px, py in self.pockets:
                    pygame.draw.circle(screen, (10, 10, 10), (int(px), int(py)), int(self.pocket_radius))
                for body, color in body_colors.items():
                    pygame.draw.circle(screen, color, (int(body.position.x), int(body.position.y)), 10)
                    
                pygame.display.flip()
                clock.tick(60) 
            else: pygame.event.pump()
            
            # --- MOVEMENT VERIFICATION ---
            moving = False
            for body in space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC and body.velocity.length > 2.0:
                    moving = True
                    break

        # Final position update for the next shot
        def update_ball(ball_obj, phys_body):
            if phys_body is None: return
            if ball_obj.is_potted: return # We no longer update fallen balls
            ball_obj.x = phys_body.position.x
            ball_obj.y = phys_body.position.y

        update_ball(white_ball, white_body)
        update_ball(black_ball, black_body)
        for ball_obj, phys_body in my_bodies: update_ball(ball_obj, phys_body)
        for ball_obj, phys_body in opponent_bodies: update_ball(ball_obj, phys_body)
