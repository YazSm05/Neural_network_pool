import numpy as np
import copy
import math
import pygame
import sys
import random

#NeuralNetwork(32, 128, 2) for input: 16 balls with 2 coords, outputs: hit force and direction, 128 hidden layers for deep learning
class NeuralNetwork:
  def __init__(self, in_nodes = 32, hid_nodes = 128, out_nodes = 2):
      self.in_nodes = in_nodes
      self.hid_nodes = hid_nodes
      self.out_nodes = out_nodes
      self.w_inputtohid = np.random.uniform(-1, 1, (hid_nodes, in_nodes))
      self.w_hidtooutput = np.random.uniform(-1, 1, (out_nodes, hid_nodes))
      self.score = 0.0
  
  def feed_forward(self, inputs):
      """
      input is a list of 32 integers (x,y) of each ball unknowingly that
        Ball 1 : white ball
        Ball 2 : black ball
        Balls 3-9 : player's balls
        Balls 10-16: opponent's balls
      """
    
      inputs = np.array(inputs).reshape(-1, 1)
      hidden = np.tanh(np.dot(self.w_inputtohid, inputs))
      output = np.tanh(np.dot(self.w_hidtooutput, hidden))
  
      angle = (output[0][0] + 1.0) * 180.0 
      force = (output[1][0] + 1.0) * 50.0
    
  def mutate(self, rate=0.05):
      mutate_inputtohid = np.random.rand(*self.w_inputtohid.shape) < rate
      mutate_hidtooutput = np.random.rand(*self.w_hidtooutput.shape) < rate
      self.w_inputtohid += np.random.normal(0, 0.1, self.w_inputtohid.shape) * mutate_inputtohid
      self.w_hidtooutput += np.random.normal(0, 0.1, self.w_hidtooutput.shape) * mutate_hidtooutput

class Ball:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.is_potted = False
      
class BilliardEnvironment:
  def __init__(self):
      self.pool_w = 800.0
      self.pool_h = 400.0
      self.ball_radius = 10.0
      self.pocket_radius = 30.0 # Danger zone around the pockets to avoid generating the balls too close to the pockets
      self.pockets = [
          (0, 0), (self.pool_w / 2, 0), (self.pool_w, 0),
          (0, self.pool_h), (self.pool_w / 2, self.pool_h), (self.pool_w, self.pool_h)
      ]
      self.placed_positions = [] # List to avoid the balls to randomly generate on top of each other

      # Random generations of the balls
      self.white_ball = self._create_random_ball()
      self.black_ball = self._create_random_ball()
      self.my_balls = [self._create_random_ball() for _ in range(7)]
      self.opponent_balls = [self._create_random_ball() for _ in range(7)]

      # Game parameters
      self.my_balls_remaining = 7
      self.opponent_balls_potted_this_shot = 0
      self.white_ball_potted = False
      self.black_ball_potted = False
      self.is_alive = True
      self.shot_force = 0.0

  def _create_random_ball(self):
      """
        Generating a ball in a valid position away from pockets and not overlapping with other balls
      """
      while True:
          # Generating within the pool area
          rx = random.uniform(self.ball_radius * 2, self.pool_w - self.ball_radius * 2)
          ry = random.uniform(self.ball_radius * 2, self.pool_h - self.ball_radius * 2)
          
          # Verification of proximity within the pockets
          in_danger_zone = False
          for px, py in self.pockets:
              dist_to_pocket = math.hypot(rx - px, ry - py) 
              if dist_to_pocket < (self.pocket_radius + self.ball_radius):
                  in_danger_zone = True
                  break
          if in_danger_zone:
              continue
          
          # Verification of the overlap with other balls
          overlap = False
          for px, py in self.placed_positions:
              dist_to_ball = math.hypot(rx - px, ry - py)
              if dist_to_ball < (self.ball_radius * 2.2): # 2.2 is a security measure
                  overlap = True
                  break
          if overlap:
              continue
          
          self.placed_positions.append((rx, ry))
          return Ball(rx, ry)

  def get_inputs(self):
      """
        Set up the inputs for the neural network
      """
      inputs = []
      
      def add_ball_data(ball):
          if ball.is_potted:
              inputs.extend([-5000.0, -5000.0]) # The ball is out of reach so the neural network ignores it
          else:
              inputs.extend([ball.x / self.pool_w, ball.y / self.pool_h]) # Simplify coordinates for easier processing by the neural network
      
      add_ball_data(self.white_ball)
      add_ball_data(self.black_ball)
      for ball in self.my_balls: add_ball_data(ball)
      for ball in self.opponent_balls: add_ball_data(ball)
      
      return inputs

  def play_shot(self, angle_deg, force_percent):
      """Simule l'impact de la queue de billard et le mouvement des billes."""
      self.shot_force = force_percent
      
      if self.shot_force < 5.0:
          self.is_alive = False
          return

      # =================================================================
      # Physics engine in another file
      # =================================================================
      
      self.is_alive = False 

  def calculate_score(self):
      """L'arbitre donne la note à la fin du tir."""
      score_final = 0.0

      if self.shot_force < 5.0:
          return -500 

      if self.white_ball_potted:
          score_final -= 1000
      score_final -= (self.opponent_balls_potted_this_shot * 500)

      if self.black_ball_potted:
          if self.my_balls_remaining == 0:
              score_final += 5000 
          else:
              return -5000 

      my_balls_potted_this_shot = 7 - self.my_balls_remaining
      score_final += (my_balls_potted_this_shot * 1000)

      return score_final
