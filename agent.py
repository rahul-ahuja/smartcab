import random
import math
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import csv
import pickle

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon=0.8, alpha=0.8, gamma=0.9,tno=1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
	#self.q = {} #initialize q table dictionary
	self.epsilon = epsilon  #exploration rate
	self.alpha = alpha  # learning rate
	self.gamma = gamma   #discount rate
	#self.lesson_counter = 0  #counts number of steps learned
	#self.steps_counter = 0 #counts steps in the trial
        self.reward_previous = None
        self.action_previous = None
        self.state_previous = None
	#self.q = {(('red',None),'right'):10,(('green',None),'forward'):15, (('green',None),'left'):20,(('red',None),'left'):25}
	self.q = {}
	self.tno = 0
	#self.success = []
	
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
	#self.steps_counter = 0  #after finishing every trial counter gets 0
	self.tno += 1 


    def getQ(self, state, action):
        return self.q.get((state, action), 10.0)  #helper function to get the q-value from the q-table.
	

 
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
	location = self.env.agent_states[self]["location"] 
	destination = self.env.agent_states[self]["destination"]
	trial_num = t
	#self.tno += 1
	#sucess = self.env.sucess()

        # TODO: Update state
	self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        #action = None
	self.actions = [None, 'left', 'right', 'forward']
	#action = self.next_waypoint
	#action = random.choice(Environment.valid_actions)
	#if random.random() < self.epsilon:    #exploration e-greedy strategy
	if random.random() < self.epsilon * 1./self.tno:    #exploration e-greedy strategy

    		action = random.choice(self.actions)
	else:
		q = [self.getQ(self.state, a) for a in self.actions] # gives out list q-values for the visited state-action pairs
		maxQ = max(q) # get the maximum values of q-values from the above list
		count = q.count(maxQ)  #count the number of maximum values
		if count > 1:
            		best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            		i = random.choice(best)  #select the action randomly from the maximum q-values index 
    		else:
        		i = q.index(maxQ)


		action = self.actions[i]
	#action = random.choice(Environment.valid_actions)

#decay rate
#Environment.valid_actions
#defaultdict


        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

	if t == 0:
		oldv = self.q.get((self.state,action), 10.0)
		self.q[(self.state, action)] = oldv   #if it's first counter run in the trial then get q-value
	else:
		oldv = self.q.get((self.state_previous, self.action_previous), 0.0)   #if it's second counter run in the trial then get the q-value from previous state-action  	
		self.q[(self.state,action)] = oldv + self.alpha * (reward - oldv + self.gamma * self.q.get((self.state, action), 10.0))
		#try:
		#	self.q[(self.state,action)] = oldv + self.alpha * (reward - oldv + self.gamma * self.q[(self.state,action)])
		#except Exception:
		#	self.q[(self.state,action)] = reward + 10   #if the state has not been visited before then except statement is triggered otherwise try statement is triggered. 
		#if oldv is None:
		#	self.q[(self.state,action)] = reward
		#else:
			#self.q[(self.state,action)] = oldv + self.alpha * (reward - oldv)# + self.gamma * self.q[(self.state, action)] - oldv)
			#self.q[(self.state,action)] = oldv + self.alpha * (reward - oldv + self.gamma)# * self.q[(self.state, action)] - oldv)
	

	uQ = self.q
	self.state_previous = self.state
        self.action_previous = action
        #self.steps_counter += 1
	#self.trial_num += 1

	
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, uQ = {}, location = {}, destination = {}, tno = {}".format(deadline, inputs, action, reward, uQ, location,destination, 1./self.tno)  # [debug]
	



def run():	
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
