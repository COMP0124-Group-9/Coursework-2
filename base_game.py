from pettingzoo.atari import warlords_v3
#import matplotlib.pyplot as plt
import agent
import typing

class Game:

    def __init__(self,agent_list:typing.List[agent.Agent]):

        self.env = warlords_v3.env(render_mode="human")
        self.env.reset(seed=42)
        self.agent_list = agent_list
        self.number_rounds = 0

    def reset(self,seed):
        self.env.reset(seed=seed)

    def getPlayerArea(observation,agent):
        # crop to game player region, reflect so player is top left
        raise NotImplementedError()

    def parseObservation(self,observation):
        player_area = getPlayerArea(observation,agent)
        return observation
    
    def findBall(self, player_area):
        pass


    def run(self):

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            parsed_observation = self.parseObservation(observation)

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = self.env.action_space(agent).sample()

            self.env.step(action)
    
    def exit(self):
        self.env.close()

if __name__ == "__main__":
    pass