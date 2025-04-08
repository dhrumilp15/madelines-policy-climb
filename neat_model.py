import neat
from neat_ai_client import NEATAIClient
import os

class NeatModel:

    def __init__(self, config_file = "neat_config.txt", checkpoint_file = None):
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )
        self.checkpoint_file = checkpoint_file
        self.output_folder = "output/"
        os.makedirs(self.output_folder, exist_ok=True)
        self.max_seconds_per_level = 20
        self.client = NEATAIClient()

    def score_function(self, session_data):
        num_completed_levels = session_data["num_completed_levels"]
        # total_time = self.max_seconds_per_level * max(1, num_completed_levels)
        # need to get the units in the right spots so that each are equally weighed
        # time_penalty = session_data["elapsed_seconds"] / total_time
        level_completion = session_data["distance_to_goal"] / session_data['level_diagonal_length']
        # assume that if the player died staying still, they would have used up the entire time
        staying_still_penalty = session_data["staying_still_penalty"]
        fitness = num_completed_levels - level_completion
        if staying_still_penalty:
            fitness -= 0.25
        print(*{
            # "time_penalty": time_penalty,
            "level completion": level_completion,
            "completed levels": num_completed_levels,
            "staying still penalty": staying_still_penalty,
            "fitness": fitness
        }.items(), sep='\n')
        return fitness

    def genome_forward(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            session_data = self.client.process_frame_and_send_move(net.activate)
            genome.fitness = self.score_function(session_data)
            print(f"fitness: {genome.fitness}")

    def run(self):
        if self.checkpoint_file:
            p = neat.Checkpointer.restore_checkpoint(self.checkpoint_file)
        else:
            p = neat.Population(self.config)

        p.add_reporter(neat.Checkpointer(1,10000, self.output_folder + "neat-checkpoint-"))
        p.add_reporter(neat.StdOutReporter(True))
        
        winner = p.run(self.genome_forward, 500)
        print(f"Best genome: {winner}")

if __name__ == "__main__":
    while True:
        try:
            model = NeatModel()
            model.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            pass


# cleared level in generation 17 (load gen #16), implement a more correct scoring func after clearing a level
# AI suggests rewarding the player with the remaining time
# This is because the player is incentivized to clear the level as fast as possible