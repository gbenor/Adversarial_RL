import click
import gym
import networkx as nx

from agents.graph_agent import GraphAgent
from gym_adversarial.envs.AdversarialCallbacks import CustomHistoryCallback

G = nx.DiGraph()
G.add_node("ToClosetTarget", action="CLOSET_CLUSTER", stop_on_target=True)
G.add_node("ReachBoundaryToClosetTarget", action="DECREASE_STEP", stop_on_target=None)
G.add_node("FromClosetTarget", action="ORIGINAL_IMAGE", stop_on_target=False)
G.add_node("ReachBoundaryFromClosetTarget", action=None, stop_on_target=None)
G.add_node("ToFarthestTarget", action="FARTHEST_CLUSTER", stop_on_target=True)
G.add_node("ReachBoundaryToFarthestTarget", action="DECREASE_STEP", stop_on_target=None)
G.add_node("FromFarthestTarget", action="ORIGINAL_IMAGE", stop_on_target=False)
G.add_node("ReachBoundaryFromFarthestTarget", action=None, stop_on_target=None)

elist=[("ToClosetTarget", "ReachBoundaryToClosetTarget"),
       ("ReachBoundaryToClosetTarget", "FromClosetTarget"),
       ("FromClosetTarget", "ReachBoundaryFromClosetTarget"),
       ("ReachBoundaryFromClosetTarget", "ToFarthestTarget"),
       ("ToFarthestTarget", "ReachBoundaryToFarthestTarget"),
       ("ReachBoundaryToFarthestTarget", "FromFarthestTarget"),
       ("FromFarthestTarget", "ReachBoundaryFromFarthestTarget"),
       ("ReachBoundaryFromFarthestTarget", "ToClosetTarget")]
G.add_edges_from(elist)


def test_policy_for_target(nb_episodes: int, target: int, test_description="simple_centers"):
    testing_env = gym.make('adver-v0', target_label=target, test_mode=True,
                           result_directory="results",
                           test_description=test_description
                           )
    simple_agent = GraphAgent(target_class=target, policy_graph=G, init_state="ToClosetTarget")
    simple_agent.test(testing_env, nb_episodes=nb_episodes, visualize=True, callbacks=[CustomHistoryCallback()])


@click.command()
@click.argument('nb_episodes', type=int)
def test_policy(nb_episodes: int):
    for target in range(10):
        test_policy_for_target(nb_episodes, target)


@click.group()
def cli():
    pass


cli.add_command(test_policy)


if __name__ == '__main__':
    cli()
