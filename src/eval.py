import copy
import math
import torch
from tqdm import tqdm

from src.envs.base_env import BaseEnv, Player, Outcome
from src.envs.connect4_env import Connect4Env


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Agent:
    def sample_actions(self, env: BaseEnv, side: Player) -> int:
        pass


class UniformAgent(Agent):
    def sample_actions(self, env: BaseEnv, side: Player) -> int:
        mask = torch.from_numpy(env.get_masks()).float().to(DEVICE)
        action = mask.multinomial(1).item()
        return action


class HumanAgent(Agent):
    def sample_actions(self, env: BaseEnv, side: Player) -> int:
        available_actions = env.get_masks().nonzero()[0].tolist()
        action = int(input("Enter action: "))
        while action not in available_actions:
            action = int(input(f"Invalid action: {action}. Enter action: "))
        return action


def play_game(
    env: BaseEnv, agent_1: Agent, agent_2: Agent, verbose: bool = False
) -> Outcome:
    if verbose:
        env.render()

    while not env.done:
        copy_env = copy.deepcopy(env)
        if env.get_curr_player() == Player.ONE:
            action = agent_1.sample_actions(copy_env, Player.ONE)
        else:
            action = agent_2.sample_actions(copy_env, Player.TWO)
        env.step(action)
        if verbose:
            env.render()

    outcome = env.outcome
    if verbose:
        print(f"Outcome: {outcome}")

    env.reset()

    return outcome


def get_matchup_stats(env: BaseEnv, agent_1: torch.nn.Module, agent_2: Agent) -> dict:
    test_res = {
        "WINS": 0,
        "DRAWS": 0,
        "LOSSES": 0,
    }
    num_games = 100

    for side in [Player.ONE, Player.TWO]:
        for _ in tqdm(range(num_games), leave=False):
            if side == Player.ONE:
                outcome = play_game(env, agent_1, agent_2)
            else:
                outcome = play_game(env, agent_2, agent_1)

            if outcome.value == side:
                test_res["WINS"] += 1
            elif outcome == Outcome.DRAW:
                test_res["DRAWS"] += 1
            else:
                test_res["LOSSES"] += 1

    ratios = {f"{name} ratio": val / (num_games * 2) for name, val in test_res.items()}
    return ratios


def ttt_optimal_percent(agent: Agent):
    def eval_player(player):
        num_optimal = 0

        states, masks, optimal_actions = torch.load(
            f"data/ttt_optimal_moves_{player}.pt"
        )
        states, masks = states.to(DEVICE), masks.to(DEVICE).float()
        with torch.no_grad():
            logits = agent(states.to(DEVICE), player)[1]
            logits = logits * masks + (1 - masks) * (-1e9)
            selected_actions = logits.argmax(dim=1)
            for i, selected_action in enumerate(selected_actions):
                if selected_action in optimal_actions[i]:
                    num_optimal += 1

        return num_optimal, states.shape[0]

    num_optimal, total = 0, 0
    for player in [0, 1]:
        res = eval_player(player)
        num_optimal += res[0]
        total += res[1]

    return num_optimal / total


def connect4_optimal_percent(agent: Agent):
    BATCH_SIZE = 1024

    res = {"Optimal": 0, "Inaccuracy": 0, "Blunder": 0, "Losing move": 0}

    vals = torch.load("data/state_vals.pt")
    for player in [0, 1]:
        curr_vals = vals[player]
        for batch in range(math.ceil(len(curr_vals) / BATCH_SIZE)):
            start, end = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
            state_batch = torch.stack(
                [torch.from_numpy(x[1]).cuda() for x in curr_vals[start:end]], dim=0
            )
            vals_batch = torch.stack([x[2].cuda() for x in curr_vals[start:end]], dim=0)

            optimal_val = vals_batch.max(dim=1).values

            # TODO CLEAN THIS UP
            with torch.no_grad():
                _, policy = agent(state_batch, player)

            selected_move = policy.argmax(dim=1)
            selected_val = vals_batch.gather(1, selected_move.unsqueeze(1)).squeeze(1)

            is_same_sign = selected_val.sign() == optimal_val.sign()

            # Optimal
            is_optimal = selected_val == optimal_val
            res["Optimal"] += (selected_val == optimal_val).sum()

            # Inaccuracy
            is_small_mistake = (selected_val - optimal_val).abs() <= 1
            is_inaccuracy = (is_same_sign & is_small_mistake) & (~is_optimal)
            res["Inaccuracy"] += is_inaccuracy.sum()

            # Blunder
            is_big_mistake = (selected_val - optimal_val).abs() > 1
            is_blunder = is_same_sign & is_big_mistake
            res["Blunder"] += is_blunder.sum()

            # Losing move
            res["Losing move"] += (~is_same_sign).sum()

    for k, v in res.items():
        res[k] = v / (len(vals[0]) + len(vals[1]))
        res[k] = res[k].item()

    return res["Optimal"]


def test_agent(env: BaseEnv, agent: Agent, opp_agent: Agent):
    agent.eval()
    res = get_matchup_stats(env, agent, opp_agent)

    if env.name == "Connect4":
        optimal_moves = connect4_optimal_percent(agent)
    elif env.name == "TicTacToe":
        optimal_moves = ttt_optimal_percent(agent)
    res["% Optimal Moves"] = optimal_moves

    agent.train()
    return res
