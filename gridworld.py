
import numpy as np

int_to_char = {
    0: "u",
    1: "r",
    2: "d",
    3: "l",
}

policy_one_step_look_ahead = {
    0: [-1, 0],
    1: [0, 1],
    2: [1, 0],
    3: [0, -1],
}


def policy_int_to_char(pi: np.ndarray, n: int) -> np.ndarray:
    """
    Converts an integer policy matrix into a character policy matrix.
    Inputs:
        pi : np.ndarray (n,n) -> integer actions in {0,1,2,3}
        n  : int              -> grid size
    Output:
        np.ndarray (n,n) with characters in {'u','r','d','l'} and empty chars on terminal states
    """
    pi_char = [""]

    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                continue
            pi_char.append(int_to_char[pi[i, j]])

    pi_char.append("")

    return np.asarray(pi_char).reshape(n, n)


def policy_evaluation(
    n: int,
    pi: np.ndarray,
    v: np.ndarray,
    Gamma: float,
    threshhold: float,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Evaluates the value function v for a given policy pi on an n x n GridWorld.
    Reward is -1 for each non-terminal move and 0 in terminal states.
    Uses iterative evaluation with a stopping criterion on delta or max_iter.
    Inputs:
        n          : int        -> grid size
        pi         : np.ndarray -> policy (n,n) with actions in {0,1,2,3}
        v          : np.ndarray -> initial value function (n,n)
        Gamma      : float      -> discount factor
        threshhold : float      -> stopping criterion on max change
        max_iter   : int        -> maximum number of iterations
    Output:
        np.ndarray (n,n) -> value function following policy pi
    """
    for _ in range(max_iter):
        delta = 0.0
        v_new = v.copy()

        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                    v_new[i, j] = 0.0
                    continue

                a = pi[i, j]
                di, dj = policy_one_step_look_ahead[a]
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j

                r = -1.0
                val = r + Gamma * v[ni, nj]
                delta = max(delta, abs(val - v[i, j]))
                v_new[i, j] = val

        v = v_new
        if delta <= threshhold:
            break

    return v


def policy_improvement(
    n: int, pi: np.ndarray, v: np.ndarray, Gamma: float
) -> tuple[np.ndarray, bool]:
    """
    Performs greedy policy improvement given a value function v.
    For each state, chooses the action that maximizes r + Gamma * v(next_state).
    Inputs:
        n     : int        -> grid size
        pi    : np.ndarray -> current policy (n,n)
        v     : np.ndarray -> current value function (n,n)
        Gamma : float      -> discount factor
    Outputs:
        new_pi    : np.ndarray -> improved policy
        pi_stable : bool       -> True if policy did not change, else False
    """
    new_pi = pi.copy()
    pi_stable = True

    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                continue

            old_a = pi[i, j]
            best_a = old_a
            best_val = -1e9

            for a in (0, 1, 2, 3):
                di, dj = policy_one_step_look_ahead[a]
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j

                r = -1.0
                val = r + Gamma * v[ni, nj]

                if val > best_val:
                    best_val = val
                    best_a = a

            new_pi[i, j] = best_a
            if best_a != old_a:
                pi_stable = False

    return new_pi, pi_stable


def policy_initialization(n: int) -> np.ndarray:
    """
    Initializes a random deterministic policy for an n x n grid.
    Each non-terminal state gets a random action in {0,1,2,3}.
    Inputs:
        n : int -> grid size
    Output:
        np.ndarray (n,n) -> initial policy
    """
    pi = np.random.randint(0, 4, size=(n, n))
    return pi


def policy_iteration(
    n: int, Gamma: float, threshhold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs policy iteration on an n x n GridWorld.
    Alternates between policy evaluation and policy improvement until convergence.
    Inputs:
        n          : int   -> grid size
        Gamma      : float -> discount factor
        threshhold : float -> stopping criterion for policy evaluation
    Outputs:
        pi : np.ndarray (n,n) -> optimal policy
        v  : np.ndarray (n,n) -> corresponding optimal value function
    """
    pi = policy_initialization(n=n)
    v = np.zeros(shape=(n, n))

    while True:
        v = policy_evaluation(n=n, v=v, pi=pi, threshhold=threshhold, Gamma=Gamma)
        pi, pi_stable = policy_improvement(n=n, pi=pi, v=v, Gamma=Gamma)
        if pi_stable:
            break

    return pi, v




n = 4

Gamma = [0.8,0.9,1]

threshhold = 1e-4

for _gamma in Gamma:

    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)
