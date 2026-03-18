"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque
import random
import numpy as np

TRACE = False
TRACE_BOARDS = False


def set_trace(enabled=True, show_boards=False):
    """Enable or disable tracing output for search algorithms.

    Args:
        enabled: Turn tracing on/off.
        show_boards: When True, also show problem-specific board displays if the
            problem provides a display_board(state) method.
    """
    global TRACE, TRACE_BOARDS
    TRACE = enabled
    TRACE_BOARDS = show_boards



def trace(*args, **kwargs):
    """Print tracing output only when TRACE is True."""
    if TRACE:
        print(*args, **kwargs)



def _fitness(problem, state):
    """Safely get a state's value when the problem defines one."""
    try:
        return problem.value(state)
    except Exception:
        return None



def _state_summary(problem, state):
    """Return a generic one-line summary for a state."""
    fitness = _fitness(problem, state)
    if fitness is None:
        return f"{state}"
    return f"{state} (Fitness = {fitness})"



def trace_state(problem, label, state):
    """Trace a state in a generic way, with optional board display."""
    if not TRACE:
        return

    print(f"{label}: {_state_summary(problem, state)}")

    if TRACE_BOARDS and hasattr(problem, 'display_board'):
        try:
            print(problem.display_board(state))
        except Exception:
            pass



def trace_neighbors(problem, neighbors, heading="Successors"):
    """Trace a list of neighboring nodes/states in a generic way."""
    if not TRACE:
        return

    print(heading + ":")
    for neighbor in neighbors:
        state = neighbor.state if hasattr(neighbor, 'state') else neighbor
        print(f"    {_state_summary(problem, state)}")
        if TRACE_BOARDS and hasattr(problem, 'display_board'):
            try:
                print(problem.display_board(state))
            except Exception:
                pass



def trace_generation(generation, fitness_values):
    """Trace GA generation statistics."""
    if not TRACE or not fitness_values:
        return

    best = max(fitness_values)
    avg = sum(fitness_values) / len(fitness_values)
    print(f"Generation {generation}: Best Fitness = {best}, Average Fitness = {avg:.2f}")
    print(f"    Population Fitness: {fitness_values}")


# ============================================================
# Minimal Utilities
# ============================================================

import random
import bisect


def shuffled(iterable):
    """Return a shuffled copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def argmax_random_tie(seq, key=lambda x: x):
    """Return element with highest key(seq[i]); break ties randomly."""
    return max(shuffled(seq), key=key)


def argmin_random_tie(seq, key=lambda x: x):
    """Return element with lowest key(seq[i]); break ties randomly."""
    return min(shuffled(seq), key=key)


def weighted_sampler(seq, weights):
    """Return a function that samples seq according to weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    def sample():
        return seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

    return sample


def is_in(elt, seq):
    """Like (elt in seq) but uses 'is' comparison."""
    return any(x is elt for x in seq)


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        raise NotImplementedError


class Node:
    """A node in a search tree."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ______________________________________________________________________________
# Local search algorithms


def hill_climbing(problem):
    """Performs the hill-climbing algorithm with optional tracing."""
    current = Node(problem.initial)

    while True:
        neighbors = current.expand(problem)
        trace_state(problem, "Current Node", current.state)

        if not neighbors:
            trace("No more neighbors to explore. Terminating.")
            break

        trace_neighbors(problem, neighbors, heading="Successors")

        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))

        if problem.value(neighbor.state) <= problem.value(current.state):
            trace_state(problem, "Current Node", current.state)
            trace("No better neighbor found. Stopping.")
            break

        trace_state(problem, "Moving to better node", neighbor.state)
        current = neighbor

    trace_state(problem, "Final Solution", current.state)
    return current.state


# ______________________________________________________________________________
# Simulated Annealing


def exp_schedule(k=10, lam=0.005, limit=5000):
    """One possible schedule function for simulated annealing."""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    """Simulated Annealing Algorithm with optional tracing."""
    current = Node(problem.initial)

    for t in range(sys.maxsize):
        T = schedule(t)

        if T == 0:
            trace(f"Simulated annealing stopped at iteration {t}: temperature reached 0.")
            trace(f"Final state: {current.state} (Fitness = {problem.value(current.state)})")
            return current.state

        neighbors = current.expand(problem)
        if not neighbors:
            trace(f"Simulated annealing stopped at iteration {t}: no neighbors available.")
            trace(f"Final state: {current.state} (Fitness = {problem.value(current.state)})")
            return current.state

        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)

        trace(f"Iteration {t}: Selected Neighbor = {next_choice.state}, Fitness Change = {delta_e}, Temperature = {T:.4f}")

        if delta_e > 0:
            current = next_choice
        else:
            probability_accept = np.exp(delta_e / T)
            random_value = random.uniform(0, 1)

            if random_value < probability_accept:
                trace(f"Accepted worse state with probability {probability_accept:.4f}")
                current = next_choice
            else:
                trace("Rejected worse state.")

    trace("Simulated annealing stopped: reached maximum iterations.")
    trace(f"Final state: {current.state} (Fitness = {problem.value(current.state)})")
    return current.state



# ______________________________________________________________________________
# Genetic Algorithm

def genetic_search(problem, ngen=200, pmut=0.1, pop_size=20):
    """
    Run genetic algorithm for N-Queens or similar problems.
    """
    gene_pool = list(range(problem.N))
    population = init_population(pop_size, gene_pool, problem.N)

    max_fitness = problem.max_value()

    trace(f"Starting genetic search with population size = {pop_size}, generations = {ngen}, mutation rate = {pmut}")
    trace(f"Target fitness = {max_fitness}")

    result = genetic_algorithm(
        population=population,
        fitness_fn=problem.value,
        gene_pool=gene_pool,
        f_thres=max_fitness,
        ngen=ngen,
        pmut=pmut
    )

    trace(f"Genetic search finished. Final result = {result} (Fitness = {problem.value(result)})")
    return result

def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """Genetic Algorithm with optional tracing."""
    ngen = max(1, int(ngen))

    for gen in range(ngen):
        fitness_values = [fitness_fn(ind) for ind in population]

        best = max(fitness_values)
        avg = sum(fitness_values) / len(fitness_values)

        trace(f"Generation {gen + 1}: Best Fitness = {best}, Average Fitness = {avg:.2f}")
        trace(f"Population Fitness: {fitness_values}")

        # Optional: remove incomplete states if your problem uses them
        population = [ind for ind in population if -1 not in ind]

        if not population:
            trace(f"Generation {gen + 1}: No valid individuals remain. Restarting population.")
            population = init_population(len(gene_pool), gene_pool, len(gene_pool))

        # Best current individual
        fittest = max(population, key=fitness_fn)

        # Stop if threshold reached
        if f_thres is not None and fitness_fn(fittest) >= f_thres:
            trace(f"Genetic algorithm stopped at generation {gen + 1}: fitness threshold reached.")
            trace(f"Best individual: {fittest} (Fitness = {fitness_fn(fittest)})")
            return fittest

        # Create next generation
        population = [
            mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
            for _ in range(len(population))
        ]

    # If no threshold reached, return the best found after ngen generations
    best_individual = max(population, key=fitness_fn)
    trace(f"Genetic algorithm stopped after {ngen} generations: generation limit reached.")
    trace(f"Best individual found: {best_individual} (Fitness = {fitness_fn(best_individual)})")
    return best_individual


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None



def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm."""
    g = len(gene_pool)
    population = []
    for _ in range(pop_number):
        new_individual = tuple(gene_pool[random.randrange(0, g)] for _ in range(state_length))
        population.append(new_individual)
    return population



def select(r, population, fitness_fn):
    """Select r individuals using weighted sampling, with fallback if needed."""
    fitnesses = list(map(fitness_fn, population))

    if not population:
        raise ValueError("Population cannot be empty in select().")

    if len(set(fitnesses)) == 1:
        return random.sample(population, r)

    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for _ in range(r)]



def recombine(x, y):
    """Crossover function to combine two parent solutions."""
    n = len(x)
    c = random.randrange(0, n)
    child = tuple(x[:c]) + tuple(y[c:])
    trace(f"Crossover: Parent1={x}, Parent2={y}, Crossover Point={c}, Child={child}")
    return child



def mutate(x, gene_pool, pmut):
    """Mutates a random gene in the individual if mutation occurs."""
    if isinstance(gene_pool, int):
        gene_pool = list(range(gene_pool))

    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    c = random.randrange(n)
    new_gene = random.choice(gene_pool)

    while new_gene == x[c]:
        new_gene = random.choice(gene_pool)

    mutated = list(x)
    mutated[c] = new_gene
    mutated_tuple = tuple(mutated)

    trace(f"Mutation: Before={x}, Mutated Position={c}, New Value={new_gene}, After={mutated_tuple}")
    return mutated_tuple
