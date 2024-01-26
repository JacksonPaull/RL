import sys
import numpy as np
from datetime import datetime
from multiprocessing import Process, Manager
import logging

# ======================== Logging Setup ==================

logger = logging.getLogger('HW1')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO) 

# ======================== Globals ========================

NOISE_STD = 0.01
N_BANDIT_RUNS = 300
N_STEPS_FOR_EACH_BANDIT = 10000

# ======================== Code ===========================

class NonStationaryBandit:
    def __init__(self, k=10, bandit_seed=0):
        self.k = k
        self.q_star = np.zeros(k)

        logger.debug('NSB Init Complete')
    
    def reset(self, episode_seed=None):
        if episode_seed is None:
            episode_seed = np.random.randint(0, 100000000)
            logger.info(f'set seed of new bandit to {episode_seed}')

        self.episode_rg = np.random.RandomState(seed=episode_seed)
        self.q_star = np.zeros(self.k)

        
    def is_best(self, a):
        return int(self.q_star.max() == self.q_star[a]) # Account for doubles with argmax

    def step(self):
        self.q_star += self.episode_rg.normal(0, NOISE_STD, self.k)


class ActionValue:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)

    def reset(self):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def update(self, a, r):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def epsilon_greedy_policy(self):
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.k) # Choose a random action from q
            return a
        else:
            a = self.q.argmax() # greedy
            return a


class SampleAverage(ActionValue):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        self.n = np.zeros(self.k)

    def reset(self):
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)

    def update(self, a, r):
        self.n[a] = self.n[a] + 1
        self.q[a] = self.q[a] + 1/self.n[a] * (r - self.q[a])


class ConstantStepSize(ActionValue):
    def __init__(self, alpha, k, epsilon):
        super().__init__(k, epsilon)
        self.alpha = alpha
        
    def reset(self):
        self.q = np.zeros(self.k)

    def update(self, a, r):
        self.q[a] = self.q[a] + self.alpha * (r - self.q[a])


def experiment(bandit, algorithm, steps, episode_seed=None, name='NO_NAME', return_list=None):
    logger.info(f'\tRunning experiment {name}, seed={episode_seed}')
    start = datetime.now()
    bandit.reset(episode_seed)
    algorithm.reset()

    rs = []
    best_action_taken = []

    for _ in range(steps):
        a = algorithm.epsilon_greedy_policy()               # 1. Choose action
        best_action_taken.append(bandit.is_best(a))         # 1.1 is it the  best action?

        r = bandit.episode_rg.normal(bandit.q_star[a], 1)           # 2. Get noisy reward
        rs.append(r)

        algorithm.update(a, r)                              # 3. Update agent
        bandit.step()                                       # 4. Step bandit problem

    if return_list is not None:
        return_list.put((np.array(rs), np.array(best_action_taken)))

    logger.info(f'\tFinished experiment {name} in {datetime.now() - start}s')

if __name__ == "__main__":
    _start = datetime.now()

    procs = []
    exp_results = {}
    for algo_name in ['sample_average', 'constant']:
        man = Manager()
        exp_results[algo_name] = man.Queue()
        for n in range(N_BANDIT_RUNS):

            if algo_name == 'sample_average':
                algo = SampleAverage(k=10, epsilon=0.1)
            else:
                algo = ConstantStepSize(k=10, epsilon=0.1, alpha=0.1)

            p = Process(target=experiment, args = (NonStationaryBandit(), algo, N_STEPS_FOR_EACH_BANDIT, np.random.randint(1000000), f'{algo_name}_{n}', exp_results[algo_name]))
            procs.append(p)
            p.start()

    # Wait for all the processes to be done         
    for i, p in enumerate(procs):
        logger.debug(f'joining proc {i}...')
        p.join()
        logger.debug('done!')

    
    logger.info('All experiments complete!')

    # Collect all the results
    for alg, q in exp_results.items():
        res = []
        while not q.empty():
            r = q.get()
            res.append(r)
        exp_results[alg] = np.asarray(res)
    logger.debug('All results collected')

    outputs = np.row_stack([exp_results['sample_average'].mean(axis=0),
                           exp_results['constant'].mean(axis=0)])
    np.savetxt(sys.argv[1], outputs)

    logger.info(f'Finished entire program in {datetime.now() - _start}')