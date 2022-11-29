#!/usr/bin/env python

import sys
import time
import os
from os.path import dirname, join, realpath
import itertools
import math
import numpy as np
# ndarray for type hints
from numpy import ndarray
from numpy import nan
import yaml
import random
import snoop
import asyncio
import requests
from requests.exceptions import Timeout, ConnectionError

# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(realpath(".")))
from common.utility import get_latency
from common.config_logging import init_logging
logger = init_logging(join(dirname(realpath(__file__)), "client.log"))

# read config-default.yml, config.yml(optional, override the default configurations) using yaml
default_config_file = "config-default.yml"
config_file = "config.yml"
logger.info(f"Loading default config file: {config_file}")
with open(join(dirname(realpath(__file__)), default_config_file), mode="r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
if os.path.exists(join(dirname(realpath(__file__)), config_file)):
    logger.info(f"Loading override config file: {config_file}")
    with open(join(dirname(realpath(__file__)), config_file), mode="r", encoding="utf-8") as file:
        # use **kwargs to merge the two dictionaries
        config = {**config, **yaml.safe_load(file)}
logger.info(f"load config: {config}")

storage_cost : list[float] = config['storage_cost']
outbound_cost : list[float] = config['outbound_cost']
read_cost : list[float] = config['read_cost']
write_cost : list[float] = config['write_cost']
cloud_providers : list[str] = config['cloud_providers']

# 
# Use the following variable names to store the data
# N: the number of clouds
# n: write_pieces
# k: read_pieces, n > k
# w0: default_window_size
# δ: delta
# b: threshold
# λ: lambda
# St: placement_policy_timed is (T,N) matrix
# τi: changed_ticks is (N,) vector
# wi: window_sizes is (N,) matrix
# ξ: xi
# ψ: psi
# Wit: windows_sizes_timed is (T,N) matrix
# lit: windows_sizes_timed is (T,N) matrix
# T: ticks, the number of ticks in the simulation
#
class AW_CUCB:
    def __init__(self, default_window_size=5, ticks=100, N=6, n=3, k=2, read=True, ψ1=0.5, ψ2=0.5, data_size=1024, ξ=0.5, b=0.5, δ=0.5, optimize_initial_exploration=True, LB=None):
        self.default_window_size = default_window_size
        self.N = N
        self.ticks=ticks
        self.n = n
        self.k = k
        self.read = read
        self.ψ1 = ψ1
        self.ψ2 = ψ2
        self.data_size = data_size
        self.ξ = ξ
        self.b = b
        self.δ = δ
        self.optimize_initial_exploration = optimize_initial_exploration
        self.LB = LB
        self.placement_count = self.k if self.read else self.n
        
        
    def processing(self):
        # initialization
        τ = np.full((self.N,),1)
        window_sizes = np.full((self.N,),self.default_window_size)
        placement_policy_timed = np.zeros((self.ticks, self.N))
        windows_sizes_timed = np.zeros((self.ticks, self.N))
        latency_cloud_timed = np.zeros((self.ticks, self.N))
        U = np.zeros((self.ticks, self.N))
        L = np.zeros((self.ticks, self.N))
        initial_optimized_placement_policy = list(itertools.combinations(range(self.N), self.placement_count))
        for tick in range(self.ticks):
            # optimize the initial exploration logic
            # change the placement policy choice which include the current tick indexed provider
            # and (placement_count - 1) providers from other randomly
            # to the combinations(N, placement_count) providers
            # Use the itertools.combinations to generate the combinations
            # initial_placement_policy is a list of tuple which contains the placement policy
            if self.optimize_initial_exploration and tick < len(initial_optimized_placement_policy):
                placement_policy = initial_optimized_placement_policy[tick]
                placement_policy_timed[tick] = [1 if i in placement_policy else 0 for i in range(self.N)]
                # make a request to the cloud and save the latency to the latency_cloud_timed
                # if the passed cloud_placements is like [0,0,1,0,1,0], then the returned latency is like [0,0,35.12,0,28.75,0]
                logger.info(f"placement_policy_timed[{tick}]: {placement_policy_timed[tick]}")
                latency_cloud = asyncio.run(get_latency(placement_policy_timed[tick], tick, self.N, self.k, cloud_providers, self.data_size, self.read))
                logger.info(f"tick: {tick}, latency_cloud: {latency_cloud}")
                latency_cloud_timed[tick] = latency_cloud
            elif tick < self.N:
                # randomly choose a super arm, the tick position is chosed
                placement_policy = list(range(self.N))
                placement_policy.pop(tick)
                placement_policy = random.choices(placement_policy, k=self.placement_count - 1) + [tick]
                placement_policy_timed[tick] = [1 if i in placement_policy else 0 for i in range(self.N)]
                # make a request to the cloud and save the latency to the latency_cloud_timed
                # if the passed cloud_placements is like [0,0,1,0,1,0], then the returned latency is like [0,0,35.12,0,28.75,0]
                latency_cloud = asyncio.run(get_latency(placement_policy_timed[tick], tick, self.N, self.k, cloud_providers, self.data_size, self.read))
                logger.info(f"tick: {tick}, latency_cloud: {latency_cloud}")
                latency_cloud_timed[tick] = latency_cloud
            else:
                # update statistics in time-window Wit
                # calculate the count of selects in each cloud in placement_policy_timed in 0~tick(exclude)
                Tiwi = np.zeros((self.N,))
                liwi = np.zeros((self.N,))
                eit = np.zeros((self.N,))
                u_hat_it = np.zeros((self.N,))
                for clould_id in range(self.N):
                    Tiwi[clould_id] = np.sum(placement_policy_timed[:tick,clould_id])
                    latency_of_cloud_previous_ticks = latency_cloud_timed[:tick,clould_id]
                    liwi[clould_id] = 1 / Tiwi[clould_id] * np.sum(latency_of_cloud_previous_ticks)
                    if np.size(np.where(latency_of_cloud_previous_ticks != 0)) > 0:
                        LB = latency_of_cloud_previous_ticks.max() - np.delete(latency_of_cloud_previous_ticks, np.where(latency_of_cloud_previous_ticks == 0)).min() if self.LB == None else self.LB
                        eit[clould_id] = LB * math.sqrt(self.ξ * math.log(window_sizes[clould_id], 10) / Tiwi[clould_id])
                    else:
                        # should not go here!
                        logger.error(f"program logic error, check the code!")
                        sys.exit(-1)
                # estimate the utility bound of each cloud
                u_hat_it[:] = self.ψ1 * liwi + self.ψ2 * np.array(storage_cost) - eit
                # select the top n arms to added into the placement policy in ascending order
                placement_policy = np.argsort(u_hat_it)[:self.placement_count]
                placement_policy_timed[tick] = [1 if i in placement_policy else 0 for i in range(self.N)]
                latency_cloud_timed[tick] = asyncio.run(get_latency(placement_policy_timed[tick], tick))
                # play super arm St and observe the latency
                changed, changed_ticks = self.FM_PHT(U,L,tick,latency_cloud_timed)
                if any(changed):
                    τ = changed_ticks
                    # TODO: reset FM-PHT
                    if self.read:
                        # LDM(St', St), ST: current placement_policy, ST': the previous placement_policy
                        self.LDH(latency_cloud_timed[tick], placement_policy_timed[tick-1])
                    # TODO: verify correct? contains 0? ERROR!
                    # https://www.geeksforgeeks.org/numpy-minimum-in-python/
                window_sizes = np.minimum(window_sizes, tick - τ + 1)
                print(f"tick: {tick}, u_hat_it: {u_hat_it}, window_sizes: {window_sizes}")
                
    def LDM(self, current_placement_policy, previous_placement_policy):
        # convert the placement_policy to the selected cloud providers
        current_placement_policy_indices = set(np.where(current_placement_policy == 1)[0].tolist())
        previous_placement_policy_indices = set(np.where(previous_placement_policy == 1)[0].tolist())
        prepare_migrate_cloud_ids = previous_placement_policy_indices - current_placement_policy_indices
        distination_migrate_cloud_ids = current_placement_policy_indices - previous_placement_policy_indices
        logger.info(f"current_placement_policy: {current_placement_policy}, current_placement_policy_indices: {current_placement_policy_indices}, previous_placement_policy: {previous_placement_policy}, previous_placement_policy_indices: {previous_placement_policy_indices}, prepare_migrate_cloud_ids: {prepare_migrate_cloud_ids}, distination_migrate_cloud_ids: {distination_migrate_cloud_ids}")
        # initial migration gains to 0
        migration_gains = 0
        if len(prepare_migrate_cloud_ids) > 0:
            # calculate migration gains
            migration_gains = sum(map(lambda i: self.data_size / self.k * (storage_cost[i] - outbound_cost[i]) - read_cost[i], prepare_migrate_cloud_ids)) - sum(map(lambda i: self.data_size / self.k * storage_cost[i] + write_cost[i], distination_migrate_cloud_ids))
        if migration_gains > 0:
            # migrate the data from prepare_migrate_cloud_ids to distination_migrate_cloud_ids
            # TODO: process the migration
            pass
            
    def FM_PHT(self, U, L, tick, latency_cloud_timed):
        # initialzation
        # y is the exist latency of one of the cloud
        U_min = U.min(axis=0)
        L_max = L.max(axis=0)
        for cloud_id in range(self.N):
            latency_cloud = latency_cloud_timed[:,cloud_id]
            latency_cloud_exist = np.delete(latency_cloud, np.where(latency_cloud == 0))
            U[tick][cloud_id] = (tick - 1) / tick * U[tick - 1][cloud_id] + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) - self.δ)
            # U_changed is array of bools, like [False, False, True, True, False, False]
            U_changed = U[tick, :] - U_min >= self.b
            # U_changed_ticks is array of ticks, like [0, 0, 5, 6, 0, 0]
            U_changed_ticks = np.array([U_min[index] if changed else 0 for index, changed in enumerate(U_changed)])
            U_changed_ticks = np.where(U_changed, tick, 0)
            L[tick][cloud_id] = (tick - 1) / tick * L[tick - 1][cloud_id] + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) + self.δ)
            L_changed = L_max -L[tick, :] >= self.b
            L_changed_ticks = np.array([L_max[index] if changed else 0 for index, changed in enumerate(L_changed)])
            changed = U_changed + L_changed
            changed_ticks = U_changed_ticks + L_changed_ticks
            return changed, changed_ticks
        
    def print_output(self):
        pass
    
if __name__ == "__main__":
    # for read operation
    # algorithm = AW_CUCB()
    # # for write operation
    # # algorithm = AW_CUCB(read=False)
    # algorithm.processing()
    ...
