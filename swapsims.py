#Authors: Sam Taggart, Chris Ikeokwu, Adam Eck
#Main config TODOS at bottom of file.
#sfrom seeds import seed_bank

#swap rosca is using seedbank 5

import math
import time

import random
from sys import argv

import numpy as np
import pandas as pd
import gurobi as gp
from gurobipy import GRB


def opt_rosca(n, agent_vals):
    try:
        model = gp.Model("opt_rosca")
        #create decsion variables
        decision_vars = dict()
        for agent in range(n):
            decision_vars[agent] = dict()
            for round in range(n):
                decision_vars[agent][round] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{agent}_{round}")

        #creating util function
        agent_utils = []
        for agent in range(n):
            util = 0
            for round in range(n):
                util += decision_vars[agent][round] * agent_vals[agent][round]
            agent_utils.append(util)

        #objective
        model.setObjective(sum(agent_utils), GRB.MAXIMIZE)

        #agent constraints
        for agent in range(n):
            assigment_val = 0
            for round in range(n):
                assigment_val += decision_vars[agent][round]
            model.addConstr(assigment_val == 1, f"ac_{agent}")

        #round constraints
        for round in range(n):
            assigment_val = 0
            for agent in range(n):
                assigment_val += decision_vars[agent][round]
            model.addConstr(assigment_val == 1, f"rc_{round}")

            # Optimize model
        model.optimize()

        # for v in model.getVars():
        #     print('%s %g' % (v.varName, v.x))
        #
        # print('Obj: %g' % model.objVal)
        return model.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')


#computes the cost for payments p.
#if crra, then costtype will be a touple containing ("crra",W,a)
def cost(p,costtype):
    if type(costtype) == tuple:
        if costtype[0] == "crra":
            W = costtype[1]
            a = costtype[2]
            if a > 0 and p >= W: return float("inf")
            if a == 1:
                return math.log(W) - math.log(W-p)
            else: return (W**(1-a)-(W-p)**(1-a))/(1-a)




#finds+returns the smallest payment that will incentivise a for the current agent
#returns None if none exists
#params
#   current_p: the agent's current payments
#   vold: the agent's value for their current slot
#   vnew: their value for the new slot
#   cost type info.
#if crra, then costtype will be a touple containing ("crra",W,a)
# Will break if somehow we require payments of more than 100000
def findminpay(current_p,vold,vnew,costtype):
    max_val = 100000
    accuracy = .0001
    if type(costtype) == tuple:
        if costtype[0] == "crra":
            W = costtype[1]
            a = costtype[2]

                
            #is it impossible to find enough money?
            if (cost(current_p,costtype)-cost(current_p-max_val,costtype)) < vold-vnew: return None

            #binary search
            pmax = max_val
            pmin = 0
            pmid = (pmin+pmax)/2
            oldmid = max_val
            while abs(pmid-oldmid) > accuracy:
                if (cost(current_p,costtype)-cost(current_p-pmid,costtype)) < vold-vnew:
                    pmin = pmid
                    oldmid = pmid
                    pmid = (pmin+pmax)/2
                else:
                    pmax = pmid
                    oldmid = pmid
                    pmid = (pmin+pmax)/2
            return pmid




#computes whether or not two agents should swap.
#if so, returns the minimum payment (from 1 to 2) supporting such a swap.
#positive payments go from 1 to 2, negative the other way
#if not, returns None
#params
#   p1, p2: current payments of players 1 and 2
#   v11,v22: 1 and 2's value for their own slots
#   v12: 1's value for 2's slot
#   v21: 2's value for 1's slot
#if crra, then costtype will be a touple containing ("crra",W,a)
def validswap(p1,p2,v11,v22,v12,v21,costtype):
    #mutually beneficial; no payments needed
    #this won't actually happen in roscas...
    if v12 > v11 and v21 > v22:
        return 0

    #1 wants to trade, 2 wants cash
    elif v12 > v11:
        minpay = findminpay(p2,v22,v21,costtype)
        if minpay == None: return None
        elif v12 - cost(p1+minpay+.0001,costtype) > v11 - cost(p1,costtype): return minpay+.0001
        else: return None

    #2 wants to trade, 1 wants cash
    elif v21 > v22:
        minpay = findminpay(p1,v11,v12,costtype)
        if minpay == None: return None
        elif v21 - cost(p2+minpay+.0001,costtype) > v22 - cost(p2,costtype): return (-1)*(minpay+.0001)
        else: return None



#goes through the list of agents, grabs pairs, and makes them swap if it's beneficial.
#format for agents = [[value_vector,currentpayments,current_slot],[etc]]
#test case agents = [[[10,5,0],0,1],[[5,4,3],0,0],[[7,6,2],0,2]]
#if crra, then costtype will be a touple containing ("crra",W,a)
def findtrade(agents,costtype):
    swapped = False
    swaps = 0
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                swap_pay = validswap(agents[i][1],agents[j][1],agents[i][0][agents[i][2]],agents[j][0][agents[j][2]],agents[i][0][agents[j][2]],agents[j][0][agents[i][2]],costtype)
                if swap_pay != None:
                    swapped = True
                    agents[i][2],agents[j][2] = agents[j][2],agents[i][2]
                    agents[i][1] += swap_pay
                    agents[j][1] -= swap_pay
                    swaps += 1
    return swapped, swaps


#finds trades until there aren't any more.
#format for agents = [[value_vector,currentpayments,current_slot],[etc]]
#test case agents = [[[10,5,0],0,1],[[5,4,3],0,0],[[7,6,2],0,2]]
#if crra, then costtype will be a touple containing ("crra",W,a)
def runtrades(agents,costtype):
    swapped = True
    swaps = 0
    while swapped:
        swapped, s = findtrade(agents,costtype)
        swaps += s
    return swaps


#computes a full run of the swap rosca, and outputs the welfare at the end.
#format for agents = [[value_vector,currentpayments,current_slot],[etc]]
#test case agents = [[[10,5,0],0,1],[[5,4,3],0,0],[[7,6,2],0,2]]
#if crra, then costtype will be a touple containing ("crra",W,a)
def fullrun(agents,costtype):
    welfare = 0
    swaps = 0
    for t in range(len(agents)):
        swaps += runtrades(agents,costtype)

        #subtracts disutilities for payments from the welfare
        for agent in agents:
            welfare -= cost(agent[1],costtype)

        #adds the value of this round's winner
        for agent in agents:
            if agent[2] == t:
                welfare += agent[0][t]
                agents.remove(agent)

        #resets current payments to 0
        for agent in agents:
            agent[1] = 0

    return welfare, swaps


#set up a full run of n agents, with types drawn uniformly at random from value_dist
#the initial allocation is uniformly at random.
def initialize(value_dist,n):
    #fill out an array with n agents.
    order = list(range(n))
    random.shuffle(order)

    agents = []
    for i in range(n):
        agent_i = order[i]
        agent = value_dist[agent_i]
        agents.append([agent,0,i])
    return agents


def estimate(value_dist,n, numruns,costtype):
    average = 0.0
    average_swaps = 0
    average_uniform = 0.0
    for i in range(numruns):
        agents = initialize(value_dist,n)
        unif = calculate_uniform(agents)

        swapwelf, swaps = fullrun(agents,costtype)

        average = average + (swapwelf - average)/(i+1)
        average_swaps = average_swaps + (swaps - average_swaps) / (i + 1)
        average_uniform = average_uniform + (unif - average_uniform) / (i + 1)

    return average, average_swaps, average_uniform


def calculate_uniform(agents):
    welfare = 0.0

    n = len(agents)
    for i in range(n):
        welfare += agents[i][0][i]

        if agents[i][-1] != i:
            print("Oh no!", agents[i][-1], i)

    return welfare



def welfare(value_dist):
    opt_welf = 0
    opt_perm = None
    avg_welf = 0

    perm_i = 1
    for perm in permutation(value_dist):
        perm_welf = 0
        for i in range(len(perm)):
            perm_welf += perm[i][i]

        if perm_welf > opt_welf:
            opt_welf = perm_welf
            opt_perm = perm

        avg_welf = avg_welf + (perm_welf - avg_welf) / (perm_i)
        perm_i += 1
    # print(opt_perm)
    return opt_welf, avg_welf


#other peoples' code
# Python function to print permutations of a given list
def permutation(lst):
    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []
 
    # If there is only one element in lst then, only
    # one permutation is possible
    if len(lst) == 1:
        return [lst]
 
    # Find the permutations for lst if there are
    # more than 1 characters
 
    l = [] # empty list that will store current permutation
 
    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
       m = lst[i]
 
       # Extract lst[i] or m from the list.  remLst is
       # remaining list
       remLst = lst[:i] + lst[i+1:]
 
       # Generating all permutations where m is first
       # element
       for p in permutation(remLst):
           l.append([m] + p)
    return l
 



#experiments.
# CRRA
# a = [0, .1, .2, .3, .5, .75, 1, 1.5, 2]
# W = [1,2,3,4,4]
#[[2,0,0,0,0,0,0,0,0],[2,2,2,2,2,0,0,0,0,],[5,5,0,0,0,0,0,0,0],[5,5,5,5,5,5,0,0,0],[8,8,8,0,0,0,0,0,0],[8,8,8,8,8,8,8,0,0],
#[8,8,8,5,5,5,2,2,2],[8,8,6,6,4,4,2,2,0],[8,7,6,5,4,3,2,1,0]]

if __name__ == '__main__':
    seed = int(math.e * 1000000)
    random.seed(seed)

#Here's all the code that remains to write:
# - add your favorite method for generating distributions. 
        #Each distribution should be a list of value vectors. 
        #We'll be drawing from the distribution using random.choice()
# - generate a distribution.
# - do the following 1000 times:
#       . generate a population from the distribution using initialize()
#       . run the rosca welfare using fullrun(), add to total
#       . divide total by number of runs

    print(argv)
    print(argv[1], argv[1] == 'crra')
    if argv[1] == "crra":
        a_values = [0, .1, .2, .3, .5, .75, 1, 1.5, 2]
        W_values = [1, 2, 3, 4, 5]
        # value_dist = [
        #     [2,0,0,0,0,0,0,0,0],
        #     [2,2,2,2,2,0,0,0,0],
        #     [5,5,0,0,0,0,0,0,0],
        #     [5,5,5,5,5,5,0,0,0],
        #     [8,8,8,0,0,0,0,0,0],
        #     [8,8,8,8,8,8,8,0,0],
        #     [8,8,8,5,5,5,2,2,2],
        #     [8,8,6,6,4,4,2,2,0],
        #     [8,7,6,5,4,3,2,1,0]
        # ]

        value_dist = []
        n = 30

        for i in range(n // 3):
            height = (n - 1) - 3 * i
            width = n - 2 - i
            remain = n - width
            agent = [height] * width + [0] * remain
            value_dist.append(agent)

            width = n - 2 - i - (n // 2)
            remain = n - width
            agent = [height] * width + [0] * remain
            value_dist.append(agent)
        value_dist.reverse()

        steps = []
        for i in range(1, n // 3 + 1):
            agent = []
            for j in range(n):
                val = (n - 1) - i * (j // i)
                agent.append(val)
            steps.append(agent)
        steps.reverse()
        value_dist.extend(steps)

        numruns = 10000

        # first calculate opt
        # opt, average = welfare(value_dist)
        # print("Old opt:", opt, "Average:", average)

        opt = opt_rosca(n, value_dist)
        print("OPT:", opt)
        # print("Average:", average)

        # now calculate the welfares from swapping
        for a in a_values:
            for W in W_values:
                start = time.process_time()
                wel, swaps, average_uniform = estimate(value_dist, len(value_dist), numruns, ("crra", W, a))
                end = time.process_time()
                duration = end - start
                poa = opt / wel
                print(f"{W},{a},{wel},{poa},{swaps},{average_uniform},{duration}")
                
    elif argv[1] == "distributions":                
        n = 30
        degen = []
        #welfare = 32
        for i in range(n):
            templist = ([4]*i)
            templist.extend([0]*(n-i))
            degen.append(templist)

        #welfare = 36
        unif_incr = []
        for i in range(n):
            templist = ([i]*(i+1))
            templist.extend([0]*(n-i-1))
            unif_incr.append(templist)

        #welfare = 45
        unif_decr = []
        for i in range(n):
            templist = ([n-i]*(i+1))
            templist.extend([0]*(n-i-1))
            unif_decr.append(templist)

        #pareto = [9,4.5,3,2.25,1.8,1.5,1.29,1.125,1]
        pareto = []

        if n == 9:
            scale = 1.4137050854113489
        elif n == 30:
            scale = 10 / 3
        for i in range(n):
            val = n / (i + 1)
            pareto.append(scale * val)

        #welfare = 25.465
        pareto_decr = []
        for i in range(n):
            templist = ([pareto[i]]*(i+1))
            templist.extend([0]*(n-i-1))
            pareto_decr.append(templist)

        #welfare = 24.465
        pareto_incr = []
        for i in range(n):
            templist = ([pareto[n-i-1]]*i)
            templist.extend([0]*(n-i))
            pareto_incr.append(templist)

        #unimodal = [9,7,7,5,5,5,3,3,1]
        unimodal = []
        if n == 9:
            scale = 0.8
        elif n == 30:
            scale = 4 / 3

        num = 1
        val = 1
        i = 0
        while i < n // 2:
            unimodal.extend([scale * val] * num)
            i += num
            val += 2
            num += 1

        num -= 1
        if n == 9:
            num -= 1

        while i < n:
            unimodal.extend([scale * val] * num)
            i += num
            val += 2
            num -= 1
        unimodal.reverse()

        #welfare = 25.465
        unimodal_decr = []
        for i in range(n):
            templist = ([unimodal[i]]*(i+1))
            templist.extend([0]*(n-i-1))
            unimodal_decr.append(templist)

        #welfare = 35.2
        unimodal_incr = []
        for i in range(n):
            templist = ([unimodal[n-i-1]]*i)
            templist.extend([0]*(n-i))
            unimodal_incr.append(templist)

        runs = [degen,unif_decr,unif_incr,pareto_decr,pareto_incr,unimodal_decr,unimodal_incr]
        run_names = ["degen","unif_decr","unif_incr","pareto_decr","pareto_incr","unimodal_decr","unimodal_incr"]
        crra = [("crra",0,0),("crra",4,.5)]
        numruns = 10000

        for i in range(len(runs)):
            print("run name:",run_names[i])
            print("opt:", opt_rosca(n, runs[i]))
            for config in crra:
                print("crra config",config)
                wel, swaps, average_uniform = estimate(runs[i], n, numruns, config)
                print("swap welfare",wel)
                print("avg welfare", average_uniform)
                print("swaps", swaps)
                print()

