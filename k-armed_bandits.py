# We'll use numpy to code our algorithm and matplotlib to plot results
import numpy as np
import matplotlib.pyplot as plt 

# A function that generates one k-armed bandit problem
# that returns Q*(a) for that problem
def k_armed_bandit(num_bandit_probs=500,k=7):
    return np.random.normal(0,1,(num_bandit_probs,k))

# A function that plays a given k-armed bandit problem
# this function includes greedy, epsilon-greedy and UCB action selection strategies.

num_bandit_probs=500
k=7
def play_k_armed_bandit(q_values,prob_num=0,time_steps=1000,\
                        strategy='greedy',epsilon=.1,c=1,k=7):
    
    q=np.zeros(k)
    rewards=np.zeros(time_steps)
    act=np.zeros(time_steps)
    actions=np.zeros(k)
    
    if strategy=='greedy':
        for i in range(time_steps):
            action=np.argmax(q)
            act[i]=action
            rewards[i]=np.random.normal(q_values[prob_num][action],1)
            actions[action]+=1
            #incremental implementation
            q[action]=q[action]+1/actions[action]*(rewards[i]-q[action])
            
        #calculates optimum action
        optimum_action = np.argmax(q_values[prob_num])
        #outputs a vector marking each time steps 1 for optimum action 0 otherwise
        opt_act=act==optimum_action
        return act,rewards,opt_act
    
    elif strategy=='epsilon_greedy':
        for i in range(time_steps):
            if np.random.uniform(0,1)<epsilon:
                action=np.random.randint(0,k)
                act[i]=action
                rewards[i]=np.random.normal(q_values[prob_num][action],1)
                actions[action]+=1
                q[action]=q[action]+1/actions[action]*(rewards[i]-q[action])
            else:
                action=np.argmax(q)
                act[i]=action
                rewards[i]=np.random.normal(q_values[prob_num][action],1)
                actions[action]+=1
                q[action]=q[action]+1/actions[action]*(rewards[i]-q[action])
        
        optimum_action = np.argmax(q_values[prob_num])
        opt_act=act==optimum_action
        return act,rewards,opt_act
    
    elif strategy=='ucb':
        for i in range(time_steps):
            temp=np.zeros(k)
            for j in range(k):
                temp[j]=q[j]+c*np.sqrt(np.log(i+1)/actions[j])
            action=np.argmax(temp)
            act[i]=action
            
            rewards[i]=np.random.normal(q_values[prob_num][action],1)
            actions[action]+=1
            q[action]=q[action]+1/actions[action]*(rewards[i]-q[action])
            
        optimum_action = np.argmax(q_values[prob_num])
        opt_act=act==optimum_action
        return act,rewards,opt_act

# A function that plays all the above strategies

def play_all(num_arms,num_runs, num_timesteps):
    
    #variables to store output of various methods
    greedy_rewards=np.zeros((num_runs,num_timesteps))
    greedy_actions=np.zeros((num_runs,num_timesteps))
    greedy_opt=np.zeros((num_runs,num_timesteps))
    epsilon01_rewards = np.zeros((num_runs,num_timesteps))
    epsilon1_rewards=np.zeros((num_runs,num_timesteps))
    epsilon01_actions=np.zeros((num_runs,num_timesteps))
    epsilon1_actions=np.zeros((num_runs,num_timesteps))
    epsilon01_opt=np.zeros((num_runs,num_timesteps))
    epsilon1_opt=np.zeros((num_runs,num_timesteps))
    ucb1_rewards=np.zeros((num_runs,num_timesteps))
    ucb2_rewards=np.zeros((num_runs,num_timesteps))
    ucb1_actions=np.zeros((num_runs,num_timesteps))
    ucb2_actions=np.zeros((num_runs,num_timesteps))
    ucb1_opt=np.zeros((num_runs,num_timesteps))
    ucb2_opt=np.zeros((num_runs,num_timesteps))
    
    print('calculating greedy policy')
    q=k_armed_bandit(num_runs,num_arms)
    for i in range(num_runs):
        greedy_actions[i],greedy_rewards[i],greedy_opt[i]=play_k_armed_bandit(q,prob_num=i,\
                time_steps=num_timesteps,strategy='greedy',k=num_arms)
    
    print('calculating epsilon greedy policy 1')
    q=k_armed_bandit(num_runs,num_arms)
    for i in range(num_runs):
        epsilon01_actions[i],epsilon01_rewards[i],epsilon01_opt[i]=play_k_armed_bandit(q,prob_num=i,\
                time_steps=num_timesteps,strategy='epsilon_greedy',epsilon=.01,k=num_arms)
    
    print('calculating epsilon greedy policy 2')
    q=k_armed_bandit(num_runs,num_arms)
    for i in range(num_runs):
        epsilon1_actions[i],epsilon1_rewards[i],epsilon1_opt[i]=play_k_armed_bandit(q,prob_num=i,\
                time_steps=num_timesteps,strategy='epsilon_greedy',epsilon=.1,k=num_arms)
    
    print('calculating UCB policy 1')
    q=k_armed_bandit(num_runs,num_arms)
    for i in range(num_runs):
        ucb1_actions[i],ucb1_rewards[i],ucb1_opt[i]=play_k_armed_bandit(q,prob_num=i,\
                time_steps=num_timesteps,strategy='ucb',c=1,k=num_arms)
    
    print('calculating UCB policy 2')
    q=k_armed_bandit(num_runs,num_arms)
    for i in range(num_runs):
        ucb2_actions[i],ucb2_rewards[i],ucb2_opt[i]=play_k_armed_bandit(q,prob_num=i,\
                time_steps=num_timesteps,strategy='ucb',c=2,k=num_arms)
    
    return greedy_rewards,epsilon01_rewards,epsilon1_rewards,ucb1_rewards,ucb2_rewards,\
greedy_opt,epsilon01_opt,epsilon1_opt,ucb1_opt,ucb2_opt

#running the experiments
num_arms = 7
num_strategies = 5
num_runs = 500
num_timesteps = 1000

greedy_rewards,epsilon01_rewards,epsilon1_rewards,ucb1_rewards,ucb2_rewards,\
greedy_opt,epsilon01_opt,epsilon1_opt,ucb1_opt,ucb2_opt\
=play_all(num_arms,num_runs,num_timesteps)

# Plotting the results reward vs timesteps plot
plt.figure(figsize=(20,8))
plt.title('Average Reward vs Time steps')
plt.xlabel('timesteps', fontsize=18)
plt.ylabel('reward', fontsize=16)
plt.plot(greedy_rewards.mean(axis=0), label="greedy")
plt.plot(epsilon01_rewards.mean(axis=0), label="Epsilon greedy .01")
plt.plot(epsilon1_rewards.mean(axis=0), label="Epsilon greedy .1")
plt.plot(ucb1_rewards.mean(axis=0), label="UCB C=1")
plt.plot(ucb2_rewards.mean(axis=0), label="UCB C=2")
plt.legend()
plt.show()

# Plotting the results %age of optimal actions vs timesteps
plt.figure(figsize=(20,8))
plt.title('Optimal Action % vs Time steps')
plt.xlabel('timesteps', fontsize=18)
plt.ylabel('% of optimal actions', fontsize=16)
plt.plot(greedy_opt.mean(axis=0)*100, label="greedy")
plt.plot(epsilon01_opt.mean(axis=0)*100, label="Epsilon greedy .01")
plt.plot(epsilon1_opt.mean(axis=0)*100, label="Epsilon greedy .1")
plt.plot(ucb1_opt.mean(axis=0)*100, label="UCB C=1")
plt.plot(ucb2_opt.mean(axis=0)*100, label="UCB C=2")
plt.legend()
plt.show()