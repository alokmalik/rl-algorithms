v_s=np.zeros(101)

def actions(state):
    if state<=50:
        return np.arange(1,state+1)
    else:
        return np.arange(1,101-state)

theta=.00000001

optimum_action=np.zeros(99)

v_s[0]=0
v_s[100]=1
delta=1
p=.4
state_record=[]
while delta>theta:
    delta=0
    for i in range(1,100):
        current_actions=actions(i)
        old_state=v_s[i]
        new_states=np.zeros(len(current_actions))
        for j,action in enumerate(current_actions):
            new_states[j]=p*v_s[i+action]+(1-p)*v_s[i-action]


        update=np.argmax(np.round(new_states, decimals=5))
        optimum_action[i-1]=update+1
        v_s[i]=new_states[update]
        delta=max(delta,abs(old_state-v_s[i]))
    state_record.append(tuple(v_s))

#Plot of value estimates vs capital
plt.figure(figsize=(20,8))
plt.title("Solution to gambler's problem for $p_h=.4$")
plt.xlabel('Capital', fontsize=18)
plt.ylabel('Value Estimates', fontsize=16)
plt.plot(state_record[0],label='Sweep 1')
plt.plot(state_record[1],label='Sweep 2')
plt.plot(state_record[2],label='Sweep 3')
plt.plot(state_record[-1],label='Final Sweep')
plt.legend()
plt.show()

#Plot of final optimal policy vs capital
plt.figure(figsize=(20,8))
plt.title("Solution to gambler's problem for $p_h=.4$")
plt.xlabel('Capital', fontsize=18)
plt.ylabel('Final Policy', fontsize=16)
plt.bar(np.arange(1,100),optimum_action)
plt.xticks([1,25,50,75,99],[1,25,50,75,99])
plt.legend()
plt.show()