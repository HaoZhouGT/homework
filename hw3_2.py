import numpy as np

# we need a function that tells which is the best action given a state


prob_matrix = np.zeros([8,7])
prob_matrix[0,0] = -1
prob_matrix[0,1] = 3
prob_matrix[0,2] = 0
prob_matrix[0,3] = 0
prob_matrix[0,4] = 1
prob_matrix[0,5] = 0
prob_matrix[0,6] = -1
prob_matrix[1,0] = 2
prob_matrix[1,1] = 1
prob_matrix[1,2] = 1
prob_matrix[1,3] = 3
prob_matrix[1,4] = -1
prob_matrix[1,5] = -1
prob_matrix[1,6] = 1
prob_matrix[2,0] = 2
prob_matrix[2,1] = 1
prob_matrix[2,2] = 2
prob_matrix[2,3] = 2
prob_matrix[2,4] = 0
prob_matrix[2,5] = 1
prob_matrix[2,6] = 1
prob_matrix[3,0] = 1
prob_matrix[3,1] = 2
prob_matrix[3,2] = -1
prob_matrix[3,3] =-1
prob_matrix[3,4] = 2
prob_matrix[3,5] = 0
prob_matrix[3,6] = 0
prob_matrix[4,0] = 0
prob_matrix[4,1] = 3
prob_matrix[4,2] = 2
prob_matrix[4,3] = -1
prob_matrix[4,4] = 1
prob_matrix[4,5] = 0
prob_matrix[4,6] = 3
prob_matrix[5,0] = -1
prob_matrix[5,1] = 3
prob_matrix[5,2] = 2
prob_matrix[5,3] = 1
prob_matrix[5,4] = 2
prob_matrix[5,5] = -1
prob_matrix[5,6] = -1
prob_matrix[6,0] = 0
prob_matrix[6,1] = 3
prob_matrix[6,2] = 3
prob_matrix[6,3] = 1
prob_matrix[6,4] = 0
prob_matrix[6,5] = 0
prob_matrix[6,6] =0
prob_matrix[7,0] = 2
prob_matrix[7,1] = -1
prob_matrix[7,2] = 3
prob_matrix[7,3] = -1
prob_matrix[7,4] = 3
prob_matrix[7,5] = 2
prob_matrix[7,6] =3

# the wind makes the system become stochastic
# but the action policy is still determinisitic policy
p0 = [0.75,0.1,0.1,0.05,0.0]
p1 = [0.2,0.6,0.05,0.15,0.0]
p2 = [0.05,0.2,0.2,0.55,0.0]
p3 = [0.2,0.05,0.6,0.15,0.0]
p = [0.0,0.0,0.0,0.0,1.0]
# there are five possible next states


# it defines after you take action and hope to stay in such a cell [i][j], the probability of your real next state due to
# the disturbance or randomness of the system
def prob(i,j):
    # it not only returns the prob model, also returns the arrow direction for you to find the next state index
    if prob_matrix[i,j] == 0:
        if j == 0:
            return ([0.0,0.1,0.1,0.05,0.75],0)
        else:
            return (p0,0)
    if prob_matrix[i,j] == 1:
        if i ==7:
            return ([0.0,0.6,0.05,0.15,0.2],1)
        else:
            return (p1,1)
    if prob_matrix[i,j] == 2:
        if j == 6:
            return ([0.0,0.2,0.2,0.55,0.05],2)
        else:
            return (p2,2)
    if prob_matrix[i,j] == 3:
        if i == 0:
            return ([0.0,0.02,0.6,0.15,0.2],3)
        else:
            return (p3,3)
    if prob_matrix[i,j] == -1:
        return (p,-1)

print('the probability matrix is given by wind, which are ','\\n', prob_matrix)


## notice that we have a reward matrix, we can set the out-of-boundary reward to negative value
def reward(i,j):
    if (0<=i and i<=7) and (0<=j and j<=6):
        if i==0 and j==0:
            return -10
        if i==1 and j==4:
            return -10
        if i==1 and j==5:
            return -10
        if i==3 and j==2:
            return -10
        if i==3 and j==3:
            return -10
        if i==4 and j==3:
            return -10
        if i==5 and j==0:
            return -10
        if i==5 and j==5:
            return -10
        if i==5 and j==6:
            return -10
        if i==7 and j==3:
            return -10
        else:
            return -1
    else:
        return -300











def value_iteration(theta=0.0001, discount_factor=1.0):

    def one_step_lookahead(i,j, V):  # compute the value of an action
        """
        Helper function to calculate the value for all action in a given state.
        66, to compute the value of an action, you still need the value of state
        Args:
            state: The state to consider (int), state is denoted by i and j
            V: The value to use as an estimator, Vector of length env.nS
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        if i==0 and j==0:
            A = np.zeros(4)  # here we always have 4 actions
            ## the only action is to moving right 1 and moving down 2. let's denote the other two as -10000
            # thus it would never be taken
            A[0] = -10000
            # for moving right 1
            # first retrieve the probability model after taking the action
            transition_prob = prob(i+1,j)[0]
            # compute the value of action 1, aka moving to the right
            i = i+1
            A[1] += transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            i = i-1 # move back the i value

            # compute the value of action 2, aka moving down
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j+1)[0]
            arrow_direction = prob(i,j+1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j+1
            A[2] += transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            j = j-1
            A[3] = -10000

        if i == 7 and j == 0:
            A = np.zeros(4)  # here we always have 4 actions
            A[0] = -10000
            A[1] = -10000
            # compute the value of action 3, aka moving left
            # first retrieve the probability model after taking the action
            transition_prob = prob(i-1,j)[0]
            arrow_direction = prob(i-1,j)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            i = i-1
            if arrow_direction == 0:
                A[0] += transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j])+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[0] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[0] += transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[0] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            i=i+1

            # compute the value of action 2, aka moving down
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j+1)[0]
            arrow_direction = prob(i,j+1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j+1
            if arrow_direction == 0:
                A[2] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[2] += transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[2] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j] + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[2] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1])  + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            j= j-1



        if i == 0 and j == 6:
            A = np.zeros(4)  # here we always have 4 actions
            A[2] = -10000
            A[3] = -10000
            # first compute the value of action 0, aka moving up
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j-1)[0]
            arrow_direction = prob(i,j-1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j-1
            if arrow_direction == 0:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j] + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[0] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) +  transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) +  transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[0] += transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            j= j+1


            # compute the value of action 1, aka moving to the right
            # first retrieve the probability model after taking the action
            transition_prob = prob(i+1,j)[0]
            arrow_direction = prob(i+1,j)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            i = i+1
            if arrow_direction == 0:
                A[1] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) +  transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[1] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[1] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j])  + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[1] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            i = i-1



        if i == 7 and j == 6:
            A = np.zeros(4)  # here we always have 4 actions
            A[1] = -10000
            A[2] = -10000
            # first compute the value of action 0, aka moving up
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j-1)[0]
            arrow_direction = prob(i,j-1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j-1
            if arrow_direction == 0:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[0] +=  transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) +  transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[0] += transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            j= j+1


            # compute the value of action 3, aka moving left
            # first retrieve the probability model after taking the action
            transition_prob = prob(i-1,j)[0]
            arrow_direction = prob(i-1,j)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            i = i-1
            if arrow_direction == 0:
                A[3] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[3] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[3] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[3] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1] + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            i=i+1




        if (1<=i and i<=6) and (1<=j and j<=5):
            A = np.zeros(4)  # here we always have 4 actions
            # if action not feasible, make the reward to be negative infinity
            # we need the probability matrix, so we can know which next state to go
            # first compute the value of action 0, aka moving up
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j-1)[0]
            arrow_direction = prob(i,j-1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j-1
            if arrow_direction == 0:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[0] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[0] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[0] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            j= j+1

            # compute the value of action 1, aka moving to the right
            # first retrieve the probability model after taking the action
            transition_prob = prob(i+1,j)[0]
            arrow_direction = prob(i+1,j)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            i = i+1
            if arrow_direction == 0:
                A[1] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[1] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[1] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[1] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            i = i-1

            # compute the value of action 2, aka moving down
            # first retrieve the probability model after taking the action
            transition_prob = prob(i,j+1)[0]
            arrow_direction = prob(i,j+1)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            j = j+1
            if arrow_direction == 0:
                A[2] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[2] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[2] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[2] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            j= j-1


            # compute the value of action 3, aka moving left
            # first retrieve the probability model after taking the action
            transition_prob = prob(i-1,j)[0]
            arrow_direction = prob(i-1,j)[1]
            #the current cell after taking action becomes i,j-1, then consider the disturbance impact and next cell
            i = i-1
            if arrow_direction == 0:
                A[3] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i+1,j) + discount_factor * V[i+1,j]+ transition_prob[2]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 1:
                A[3] += transition_prob[0]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[1]*(reward(i,j+1) + discount_factor * V[i,j+1]+ transition_prob[2]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[3]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 2:
                A[3] += transition_prob[0]*(reward(i,j-1) + discount_factor * V[i,j-1]) + transition_prob[1]*(reward(i-1,j) + discount_factor * V[i-1,j]+ transition_prob[2]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[3]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[4]*(reward(i,j) + discount_factor * V[i,j])
            if arrow_direction == 3:
                A[3] += transition_prob[0]*(reward(i-1,j) + discount_factor * V[i-1,j]) + transition_prob[1]*(reward(i,j-1) + discount_factor * V[i,j-1]+ transition_prob[2]*(reward(i,j+1) + discount_factor * V[i,j+1]) + transition_prob[3]*(reward(i+1,j) + discount_factor * V[i+1,j]) + transition_prob[4]*(reward[i,j] + discount_factor * V[i,j])
            i=i+1


        return A
        # A is the action list, which returns the list of action value for a given state, and a given state value matrix
        # note that the optimal control policy must pick the best action from the action list
        # also notice that A also depends on the state value matrix it uses


    ## create dictionary for value iteration, because the matrix index is limited
    ## also need to give value to out-of-boundary
    # V0 = {(i,j):0 for i in range(-1,8,1) and for j in range(-1,7,1)}
    # print('initial value function is ',V)

    while True:
        delta = 0
        V = V0
        for i in range(7):
            for j in range(8):
                V0[i][j] = R[i][j] + discount_factor * # averaged next state value by taking the best action
                # here we need to compute the best action for each state
                # and we need the probability transition matrix to get the


#
value_iteration()
