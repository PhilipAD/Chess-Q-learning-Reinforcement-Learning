import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *

size_board = 4


def main():
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1):
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    """
    Initialization
    Define the size of the layers and initialization
    FILL THE CODE
    Define the network, the number of the nodes of the hidden layer should be 200, you should know the rest. The weights
    should be initialised according to a uniform distribution and rescaled by the total number of connections between
    the considered two layers. For instance, if you are initializing the weights between the input layer and the hidden
    layer each weight should be divided by (n_input_layer x n_hidden_layer), where n_input_layer and n_hidden_layer
    refer to the number of nodes in the input layer and the number of nodes in the hidden layer respectively. The biases
     should be initialized with zeros.
    """
    n_input_layer = 52  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the
    output layer according to the instructions. Define also the biases.
    """

    W1 = np.random.normal(scale=0.1, size=(n_input_layer,n_hidden_layer))
    W2 = np.random.normal(scale=0.1, size=(n_hidden_layer,n_output_layer))

    bias_W1 = np.zeros((1,n_hidden_layer))
    bias_W2 = np.zeros((1,n_output_layer))
    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035      #learning rate
    N_episodes = 100000     #Number of games, each game ends when we have a checkmate or a draw

    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction,
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])

    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.

    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])

    # END OF SUGGESTIONS


    for n in range(N_episodes):
        #print(n,W1,"W2",W2)

        epsilon_f = epsilon_0 / (1 + beta * n) #psilon is discounting per iteration to have less probability to explore

        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        # :return: dfK1: Degrees of Freedom of King 1, a_k1: Allowed actions for King 1, dfK1_: Squares the King1 is threatening
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        # :return: dfQ1: Degrees of Freedom of the Queen, a_q1: Allowed actions for the Queen, dfQ1_: Squares the Queen is threatening

        fQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # Actions & allowed_actions
            #Directions: down, up, right, left, down-right, down-left, up-right, up-left p1 king and queen directions to move
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])

            # Index postions of each available action in tge list of directions in a
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)


            # FILL THE CODE
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested.

            #x = np.array([x[0:16],x[16:32],x[32:48],np.asarray(x[48]),np.asarray(x[49])])
            Q, secondWB, firstRelu , firstWB= Q_values(x, W1, W2, bias_W1, bias_W2)
            """
            YOUR CODE STARTS HERE

            FILL THE CODE
            Implement epsilon greedy policy by using the vector a and a_allowed vector: be careful that the action must
            be chosen from the a_allowed vector. The index of this action must be remapped to the index of the vector a,
            containing all the possible actions. Create a vector called a_agent that contains the index of the action
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third action, a_agent=32 not 3.
            """

            #Max Qvalue from the network that the player can move
            predictedMove = 0
            sortedOutputs = np.argsort(Q)[::-1]
            for topProb in sortedOutputs[0]:
                if topProb in allowed_a:
                    predictedMove = topProb
                    break;

            #Exploration vs exploitation
            eGreedy = 0
            eGreedy = int(np.random.rand() < epsilon_f)  # with probability epsilon choose action at random if epsilon=0 then always choose Greedy
            if eGreedy:
                a_agent = np.random.choice(allowed_a)  # if epsilon > 0 (e-Greedy, chose at random with probability epsilon) choose one at random
            else:

                a_agent = predictedMove # will result will be Qvalue outputted from network

            #THE CODE ENDS HERE.


            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]
                N_moves_save[n-1,0] +=1

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]
                N_moves_save[n-1,0] +=i

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate

                R_save[n-1,0] = R


                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last
                iteration of the episode, the agent gave checkmate.
                """

                # ReLU derivative
                def dReLU(input):
                    return 1. * (input > 0)

                newQ = Q.copy()

                # apply reward to q value
                newQ[0][a_agent] = R

                #backpropagation
                dL2o = Q - newQ
                dU2 = dReLU(secondWB)

                #Second layer
                gL2 = np.dot(firstRelu.T, dU2 * dL2o)
                dL2b = dL2o * dU2

                #First layer
                dL1o = np.dot(dL2o , W2.T)
                dU1 = dReLU(firstWB)

                #convert into readable array
                newArray = np.zeros((52,1))
                count=0
                for g in np.nditer(x):
                    newArray[count]=g
                    count+=1

                gL1 = np.dot(newArray, dU1 * dL1o)
                dL1b = dL1o * dU1

                #Update weights and biases
                W1 -= eta * gL1
                bias_W1 -= eta * dL1b.sum(axis=0)

                W2 -= eta * gL2
                bias_W2 -= eta * dL2b.sum(axis=0)

                # THE CODE ENDS HERE

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1

                R_save[n-1,0] += R

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last
                iteration of the episode, it is a draw.
                """
                # ReLU derivative
                def dReLU(input):
                    return 1. * (input > 0)

                newQ = Q.copy()

                # apply reward to q value
                newQ[0][a_agent] = R

                #backpropagation
                dL2o = Q - newQ
                dU2 = dReLU(secondWB)

                #Second layer
                gL2 = np.dot(firstRelu.T, dU2 * dL2o)
                dL2b = dL2o * dU2

                #First layer
                dL1o = np.dot(dL2o , W2.T)
                dU1 = dReLU(firstWB)

                newArray = np.zeros((52,1))
                count=0
                for g in np.nditer(x):
                    newArray[count]=g
                    count+=1

                gL1 = np.dot(newArray, dU1 * dL1o)
                dL1b = dL1o * dU1

                #Update weights and biases
                W1 -= eta * gL1
                bias_W1 -= eta * dL1b.sum(axis=0)

                W2 -= eta * gL2
                bias_W2 -= eta * dL2b.sum(axis=0)


                # YOUR CODE ENDS HERE

                if draw:
                    break
            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]
                N_moves_save[n-1,0] +=i
            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _,_,_ = Q_values(x_next, W1, W2, bias_W1, bias_W2)

            """
            FILL THE CODE
            Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
            rectified linear function as activation function (see supplementary materials). Exploit the Q value for
            the action made. You computed previously Q values in the Q_values function. Be careful: this is not the last
            iteration of the episode, the match continues.
            """
            # Uncomment this to use SARSA algorithm
            #Max Qvalue from the network that the player can move
            #predictedMove = 0
            #sortedOutputs = np.argsort(Q)[::-1]
            #for topProb in sortedOutputs[0]:
            #    if topProb in allowed_a:
            #        predictedMove = topProb
            #        break;

            #Exploration vs exploitation
            #eGreedy = 0
            #eGreedy = int(np.random.rand() < epsilon_f)  # with probability epsilon choose action at random if epsilon=0 then always choose Greedy
            #if eGreedy:
            #    a_agent = np.random.choice(allowed_a)  # if epsilon > 0 (e-Greedy, chose at random with probability epsilon) choose one at random
            #else:

            #    a_agent = predictedMove # will result will be Qvalue outputted from network

            # ReLU derivative
            def dReLU(input):
                return 1. * (input > 0)

            newQ = Q.copy()
            modelPred = Q_next

            # apply reward to q value- this is q-learning algorithm
            newQ[0][a_agent] = R + gamma * np.max(modelPred)

            #backpropagation
            dL2o = Q - newQ
            dU2 = dReLU(secondWB)

            #Second layer
            gL2 = np.dot(firstRelu.T, dU2 * dL2o)
            dL2b = dL2o * dU2

            #First layer
            dL1o = np.dot(dL2o , W2.T)
            dU1 = dReLU(firstWB)

            newArray = np.zeros((52,1))
            count=0
            for g in np.nditer(x):
                newArray[count]=g
                count+=1

            gL1 = np.dot(newArray, dU1 * dL1o)
            dL1b = dL1o * dU1


            W1 -= eta * gL1
            bias_W1 -= eta * dL1b.sum(axis=0)

            W2 -= eta * gL2
            bias_W2 -= eta * dL2b.sum(axis=0)


            # YOUR CODE ENDS HERE

            i += 1


    fontSize = 18
    repetitions = 1   # should be integer, greater than 0; for statistical reasons
    totalRewards = np.zeros((repetitions,N_episodes))
    totalMoves = np.zeros((repetitions,N_episodes))


    totalRewards[0,:] = R_save.T
    totalMoves[0,:] = N_moves_save.T
    print(totalRewards.mean())

    newArray2 = np.zeros((52,1))
    count=0
    for g in np.nditer(x):
        newArray2[count]=g
        count+=1

        #Exponentially weighted moving average with alpha input
        def ewma(v, a):

            # Conform to array
            v = np.array(v)
            t = v.size

            # initialise matrix with 1-alpha
            # and a matrix to increse the weights
            wU = np.ones(shape=(t,t)) * (1-a)
            p = np.vstack([np.arange(i,i-t,-1)for
                i in range(t)])

            # Produce new weight matrix
            z = np.tril(wU**p,0)

            # return Exponentially moved average
            return np.dot(z, v[::np.newaxis].T)/z.sum(axis=1)


    # Plot the average reward as a function of the number of trials --> the average has to be performed over the episodes
    plt.figure()
    means = np.mean(ewma(totalRewards, 0.0001), axis = 0)
    errors = 2 * np.std(ewma(totalRewards, 0.0001), axis = 0)  # errorbars are equal to twice standard error i.e. std/sqrt(samples)

    plt.plot(np.arange(N_episodes), means)
    plt.xlabel('Episode',fontsize = fontSize)
    plt.ylabel('Average Moves',fontsize = fontSize)
    plt.axis((-(N_episodes/10.0),N_episodes,-0.1,1.1))
    plt.tick_params(axis = 'both', which='major', labelsize = 14)
    plt.show()

    plt.figure()
    means2 = np.mean(totalMoves, axis = 0)
    errors = 2 * np.std(ewma(totalMoves, 0.0001), axis = 0)  # errorbars are equal to twice standard error i.e. std/sqrt(samples)

    plt.plot(np.arange(N_episodes), means2)
    plt.xlabel('Episode',fontsize = fontSize)
    plt.ylabel('Moves',fontsize = fontSize)
    plt.axis((-(N_episodes/10.0),N_episodes,-0.1,1.1))
    plt.tick_params(axis = 'both', which='major', labelsize = 14)
    plt.show()
if __name__ == '__main__':
    main()
