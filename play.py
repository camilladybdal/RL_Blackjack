from Blackjack import *
from DQN import *

CHECKPOINT_DIR = "checkpoints/"

def main():
    #dqn_train_loop(CHECKPOINT_DIR, 10)
    
    #CHECKPOINT_DIR = checkpoint_dir
    EPISODES = 5000000
    rewards = np.empty(EPISODES)
    losses =  np.empty(EPISODES)

    copy_step = 25
    num_actions = 4
    num_states = 4
    input = (num_states,)
    iterations = 0
    gamma = 0.95
    eps_decay = 0.9999
    eps_min = 0.1
    epsilon = 0.99
    loss = 0


    env = Blackjack()
    TargetNet = DQN(input, num_actions)
    TrainNet = DQN(input, num_actions)

    for episode in range(EPISODES):
        if episode < 0.009 * EPISODES:
            epsilon = max(eps_min, epsilon * eps_decay)
            
        state = env.reset()

        while not env.game_over:
            action = TrainNet.get_action(state, epsilon)
            prev_state = state
            state, reward, done = env.step(action)

            experience = {'state': prev_state, 'action': action, 'reward': reward, 'next_state': state, 'done': done}
            TrainNet.memory.add_experience(experience)
            loss = TrainNet.train(TargetNet, gamma)
            
            iterations += 1
            if iterations % copy_step == 0:
                TargetNet.copy_weights(TrainNet)
            
        rewards[episode] = reward
        if loss == None: loss = 0
        losses[episode] = loss

        if episode % 100 == 0:
            print("saved model:" , episode)
            #TrainNet.save_weights(CHECKPOINT_DIR)
            print("episode loss ", loss)
            TrainNet.model.save("my_model")
            save_strategy(TrainNet, "strategy.csv")

    print("average reward: ", np.mean(rewards))
    TrainNet.save_weights(CHECKPOINT_DIR )
    save_strategy(TrainNet, "strategy.csv")


    dqn_test_loop(10000)

if __name__ == "__main__":
    main()