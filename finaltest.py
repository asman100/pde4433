# import dependencies
import gymnasium as gym
from stable_baselines3 import SAC
import os
import cv2
import csv

# setup the environment and SAC model
log_path = os.path.join("Training", "Logs")
sac_path = os.path.join("Training", "Saved")
env = gym.make("Humanoid-v4", render_mode="rgb_array", camera_id=0)
model = SAC("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log=log_path)
model = SAC.load(os.path.join("Training", "Saved", "SAC_model.zip"))

# CSV File setup
csv_file_path = "scorechart.csv"
# Check if the CSV file exists already and setup the headers
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Score", "Reward"])
#


def testmodel():

    # observe model performance
    obs = env.reset()[0]
    # initialize tracking variables
    score = 0
    step = 0
    # intialize steps counter for testing, increasing steps will increase the time of testing (episode truncates after 100 steps by default)
    steps = 1000

    while True:
        # intialize model prediction and step through the environment
        action, _states = model.predict(obs)
        obs, rewards, trun, dones, info = env.step(action)
        # update the score and steps for data tracking
        score += rewards
        step += 1
        # render the environment and save the data to the csv file
        cv2.imshow("Frame", env.render())
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([step, score, rewards])
        # check for exit conditions and break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            env.close()
            break
        # check for the end of the episode and deccrement the steps counter
        if dones:
            steps -= 1
            if steps <= 0:
                break
    # print(info)
    # print(step)
    # print(steps)
    print("Final Score:{}".format(score))
    print("Test Complete")
    # close the environment
    env.close()


# run testing sequence
testmodel()
