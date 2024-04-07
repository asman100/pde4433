# import dependencies
import gymnasium as gym
from stable_baselines3 import SAC
import os
import cv2

log_path = os.path.join("Training", "Logs")
sac_path = os.path.join("Training", "Saved")
env = gym.make("Humanoid-v4", render_mode="rgb_array", camera_id=0)
model = SAC("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log=log_path)
model = SAC.load(os.path.join("Training", "Saved", "SAC_model.zip"))


def testmodel():

    # observe model performance
    obs = env.reset()[0]
    score = 0
    steps = 1000
    while True:
        action, _states = model.predict(obs)
        obs, rewards, trun, dones, info = env.step(action)
        cv2.imshow("Frame", env.render())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        score += rewards
        if dones:
            steps -= 1
            if steps <= 0:
                break
    print(info)
    print("Final Score:{}".format(score))
    print("Test Complete")
    env.close()


testmodel()
