import bird_play
import bird_train

def play():
    bird_play.MODEL_FILE = 'pretrain_model/DQN_KERAS'
    bird_play.play()

if __name__ == "__main__":
    bird_train.init()
    play()
