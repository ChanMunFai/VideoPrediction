import torch
from data.MovingMNIST import MovingMNIST

class LinearScheduler: 
    """Basic linear scheduler
    """
    def __init__(self, 
                training_steps, 
                start_value, 
                end_value):

        self.training_steps = training_steps 
        self.start_value = start_value 
        self.end_value = end_value 

        self.increment = (self.end_value - self.start_value)/self.training_steps
        self.counter = 0
        self.value = self.start_value

    def step(self):
        self.value += self.increment
        self.counter += 1 
        return self.value 

def test_scheduler(): 
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=16,
                shuffle=True)

    epochs = 50
    num_training_steps = len(train_loader) * epochs

    scheduler = LinearScheduler(training_steps = num_training_steps, 
                                start_value = 0, 
                                end_value = 0.001)

    for epoch in range(epochs): 
        for data, _ in train_loader: 
            beta = scheduler.step() 
            print(beta)

if __name__ == "__main__": 
    test_scheduler()

    


        


     

