class Trainer:
    def __init__(self, model, lr, epochs, loss):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.loss = loss

    def train(self, x, y):
        for _ in range(self.epochs):
            y_pred = self.model.forward(x)
            self.loss.forward(y_pred, y)
            self.model.backward(self.loss.backward(), self.lr)
