import torch.nn as nn
import torch


class FeedForwardTorch(nn.Module):
    def __init__(self, dimensions, activations, loss_fun, optimizer):
        super(FeedForwardTorch, self).__init__()

        self.layer_count = len(dimensions) - 1
        if self.layer_count == 1:
            self.ffw = nn.Linear(dimensions[0], dimensions[1])
            if activations and activations[0]:
                self.ffw = activations[0](self.ffw)
        else:
            layers = []
            if activations:
                for i in range(self.layer_count):
                    layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
                    if activations[i]:
                        layers.append(activations[i]())
            else:
                for i in range(self.layer_count):
                    layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))

            self.ffw = nn.Sequential(*layers)

        self.loss_fun = loss_fun
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        return self.ffw(x)

    def train(self, mini_batches, stop_loss=1e-6, max_iter=1000, mode=True):
        self.ffw.train()
        super(FeedForwardTorch, self).train(mode)
        loss = None
        if mini_batches:
            for _ in range(max_iter):
                for x, y in mini_batches:
                    self.zero_grad()
                    loss = self.loss_fun(self.__call__(x), y)
                    if loss < stop_loss:
                        break
                    loss.backward()
                    self.optimizer.step()
        return loss

    def predict(self, x):
        self.ffw.eval()
        return self.__call__(x)

    def save(self, path: str):
        torch.save(self.ffw.state_dict(), path)

    def load(self, path: str):
        self.ffw.load_state_dict(torch.load(path))
        self.ffw.eval()

