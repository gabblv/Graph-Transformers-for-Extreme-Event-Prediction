import numpy as np

# #         # # # TEMP
# #         # print('\nlast train avg', last_train_avg)
# #         # print('best train avg', self.best_train)
# #         # print('last val avg', last_val_avg)
# #         # print('best val avg', self.best_val)
# #         # print('count', self.counter)
# #         # print('itr', self.iter, 'best itr', self.best_iter)



class EarlyStopping:
    def __init__(self, times=1, tolerance=1.00, plateau=2):

        self.times = times
        self.tolerance = tolerance
        self.best_val = np.inf
        self.iter = 0
        self.counter = 0
        self.plateau = plateau
        self.early_stop = False

    def __call__(self, last_val_losses):

        print('\nTesting for early stopping')

        self.iter += 1
        last_val_avg = sum(last_val_losses) / len(last_val_losses)

        if last_val_avg / self.best_val > self.tolerance:
            self.counter += 1

        if last_val_avg < self.best_val:
            self.best_val = last_val_avg
            # Reset the counter
            self.counter = 0
            self.best_iter = self.iter

        if self.counter >= self.times or (self.iter - self.best_iter) == self.plateau:
            self.early_stop = True
