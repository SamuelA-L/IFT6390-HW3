import random
import numpy as np
import torch
from typing import Tuple, Callable, List, NamedTuple
import torchvision
import tqdm
# import matplotlib.pyplot as plt
# import json


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    paddings: Tuple[int, ...] = (0, 0, 0)
    dense_hiddens: Tuple[int, ...] = (256, 256)


# Pytorch preliminaries
def gradient_norm(function: Callable, *tensor_list: List[torch.Tensor]) -> float:

    loss = function(*tensor_list)
    loss.backward()
    grads = torch.stack(tuple([tensor.grad for tensor in tensor_list]))
    norm = torch.norm(grads)

    return norm


def jacobian_norm(function: Callable, input_tensor: torch.Tensor) -> float:

    jacobian = torch.autograd.functional.jacobian(function, input_tensor)
    norm = torch.norm(jacobian)

    return norm


class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 datapath: str = './data',
                 n_classes: int = 10,
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 activation_name: str = "relu",
                 normalization: bool = True):
        self.train, self.valid, self.test = self.load_dataset(datapath)
        if normalization:
            self.train, self.valid, self.test = self.normalize(self.train, self.valid, self.test)
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0][0].shape
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], net_config,
                                           n_classes, activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], net_config, n_classes, activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = 1e-9

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': [],
                           'train_gradient_norm': []}

    @staticmethod
    def load_dataset(datapath: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        trainset = torchvision.datasets.FashionMNIST(root=datapath,
                                                     download=True, train=True)
        testset = torchvision.datasets.FashionMNIST(root=datapath,
                                                    download=True, train=False)

        X_train = trainset.data.view(-1, 1, 28, 28).float()
        y_train = trainset.targets

        X_ = testset.data.view(-1, 1, 28, 28).float()
        y_ = testset.targets

        X_val = X_[:2000]
        y_val = y_[:2000]

        X_test = X_[2000:]
        y_test = y_[2000:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        layer_dim = list(net_config.dense_hiddens)
        mlp_list = [torch.nn.Flatten(),
                    torch.nn.Linear(input_dim, layer_dim[0]),
                    activation
                    ]

        layer_dim.append(n_classes)
        for i, n in enumerate(layer_dim):
            if i != len(layer_dim)-1:
                mlp_list.append(torch.nn.Linear(n, layer_dim[i+1]))
                mlp_list.append(activation)

        mlp_list.append(torch.nn.Softmax(dim=1))
        mlp = torch.nn.Sequential(*mlp_list)

        return mlp


    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.

        class NetworkConfiguration(NamedTuple):
            n_channels: Tuple[int, ...] = (16, 32, 48)
            kernel_sizes: Tuple[int, ...] = (3, 3, 3)
            strides: Tuple[int, ...] = (1, 1, 1)
            paddings: Tuple[int, ...] = (0, 0, 0)
            dense_hiddens: Tuple[int, ...] = (256, 256)

        """
        n_channels = list(net_config.n_channels)
        n_channels = [in_channels] + n_channels
        kernel_sizes = list(net_config.kernel_sizes)
        strides = list(net_config.strides)
        paddings = list(net_config.paddings)
        layer_dim = list(net_config.dense_hiddens)
        maxpool = torch.nn.MaxPool2d(kernel_size=2)

        cnn_list = []
        for i, n in enumerate(n_channels):
            if i != len(n_channels) - 1:
                cnn_list.append(torch.nn.Conv2d(in_channels=n, out_channels=n_channels[i+1], kernel_size=kernel_sizes[i], padding=paddings[i], stride=strides[i]))
                cnn_list.append(activation)
                if i != len(n_channels) - 2:
                    cnn_list.append(maxpool)

        cnn_list.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
        cnn_list.append(torch.nn.Flatten())
        cnn_list.append(activation)
        cnn_list.append(torch.nn.Linear(n_channels[-1]*4*4, layer_dim[0]))
        cnn_list.append(activation)
        layer_dim.append(n_classes)

        for i, n in enumerate(layer_dim):
            if i != len(layer_dim)-1:
                cnn_list.append(torch.nn.Linear(n, layer_dim[i+1]))
                cnn_list.append(activation)

        cnn_list.append(torch.nn.Softmax(dim=1))

        cnn = torch.nn.Sequential(*cnn_list)

        return cnn

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        layer = None
        if activation_str == "relu":
            layer = torch.nn.ReLU()
        elif activation_str == "tanh":
            layer = torch.nn.Tanh()
        elif activation_str == "sigmoid":
            layer = torch.nn.Sigmoid()

        return layer

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        one_hot = np.zeros((len(y), self.n_classes), dtype=int)
        y_list = y.tolist()
        for i, label in enumerate(y_list):
            one_hot[i, int(label)] = 1

        return torch.tensor(one_hot, dtype=float)

    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:

        y = y.to(torch.float64)
        predictions = self.network(X)
        pred_scaled = torch.clamp(predictions, min=self.epsilon, max=(1-self.epsilon))

        loss_fn = torch.nn.NLLLoss()
        loss_val = loss_fn(torch.log(pred_scaled), y.argmax(dim=1))

        n_samples = len(y)
        good = 0
        for i, pred in enumerate(pred_scaled):
            if pred.argmax() == y[i].argmax():
                good += 1

        accuracy = good / n_samples

        return loss_val, accuracy

    @staticmethod
    def compute_gradient_norm(network: torch.nn.Module) -> float:

        # Compute the Euclidean norm of the gradients of the parameters of the network
        # with respect to the loss function.

        grads = tuple([torch.flatten(p.grad) for p in network.parameters()])
        concat_grads = torch.cat(grads)
        norm = torch.linalg.norm(concat_grads)

        return norm.item()

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:

        loss_fn = torch.nn.NLLLoss()
        preds = self.network(X_batch)
        pred_scaled = torch.clamp(preds, min=self.epsilon, max=(1-self.epsilon))
        loss = loss_fn(torch.log(pred_scaled), y_batch.argmax(dim=1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        grad_norm = self.compute_gradient_norm(self.network)

        return grad_norm

    def log_metrics(self, X_train: torch.Tensor, y_train_oh: torch.Tensor,
                    X_valid: torch.Tensor, y_valid_oh: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train_oh)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid_oh)
        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(float(train_loss))
        self.train_logs['validation_loss'].append(float(valid_loss))

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        y_train_oh = self.one_hot(y_train)
        X_valid, y_valid = self.valid
        y_valid_oh = self.one_hot(y_valid)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train_oh[self.batch_size * batch:self.batch_size * (batch + 1), :]
                gradient_norm = self.training_step(minibatchX, minibatchY)
            # Just log the last gradient norm
            self.train_logs['train_gradient_norm'].append(gradient_norm)
            self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:

        print(y.size()[0])
        y = self.one_hot(y)
        with torch.no_grad():
            loss, accuracy = self.compute_loss_and_accuracy(X, y)

        return loss, accuracy


    @staticmethod
    def normalize(train: Tuple[torch.Tensor, torch.Tensor],
                  valid: Tuple[torch.Tensor, torch.Tensor],
                  test: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor]]:

        mean = train[0].mean(axis=0)
        std = train[0].std(axis=0)
        new_train = (train[0] - mean) / std
        new_valid = (valid[0] - mean) / std
        new_test = (test[0] - mean) / std


        return (
            (new_train, train[1]),
            (new_valid, valid[1]),
            (new_test, test[1])
         )


    def test_equivariance(self):
        from functools import partial
        test_im = self.train[0][0]/255.
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(torchvision.transforms.functional.affine, angle=0,
                        translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(torchvision.transforms.functional.affine, angle=90,
                           translate=(0, 0), scale=1, shear=0)

        def show_image(tensor):

            plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
            plt.show()

        def q1() :
            # first figure
            show_image(self.train[0][0])

            # second image
            cnn_image = model(torch.tensor(torch.unsqueeze(self.train[0][0].clone().detach(), 0)))
            show_image(torch.squeeze(cnn_image.clone().detach(), 0))

            # third image

            shifted_output = shift(model(torch.tensor(torch.unsqueeze(self.train[0][0].clone().detach(), 0))))
            show_image(torch.squeeze(shifted_output.clone().detach(), 0))

            shifted_input = shift(self.train[0][0].clone().detach())
            cnn_shifted_input = model(torch.unsqueeze(shifted_input.clone().detach(), 0))
            show_image(torch.squeeze(cnn_shifted_input.clone().detach(), 0))

            diff_image = torch.sub(shifted_output, cnn_shifted_input)
            show_image(torch.squeeze(diff_image.clone().detach(), 0))

            # fourth image

            rot_output = rotation(cnn_image.clone().detach())
            show_image(torch.squeeze(rot_output.clone().detach(), 0))

            rot_input = rotation(self.train[0][0].clone().detach())
            cnn_rot_input = model(torch.unsqueeze(rot_input.clone().detach(), 0))
            show_image(torch.squeeze(cnn_rot_input.clone().detach(), 0))

            diff_image = torch.sub(rot_output, cnn_rot_input)
            show_image(torch.squeeze(diff_image.clone().detach(), 0))

        q1()

#
#  # q1
# # trainer = Trainer(normalization=True)
# # trainer.test_equivariance()
#
# #q2
#
# def make_training_logs(lr_list, file : bool = False):
#     # no normalization
#     logs_dict = {}
#     for i, learning_rate in enumerate(lr_list) :
#         trainer = Trainer(batch_size=128, activation_name="relu", network_type="mlp", normalization=False)
#         print('nb_params : ', sum(p.numel() for p in trainer.network.parameters() if p.requires_grad))
#         logs_dict[str(i)] = trainer.train_loop(n_epochs=50)
#
#     if file:
#         with open('q2_mlp_no_norm.json', 'w') as file:
#             json.dump(logs_dict, file)
#
#     # normalization
#
#     logs_dict = {}
#     for i, learning_rate in enumerate(lr_list):
#         trainer = Trainer(batch_size=128, activation_name="relu", network_type="mlp", normalization=True)
#         print('nb_params : ', sum(p.numel() for p in trainer.network.parameters() if p.requires_grad))
#         logs_dict[str(i)] = trainer.train_loop(n_epochs=50)
#
#     if file:
#         with open('q2_mlp_norm.json', 'w') as file:
#             json.dump(logs_dict, file)
#
#
# def accuracy_graphs():
#     with open('q2_mlp_no_norm.json') as json_file:
#         no_norm_dict = json.load(json_file)
#
#     with open('q2_mlp_norm.json') as json_file:
#         w_norm_dict = json.load(json_file)
#
#     plt.title("Training mlp without normalization by learning rate")
#     [plt.plot(v['validation_accuracy'], label=lr_list[int(k)]) for k, v in no_norm_dict.items()]
#     plt.xlabel('epochs')
#     plt.ylabel('Validation accuracy')
#     plt.legend()
#     plt.show()
#
#     plt.title("Training mlp with normalization by learning rate")
#     [plt.plot(v['validation_accuracy'], label=lr_list[int(k)]) for k, v in w_norm_dict.items()]
#     plt.xlabel('epochs')
#     plt.ylabel('Validation accuracy')
#     plt.legend()
#     plt.show()
#
# lr_list = [0.01, 0.0001, 0.00000001]
# # make_training_logs(lr_list, file=False)
# # accuracy_graphs()
#
#
#
#
# #q3
#
# def train_network(file=False):
#     config = NetworkConfiguration(dense_hiddens=tuple([64]*54))
#     trainer = Trainer(net_config=config)
#     print('nb_params : ', sum(p.numel() for p in trainer.network.parameters() if p.requires_grad))
#     logs_dict = trainer.train_loop(n_epochs=50)
#
#     if file:
#         with open('q3_default.json', 'w') as file:
#             json.dump(logs_dict, file)
#
#
#
# def plot_comparison():
#     with open('q2_mlp_norm.json') as json_file:
#         w_norm_dict = json.load(json_file)
#
#     with open('q3_default.json') as json_file:
#         mlp_64_dict = json.load(json_file)
#
#     plt.title("Training mlp  53x64 vs 2x256 (layesrxfilters)")
#     plt.plot(mlp_64_dict['validation_accuracy'], label=' validation accuracy mlp (64x53)')
#     plt.plot(mlp_64_dict['train_accuracy'], label=' train accuracy mlp (64x53)')
#     plt.plot(w_norm_dict['1']['validation_accuracy'], label=' validation accuracy mlp (256x2)')
#     plt.plot(w_norm_dict['1']['train_accuracy'], label=' train accuracy mlp (256x2)')
#     plt.xlabel('epochs')
#     plt.ylabel('metric')
#     plt.legend()
#     plt.show()
#
#     plt.title("Training gradient norm of mlp 53x64 vs 2x256 (layesrxfilters)")
#     plt.plot(mlp_64_dict['train_gradient_norm'], label='mlp (64x53)')
#     plt.plot(w_norm_dict['1']['train_gradient_norm'], label='mlp (256x2)')
#     plt.xlabel('epochs')
#     plt.ylabel('train_gradient_norm')
#     plt.legend()
#     plt.show()
#
# train_network(file=True)
# plot_comparison()


# q4

# config = NetworkConfiguration(n_channels=(16, 32, 45), kernel_sizes=(3, 3, 3))
# trainer = Trainer(batch_size=128, activation_name="relu", network_type="cnn", normalization=False, net_config=config)
# print('nb_params : ', sum(p.numel() for p in trainer.network.parameters() if p.requires_grad))
# print(trainer.network)
# logs = trainer.train_loop(n_epochs=50)
