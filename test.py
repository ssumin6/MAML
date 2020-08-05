import torch 
import argparse
import matplotlib.pyplot as plt 
import numpy as np

from model import Net
from sinusoid import Sinusoid 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    ckpt = torch.load(args.path)
    net = Net().to(device, torch.double)
    net.load_state_dict(ckpt["model"])
    print("Load Checkpoint trained for epochs %d." %(ckpt["epoch"]))

    # test set : [1, K, 1] K : number of points
    input_range = np.array([-5.0, 5.0])
    amplitude_range = np.array([0.1, 5.0])
    phase_range = np.array([0., np.pi])

    amplitude = np.random.uniform(amplitude_range[0],
        amplitude_range[1])
    phase = np.random.uniform(phase_range[0], phase_range[1])

    # input shape [100]
    inputs = np.linspace(input_range[0], input_range[1], num=100)
    # output shape [100]
    targets = amplitude * np.sin(inputs + phase)

    # Output tensor
    inputs = torch.tensor(inputs).double().to(device).unsqueeze(1)
    print(inputs.size())

    test_output = net(inputs.unsqueeze(0)) # output shape [1, 100, 1]
    if torch.cuda.is_available():
        test_output = test_output.cpu()
        inputs = inputs.cpu()
    inputs = inputs.view(-1).detach().numpy()

    criterion = torch.nn.MSELoss()

    loss = criterion(test_output.squeeze(-1), torch.tensor(targets).unsqueeze(0))
    print("loss: ", loss.item())

    test_output = test_output.view(-1).detach().numpy()

    plt.plot(inputs, test_output, label="MAML")
    plt.plot(inputs, targets, label="GT")
    plt.legend()
    plt.savefig("sin_graph.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAML')
    
    parser.add_argument(
        '--path',
        type=str,
        default="best.ckpt")

    args = parser.parse_args()

    main(args)