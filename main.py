import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch.utils.data import DataLoader

from model import Net
from sinusoid import Sinusoid

import signal
import sys

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    M = 10
    K = 10

    train_dataset = Sinusoid(k_shot=K, q_query=15, num_tasks=2000000)
    train_loader = DataLoader(train_dataset, batch_size=M, shuffle=True, pin_memory=True)

    inner_alpha = 1e-2 
    iterations = 70000
    losses = []
    save_loss = 150
    loss_for_graph = []

    def signal_handler(sig, frame):
        print("CTRL-C")
        plt.plot(loss_for_graph)
        plt.xlabel("iterations / 50")
        plt.ylabel("MSE Loss")
        plt.savefig("test_loss.png")
        sys.exit(0)

    net = Net().to(device, torch.double)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)

    if args.model_load:
        ckpt = torch.load("best.ckpt")
        net.load_state_dict(ckpt["model"])
        print("Load Checkpoint trained for epochs %d." %(ckpt["epoch"]))

    net.train()
    
    signal.signal(signal.SIGINT, signal_handler)

    # Shape: (M, K, 1), (M, K, 1), (M, Q, 1), (M, Q, 1) where M:batch_size
    for i in range(iterations):
        train_inputs, train_targets, test_inputs, test_targets = next(iter(train_loader))  
        if torch.cuda.is_available():
            # LOAD DATASET ON CUDA
            train_inputs = train_inputs.cuda()
            train_targets = train_targets.cuda()
            test_targets = test_targets.cuda()
            test_inputs = test_inputs.cuda()
        
        test_inputs = test_inputs.double()
        train_inputs = train_inputs.double()

        outer_loss = 0

        # 1 step inner optimization (10 TASK)
        for j in range(M):
            net.zero_grad()
            # clone parameters into model_dict
            model_dict = OrderedDict()
            for k, v in net.named_parameters():
                model_dict[k] = v.clone()
            train_output = net(train_inputs[j].unsqueeze(0))
            loss = criterion(train_output, train_targets[j].unsqueeze(0))
            grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
           
            # parameter update
            for idx, key in enumerate(model_dict.keys()):
                model_dict[key] = model_dict[key] - inner_alpha * grad_params[idx]

            test_output = net(test_inputs[j].unsqueeze(0), params=model_dict)
            outer_loss += criterion(test_output, test_targets[j].unsqueeze(0))
            
        # OUTER LOOP
        outer_loss.div_(M)
        optimizer.zero_grad()
        outer_loss.backward()
        optimizer.step()
        
        losses.append(outer_loss.item())

        if (outer_loss < save_loss):
            save_loss = outer_loss
            torch.save({"epoch": i+1, "model":net.state_dict()}, "best.ckpt")
            print("Model Saved at Checkpoint")

        if (i % 50 == 0):
            tmp = sum(losses)/len(losses)
            print("Loss : %f" %(tmp))

            loss_for_graph.append(tmp)
            losses = []

    # DRAW Loss GRAPH AND SAVE INTO A FILE
    plt.plot(loss_for_graph)
    plt.xlabel("iterations / 50")
    plt.ylabel("MSE Loss")
    plt.savefig("test_loss.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAML')
    
    parser.add_argument(
        '--model_load',
        action='store_true')
    args = parser.parse_args()

    main(args)