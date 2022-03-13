import sys, os
import math
import torch.optim as optim

sys.path.append(os.path.abspath(".."))
import fss_torch


# the square Ising model
tc_true = 0.5 * math.log(1 + math.sqrt(2))
# c1_true, c2_true = (1.0, 0)
# fname = "Data/Ising2D/ising-square-B.dat"
c1_true, c2_true = (1.0, 0.125)
fname = "./Data/Ising2D/ising-square-M.dat"
# c1_true, c2_true = (1.0, -1.75)
# fname = "Data/Ising2D/ising-square-X.dat"

# Dataset
dataset = fss_torch.fss.Dataset.fromFile(fname=fname)
# Transformer
rtc, rc1, rc2 = 0.97, 0.9, 0.9
initial_values = [dataset.transform_t(tc_true * rtc), c1_true * rc1, c2_true * rc2]
transform = fss_torch.fss.Transform(initial_values)
# Model
model = fss_torch.nsa_util.MLP(hidden_sizes=[50, 50])
# Optimizer
optimizer = optim.Adam(params=model.parameters())
optimizer.add_param_group({"params": transform.parameters(), "lr": 0.01})

# Doing FSS by NN
tc, c1, c2 = fss_torch.nsa_util.do_fss(dataset, model, optimizer, transform)
# Results
print(
    "%g %g %g %g %g %g"
    % (
        dataset.inv_transform_t(tc),
        c1,
        c2,
        dataset.inv_transform_t(initial_values[0]),
        initial_values[1],
        initial_values[2],
    ),
    flush=True,
)
