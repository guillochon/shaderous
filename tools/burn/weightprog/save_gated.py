"""Program round 30 with the tight Si/S rate gate (strength 0.6) and save it as an ordinary Burner
checkpoint, every hidden layer padded with zero units to the widest (144)."""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train
from weightprog import program
import try_gate
base = train.load_model("model_dagger_30.pt")
m = program(base, try_gate.PROGRAMS["qt06s"])
L = [x for x in m.net if isinstance(x, nn.Linear)]
W = max(l.out_features for l in L[:-1])
out = train.Burner(W, len(L) - 1, base.arch).to(train.dev)
O = [x for x in out.net if isinstance(x, nn.Linear)]
with torch.no_grad():
    for a, b in zip(L, O):
        b.weight.zero_(); b.bias.zero_()
        o, i = a.weight.shape
        b.weight[:o, :i] = a.weight
        b.bias[:o] = a.bias
    # padded units are silu(0) = 0 and read by nothing: the outputs are unchanged
    x = train.features(torch.full((5,), 7.5, device=train.dev), torch.full((5,), 9.55, device=train.dev),
                       torch.full((5,), -3.0, device=train.dev), torch.softmax(torch.randn(5, 13, device=train.dev), 1), "full")
    print("max output difference after padding:", (m.net(x) - out.net(x)).abs().max().item())
torch.save({"state": out.state_dict(), "width": W, "depth": len(L) - 1, "arch": base.arch}, train.HERE / "model_dagger_30_gated.pt")
print("wrote model_dagger_30_gated.pt, width", W)
