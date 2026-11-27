"""Build the 13-isotope alpha-chain networks with pynucastro and write them as Python modules next
to this file.

aprox13.py, the reference network that burner.py integrates: pynucastro's aprox13 (Timmes et al.
2000). The chain he4, c12, o16, ..., ni56 with the alpha captures and their reverses, each
(a,p)(p,g) pair folded into an effective (a,g) through the intermediate nucleus, and C12 + C12,
C12 + O16 and O16 + O16 with both of their exits: the alpha exit, and the proton exit folded
through Na23, Al27 or P31 (the proton is captured again at once, by (p,a) or (p,g)). The proton
exits carry 45%, about 60% and about 65% of the three totals at 1-4e9 K.

aprox13_features.py, the network the surrogate's inputs are read from (train.reaclib_sets, the
REACLIB fits of the rate terms that the surrogate takes as inputs, also in src/burn/network.ts):
the first reference, which kept only the alpha exits of the three heavy-ion reactions and
dropped the protons with the intermediate nuclei. The inputs stay those of the shipped network;
a rate term that leaves out a near-constant share of its total is still a good input.

    tools/burn/.venv/Scripts/python tools/burn/build_net.py
"""
from pathlib import Path

import pynucastro as pyna
from pynucastro.networks.common_networks import aprox13

HERE = Path(__file__).parent
CHAIN = ["he4", "c12", "o16", "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "fe52", "ni56"]
INTERMEDIATE = ["al27", "p31", "cl35", "k39", "sc43", "v47", "mn51", "co55"]


def build():
    """The reference: pynucastro's aprox13."""
    return aprox13()


def build_features():
    """The first reference, whose rate terms are the surrogate's inputs."""
    rl = pyna.ReacLibLibrary()
    lib = rl.linking_nuclei(["p"] + CHAIN + INTERMEDIATE, with_reverse=True)
    net = pyna.PythonNetwork(libraries=[lib])
    net.make_ap_pg_approx(intermediate_nuclei=INTERMEDIATE)
    net.remove_nuclei(INTERMEDIATE)
    # Rates that still make or use free protons (the heavy-ion proton exits) go with them.
    net.remove_nuclei(["p"])
    return net


if __name__ == "__main__":
    for name, net in (("aprox13.py", build()), ("aprox13_features.py", build_features())):
        print(name, len(net.unique_nuclei), "nuclei:", " ".join(str(n) for n in net.unique_nuclei))
        print(len(net.rates), "rates")
        for r in net.rates:
            print("  ", r.fname)
        net.write_network(str(HERE / name))
