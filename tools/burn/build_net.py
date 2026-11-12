"""Build the 13-isotope alpha-chain network (like aprox13) with pynucastro and write it as a
Python module, aprox13.py, next to this file.

The chain he4, c12, o16, ..., ni56 with the alpha captures and their reverses, C12 + C12,
C12 + O16 and O16 + O16, and each (a,p)(p,g) pair folded into an effective (a,g) through the
intermediate nucleus, which is then dropped (Timmes et al. 2000).

    tools/burn/.venv/Scripts/python tools/burn/build_net.py
"""
from pathlib import Path

import pynucastro as pyna

HERE = Path(__file__).parent
CHAIN = ["he4", "c12", "o16", "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "fe52", "ni56"]
INTERMEDIATE = ["al27", "p31", "cl35", "k39", "sc43", "v47", "mn51", "co55"]


def build():
    rl = pyna.ReacLibLibrary()
    lib = rl.linking_nuclei(["p"] + CHAIN + INTERMEDIATE, with_reverse=True)
    net = pyna.PythonNetwork(libraries=[lib])
    net.make_ap_pg_approx(intermediate_nuclei=INTERMEDIATE)
    net.remove_nuclei(INTERMEDIATE)
    # Rates that still make or use free protons (heavy-ion branches) are dropped with them.
    net.remove_nuclei(["p"])
    return net


if __name__ == "__main__":
    net = build()
    print(len(net.unique_nuclei), "nuclei:", " ".join(str(n) for n in net.unique_nuclei))
    print(len(net.rates), "rates")
    for r in net.rates:
        print("  ", r)
    net.write_network(str(HERE / "aprox13.py"))
