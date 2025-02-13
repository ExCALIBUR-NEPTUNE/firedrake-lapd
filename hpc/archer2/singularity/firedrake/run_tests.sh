#!/bin/bash

. "/home/firedrake/firedrake/bin/activate"
cd "/home/firedrake/firedrake/src/firedrake/" || exit 2
python3 -m pytest -v tests/firedrake/regression/ -k "poisson_strong or stokes_mini or dg_advection"