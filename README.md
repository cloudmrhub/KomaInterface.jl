# KomaInterface.jl

This package contains the KomaMRI interface used in CloudMR python code. Additionally, this package contains a parser for mtrk files for KomaMRI.

## Reading .mtrk sequences

In order to use the new mtrk parser, simply replace `read_seq` with `KomaInterface.read_seq`.
