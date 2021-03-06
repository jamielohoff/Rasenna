import sys, getopt
from boundary_extraction import modify_full_cremi

if __name__ == "__main__":
    print(sys.argv[1:])
    modify_full_cremi(sys.argv[1:])
