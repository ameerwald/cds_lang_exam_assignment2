def input_parse(): 
    #initialize the parser
    parser = argparse.ArgumentParser() 
    # add arguments
    parser.add_argument("--filename", type=str, default="Beautiful") # if no name arg, will return default 
    parser.add_argument("--age", type=int, required=True) # not a default but is required 
    # parse the arguments from the command line 
    args = parser.parse_args()
    return args # need this, it acts similiar to print


    # add arguments
    parser.add_argument("--filename", type = str, required=True) # loading in the data/filename 
    parser.add_argument("--hidden_layer_sizes", type = tuple, default = (20,)) # hidden layer sizes
    parser.add_argument("--max_iter", type = int, default = 1000) # max iterations
    parser.add_argument("--random_state", type = int, default = 69) # random state
    # parse the arguments from command line
    args = parser.parse_args()
    # get the variables
    return args