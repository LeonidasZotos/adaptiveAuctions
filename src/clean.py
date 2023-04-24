import os

def run(args):
    # delete all files in the results folder
    for file in os.listdir("results"):
        os.remove(os.path.join("results", file))
    
    print("Clean Completed")