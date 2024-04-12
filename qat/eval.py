import argparse


def main():
    pass
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="evaluate model checkpoints")
    parser.add_argument("--model_checkpoint", type=str, optional=False)
    parser.add_argument("--dataset_path", type=str, defualt="../dataset")
    
    
