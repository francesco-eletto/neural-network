
#building a single node neural network able to learn the y = 2x linear function
#giving him only a single training entry of input x = 2 and target 4 repeated
#until the error is acceptable

#we will suppose the b offset to be 0 to simplify the model

x = 2
target = 4
lr = 0.1 #learning rate
weight = 0.5

y = 0
err = 0
grad = 0

    
def menu_print():
    print("\n------ menù ------")
    print(f"node weight: {weight}")
    print(f"learning rate: {lr}")
    print("\n")
    print(f"commands: \n0 - Train model")
    print(f"1 - Test model")
    print(f"2 - Reset weight")
    print(f"3 - Change lr")
    print(f"10 - Exit")

def main():
    global x, target, lr, weight, y, err, grad
    
    while True:
        menu_print()
        line = input("Type your choice: ").strip()
        
        if not line:
            continue

        try:
            choice = int(line)
        except ValueError:
            print("Error! Please insert a number")
            continue
        
        if choice == 0:
            print("\n------ Train ------")
            
            count = 0
            
            while(True):
                line = input("write a tuple es. x,target where x is the test entry input (nothing to exit): ")
                try:
                    if(line is ''): break
                    
                    (x, target) = line.strip().split(",")
                    x = float(x)
                    target = float(target)
                    while True:
                        try:
                            line = input("Insert the number of training cycles (nothing to exit): ")
                            if(line is ''): break
                            count = int(line)
                            break
                        except:
                            print("Error! Please insert a valid value")
                            continue
                        
                    break
                except:
                    print("Error! Please insert a valid value")
                    continue
            
            if(line is ''): continue
            
            for i in range(count):
                y = weight * x
                err = y - target
                grad = err * x
                weight = weight - (lr * grad)
                print(f"Iteration {i} - y: {y} | err: {err} | grad: {grad} | new_weight: {weight}")
            
        elif choice == 1:
            print(f"\n------ Test ------")
            while True:
                try:
                    line = input("Type your input (nothing to exit): ")
                    if(line is ''): break
                    test_x = float(line)
                    break
                except:
                    print("Error! Please insert a valid value")
                    continue
           
            if(line is ''): continue
            
            y = weight * test_x
            
            print(f"Result: {y}")
        elif choice == 2:
            weight = 0.5
        elif choice == 3:
            while True:
                try:
                    line = input("Enter new learning rate (nothing to exit): ")
                    if(line is ''): break
                    lr = float(line)
                    break
                except:
                    print("Error! Please insert a valid value")
                    continue
           
            if(line is ''): continue
            
            
        elif choice == 10: 
            break
        else:
            print("Command not found!")

if __name__ == "__main__":
    main()

