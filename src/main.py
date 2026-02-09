from tests import *

if __name__ == "__main__":
    try:
        number_of_test = int(input("Enter the test number: "))
        func_name = f"test{number_of_test}"
        if func_name in globals():
            globals()[func_name]()
        else:
            print(f"Test {number_of_test} is not defined.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
