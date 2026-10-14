from task1_banknote_classification import task1_banknote_classification
from task2_decision_boundary import task2_decision_boundary
from task3_feature_pairs import task3_feature_pairs
from task4_sample_order import task4_sample_order
from task5_borderline import task5_borderline
from task6_xor import task6_xor
from task7_feature_removal import task7_feature_removal
from task8_report import task8_report
from common import FEATURES

def ask_feature(prompt):
    while True:
        s = input(prompt).strip()
        if s == '':
            return None
        if s.isdigit() and 0 <= int(s) < 4:
            return int(s)
        print("Enter a number 0-3 (or press Enter for default).")

def menu():
    while True:
        print("\n===== PERCEPTRON MENU =====")
        print("1. Banknote Classification using Perceptron")
        print("2. Decision Boundary Explorer")
        print("3. Feature Pair Separability Challenge")
        print("4. Training Sample Order Challenge")
        print("5. Borderline Sample Detector")
        print("6. XOR Limitation Demonstrator")
        print("7. Perceptron Feature Removal Challenge")
        print("8. Perceptron Classification Report")
        print("9. Exit")
        ch = input("Enter your choice (1-9): ").strip()
        if ch == '1':
            task1_banknote_classification()
        elif ch == '2':
            print("Features: " + ", ".join(f"{i}={f}" for i, f in enumerate(FEATURES)))
            f1 = ask_feature("First feature index (Enter = 0): ")
            f2 = ask_feature("Second feature index (Enter = 1): ")
            f1 = 0 if f1 is None else f1
            f2 = 1 if f2 is None else f2
            if f1 == f2:
                print("Choose two different features.")
            else:
                task2_decision_boundary(f1, f2)
        elif ch == '3':
            task3_feature_pairs()
        elif ch == '4':
            task4_sample_order()
        elif ch == '5':
            task5_borderline()
        elif ch == '6':
            task6_xor()
        elif ch == '7':
            task7_feature_removal()
        elif ch == '8':
            task8_report()
        elif ch == '9':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == '__main__':
    menu()
