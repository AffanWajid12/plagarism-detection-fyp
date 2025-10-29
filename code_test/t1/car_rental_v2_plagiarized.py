# Car Hire Management (Plagiarized Version)

vehicles = {
    "Toyota": {"available": 3, "price": 80},
    "Honda": {"available": 2, "price": 100},
    "Suzuki": {"available": 4, "price": 60}
}

borrowed = {}

def show_vehicles():
    print("\nCars You Can Rent:")
    for model, details in vehicles.items():
        print(f"{model} - {details['available']} left at ${details['price']} per day")

def hire_vehicle():
    model = input("Which car would you like to hire? ")
    if model in vehicles and vehicles[model]["available"] > 0:
        days = int(input("Days of rental: "))
        bill = vehicles[model]["price"] * days
        vehicles[model]["available"] -= 1
        borrowed[model] = days
        print(f"You hired {model} for {days} days. Bill: ${bill}")
    else:
        print("Sorry, not available!")

def give_back():
    model = input("Enter name of car to return: ")
    if model in borrowed:
        vehicles[model]["available"] += 1
        del borrowed[model]
        print(f"{model} successfully returned.")
    else:
        print("That car was not rented!")

while True:
    print("\n1. Show Cars\n2. Rent\n3. Return\n4. Quit")
    opt = input("Enter choice: ")
    if opt == "1":
        show_vehicles()
