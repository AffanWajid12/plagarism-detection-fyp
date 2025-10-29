# Car Rental System - Basic Version

cars = {
    "Toyota": {"count": 3, "rate": 80},
    "Honda": {"count": 2, "rate": 100},
    "Suzuki": {"count": 4, "rate": 60}
}

rented_cars = {}

def display_cars():
    print("\nAvailable Cars:")
    for name, data in cars.items():
        print(f"{name} - {data['count']} available at ${data['rate']} per day")

def rent_car():
    name = input("Enter car name to rent: ")
    if name in cars and cars[name]["count"] > 0:
        days = int(input("Enter number of days: "))
        cost = cars[name]["rate"] * days
        cars[name]["count"] -= 1
        rented_cars[name] = days
        print(f"You rented {name} for {days} days. Total cost: ${cost}")
    else:
        print("Car not available!")

def return_car():
    name = input("Enter car name to return: ")
    if name in rented_cars:
        cars[name]["count"] += 1
        del rented_cars[name]
        print(f"{name} returned successfully!")
    else:
        print("You did not rent this car!")

while True:
    print("\n1. Display Cars\n2. Rent Car\n3. Return Car\n4. Exit")
    ch = input("Enter your choice: ")
    if ch == "1":
        display_cars()
    elif ch == "2":
        rent_car()
    elif ch == "3":
        return_car()
    elif ch == "4":
        print("Goodbye!")
        break
    else:
        print("Invalid choice!")
